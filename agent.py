"""
프로젝트-AI-kha-gi: env 생성 + 관측 설계 + PPO 리그 학습 (예선~본선) + 시즌(episode) 반복

1 episode =
    1) Elo 기반 예선 1회 (리그 스크림)
    2) 본선 1회 (Swiss + Knockout, 롤드컵 포맷)
    3) 챔피언 팀 추출
    4) 챔피언 모델 파일 저장
    5) 챔피언 기반 초기 파라미터로 모든 팀 재초기화

이 episode를 원하는 횟수(예: 8000번) 반복.
"""

import os
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import gymnasium as gym
import kymnasium  # env 등록용
import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# 0. 기본 설정 / 전역 상수
# ================================================================
N_STONES = 3
N_OBS = 3

BOARD_W = 600.0
BOARD_H = 600.0

STATE_DIM = 52  # encode_state_fe_alkkagi 출력 차원

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# 1. 환경 생성
# ================================================================
def make_env():
    env = gym.make(
        "kymnasium/AlKkaGi-3x3-v0",
        obs_type="custom",
        render_mode=None,
        bgm=False,
    )
    return env


# ================================================================
# 2. 관측 설계 (Feature Engineering)
#   - 여기서 "플레이어 시점" 좌표계 적용:
#     * my_color == 1(백)일 때, y 방향 뒤집기
#     * 장애물도 y+h 기준으로 미러링
# ================================================================
def split_me_opp(obs, my_color: int):
    """
    obs: env에서 받은 dict
      - obs["black"] : (3, 3) = [x, y, alive]
      - obs["white"] : (3, 3)
      - obs["obstacles"]: (3, 4) = [x, y, w, h]
      - obs["turn"]  : 0 (흑 차례) or 1 (백 차례)

    my_color: 0=흑, 1=백

    return:
      me        : 내 돌 (3, 3)
      opp       : 상대 돌 (3, 3)
      obstacles : 장애물 (3, 4)
      turn      : float(0.0 or 1.0)  (env 기준 턴)
    """
    if my_color == 0:  # 내가 흑
        me = np.array(obs["black"], dtype=np.float32)
        opp = np.array(obs["white"], dtype=np.float32)
    else:              # 내가 백
        me = np.array(obs["white"], dtype=np.float32)
        opp = np.array(obs["black"], dtype=np.float32)

    obstacles = np.array(obs["obstacles"], dtype=np.float32)
    turn = float(obs["turn"])
    return me, opp, obstacles, turn


def normalize_stones(stones, board_w, board_h):
    """
    stones: (N_STONES, 3) = [x, y, alive]
    x, y를 [0,1]로 정규화 + clip
    """
    out = stones.copy()
    out[:, 0] /= board_w  # x
    out[:, 1] /= board_h  # y
    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
    return out


def normalize_obstacles(obs_arr, board_w, board_h):
    """
    obs_arr: (N_OBS, 4) = [x, y, w, h]
    x, y, w, h를 [0,1] 근처 값으로 정규화 + clip
    """
    out = obs_arr.copy()
    out[:, 0] /= board_w  # x
    out[:, 1] /= board_h  # y
    out[:, 2] /= board_w  # w
    out[:, 3] /= board_h  # h

    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
    out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)
    out[:, 3] = np.clip(out[:, 3], 0.0, 1.0)
    return out


def get_player_centric_norms(
    obs,
    my_color: int,
    board_w: float,
    board_h: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    플레이어 시점 정규화:
      - 흑(my_color=0): 원래 좌표 사용
      - 백(my_color=1): y축을 보드 중앙 기준으로 뒤집어서,
        항상 "내가 아래"에 있는 것처럼 보이게 함.
    """
    me, opp, obstacles, turn = split_me_opp(obs, my_color)

    me_norm = normalize_stones(me, board_w, board_h)          # (3, 3)
    opp_norm = normalize_stones(opp, board_w, board_h)        # (3, 3)
    obs_norm = normalize_obstacles(obstacles, board_w, board_h)  # (3, 4)

    if my_color == 1:
        # 돌: y -> 1 - y
        me_norm[:, 1] = 1.0 - me_norm[:, 1]
        opp_norm[:, 1] = 1.0 - opp_norm[:, 1]

        # 장애물: top-left y와 height를 이용해서 전체 박스를 수직 미러링
        y = obs_norm[:, 1]
        h = obs_norm[:, 3]
        # 기존 박스: [y, y + h] -> 뒤집으면 [1 - (y + h), 1 - y]
        obs_norm[:, 1] = 1.0 - (y + h)
        obs_norm[:, 1] = np.clip(obs_norm[:, 1], 0.0, 1.0)

    return me_norm, opp_norm, obs_norm, float(turn)


def encode_state_basic_alkkagi(
    obs,
    my_color: int,
    board_w: float,
    board_h: float,
) -> np.ndarray:
    """
    Baseline 관측 인코더 (플레이어 시점 좌표계 포함)
    - turn (1)
    - me_norm  : 3 * (x, y, alive) = 9
    - opp_norm : 3 * (x, y, alive) = 9
    - obs_norm : 3 * (x, y, w, h) = 12

    => 최종 shape: (31,)
    """
    me_norm, opp_norm, obs_norm, turn = get_player_centric_norms(
        obs, my_color, board_w, board_h
    )

    feat = np.concatenate(
        [
            np.array([turn], dtype=np.float32),  # (1,)
            me_norm.flatten(),                   # (9,)
            opp_norm.flatten(),                  # (9,)
            obs_norm.flatten(),                  # (12,)
        ]
    ).astype(np.float32)

    return feat


def group_stats(stones_norm: np.ndarray) -> np.ndarray:
    """
    stones_norm: (N_STONES, 3) = [x_norm, y_norm, alive]
    alive 돌만 사용해서:
      - center_x, center_y
      - var_x, var_y
    반환 shape: (4,)
    """
    alive_mask = stones_norm[:, 2] > 0.5
    if not np.any(alive_mask):
        return np.zeros(4, dtype=np.float32)

    xs = stones_norm[alive_mask, 0]
    ys = stones_norm[alive_mask, 1]

    cx = xs.mean()
    cy = ys.mean()
    var_x = xs.var()
    var_y = ys.var()

    return np.array([cx, cy, var_x, var_y], dtype=np.float32)


def min_edge_dist(stones_norm: np.ndarray) -> float:
    """
    alive 돌들 중에서 보드 엣지까지의 최소 거리.
    좌표는 [0,1] 기준이라고 가정.
    """
    alive_mask = stones_norm[:, 2] > 0.5
    if not np.any(alive_mask):
        return 0.0

    xs = stones_norm[alive_mask, 0]
    ys = stones_norm[alive_mask, 1]

    edge_dists = np.minimum.reduce([xs, 1.0 - xs, ys, 1.0 - ys])
    return float(edge_dists.min())


def min_pairwise_dist(me_norm: np.ndarray, opp_norm: np.ndarray) -> float:
    """
    alive인 내 돌 vs alive인 상대 돌 사이의 유클리드 거리 중 최소값.
    """
    me_alive = me_norm[me_norm[:, 2] > 0.5]
    opp_alive = opp_norm[opp_norm[:, 2] > 0.5]

    if me_alive.size == 0 or opp_alive.size == 0:
        return 0.0

    min_d = 1e9
    for a in me_alive:
        for b in opp_alive:
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < min_d:
                min_d = d

    return float(min_d)


def obstacle_summary(obs_norm: np.ndarray) -> np.ndarray:
    """
    obs_norm: (N_OBS, 4) = [x_norm, y_norm, w_norm, h_norm]

    - count
    - center_x_mean, center_y_mean
    - w_mean, h_mean

    반환 shape: (5,)
    """
    if obs_norm.size == 0:
        return np.zeros(5, dtype=np.float32)

    cx = obs_norm[:, 0] + obs_norm[:, 2] / 2.0
    cy = obs_norm[:, 1] + obs_norm[:, 3] / 2.0
    w = obs_norm[:, 2]
    h = obs_norm[:, 3]

    cnt = float(obs_norm.shape[0])
    return np.array(
        [cnt, cx.mean(), cy.mean(), w.mean(), h.mean()],
        dtype=np.float32,
    )


def encode_state_fe_alkkagi(
    obs,
    my_color: int,
    board_w: float,
    board_h: float,
) -> np.ndarray:
    """
    Feature engineering 버전 state encoder (플레이어 시점 좌표계).

    - baseline feature: 31차원
    - 추가 feature:
        * turn_is_me, my_alive_cnt, opp_alive_cnt, alive_diff, alive_ratio (5)
        * my_center_x, my_center_y, my_var_x, my_var_y (4)
        * op_center_x, op_center_y, op_var_x, op_var_y (4)
        * my_min_edge, op_min_edge, min_my_op_dist (3)
        * obs_cnt, obs_cx_mean, obs_cy_mean, obs_w_mean, obs_h_mean (5)

      => 추가 21차원
      => 총 31 + 21 = 52차원
    """
    # --- 1) baseline feature (31차원) ---
    base_feat = encode_state_basic_alkkagi(obs, my_color, board_w, board_h)

    # --- 2) 플레이어 시점 정규화된 돌/장애물 ---
    me_norm, opp_norm, obs_norm, turn_raw = get_player_centric_norms(
        obs, my_color, board_w, board_h
    )

    # --- 3) scalar feature들 ---
    my_alive_cnt = float((me_norm[:, 2] > 0.5).sum())
    opp_alive_cnt = float((opp_norm[:, 2] > 0.5).sum())
    alive_diff = my_alive_cnt - opp_alive_cnt
    denom = my_alive_cnt + opp_alive_cnt
    alive_ratio = my_alive_cnt / denom if denom > 0 else 0.0

    # env 기준 턴(0=흑,1=백)과 my_color 비교
    turn_is_me = 1.0 if int(turn_raw) == int(my_color) else 0.0

    scalar_feats = np.array(
        [turn_is_me, my_alive_cnt, opp_alive_cnt, alive_diff, alive_ratio],
        dtype=np.float32,
    )

    # --- 4) 그룹 요약 (무게중심 + 분산) ---
    my_stats = group_stats(me_norm)   # (4,)
    op_stats = group_stats(opp_norm)  # (4,)

    # --- 5) 관계 feature ---
    my_min_edge = min_edge_dist(me_norm)
    op_min_edge = min_edge_dist(opp_norm)
    min_my_op = min_pairwise_dist(me_norm, opp_norm)

    relation_feats = np.array(
        [my_min_edge, op_min_edge, min_my_op],
        dtype=np.float32,
    )

    # --- 6) 장애물 요약 ---
    obs_stats = obstacle_summary(obs_norm)  # (5,)

    # --- 7) 전부 concat ---
    extra_feats = np.concatenate(
        [
            scalar_feats,   # 5
            my_stats,       # 4
            op_stats,       # 4
            relation_feats, # 3
            obs_stats,      # 5
        ]
    ).astype(np.float32)

    feat = np.concatenate([base_feat, extra_feats]).astype(np.float32)

    return feat  # (52,)


# ================================================================
# 3. 보상 설계
# ================================================================
def get_alive_counts(obs) -> Tuple[int, int]:
    """obs에서 흑/백 살아있는 돌 개수 반환"""
    black_alive = sum(1 for s in obs["black"] if s[2] > 0.5)
    white_alive = sum(1 for s in obs["white"] if s[2] > 0.5)
    return black_alive, white_alive


def compute_team_reward(prev_obs, next_obs, my_color: int,
                        terminated: bool, truncated: bool) -> float:
    """
    팀 관점 shaping reward.
    - 상대 돌이 줄어들면 +1
    - 내 돌이 줄어들면 -1
    - 게임 종료 시:
        * 내가 돌을 더 많이 가지고 있으면 +5
        * 내가 더 적으면 -5
    """
    prev_black, prev_white = get_alive_counts(prev_obs)
    next_black, next_white = get_alive_counts(next_obs)

    if my_color == 0:
        my_prev, my_next = prev_black, next_black
        opp_prev, opp_next = prev_white, next_white
    else:
        my_prev, my_next = prev_white, next_white
        opp_prev, opp_next = prev_black, next_black

    delta_my = my_next - my_prev
    delta_opp = opp_prev - opp_next

    r = 0.0
    r += 1.0 * delta_opp   # 상대 돌 줄어들면 +
    r += 1.0 * delta_my    # 내 돌 줄어들면 - (delta_my<0일 때)

    if terminated or truncated:
        if my_next > opp_next:
            r += 5.0
        elif my_next < opp_next:
            r -= 5.0

    return float(r)


# ================================================================
# 4. PPO용 Actor-Critic 모델
# ================================================================
class ActorCritic(nn.Module):
    """
    입력: state (52차원)
    출력:
      - stone_logits: (batch, 3)
      - power_mean/log_std: (batch, 1) each
      - angle_mean/log_std: (batch, 1) each
      - value: (batch, 1)
    """
    def __init__(self, state_dim=STATE_DIM, n_stones=N_STONES, hidden_dim=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # discrete stone head
        self.stone_head = nn.Linear(hidden_dim, n_stones)

        # continuous heads
        self.power_mean = nn.Linear(hidden_dim, 1)
        self.power_log_std = nn.Parameter(torch.zeros(1))

        self.angle_mean = nn.Linear(hidden_dim, 1)
        self.angle_log_std = nn.Parameter(torch.zeros(1))

        # critic
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.body(x)
        stone_logits = self.stone_head(h)
        power_mean = self.power_mean(h)
        angle_mean = self.angle_mean(h)
        value = self.value_head(h)
        return stone_logits, power_mean, angle_mean, value


# ================================================================
# 5. PPO 버퍼 및 팀 에이전트
# ================================================================
@dataclass
class PPOBuffer:
    states: List[np.ndarray] = field(default_factory=list)
    stone_idx: List[int] = field(default_factory=list)
    power: List[float] = field(default_factory=list)
    angle: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def clear(self):
        self.states.clear()
        self.stone_idx.clear()
        self.power.clear()
        self.angle.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.rewards)


class TeamAgent:
    """
    롤드컵 팀 = 독립 PPO 에이전트
    """
    def __init__(
        self,
        team_id: int,
        state_dim: int = STATE_DIM,
        n_stones: int = N_STONES,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.team_id = team_id
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr

        self.model = ActorCritic(state_dim=state_dim, n_stones=n_stones).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.buffer = PPOBuffer()

        # Elo 초기값
        self.elo = 1500.0

    # ------------ 정책에서 액션 샘플 & log_prob 계산 ------------

    def _sample_action_from_dist(
        self,
        stone_logits: torch.Tensor,
        power_mean: torch.Tensor,
        angle_mean: torch.Tensor,
    ):
        """
        각 분포에서 샘플 + log_prob 계산.
        """
        # Categorical for stone
        stone_dist = torch.distributions.Categorical(logits=stone_logits)
        stone_action = stone_dist.sample()
        stone_log_prob = stone_dist.log_prob(stone_action)

        # Gaussian for power
        power_std = torch.exp(self.model.power_log_std)
        power_dist = torch.distributions.Normal(power_mean, power_std)
        power_raw = power_dist.rsample()
        power_log_prob = power_dist.log_prob(power_raw)

        # Gaussian for angle
        angle_std = torch.exp(self.model.angle_log_std)
        angle_dist = torch.distributions.Normal(angle_mean, angle_std)
        angle_raw = angle_dist.rsample()
        angle_log_prob = angle_dist.log_prob(angle_raw)

        # action post-process: tanh → [-1,1]
        power = torch.tanh(power_raw)
        angle = torch.tanh(angle_raw)

        # log_prob 합산 (continuous 1D라 squeeze)
        log_prob = (
            stone_log_prob
            + power_log_prob.squeeze(-1)
            + angle_log_prob.squeeze(-1)
        )

        return (
            stone_action.squeeze(-1),
            power.squeeze(-1),
            angle.squeeze(-1),
            log_prob,
            power_raw.squeeze(-1),
            angle_raw.squeeze(-1),
        )

    def act(self, state: np.ndarray, eval_mode: bool = False):
        """
        state: np.ndarray (52,)
        return:
          stone_idx (int), power_raw(float), angle_raw(float), log_prob(float), value(float)
          - power_raw, angle_raw ∈ R (Gaussian 샘플) → tanh 후 env에 사용
        """
        state_t = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        stone_logits, power_mean, angle_mean, value = self.model(state_t)

        if eval_mode:
            with torch.no_grad():
                stone_idx = stone_logits.argmax(dim=-1)
                power_raw = power_mean
                angle_raw = angle_mean

                power_std = torch.exp(self.model.power_log_std)
                angle_std = torch.exp(self.model.angle_log_std)
                stone_dist = torch.distributions.Categorical(logits=stone_logits)
                stone_log_prob = stone_dist.log_prob(stone_idx)
                power_dist = torch.distributions.Normal(power_mean, power_std)
                angle_dist = torch.distributions.Normal(angle_mean, angle_std)
                power_log_prob = power_dist.log_prob(power_raw)
                angle_log_prob = angle_dist.log_prob(angle_raw)
                log_prob = (
                    stone_log_prob
                    + power_log_prob.squeeze(-1)
                    + angle_log_prob.squeeze(-1)
                )
        else:
            (
                stone_idx,
                power,   # tanh 후 값이지만 여기선 안 씀
                angle,   # tanh 후 값이지만 여기선 안 씀
                log_prob,
                power_raw,
                angle_raw,
            ) = self._sample_action_from_dist(stone_logits, power_mean, angle_mean)

        stone_idx = int(stone_idx.item())
        power_raw = float(power_raw.item())
        angle_raw = float(angle_raw.item())
        log_prob = float(log_prob.item())
        value = float(value.item())
        return stone_idx, power_raw, angle_raw, log_prob, value

    # ------------ 버퍼 기록 & PPO 업데이트 ------------

    def store_transition(
        self,
        state: np.ndarray,
        stone_idx: int,
        power_raw: float,
        angle_raw: float,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
    ):
        self.buffer.states.append(state)
        self.buffer.stone_idx.append(stone_idx)
        self.buffer.power.append(power_raw)
        self.buffer.angle.append(angle_raw)
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)

    def _compute_gae(self):
        rewards = np.array(self.buffer.rewards, dtype=np.float32)
        values = np.array(self.buffer.values + [0.0], dtype=np.float32)
        dones = np.array(self.buffer.dones, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = (
                rewards[t]
                + self.gamma * values[t + 1] * (1.0 - dones[t])
                - values[t]
            )
            gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, batch_size: int = 64, epochs: int = 4):
        if len(self.buffer) == 0:
            return

        states = torch.tensor(
            np.array(self.buffer.states), dtype=torch.float32, device=DEVICE
        )
        stone_idx = torch.tensor(
            self.buffer.stone_idx, dtype=torch.long, device=DEVICE
        )
        power_raw = torch.tensor(
            self.buffer.power, dtype=torch.float32, device=DEVICE
        ).unsqueeze(-1)
        angle_raw = torch.tensor(
            self.buffer.angle, dtype=torch.float32, device=DEVICE
        ).unsqueeze(-1)
        old_log_probs = torch.tensor(
            self.buffer.log_probs, dtype=torch.float32, device=DEVICE
        )

        advantages, returns = self._compute_gae()
        advantages = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                batch_states = states[idx]
                batch_stone = stone_idx[idx]
                batch_power_raw = power_raw[idx]
                batch_angle_raw = angle_raw[idx]
                batch_old_logp = old_log_probs[idx]
                batch_adv = advantages[idx]
                batch_ret = returns[idx]

                stone_logits, power_mean, angle_mean, value_pred = self.model(
                    batch_states
                )

                # 새 분포
                stone_dist = torch.distributions.Categorical(logits=stone_logits)
                power_std = torch.exp(self.model.power_log_std)
                angle_std = torch.exp(self.model.angle_log_std)

                power_dist = torch.distributions.Normal(power_mean, power_std)
                angle_dist = torch.distributions.Normal(angle_mean, angle_std)

                # log_prob 재계산 (raw 값 기준)
                log_prob_stone = stone_dist.log_prob(batch_stone)
                log_prob_power = power_dist.log_prob(batch_power_raw).squeeze(-1)
                log_prob_angle = angle_dist.log_prob(batch_angle_raw).squeeze(-1)

                log_prob = log_prob_stone + log_prob_power + log_prob_angle

                ratio = torch.exp(log_prob - batch_old_logp)

                surr1 = ratio * batch_adv
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss
                value_pred = value_pred.squeeze(-1)
                critic_loss = F.mse_loss(value_pred, batch_ret)

                # entropy bonus
                entropy = (
                    stone_dist.entropy().mean()
                    + power_dist.entropy().mean()
                    + angle_dist.entropy().mean()
                )

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()

    # ------------ 챔피언 기반 재초기화 ------------

    def reset_from_champion(self, champion_state_dict: Dict[str, torch.Tensor],
                            noise_scale: float = 0.01,
                            copy_exact: bool = False):
        """
        챔피언의 파라미터를 기반으로 이 에이전트 초기화.
        copy_exact=True 이면 그대로 복사, False면 약간의 노이즈 추가.
        """
        # deepcopy + load
        copied = {k: v.clone() for k, v in champion_state_dict.items()}
        self.model.load_state_dict(copied)

        if not copy_exact:
            with torch.no_grad():
                for p in self.model.parameters():
                    p.add_(torch.randn_like(p) * noise_scale)

        # 옵티마도 리셋
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # 버퍼/elo 리셋은 League 쪽에서 처리


# ================================================================
# 6. 리그 / 스위스 / 토너먼트 스케줄러
# ================================================================
@dataclass
class MatchResult:
    winner_id: Optional[int]  # None이면 무승부
    loser_id: Optional[int]


class League:
    """
    K개 팀으로 구성된 리그.
    예선(Elo 기반 리그) + 본선(Swiss + 토너먼트)을 모두 관리.
    """
    def __init__(self, num_teams: int, env=None):
        self.num_teams = num_teams
        self.teams: List[TeamAgent] = [
            TeamAgent(team_id=i) for i in range(num_teams)
        ]
        self.env = env or make_env()

        # 본선용 Swiss 상태
        self.swiss_wins: Dict[int, int] = {}
        self.swiss_losses: Dict[int, int] = {}

    # ---------- ELO 헬퍼 ----------

    def _expected_score(self, Ra: float, Rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((Rb - Ra) / 400.0))

    def update_elo(self, i: int, j: int, winner: Optional[int], K: float = 24.0):
        """
        winner: 팀 id 혹은 None(무승부)
        """
        team_i = self.teams[i]
        team_j = self.teams[j]

        Ra, Rb = team_i.elo, team_j.elo
        Ea = self._expected_score(Ra, Rb)
        Eb = self._expected_score(Rb, Ra)

        if winner is None:
            Sa, Sb = 0.5, 0.5
        elif winner == i:
            Sa, Sb = 1.0, 0.0
        else:
            Sa, Sb = 0.0, 1.0

        team_i.elo = Ra + K * (Sa - Ea)
        team_j.elo = Rb + K * (Sb - Eb)

    # ---------- 한 경기 실행 ----------

    def _obs_to_state(self, obs, my_color: int) -> np.ndarray:
        return encode_state_fe_alkkagi(
            obs, my_color=my_color, board_w=BOARD_W, board_h=BOARD_H
        )

    def run_match(self, team_id_a: int, team_id_b: int,
                  training: bool = True) -> MatchResult:
        """
        팀 A vs 팀 B 한 판.
        training=True이면 PPO용 데이터 버퍼에 기록.
        """
        env = self.env
        obs, info = env.reset()
        done = False
        terminated = False
        truncated = False

        # 흑/백 랜덤 배정
        if random.random() < 0.5:
            color_to_team = {0: team_id_a, 1: team_id_b}
        else:
            color_to_team = {0: team_id_b, 1: team_id_a}

        while not done:
            turn_color = int(obs["turn"])  # 0=흑, 1=백
            team_id = color_to_team[turn_color]
            agent = self.teams[team_id]

            # state 인코딩 (현재 차례 팀 기준 my_color = turn_color)
            state = self._obs_to_state(obs, my_color=turn_color)

            stone_idx, power_raw, angle_raw, log_prob, value = agent.act(
                state, eval_mode=not training
            )

            # --------- 플레이어 시점 → env 전역 좌표계로 변환 ---------
            # power: [-1,1] → [500,2000]
            power = 500.0 + (math.tanh(power_raw) + 1.0) * 0.5 * (2000.0 - 500.0)

            # angle_local: [-1,1] -> [-180,180] (플레이어 기준 각도)
            angle_local = 180.0 * math.tanh(angle_raw)

            # 백일 때 y축 대칭: angle_global = -angle_local
            if turn_color == 1:
                angle = -angle_local
            else:
                angle = angle_local
            # -------------------------------------------------------

            # 선택한 돌이 죽었으면 살아있는 돌 중 하나로 바꿔주기 (간단 처리)
            stones = obs["black"] if turn_color == 0 else obs["white"]
            if stones[stone_idx][2] <= 0.5:
                alive_indices = [i for i, s in enumerate(stones) if s[2] > 0.5]
                if alive_indices:
                    stone_idx = random.choice(alive_indices)

            action = {
                "turn": turn_color,
                "index": int(stone_idx),
                "power": float(power),
                "angle": float(angle),
            }

            next_obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            reward = compute_team_reward(
                prev_obs=obs,
                next_obs=next_obs,
                my_color=turn_color,
                terminated=terminated,
                truncated=truncated,
            )

            if training:
                agent.store_transition(
                    state=state,
                    stone_idx=stone_idx,
                    power_raw=power_raw,
                    angle_raw=angle_raw,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=done,
                )

            obs = next_obs

        # 승패 판정 (흑/백 관점 → 팀 id로 변환)
        black_alive, white_alive = get_alive_counts(obs)
        if black_alive > white_alive:
            winner_team = color_to_team[0]
            loser_team = color_to_team[1]
        elif white_alive > black_alive:
            winner_team = color_to_team[1]
            loser_team = color_to_team[0]
        else:
            winner_team = None
            loser_team = None

        # Elo 업데이트
        self.update_elo(team_id_a, team_id_b, winner=winner_team)

        return MatchResult(winner_id=winner_team, loser_id=loser_team)

    # ---------- 예선(리그) ----------

    def run_preliminary_epoch(
        self,
        num_matches: int = 50,
        ppo_batch_size: int = 128,
        ppo_epochs: int = 4,
    ):
        """
        예선 1 epoch:
        - num_matches 번의 매치를 돌리고
        - 각 팀별 PPO 업데이트 한 번 수행
        """
        team_indices = list(range(self.num_teams))

        for m in range(num_matches):
            # Elo 기준으로 정렬해서 비슷한 팀끼리 샘플
            team_indices.sort(key=lambda i: self.teams[i].elo)
            # 인접한 구간에서 랜덤하게 뽑는 간단한 방식
            i = random.randint(0, self.num_teams - 2)
            j = i + 1

            team_i = team_indices[i]
            team_j = team_indices[j]

            result = self.run_match(team_i, team_j, training=True)
            print(
                f"[PRELIM] Match {m+1}/{num_matches} : "
                f"{team_i} vs {team_j}, winner = {result.winner_id}, "
                f"Elo=({self.teams[team_i].elo:.1f},{self.teams[team_j].elo:.1f})"
            )

        # 매치 끝나면 PPO 업데이트
        for agent in self.teams:
            agent.update(batch_size=ppo_batch_size, epochs=ppo_epochs)

    def get_top_teams_by_elo(self, N: int) -> List[int]:
        sorted_ids = sorted(
            range(self.num_teams), key=lambda i: self.teams[i].elo, reverse=True
        )
        return sorted_ids[:N]

    # ---------- 본선: Swiss + 토너먼트 ----------

    def init_swiss(self, finalists: List[int]):
        self.swiss_wins = {tid: 0 for tid in finalists}
        self.swiss_losses = {tid: 0 for tid in finalists}

    def swiss_alive(self) -> List[int]:
        return [
            tid
            for tid in self.swiss_wins.keys()
            if self.swiss_wins[tid] < 3 and self.swiss_losses[tid] < 3
        ]

    def run_swiss_round(
        self,
        ppo_batch_size: int = 128,
        ppo_epochs: int = 4,
    ):
        """
        Swiss 라운드 1회.
        - 같은 전적끼리 매칭
        - 각 매치에서 학습 데이터 수집
        - 무승부일 경우 Elo 높은 쪽에게 강제 승리 부여(기록용)
        - 라운드 끝나고 PPO 업데이트
        """
        alive_teams = self.swiss_alive()
        # 전적 그룹핑
        record_groups: Dict[Tuple[int, int], List[int]] = {}
        for tid in alive_teams:
            rec = (self.swiss_wins[tid], self.swiss_losses[tid])
            record_groups.setdefault(rec, []).append(tid)

        # 각 그룹별로 매칭
        for rec, group in record_groups.items():
            random.shuffle(group)
            if len(group) < 2:
                continue
            for i in range(0, len(group) - 1, 2):
                t1 = group[i]
                t2 = group[i + 1]
                result = self.run_match(t1, t2, training=True)

                if result.winner_id is None:
                    # ---- 핵심 수정: 무승부일 때도 강제로 승/패 부여 ----
                    # Elo 높은 쪽에게 승리
                    if self.teams[t1].elo >= self.teams[t2].elo:
                        w, l = t1, t2
                    else:
                        w, l = t2, t1
                    print(
                        f"[SWISS] Draw between {t1} and {t2} at record {rec} "
                        f"-> forced winner {w} by Elo"
                    )
                else:
                    w = result.winner_id
                    l = result.loser_id
                    print(
                        f"[SWISS] {t1} vs {t2}, winner={w}, "
                        f"record_before={rec}"
                    )

                self.swiss_wins[w] += 1
                self.swiss_losses[l] += 1

        # 라운드 끝나고 PPO 업데이트
        for agent in self.teams:
            agent.update(batch_size=ppo_batch_size, epochs=ppo_epochs)

    def run_full_swiss(
        self,
        finalists: List[int],
        ppo_batch_size: int = 128,
        ppo_epochs: int = 4,
        max_rounds: int = 10,  # <-- Swiss 최대 라운드
    ) -> List[int]:
        """
        Swiss 전체 진행.
        3승 진출 / 3패 탈락 규칙을 기본으로 하되,
        - max_rounds를 넘기면 강제로 종료
        - 최종적으로는 (wins, -losses, Elo) 기준으로 상위 8팀을 선택
        """
        self.init_swiss(finalists)

        round_idx = 0
        while True:
            round_idx += 1
            print(f"\n[SWISS] Round {round_idx} start")

            self.run_swiss_round(
                ppo_batch_size=ppo_batch_size,
                ppo_epochs=ppo_epochs,
            )

            alive = self.swiss_alive()
            qualified_now = [
                tid for tid in finalists if self.swiss_wins[tid] >= 3
            ]
            eliminated_now = [
                tid for tid in finalists if self.swiss_losses[tid] >= 3
            ]

            print(
                f"[SWISS] after round {round_idx}: "
                f"qualified={qualified_now}, "
                f"eliminated={eliminated_now}, "
                f"alive={alive}"
            )

            # 종료 조건:
            # 1) 3승팀이 충분히 많이 생겼거나
            # 2) 더 붙일 alive 팀이 1 이하이거나
            # 3) max_rounds를 초과했을 때
            if len(qualified_now) >= 8 or len(alive) <= 1 or round_idx >= max_rounds:
                break

        # ---- 최종 8강 선발: 항상 딱 8팀 뽑히도록 정렬 후 슬라이싱 ----
        # 기준: (wins, -losses, Elo) 내림차순
        sorted_finalists = sorted(
            finalists,
            key=lambda tid: (
                self.swiss_wins.get(tid, 0),
                -self.swiss_losses.get(tid, 0),
                self.teams[tid].elo,
            ),
            reverse=True,
        )

        finalists_8 = sorted_finalists[: min(8, len(sorted_finalists))]
        print(f"[SWISS] Final qualified for Knockout: {finalists_8}")
        return finalists_8

    def run_knockout(
        self,
        finalists_8: List[int],
        ppo_batch_size: int = 128,
        ppo_epochs: int = 4,
        series_games: int = 5,
    ) -> int:
        """
        KO 토너먼트 (싱글 엘리미네이션, BOx 시리즈)

        - finalists_8: 2, 4, 8 ... 어떤 팀 수든 상관 없음
        - series_games: BOx (예: 5 -> BO5)
        - return: 최종 챔피언 team_id

        KO 단계에서는 학습(ppo update)은 하지 않고,
        run_match(..., training=False)로 순수 평가만 함.
        """
        teams = list(finalists_8)
        assert len(teams) >= 2, "Knockout needs at least 2 teams"

        # 시리즈 승리 조건: 절반 초과
        need_wins = (series_games + 1) // 2

        round_idx = 1
        while len(teams) > 1:
            print(f"\n[KO] Round {round_idx} start, teams={teams}")
            next_round: List[int] = []

            # 필요하면 섞어서 브래킷 구성
            random.shuffle(teams)

            for i in range(0, len(teams), 2):
                if i + 1 >= len(teams):
                    # 홀수 팀이면 bye
                    bye_team = teams[i]
                    print(f"[KO] team {bye_team} gets a bye")
                    next_round.append(bye_team)
                    continue

                t1 = teams[i]
                t2 = teams[i + 1]

                wins1 = 0
                wins2 = 0
                game_idx = 0

                while wins1 < need_wins and wins2 < need_wins:
                    game_idx += 1
                    result = self.run_match(t1, t2, training=False)

                    if result.winner_id is None:
                        # 무승부 -> Elo 높은 쪽에게 승리 1판 강제
                        if self.teams[t1].elo >= self.teams[t2].elo:
                            wins1 += 1
                            forced = t1
                        else:
                            wins2 += 1
                            forced = t2
                        print(
                            f"[KO] series {t1} vs {t2}, "
                            f"game {game_idx}: draw -> forced winner {forced} by Elo"
                        )
                    elif result.winner_id == t1:
                        wins1 += 1
                        print(
                            f"[KO] series {t1} vs {t2}, "
                            f"game {game_idx}: winner = {t1}"
                        )
                    else:
                        wins2 += 1
                        print(
                            f"[KO] series {t1} vs {t2}, "
                            f"game {game_idx}: winner = {t2}"
                        )

                if wins1 > wins2:
                    winner = t1
                else:
                    winner = t2

                print(
                    f"[KO] series result {t1} vs {t2} -> winner {winner} "
                    f"({wins1}:{wins2})"
                )
                next_round.append(winner)

            teams = next_round
            round_idx += 1

        champion = teams[0]
        print(f"[KO] Champion = team {champion}")
        return champion

    # ---------- 챔피언 기반 리셋 ----------

    def reset_from_champion(self, champion_id: int,
                            noise_scale: float = 0.01):
        """
        챔피언 파라미터를 기반으로 모든 팀 재초기화.
        - 챔피언 팀은 copy_exact=True (그대로)
        - 나머지 팀은 노이즈를 약간 섞어서 초기화
        - Elo는 모두 1500으로 리셋
        """
        champion_agent = self.teams[champion_id]
        champ_sd = champion_agent.model.state_dict()

        for tid, agent in enumerate(self.teams):
            if tid == champion_id:
                agent.reset_from_champion(champ_sd, noise_scale=0.0, copy_exact=True)
            else:
                agent.reset_from_champion(champ_sd, noise_scale=noise_scale, copy_exact=False)
            agent.elo = 1500.0

        # swiss 상태도 초기화
        self.swiss_wins.clear()
        self.swiss_losses.clear()
        print(f"[LEAGUE] reset_from_champion: champion={champion_id}, all Elo reset to 1500")


# ================================================================
# 7. 에피소드(시즌) 단위 학습 루프
# ================================================================
def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_one_episode(
    league: League,
    episode_idx: int,
    prelim_matches: int,
    finalists_N: int,
    ppo_batch_size: int = 128,
    ppo_epochs: int = 4,
    ko_series_games: int = 3,
    save_dir: str = "./checkpoints_alkkagi",
) -> int:
    """
    1 episode:
      1) 예선(리그) 1회
      2) 본선(Swiss + Knockout) 1회
      3) 챔피언 추출 및 저장
      4) 챔피언 기반 리셋은 이 함수 밖에서 수행
    """
    print(f"\n\n==================== EPISODE {episode_idx} START ====================")

    # 1) 예선 1회
    print("[EP] Preliminary (Elo league) start")
    league.run_preliminary_epoch(
        num_matches=prelim_matches,
        ppo_batch_size=ppo_batch_size,
        ppo_epochs=ppo_epochs,
    )
    print("[EP] Preliminary done. Elo standings:")
    for tid, agent in enumerate(league.teams):
        print(f"  team {tid}: Elo = {agent.elo:.1f}")

    # 상위 N팀 본선 진출
    finalists = league.get_top_teams_by_elo(finalists_N)
    print(f"[EP] Finalists for Swiss/KO (top {finalists_N} by Elo): {finalists}")

    # 2) 본선: Swiss + Knockout
    finalists_8 = league.run_full_swiss(
        finalists,
        ppo_batch_size=ppo_batch_size,
        ppo_epochs=ppo_epochs,
    )
    champion_id = league.run_knockout(
        finalists_8,
        ppo_batch_size=ppo_batch_size,
        ppo_epochs=ppo_epochs,
        series_games=ko_series_games,
    )

    print(f"[EP] Episode {episode_idx} champion team id = {champion_id}")

    # 3) 챔피언 모델 저장
    os.makedirs(save_dir, exist_ok=True)
    champion_agent = league.teams[champion_id]
    ckpt_path = os.path.join(
        save_dir,
        f"episode{episode_idx:04d}_champion_team{champion_id}.pt"
    )
    torch.save(champion_agent.model.state_dict(), ckpt_path)
    print(f"[EP] Champion model saved to: {ckpt_path}")

    print(f"==================== EPISODE {episode_idx} END ====================\n")
    return champion_id


def train_league_multi_episode(
    num_episodes: int = 10,
    num_teams: int = 8,
    prelim_matches_per_episode: int = 50,
    finalists_N: int = 8,
    ppo_batch_size: int = 128,
    ppo_epochs: int = 4,
    ko_series_games: int = 3,
    champion_noise_scale: float = 0.01,
    seed: int = 42,
    save_dir: str = "./checkpoints_alkkagi",
):
    """
    전체 학습:
      for episode in 1..num_episodes:
        예선(1회) → 본선(1회) → 챔피언 → 저장 → 챔피언 기반 리셋
    """
    set_global_seed(seed)
    league = League(num_teams=num_teams)

    for ep in range(num_episodes):
        champion_id = run_one_episode(
            league=league,
            episode_idx=ep,
            prelim_matches=prelim_matches_per_episode,
            finalists_N=finalists_N,
            ppo_batch_size=ppo_batch_size,
            ppo_epochs=ppo_epochs,
            ko_series_games=ko_series_games,
            save_dir=save_dir,
        )

        # 다음 에피소드를 위해 챔피언 기반 리셋
        league.reset_from_champion(champion_id, noise_scale=champion_noise_scale)

    print("\n[TRAIN] Multi-episode training finished.")
    print("[TRAIN] Final Elo standings:")
    for tid, agent in enumerate(league.teams):
        print(f"  team {tid}: Elo = {agent.elo:.1f}")

    return league


# ================================================================
# 8. 메인
# ================================================================
if __name__ == "__main__":
    # 여기서 에피소드 수만 바꾸면 됨
    NUM_EPISODES = 10        # 실제로는 8000까지도 가능 (시간/자원 문제만 감당하면 됨)
    NUM_TEAMS = 8
    PRELIM_MATCHES_PER_EP = 50
    FINALISTS_N = 8
    PPO_BATCH_SIZE = 128
    PPO_EPOCHS = 4
    KO_SERIES_GAMES = 3
    CHAMPION_NOISE_SCALE = 0.01
    SEED = 42
    SAVE_DIR = "./checkpoints_alkkagi"

    print(f"[RUN] device = {DEVICE}")
    print(f"[RUN] episodes = {NUM_EPISODES}, teams = {NUM_TEAMS}")

    train_league_multi_episode(
        num_episodes=NUM_EPISODES,
        num_teams=NUM_TEAMS,
        prelim_matches_per_episode=PRELIM_MATCHES_PER_EP,
        finalists_N=FINALISTS_N,
        ppo_batch_size=PPO_BATCH_SIZE,
        ppo_epochs=PPO_EPOCHS,
        ko_series_games=KO_SERIES_GAMES,
        champion_noise_scale=CHAMPION_NOISE_SCALE,
        seed=SEED,
        save_dir=SAVE_DIR,
    )
