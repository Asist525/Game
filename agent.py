from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import kymnasium as kym


# ------------------------------------------------
# 기본 설정
# ------------------------------------------------
N_STONES = 3
N_OBS = 3

BOARD_W = 600
BOARD_H = 600

# --- 액션 디스크리타이제이션 ---
N_ANGLES = 16
N_POWERS = 3
N_ACTIONS = N_STONES * N_ANGLES * N_POWERS  # 전체 디스크리트 액션 수

# --- 리워드/셰이핑 관련 ---
STEP_PENALTY = 0.005       # 턴당 -0.005 (너무 길게 끄는 걸 약하게 억제)
POTENTIAL_ALPHA = 0.25     # 위치 기반 potential Φ_geo(s) 스케일


# ------------------------------------------------
# 환경 생성
# ------------------------------------------------
def make_env(render_mode=None, bgm: bool = False):
    env = gym.make(
        "kymnasium/AlKkaGi-3x3-v0",
        obs_type="custom",
        render_mode=render_mode,   # gym에 render_mode 전달
        bgm=bgm,
    )

    # OrderEnforcing wrapper 때문에 render_mode는 setter가 없어서
    # 직접 저장할 때는 다른 이름으로 저장해야 한다.
    env._render_mode_debug = render_mode

    return env


# ------------------------------------------------
# 공통 전처리 함수들
# ------------------------------------------------
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
    # alive는 그대로
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


# ------------------------------------------------
# Baseline 인코더 (31차원, 필요시 사용)
# ------------------------------------------------
def encode_state_basic_alkkagi(
    obs,
    my_color: int,
    board_w: float,
    board_h: float,
) -> np.ndarray:
    """
    Baseline 관측 인코더 (canonical, me/opp 기준)

    첫 번째 스칼라는 env 절대 턴(0/1)이 아니라,
    "지금 내 턴이냐?"를 나타내는 turn_is_me ∈ {0,1}.

    실제 학습에서는 encode_state_fe_alkkagi를
    (obs, my_color=obs["turn"]) 으로만 호출하므로
    turn_is_me는 항상 1.0이 된다.
    """
    me, opp, obstacles, turn_raw = split_me_opp(obs, my_color)

    # canonical actor 관점: 내 턴이면 1, 아니면 0
    turn_is_me = 1.0 if int(turn_raw) == int(my_color) else 0.0

    me_norm = normalize_stones(me, board_w, board_h)
    opp_norm = normalize_stones(opp, board_w, board_h)
    obs_norm = normalize_obstacles(obstacles, board_w, board_h)

    feat = np.concatenate([
        np.array([turn_is_me], dtype=np.float32),  # (1,)
        me_norm.flatten(),
        opp_norm.flatten(),
        obs_norm.flatten(),
    ]).astype(np.float32)

    return feat


# ------------------------------------------------
# Feature Engineering용 helper들
# ------------------------------------------------
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


# ------------------------------------------------
# 1) 로컬(Actor용) FE 인코더 (canonical, 51차원)
# ------------------------------------------------
def encode_state_fe_alkkagi(
    obs,
    my_color: int,
    board_w: float,
    board_h: float,
) -> np.ndarray:
    """
    Feature engineering 버전 state encoder (Actor 입력용, canonical).

    - baseline feature: 31차원
    - 추가 feature:
        * my_alive_cnt, opp_alive_cnt, alive_diff, alive_ratio (4)
        * my_center_x, my_center_y, my_var_x, my_var_y (4)
        * op_center_x, op_center_y, op_var_x, op_var_y (4)
        * my_min_edge, op_min_edge, min_my_op_dist (3)
        * obs_cnt, obs_cx_mean, obs_cy_mean, obs_w_mean, obs_h_mean (5)

      => 추가 20차원
      => 총 31 + 20 = 51차원
    """
    # --- 1) baseline feature (31차원) ---
    base_feat = encode_state_basic_alkkagi(obs, my_color, board_w, board_h)

    # --- 2) 정규화된 돌/장애물 ---
    me, opp, obstacles, _ = split_me_opp(obs, my_color)
    me_norm = normalize_stones(me, board_w, board_h)
    opp_norm = normalize_stones(opp, board_w, board_h)
    obs_norm = normalize_obstacles(obstacles, board_w, board_h)

    # --- 3) scalar feature들 ---
    my_alive_cnt = float((me_norm[:, 2] > 0.5).sum())
    opp_alive_cnt = float((opp_norm[:, 2] > 0.5).sum())
    alive_diff = my_alive_cnt - opp_alive_cnt
    denom = my_alive_cnt + opp_alive_cnt
    alive_ratio = my_alive_cnt / denom if denom > 0 else 0.0

    scalar_feats = np.array(
        [my_alive_cnt, opp_alive_cnt, alive_diff, alive_ratio],
        dtype=np.float32,
    )

    # --- 4) 그룹 요약 ---
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
    extra_feats = np.concatenate([
        scalar_feats,   # 4
        my_stats,       # 4
        op_stats,       # 4
        relation_feats, # 3
        obs_stats,      # 5
    ]).astype(np.float32)

    feat = np.concatenate([base_feat, extra_feats]).astype(np.float32)

    return feat  # (51,)


ACTOR_STATE_DIM = 51


def encode_state_fe_tensor(
    obs,
    my_color: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    feature engineering encoder → torch.Tensor
    shape: (ACTOR_STATE_DIM,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_np = encode_state_fe_alkkagi(obs, my_color, BOARD_W, BOARD_H)
    feat_t = torch.from_numpy(feat_np).to(device=device, dtype=torch.float32)
    return feat_t


# ------------------------------------------------
# 2) 중앙 관측(Critic용) 인코더 (non-canonical, 31차원)
# ------------------------------------------------
def encode_state_central_alkkagi(
    obs,
    board_w: float,
    board_h: float,
) -> np.ndarray:
    """
    Critic 입력용 중앙 인코더.
    - turn (1)
    - black_norm : 3 * (x, y, alive) = 9
    - white_norm : 3 * (x, y, alive) = 9
    - obs_norm   : 3 * (x, y, w, h)   = 12
    => 1 + 9 + 9 + 12 = 31차원
    """
    black = np.array(obs["black"], dtype=np.float32)
    white = np.array(obs["white"], dtype=np.float32)
    obstacles = np.array(obs["obstacles"], dtype=np.float32)
    turn = float(obs["turn"])

    black_norm = normalize_stones(black, board_w, board_h)
    white_norm = normalize_stones(white, board_w, board_h)
    obs_norm = normalize_obstacles(obstacles, board_w, board_h)

    feat = np.concatenate([
        np.array([turn], dtype=np.float32),
        black_norm.flatten(),   # 9
        white_norm.flatten(),   # 9
        obs_norm.flatten(),     # 12
    ]).astype(np.float32)
    return feat  # (31,)


CRITIC_STATE_DIM = 31


def encode_state_central_tensor(
    obs,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    중앙 인코더 → torch.Tensor
    shape: (CRITIC_STATE_DIM,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_np = encode_state_central_alkkagi(obs, BOARD_W, BOARD_H)
    feat_t = torch.from_numpy(feat_np).to(device=device, dtype=torch.float32)
    return feat_t


# ------------------------------------------------
# 액션 디코딩
# ------------------------------------------------
def decode_action_index(action_idx: int):
    """
    디스크리트 액션 인덱스 -> (stone_id, angle, power)로 변환.

    - stone_id ∈ {0,1,2}
    - angle ∈ [-180, 180] 근처의 균일 그리드
    - power ∈ [500, 2500] 범위에서 균일 그리드
    """
    per_stone = N_ANGLES * N_POWERS
    stone_id = action_idx // per_stone
    rem = action_idx % per_stone

    angle_id = rem // N_POWERS
    power_id = rem % N_POWERS

    # angle 그리드
    angle_step = 360.0 / N_ANGLES
    angle = -180.0 + (angle_id + 0.5) * angle_step  # 중앙값

    # power 그리드
    power_min, power_max = 500.0, 2500.0
    if N_POWERS == 1:
        power = (power_min + power_max) / 2.0
    else:
        ratio = power_id / (N_POWERS - 1)
        power = power_min + ratio * (power_max - power_min)

    return int(stone_id), float(angle), float(power)


def map_stone_to_alive_index(
    observation: dict,
    my_color: int,
    stone_id: int,
) -> int:
    """
    디스크리트 action에서 나온 stone_id(0,1,2)를
    실제로 alive 상태인 돌들 중 하나로 매핑한다.

    - my_color == 0 → obs["black"]
    - my_color == 1 → obs["white"]
    - alive 인덱스만 모아서, stone_id를 alive 개수로 mod 후
      그 번째 alive 돌을 선택.
    """
    if my_color == 0:
        stones = observation["black"]
    else:
        stones = observation["white"]

    # stones: (3, 3) = [x, y, alive]
    alive_indices = [i for i, s in enumerate(stones) if s[2] > 0.5]

    # alive 돌이 하나도 없으면 그냥 0으로 (어차피 게임 끝 직전 상태일 것)
    if not alive_indices:
        return 0

    stone_id = stone_id % len(alive_indices)
    return alive_indices[stone_id]


# ------------------------------------------------
# Policy / Value Network 분리 (CTDE)
# ------------------------------------------------
class ActorNet(nn.Module):
    def __init__(self, state_dim: int = ACTOR_STATE_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        h1, h2, h3 = 256, 256, 256
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.policy_head = nn.Linear(h3, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        logits = self.policy_head(h)
        return logits  # (B, n_actions)


class CriticNet(nn.Module):
    def __init__(self, state_dim: int = CRITIC_STATE_DIM):
        super().__init__()
        h1, h2 = 256, 256
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.value_head = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        value = self.value_head(h).squeeze(-1)  # (B,)
        return value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    update_epochs: int = 2
    batch_size: int = 512


# ------------------------------------------------
# 리워드 / Potential-based shaping / GAE
# ------------------------------------------------
def compute_alive_diff(obs, my_color: int) -> float:
    """
    any state에서 my_color 입장에서 alive_diff 계산.
    """
    me, opp, _, _ = split_me_opp(obs, my_color)
    me = np.array(me, dtype=np.float32)
    opp = np.array(opp, dtype=np.float32)

    my_alive = float((me[:, 2] > 0.5).sum())
    opp_alive = float((opp[:, 2] > 0.5).sum())
    alive_diff = my_alive - opp_alive  # [-3, +3] 범위

    return alive_diff


def potential(obs, my_color: int) -> float:
    """
    위치 기반 형세 potential Φ_geo(s).

    - 내 돌은 보드 중앙 쪽(엣지에서 멀게)
    - 상대 돌은 보드 엣지 쪽(엣지에서 가깝게)

    로 두고 싶다는 신호:
      Φ_geo(s) = POTENTIAL_ALPHA * (my_min_edge - opp_min_edge)
    """
    me, opp, obstacles, _ = split_me_opp(obs, my_color)
    me_norm = normalize_stones(me, BOARD_W, BOARD_H)
    opp_norm = normalize_stones(opp, BOARD_W, BOARD_H)

    my_min_edge = min_edge_dist(me_norm)
    opp_min_edge = min_edge_dist(opp_norm)

    geo = my_min_edge - opp_min_edge
    return POTENTIAL_ALPHA * geo


def compute_gae_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
    bootstrap_value_last: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    단일 에피소드에 대해 GAE + return 계산.
    rewards: (T,)
    values : (T,)   rollout 동안 critic(s_t)
    bootstrap_value_last : truncated일 때 V(s_{T}), terminated면 0.0
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)

    gae = 0.0
    next_value = bootstrap_value_last

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
        next_value = values[t]

    return advantages, returns


# ------------------------------------------------
# PPO Policy (Actor/ Critic 분리)
# ------------------------------------------------
class PPOPolicy:
    """
    하나의 ActorNet(로컬/캐노니컬) + CriticNet(중앙)을 흑/백 모두가 공유하는 PPO 정책.
    - act_eval : 평가용 (deterministic argmax or sampling)
    - act_train: 학습용 (logprob, actor_state, critic_state 반환)
    """

    def __init__(self, device: torch.device | None = None, lr: float = 3e-4):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.actor = ActorNet(state_dim=ACTOR_STATE_DIM, n_actions=N_ACTIONS).to(self.device)
        self.critic = CriticNet(state_dim=CRITIC_STATE_DIM).to(self.device)

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr,
        )

    # ===== 저장/로드 =====
    def save(self, path: str) -> None:
        ckpt = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            # 이어학습을 위해 optimizer state도 같이 저장
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(ckpt, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device | None = None,
        lr: float = 3e-4,
    ) -> "PPOPolicy":
        policy = cls(device=device, lr=lr)
        ckpt = torch.load(path, map_location=policy.device)

        if isinstance(ckpt, dict) and "actor_state_dict" in ckpt and "critic_state_dict" in ckpt:
            policy.actor.load_state_dict(ckpt["actor_state_dict"])
            policy.critic.load_state_dict(ckpt["critic_state_dict"])

            # optimizer state가 있으면 같이 로드 (없으면 무시: 구버전 호환)
            if "optimizer_state_dict" in ckpt:
                try:
                    policy.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    # shape 안 맞거나 하면 그냥 무시하고 새 optimizer 사용
                    pass
        else:
            # 구버전 호환: 하나의 state_dict만 있을 경우 actor만 로드
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            policy.actor.load_state_dict(state_dict)

        return policy

    @torch.no_grad()
    def act_eval(
        self,
        observation: dict,
        my_color: int,
        greedy: bool = False,
        temperature: float = 1,
    ) -> dict:
        """
        평가용 act.
        - greedy=True  이면 argmax 정책 (대회/리더보드용)
        - greedy=False 이면 샘플링 정책 (디버그/관찰용)
        - temperature != 1.0 이면 logits / T 후 softmax (T>1: 더 random)
        """
        self.actor.eval()

        state_t = encode_state_fe_tensor(
            observation,
            my_color=my_color,
            device=self.device,
        ).unsqueeze(0)  # (1, state_dim)

        logits = self.actor(state_t)
        if temperature != 1.0:
            logits = logits / temperature

        dist = torch.distributions.Categorical(logits=logits)

        if greedy:
            probs = dist.probs  # (1, A)
            action_idx_t = probs.argmax(dim=-1)  # (1,)
        else:
            action_idx_t = dist.sample()  # (1,)

        action_idx = int(action_idx_t.item())

        # 디스크리트 → (stone_id, angle, power)
        stone_id, angle, power = decode_action_index(action_idx)

        # alive 돌로 매핑
        stone_id = map_stone_to_alive_index(
            observation=observation,
            my_color=my_color,
            stone_id=stone_id,
        )

        action = {
            "turn": int(observation["turn"]),
            "index": int(stone_id),
            "power": float(power),
            "angle": float(angle),
        }
        return action

    def act_train(
        self,
        observation: dict,
        my_color: int,
    ) -> tuple[dict, int, float, torch.Tensor, torch.Tensor]:
        """
        학습용 act.
        반환:
        - action dict        : env.step()에 넣을 액션
        - action_idx         : 디스크리트 액션 인덱스 (0 ~ N_ACTIONS-1)
        - logprob            : log π(a|s)
        - actor_state_vec    : (ACTOR_STATE_DIM,) canonical 상태 벡터
        - critic_state_vec   : (CRITIC_STATE_DIM,) central 상태 벡터
        """
        self.actor.train()
        self.critic.train()

        # Actor용 canonical 인코딩
        actor_state_t = encode_state_fe_tensor(
            observation,
            my_color=my_color,
            device=self.device,
        ).unsqueeze(0)

        # Critic용 central 인코딩
        critic_state_t = encode_state_central_tensor(
            observation,
            device=self.device,
        ).unsqueeze(0)

        # policy 분포
        logits = self.actor(actor_state_t)
        dist = torch.distributions.Categorical(logits=logits)

        # 학습 때는 샘플링 기반
        action_idx_t = dist.sample()
        logprob_t = dist.log_prob(action_idx_t)

        action_idx = int(action_idx_t.item())
        logprob = float(logprob_t.item())

        # 디스크리트 → (stone_id, angle, power)
        stone_id, angle, power = decode_action_index(action_idx)
        stone_id = map_stone_to_alive_index(
            observation=observation,
            my_color=my_color,
            stone_id=stone_id,
        )

        action = {
            "turn": int(observation["turn"]),
            "index": int(stone_id),
            "power": float(power),
            "angle": float(angle),
        }

        actor_state_vec = actor_state_t.squeeze(0).detach()
        critic_state_vec = critic_state_t.squeeze(0).detach()

        return action, action_idx, logprob, actor_state_vec, critic_state_vec


# ------------------------------------------------
# PPO 업데이트 (Actor/ Critic 분리, CTDE)
# ------------------------------------------------
def ppo_update(
    policy: PPOPolicy,
    actor_states: List[torch.Tensor],
    critic_states: List[torch.Tensor],
    actions: List[int],
    old_logprobs: List[float],
    advantages: List[float],
    returns: List[float],
    config: PPOConfig,
):
    device = policy.device
    policy.actor.train()
    policy.critic.train()

    actor_states_t = torch.stack(actor_states).to(device)    # (N, ACTOR_STATE_DIM)
    critic_states_t = torch.stack(critic_states).to(device)  # (N, CRITIC_STATE_DIM)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)             # (N,)
    old_logprobs_t = torch.tensor(old_logprobs, dtype=torch.float32, device=device)  # (N,)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)      # (N,)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)            # (N,)

    # advantage 정규화
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    N = actor_states_t.size(0)
    batch_size = min(config.batch_size, N)

    for _ in range(config.update_epochs):
        idx = torch.randperm(N, device=device)

        for start in range(0, N, batch_size):
            mb_idx = idx[start:start + batch_size]

            mb_actor_states = actor_states_t[mb_idx]
            mb_critic_states = critic_states_t[mb_idx]
            mb_actions = actions_t[mb_idx]
            mb_old_logprobs = old_logprobs_t[mb_idx]
            mb_adv = advantages_t[mb_idx]
            mb_returns = returns_t[mb_idx]

            # Policy
            logits = policy.actor(mb_actor_states)      # (B, A)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(mb_actions)    # (B,)
            entropy = dist.entropy().mean()

            ratio = (new_logprobs - mb_old_logprobs).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(
                ratio,
                1.0 - config.clip_coef,
                1.0 + config.clip_coef,
            ) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value (central critic)
            values = policy.critic(mb_critic_states)    # (B,)
            value_loss = F.mse_loss(values, mb_returns)

            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

            policy.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.actor.parameters()) + list(policy.critic.parameters()),
                config.max_grad_norm,
            )
            policy.optimizer.step()


# ------------------------------------------------
# Policy 클론
# ------------------------------------------------
def clone_policy(src: PPOPolicy, new_lr: float | None = None) -> PPOPolicy:
    """
    learner policy를 snapshot으로 복사.
    """
    device = src.device
    if new_lr is None:
        lr = src.optimizer.param_groups[0]["lr"]
    else:
        lr = new_lr

    new_policy = PPOPolicy(device=device, lr=lr)
    new_policy.actor.load_state_dict(src.actor.state_dict())
    new_policy.critic.load_state_dict(src.critic.state_dict())
    return new_policy


# ------------------------------------------------
# ELO / 리그 (상대 샘플링)
# ------------------------------------------------
@dataclass
class RatedPolicy:
    """
    ELO 리그에서 하나의 정책 엔트리.
    """
    id: str
    policy: PPOPolicy
    rating: float = 1500.0
    games: int = 0


def elo_expected(ra: float, rb: float) -> float:
    """
    E_A = 1 / (1 + 10^((Rb - Ra)/400))
    """
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def elo_update(
    ra: float,
    rb: float,
    score_a: float,
    k: float = 32.0,
) -> Tuple[float, float]:
    """
    score_a: A의 실제 결과 (1=승, 0.5=무, 0=패)
    """
    ea = elo_expected(ra, rb)
    eb = 1.0 - ea

    ra_new = ra + k * (score_a - ea)
    rb_new = rb + k * ((1.0 - score_a) - eb)
    return ra_new, rb_new


class EloLeague:
    """
    ELO 기반 self-play 리그 구조.
    rating 차이에 대한 softmax(-|Δrating| / temperature)로 opponent 샘플링.
    """

    def __init__(self, players: List[RatedPolicy], k: float = 32.0, temperature: float = 200.0):
        self.players = players
        self.k = k
        self.temperature = temperature

    def find_index(self, player_id: str) -> int:
        for i, p in enumerate(self.players):
            if p.id == player_id:
                return i
        raise ValueError(f"player_id '{player_id}' not found")

    def choose_opponent(self, learner_id: str) -> RatedPolicy:
        """
        softmax(-|Δrating| / T) 분포로 opponent 샘플링.
        rating 가까운 상대를 자주, 가끔은 먼 상대도 선택.
        """
        li = self.find_index(learner_id)
        learner = self.players[li]

        candidates = [
            p for p in self.players
            if p.id != learner_id
        ]
        if not candidates:
            raise RuntimeError("opponent candidates가 없습니다.")

        diffs = np.array([abs(p.rating - learner.rating) for p in candidates], dtype=np.float32)
        weights = np.exp(-diffs / self.temperature)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()

        idx = np.random.choice(len(candidates), p=weights)
        opponent = candidates[idx]
        return opponent

    def update_result(self, a_id: str, b_id: str, score_a: float) -> None:
        """
        경기 결과를 기록하고, 양쪽 rating 업데이트.
        """
        ai = self.find_index(a_id)
        bi = self.find_index(b_id)
        pa = self.players[ai]
        pb = self.players[bi]

        ra_new, rb_new = elo_update(pa.rating, pb.rating, score_a, k=self.k)

        pa.rating = ra_new
        pb.rating = rb_new
        pa.games += 1
        pb.games += 1


# ------------------------------------------------
# 체크포인트 / 로그
# ------------------------------------------------
def save_checkpoint_policy(
    policy: PPOPolicy,
    epoch: int,
    checkpoint_dir: str = "checkpoints",
    max_keep: int = 20,
) -> str:
    """
    policy 파라미터를 checkpoint_dir에 저장하고,
    최근 max_keep개만 남기고 나머지는 삭제한다.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch:06d}.pt")
    policy.save(ckpt_path)

    # 오래된 것 정리
    ckpts = sorted(Path(checkpoint_dir).glob("policy_epoch_*.pt"))
    if len(ckpts) > max_keep:
        for old in ckpts[:-max_keep]:
            try:
                old.unlink()
            except OSError:
                pass

    return ckpt_path


def append_epoch_log(
    epoch: int,
    episodes: int,
    wins: int,
    draws: int,
    losses: int,
    avg_reward: float,
    avg_steps: float,
    learner_rating: float,
    num_players: int,
    log_path: str = "training_metrics.csv",
) -> None:
    """
    에폭 단위 학습 품질 로그를 CSV로 남긴다.
    """
    file_exists = os.path.exists(log_path)

    if not file_exists:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "epoch,episodes,wins,draws,losses,win_rate,avg_reward,avg_steps,learner_rating,num_players\n"
            )

    win_rate = wins / episodes if episodes > 0 else 0.0

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch},{episodes},{wins},{draws},{losses},"
            f"{win_rate:.6f},{avg_reward:.6f},{avg_steps:.6f},"
            f"{learner_rating:.6f},{num_players}\n"
        )


# ★★★ 이어학습용: 최신 체크포인트 찾기 ★★★
def find_latest_checkpoint(
    checkpoint_dir: str = "checkpoints",
) -> tuple[str | None, int]:
    """
    checkpoint_dir 안에서 policy_epoch_XXXXXX.pt 중
    가장 큰 epoch 번호를 가진 파일을 찾아서 (path, epoch)를 반환.
    없으면 (None, 0).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpts = sorted(Path(checkpoint_dir).glob("policy_epoch_*.pt"))
    if not ckpts:
        return None, 0

    latest = ckpts[-1]
    stem = latest.stem  # e.g. "policy_epoch_001470"
    epoch = 0
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        epoch = int(parts[-1])
    else:
        # 실패하면 마지막 6자리만 시도
        try:
            epoch = int(stem[-6:])
        except ValueError:
            epoch = 0

    return str(latest), epoch


# ------------------------------------------------
# 리그 self-play 학습 루프 (CTDE + potential shaping)
# ------------------------------------------------
def train_league_selfplay(
    num_epochs: int = 5,
    episodes_per_epoch: int = 20,
    snapshot_interval: int = 2,
    config: PPOConfig | None = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 5000,
    max_checkpoints: int = 20,
    log_path: str = "training_metrics.csv",
    max_players: int = 50,
    # ★ 이어학습용: 외부에서 learner 넘겨받기 + epoch 시작 위치
    existing_learner: PPOPolicy | None = None,
    start_epoch: int = 0,
) -> tuple[PPOPolicy, EloLeague]:
    """
    ELO 리그 기반 self-play 학습 루프 (CTDE + potential-based shaping + final diff 보정 + 디버깅).

    - existing_learner가 None이면 새로 PPOPolicy 생성
    - start_epoch부터 epoch 번호를 이어서 사용
    """

    if config is None:
        config = PPOConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ★ 인간 렌더링 모드로 환경 생성
    env = make_env(render_mode=None, bgm=False)

    # learner 초기화 또는 이어받기
    if existing_learner is None:
        learner = PPOPolicy(device=device, lr=config.learning_rate)
        print("[TRAIN] No existing learner, initialize new policy.")
    else:
        learner = existing_learner
        # 혹시 device가 바뀌었을 수 있으니 to(device)
        learner.device = device
        learner.actor.to(device)
        learner.critic.to(device)
        print(f"[TRAIN] Using existing learner, start_epoch={start_epoch}")

    # 초기 snapshot 생성 (현재 learner 기준)
    snap0 = clone_policy(learner)

    league = EloLeague(
        players=[
            RatedPolicy(id="learner", policy=learner, rating=1500.0),
            RatedPolicy(id="snap_000", policy=snap0, rating=1500.0),
        ],
        k=32.0,
        temperature=200.0,
    )
    snapshot_counter = 1

    # ★ epoch 인덱스를 start_epoch에서 이어서 진행
    for epoch_idx in range(start_epoch + 1, start_epoch + num_epochs + 1):
        all_actor_states = []
        all_critic_states = []
        all_actions = []
        all_logprobs = []
        all_advantages = []
        all_returns = []

        # epoch 메트릭
        ep_wins = ep_draws = ep_losses = 0
        ep_reward_sum = 0.0
        ep_step_sum = 0
        episodes_this_epoch = 0

        # ===========================================================
        #                   에피소드 반복
        # ===========================================================
        for ep in range(episodes_per_epoch):

            opponent = league.choose_opponent("learner")

            learner_color = int(np.random.randint(0, 2))

            # seed도 epoch_idx 기준으로
            global_seed = (epoch_idx - 1) * episodes_per_epoch + ep
            obs, info = env.reset(seed=global_seed)

            done = False
            step = 0
            truncated = False

            ep_actor_states = []
            ep_critic_states = []
            ep_actions = []
            ep_logprobs = []
            ep_rewards = []

            last_step_by_learner = False

            # ===========================================================
            #                     에피소드 진행 (step loop)
            # ===========================================================
            while not done:
                turn = int(obs["turn"])
                learner_turn = (turn == learner_color)

                if learner_turn:
                    # 위치 기반 potential: 현재 형세
                    phi_before = potential(obs, my_color=learner_color)

                    (action, action_idx, logprob,
                     actor_state_vec, critic_state_vec) = learner.act_train(
                        observation=obs,
                        my_color=learner_color
                    )

                    ep_actor_states.append(actor_state_vec.cpu())
                    ep_critic_states.append(critic_state_vec.cpu())
                    ep_actions.append(action_idx)
                    ep_logprobs.append(logprob)

                    last_step_by_learner = True

                else:
                    action = opponent.policy.act_eval(
                        observation=obs,
                        my_color=turn,
                        greedy=False
                    )
                    last_step_by_learner = False

                obs_next, reward_env, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

                # =============================
                #   Reward shaping (learner)
                # =============================
                if learner_turn:
                    # 시간 패널티
                    base_reward = -STEP_PENALTY

                    # 위치 형세 potential: Φ_geo(s'), Φ_geo(s)
                    phi_after = potential(obs_next, my_color=learner_color)

                    # potential-based shaping: r' = r_base + γ Φ(s') - Φ(s)
                    shaped_r = base_reward + config.gamma * phi_after - phi_before

                    ep_rewards.append(shaped_r)

                obs = obs_next

            # ===========================================================
            #               에피소드 종료 처리 (승패/ELO/GAE)
            # ===========================================================
            R_learner = compute_alive_diff(obs, my_color=learner_color)  # [-3, +3]

            # 로그용 (평균 alive_diff)
            ep_reward_sum += R_learner
            ep_step_sum += step
            episodes_this_epoch += 1

            # ELO용 스코어 (승/무/패)
            if R_learner > 0:
                score_a = 1.0
                ep_wins += 1
            elif R_learner < 0:
                score_a = 0.0
                ep_losses += 1
            else:
                score_a = 0.5
                ep_draws += 1

            league.update_result("learner", opponent.id, score_a)

            # -----------------------------
            #   최종 승패 보상(game_reward)
            # -----------------------------
            game_reward = 0.0
            if not truncated:
                # 단순 승/패 보상 (원하면 스케일 조정 가능)
                if R_learner > 0:
                    game_reward = 1.0
                elif R_learner < 0:
                    game_reward = -1.0
                else:
                    game_reward = 0.0  # 무승부

                # 마지막 learner 스텝에 최종 보상 얹기
                if len(ep_rewards) > 0:
                    ep_rewards[-1] += game_reward

            # ---- GAE 계산 ----
            if len(ep_rewards) > 0:

                rewards_np = np.array(ep_rewards, dtype=np.float32)

                critic_states_ep = torch.stack(ep_critic_states).to(device)
                with torch.no_grad():
                    values_t = learner.critic(critic_states_ep).cpu().numpy()

                if truncated:
                    last_state = encode_state_central_tensor(obs, device=device).unsqueeze(0)
                    with torch.no_grad():
                        bootstrap_value_last = learner.critic(last_state).item()
                else:
                    bootstrap_value_last = 0.0

                adv_np, ret_np = compute_gae_returns(
                    rewards=rewards_np,
                    values=values_t,
                    gamma=config.gamma,
                    lam=config.gae_lambda,
                    bootstrap_value_last=bootstrap_value_last
                )

                all_actor_states.extend(ep_actor_states)
                all_critic_states.extend(ep_critic_states)
                all_actions.extend(ep_actions)
                all_logprobs.extend(ep_logprobs)
                all_advantages.extend(adv_np.tolist())
                all_returns.extend(ret_np.tolist())

        # ===========================================================
        #                  Epoch summary & PPO update
        # ===========================================================
        if episodes_this_epoch > 0:
            avg_R = ep_reward_sum / episodes_this_epoch
            avg_steps = ep_step_sum / episodes_this_epoch
        else:
            avg_R = 0.0
            avg_steps = 0.0

        learner_rating = league.players[league.find_index("learner")].rating

        append_epoch_log(
            epoch=epoch_idx,
            episodes=episodes_this_epoch,
            wins=ep_wins,
            draws=ep_draws,
            losses=ep_losses,
            avg_reward=avg_R,
            avg_steps=avg_steps,
            learner_rating=learner_rating,
            num_players=len(league.players),
            log_path=log_path,
        )

        # PPO update
        if len(all_actor_states) > 0:
            ppo_update(
                policy=learner,
                actor_states=all_actor_states,
                critic_states=all_critic_states,
                actions=all_actions,
                old_logprobs=all_logprobs,
                advantages=all_advantages,
                returns=all_returns,
                config=config,
            )

        # checkpoint
        if epoch_idx % checkpoint_interval == 0:
            save_checkpoint_policy(
                learner,
                epoch=epoch_idx,
                checkpoint_dir=checkpoint_dir,
                max_keep=max_checkpoints,
            )

        # snapshot
        if epoch_idx % snapshot_interval == 0:
            snap_policy = clone_policy(learner)
            snap_id = f"snap_{snapshot_counter:03d}"
            snapshot_counter += 1

            # 새 snapshot 추가
            league.players.append(
                RatedPolicy(
                    id=snap_id,
                    policy=snap_policy,
                    rating=learner_rating,
                    games=0,
                )
            )

            # ★ max_players=50 유지: learner를 제외한 가장 오래된 snapshot부터 삭제
            while len(league.players) > max_players:
                # learner는 항상 남겨두고, 제일 앞에서부터 learner가 아닌 놈을 제거
                remove_idx = None
                for i, p in enumerate(league.players):
                    if p.id != "learner":
                        remove_idx = i
                        break

                # 이론상 항상 찾지만, 방어적으로 체크
                if remove_idx is not None:
                    del league.players[remove_idx]
                else:
                    # 혹시나 learner 하나만 남은 상태에서 꼬이면 루프 탈출
                    break

    env.close()
    return learner, league


# ------------------------------------------------
# kymnasium Agent 래핑
# ------------------------------------------------
class YourBlackAgent(kym.Agent):
    def __init__(self, policy: PPOPolicy | None = None, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if policy is None:
            self.policy = PPOPolicy(device=self.device)
        else:
            self.policy = policy

    def act(self, observation, info):
        # 흑 기준 my_color=0 (평가에서는 greedy=True 기본)
        return self.policy.act_eval(observation, my_color=0)

    def save(self, path: str) -> None:
        self.policy.save(path)

    @classmethod
    def load(cls, path: str) -> "YourBlackAgent":
        policy = PPOPolicy.load(path)
        return cls(policy=policy, device=policy.device)


class YourWhiteAgent(kym.Agent):
    def __init__(self, policy: PPOPolicy | None = None, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if policy is None:
            self.policy = PPOPolicy(device=self.device)
        else:
            self.policy = policy

    def act(self, observation, info):
        # 백 기준 my_color=1
        return self.policy.act_eval(observation, my_color=1)

    def save(self, path: str) -> None:
        self.policy.save(path)

    @classmethod
    def load(cls, path: str) -> "YourWhiteAgent":
        policy = PPOPolicy.load(path)
        return cls(policy=policy, device=policy.device)


# ------------------------------------------------
# 메인 학습 엔트리
# ------------------------------------------------
def main_train():
    """
    학습 전용 엔트리포인트.
    - ELO 리그 self-play (CTDE + potential shaping)로 learner 학습
    - 최종 learner policy를 'shared_policy.pt'로 저장
    - 체크포인트가 있으면 최신에서 이어학습, 없으면 처음부터
    """
    config = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=1e-4,
        update_epochs=3,
        batch_size=64,
    )

    checkpoint_dir = "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ★ 최신 체크포인트 탐색
    latest_ckpt, last_epoch = find_latest_checkpoint(checkpoint_dir)

    if latest_ckpt is not None:
        print(f"[MAIN] Found checkpoint: {latest_ckpt} (epoch={last_epoch})")
        learner = PPOPolicy.load(
            latest_ckpt,
            device=device,
            lr=config.learning_rate,
        )
        start_epoch = last_epoch
    else:
        print("[MAIN] No checkpoint found. Start from scratch.")
        learner = PPOPolicy(device=device, lr=config.learning_rate)
        start_epoch = 0

    learner, league = train_league_selfplay(
        num_epochs=100000000,  # 계속 이어서 갈 거라 사실상 무한
        episodes_per_epoch=50,
        snapshot_interval=5,
        config=config,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=10,
        max_checkpoints=20,
        log_path="training_metrics.csv",
        max_players=50,
        existing_learner=learner,
        start_epoch=start_epoch,
    )

    # 최종 learner 정책 저장
    weight_path = "shared_policy.pt"
    learner.save(weight_path)

    for p in league.players:
        pass


if __name__ == "__main__":
    main_train()
