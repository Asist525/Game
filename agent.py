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

# --- 리워드/셰이핑 관련 (심층리서치 철학 반영) ---
# RL 타임스텝: "내 턴 시작 상태 s_t" → "다음 내 턴 시작 상태 s_{t+1}".
# 이 사이에 발생하는 모든 env.step(내 샷 + 상대 응수)에 대해 STEP_PENALTY를 누적한다.
# 너무 큰 값이면 "빨리 지는게 덜 손해" 메타가 생기므로 스케일을 낮게 시작.
STEP_PENALTY = 0.001         # env step당 -0.001 : 장기전/캠핑 억제(약하게)
ALIVE_TRADE_WEIGHT = 0.25    # Δalive_diff 당 리워드 계수
WIN_REWARD = 1.0             # 최종 승/패 보상 (sign(alive_diff) * WIN_REWARD)

# potential-based shaping: Φ_safe + Φ_center (상태 기반, my_color 기준)
SAFE_ALPHA = 0.15            # Φ_safe 계수
CENTER_ALPHA = 0.10          # Φ_center 계수
SAFE_SATURATION_EDGE = 0.25  # 엣지 거리 saturation 임계값


# ------------------------------------------------
# 환경 생성
# ------------------------------------------------
def make_env(render_mode=None, bgm: bool = False):
    """
    kymnasium AlKkaGi-3x3 환경 생성.
    env 자체 reward는 0이라고 가정하고, 모든 보상은 이 코드에서 정의한다.
    """
    env = gym.make(
        "kymnasium/AlKkaGi-3x3-v0",
        obs_type="custom",
        render_mode=render_mode,
        bgm=bgm,
    )
    # OrderEnforcing wrapper 우회용 디버그 필드
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
    if my_color == 0:
        me = np.array(obs["black"], dtype=np.float32)
        opp = np.array(obs["white"], dtype=np.float32)
    else:
        me = np.array(obs["white"], dtype=np.float32)
        opp = np.array(obs["black"], dtype=np.float32)

    obstacles = np.array(obs["obstacles"], dtype=np.float32)
    turn = float(obs["turn"])
    return me, opp, obstacles, turn


def normalize_stones(stones, board_w, board_h):
    """
    stones: (N_STONES, 3) = [x, y, alive]
    x, y를 [0,1]로 정규화 + clip
    alive 플래그는 그대로 유지.
    """
    out = stones.copy()
    out[:, 0] /= board_w
    out[:, 1] /= board_h
    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
    return out


def normalize_obstacles(obs_arr, board_w, board_h):
    """
    obs_arr: (N_OBS, 4) = [x, y, w, h]
    x, y, w, h를 [0,1] 근처 값으로 정규화 + clip
    """
    out = obs_arr.copy()
    out[:, 0] /= board_w
    out[:, 1] /= board_h
    out[:, 2] /= board_w
    out[:, 3] /= board_h

    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
    out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)
    out[:, 3] = np.clip(out[:, 3], 0.0, 1.0)
    return out


# ------------------------------------------------
# Baseline 인코더 (31차원)
# ------------------------------------------------
def encode_state_basic_alkkagi(
    obs,
    my_color: int,
    board_w: float,
    board_h: float,
) -> np.ndarray:
    """
    Baseline 관측 인코더 (canonical, me/opp 기준)

    첫 번째 스칼라는 "지금 내 턴이냐?"를 나타내는 turn_is_me ∈ {0,1}.
    self-play 루프에서 항상 내 턴일 때만 actor를 호출하므로 실제로는 1.0 고정에 가까움.
    """
    me, opp, obstacles, turn_raw = split_me_opp(obs, my_color)

    turn_is_me = 1.0 if int(turn_raw) == int(my_color) else 0.0

    me_norm = normalize_stones(me, board_w, board_h)
    opp_norm = normalize_stones(opp, board_w, board_h)
    obs_norm = normalize_obstacles(obstacles, board_w, board_h)

    feat = np.concatenate(
        [
            np.array([turn_is_me], dtype=np.float32),
            me_norm.flatten(),
            opp_norm.flatten(),
            obs_norm.flatten(),
        ]
    ).astype(np.float32)

    return feat  # (31,)


# ------------------------------------------------
# Feature Engineering용 helper들
# ------------------------------------------------
def group_stats(stones_norm: np.ndarray) -> np.ndarray:
    """
    stones_norm: (N_STONES, 3) = [x_norm, y_norm, alive]
    alive 돌만 사용해서 center_x, center_y, var_x, var_y 생성.
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


def stone_edge_distances(stones_norm: np.ndarray) -> np.ndarray:
    """
    alive 돌들에 대해 보드 엣지까지의 거리 배열을 반환.
    """
    alive_mask = stones_norm[:, 2] > 0.5
    if not np.any(alive_mask):
        return np.zeros(0, dtype=np.float32)

    xs = stones_norm[alive_mask, 0]
    ys = stones_norm[alive_mask, 1]

    edge_dists = np.minimum.reduce([xs, 1.0 - xs, ys, 1.0 - ys])
    return edge_dists.astype(np.float32)


def obstacle_summary(obs_norm: np.ndarray) -> np.ndarray:
    """
    obs_norm: (N_OBS, 4) = [x_norm, y_norm, w_norm, h_norm]

    - count
    - center_x_mean, center_y_mean
    - w_mean, h_mean
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


# ------------------------------------------------
# Actor용 FE 인코더 (51차원)
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

      => 총 51차원
    """
    base_feat = encode_state_basic_alkkagi(obs, my_color, board_w, board_h)

    me, opp, obstacles, _ = split_me_opp(obs, my_color)
    me_norm = normalize_stones(me, board_w, board_h)
    opp_norm = normalize_stones(opp, board_w, board_h)
    obs_norm = normalize_obstacles(obstacles, board_w, board_h)

    my_alive_cnt = float((me_norm[:, 2] > 0.5).sum())
    opp_alive_cnt = float((opp_norm[:, 2] > 0.5).sum())
    alive_diff = my_alive_cnt - opp_alive_cnt
    denom = my_alive_cnt + opp_alive_cnt
    alive_ratio = my_alive_cnt / denom if denom > 0 else 0.0

    scalar_feats = np.array(
        [my_alive_cnt, opp_alive_cnt, alive_diff, alive_ratio],
        dtype=np.float32,
    )

    my_stats = group_stats(me_norm)
    op_stats = group_stats(opp_norm)

    my_min_edge = min_edge_dist(me_norm)
    op_min_edge = min_edge_dist(opp_norm)
    min_my_op = min_pairwise_dist(me_norm, opp_norm)

    relation_feats = np.array(
        [my_min_edge, op_min_edge, min_my_op],
        dtype=np.float32,
    )

    obs_stats = obstacle_summary(obs_norm)

    extra_feats = np.concatenate(
        [
            scalar_feats,
            my_stats,
            op_stats,
            relation_feats,
            obs_stats,
        ]
    ).astype(np.float32)

    feat = np.concatenate([base_feat, extra_feats]).astype(np.float32)
    return feat  # (51,)


ACTOR_STATE_DIM = 51


def encode_state_fe_tensor(
    obs,
    my_color: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_np = encode_state_fe_alkkagi(obs, my_color, BOARD_W, BOARD_H)
    feat_t = torch.from_numpy(feat_np).to(device=device, dtype=torch.float32)
    return feat_t


# ------------------------------------------------
# Critic용 중앙 인코더 (31차원)
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
    => 31차원
    """
    black = np.array(obs["black"], dtype=np.float32)
    white = np.array(obs["white"], dtype=np.float32)
    obstacles = np.array(obs["obstacles"], dtype=np.float32)
    turn = float(obs["turn"])

    black_norm = normalize_stones(black, board_w, board_h)
    white_norm = normalize_stones(white, board_w, board_h)
    obs_norm = normalize_obstacles(obstacles, board_w, board_h)

    feat = np.concatenate(
        [
            np.array([turn], dtype=np.float32),
            black_norm.flatten(),
            white_norm.flatten(),
            obs_norm.flatten(),
        ]
    ).astype(np.float32)
    return feat  # (31,)


CRITIC_STATE_DIM = 31


def encode_state_central_tensor(
    obs,
    device: torch.device | None = None,
) -> torch.Tensor:
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
    """
    per_stone = N_ANGLES * N_POWERS
    stone_id = action_idx // per_stone
    rem = action_idx % per_stone

    angle_id = rem // N_POWERS
    power_id = rem % N_POWERS

    angle_step = 360.0 / N_ANGLES
    angle = -180.0 + (angle_id + 0.5) * angle_step

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
    stone_id(0,1,2)를 실제 alive 돌 인덱스로 매핑.
    """
    if my_color == 0:
        stones = observation["black"]
    else:
        stones = observation["white"]

    alive_indices = [i for i, s in enumerate(stones) if s[2] > 0.5]
    if not alive_indices:
        return 0

    stone_id = stone_id % len(alive_indices)
    return alive_indices[stone_id]


# ------------------------------------------------
# PPO 네트워크
# ------------------------------------------------
class ActorNet(nn.Module):
    def __init__(self, state_dim: int = ACTOR_STATE_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        h1, h2, h3 = 512, 512, 512
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.policy_head = nn.Linear(h3, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        logits = self.policy_head(h)
        return logits


class CriticNet(nn.Module):
    def __init__(self, state_dim: int = CRITIC_STATE_DIM):
        super().__init__()
        h1, h2 = 512, 512
        self.fc1 = nn.Linear(state_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.value_head = nn.Linear(h2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        value = self.value_head(h).squeeze(-1)
        return value


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01      # 엔트로피 계수 약간 상향
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 1e-4
    update_epochs: int = 3
    batch_size: int = 64


# ------------------------------------------------
# 리워드 관련 함수들 (심층리서치 구조)
# ------------------------------------------------
def compute_alive_counts(obs, my_color: int) -> Tuple[float, float]:
    """
    현재 상태에서 my_color 입장에서 (my_alive, opp_alive) 반환.
    """
    me, opp, _, _ = split_me_opp(obs, my_color)
    me = np.array(me, dtype=np.float32)
    opp = np.array(opp, dtype=np.float32)

    my_alive = float((me[:, 2] > 0.5).sum())
    opp_alive = float((opp[:, 2] > 0.5).sum())
    return my_alive, opp_alive


def compute_alive_diff(obs, my_color: int) -> float:
    """
    현재 상태에서 my_color 입장에서 alive_diff 계산.
    """
    my_alive, opp_alive = compute_alive_counts(obs, my_color)
    return my_alive - opp_alive  # [-3, +3]


def safe_score_side(stones_norm: np.ndarray) -> float:
    """
    엣지로부터의 거리에 기반한 안전도 점수 (0~1).
    SAFE_SATURATION_EDGE 이상이면 포화되도록 설계.
    """
    d = stone_edge_distances(stones_norm)
    if d.size == 0:
        return 0.0
    # [0, SAFE_SATURATION_EDGE] 구간을 [0,1]로 정규화, 그 이상은 1로 포화
    h = np.minimum(d, SAFE_SATURATION_EDGE) / SAFE_SATURATION_EDGE
    return float(h.mean())


def center_score_side(stones_norm: np.ndarray) -> float:
    """
    보드 중심(0.5,0.5)으로부터 거리 기반 중앙 점유 점수 (0~1).
    중앙에 가까울수록 1, 코너에 가까울수록 0 근처.
    """
    alive_mask = stones_norm[:, 2] > 0.5
    if not np.any(alive_mask):
        return 0.0

    xs = stones_norm[alive_mask, 0]
    ys = stones_norm[alive_mask, 1]
    dx = xs - 0.5
    dy = ys - 0.5
    dist = np.sqrt(dx * dx + dy * dy)  # [0, ~0.707]
    max_dist = np.sqrt(2.0) / 2.0  # ≈0.707

    norm = 1.0 - np.clip(dist / max_dist, 0.0, 1.0)
    return float(norm.mean())


def potential(obs, my_color: int) -> float:
    """
    순수 상태 기반 potential Φ(s).
    - alive_diff는 base reward에서만 다루고, Φ에는 포함하지 않는다.
    - Φ_safe: 엣지로부터의 안전도 차이 (saturation 포함)
    - Φ_center: 중앙 점유도 차이
    Φ(s) = SAFE_ALPHA * (safe_our - safe_opp)
         + CENTER_ALPHA * (center_our - center_opp)
    """
    me, opp, _, _ = split_me_opp(obs, my_color)
    me_norm = normalize_stones(me, BOARD_W, BOARD_H)
    opp_norm = normalize_stones(opp, BOARD_W, BOARD_H)

    safe_our = safe_score_side(me_norm)
    safe_opp = safe_score_side(opp_norm)
    center_our = center_score_side(me_norm)
    center_opp = center_score_side(opp_norm)

    phi_safe = safe_our - safe_opp
    phi_center = center_our - center_opp

    return SAFE_ALPHA * phi_safe + CENTER_ALPHA * phi_center


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
    values : (T,)  (각 t에서의 V(s_t))
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
# PPO Policy (Actor / Critic 분리)
# ------------------------------------------------
class PPOPolicy:
    def __init__(self, device: torch.device | None = None, lr: float = 1e-4):
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
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(ckpt, path)

    @staticmethod
    def _partial_load_module(module: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        old_state_dict에서 가능한 부분만 잘라서 module 파라미터에 복사.
        (예: 256→512로 키울 때 상단/좌측 블록만 복사)
        """
        with torch.no_grad():
            for name, param in module.named_parameters():
                if name not in state_dict:
                    continue
                old_param = state_dict[name]

                if param.ndim != old_param.ndim:
                    continue

                if param.ndim == 2:
                    out_dim = min(param.size(0), old_param.size(0))
                    in_dim = min(param.size(1), old_param.size(1))
                    param[:out_dim, :in_dim].copy_(old_param[:out_dim, :in_dim])
                elif param.ndim == 1:
                    out_dim = min(param.size(0), old_param.size(0))
                    param[:out_dim].copy_(old_param[:out_dim])

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device | None = None,
        lr: float = 1e-4,
    ) -> "PPOPolicy":
        policy = cls(device=device, lr=lr)
        ckpt = torch.load(path, map_location=policy.device)

        def _load_actor_from_state_dict(actor_sd: Dict[str, torch.Tensor]) -> bool:
            try:
                policy.actor.load_state_dict(actor_sd)
                return True
            except RuntimeError:
                print("[PPOPolicy.load] Actor shape mismatch, applying partial migration.")
                PPOPolicy._partial_load_module(policy.actor, actor_sd)
                return False

        def _load_critic_from_state_dict(critic_sd: Dict[str, torch.Tensor]) -> bool:
            try:
                policy.critic.load_state_dict(critic_sd)
                return True
            except RuntimeError:
                print("[PPOPolicy.load] Critic shape mismatch, applying partial migration.")
                PPOPolicy._partial_load_module(policy.critic, critic_sd)
                return False

        if isinstance(ckpt, dict) and "actor_state_dict" in ckpt and "critic_state_dict" in ckpt:
            actor_ok = _load_actor_from_state_dict(ckpt["actor_state_dict"])
            critic_ok = _load_critic_from_state_dict(ckpt["critic_state_dict"])

            if "optimizer_state_dict" in ckpt and actor_ok and critic_ok:
                try:
                    policy.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass
        else:
            state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            _load_actor_from_state_dict(state_dict)

        return policy

    @torch.no_grad()
    def act_eval(
        self,
        observation: dict,
        my_color: int,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> dict:
        """
        평가용 act.
        greedy=True  → argmax (대회용)
        greedy=False → 샘플링
        """
        self.actor.eval()

        state_t = encode_state_fe_tensor(
            observation,
            my_color=my_color,
            device=self.device,
        ).unsqueeze(0)

        logits = self.actor(state_t)
        if temperature != 1.0:
            logits = logits / temperature

        dist = torch.distributions.Categorical(logits=logits)

        if greedy:
            probs = dist.probs
            action_idx_t = probs.argmax(dim=-1)
        else:
            action_idx_t = dist.sample()

        action_idx = int(action_idx_t.item())
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
        return action

    def act_train(
        self,
        observation: dict,
        my_color: int,
    ) -> Tuple[dict, int, float, torch.Tensor, torch.Tensor]:
        """
        학습용 act.
        - action dict
        - action_idx
        - logprob
        - actor_state_vec
        - critic_state_vec
        (둘 다 "내 턴 시작 상태 s_t" 기준)
        """
        self.actor.train()
        self.critic.train()

        actor_state_t = encode_state_fe_tensor(
            observation,
            my_color=my_color,
            device=self.device,
        ).unsqueeze(0)

        critic_state_t = encode_state_central_tensor(
            observation,
            device=self.device,
        ).unsqueeze(0)

        logits = self.actor(actor_state_t)
        dist = torch.distributions.Categorical(logits=logits)

        action_idx_t = dist.sample()
        logprob_t = dist.log_prob(action_idx_t)

        action_idx = int(action_idx_t.item())
        logprob = float(logprob_t.item())

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
# PPO 업데이트 (엔트로피/Top-k/ KL/ clip_frac 로그 + 액션 분포)
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
) -> Dict[str, float]:
    device = policy.device
    policy.actor.train()
    policy.critic.train()

    actor_states_t = torch.stack(actor_states).to(device)
    critic_states_t = torch.stack(critic_states).to(device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    old_logprobs_t = torch.tensor(old_logprobs, dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    N = actor_states_t.size(0)
    batch_size = min(config.batch_size, N)

    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []
    approx_kls: List[float] = []
    clip_fracs: List[float] = []
    top1_means: List[float] = []
    top5_means: List[float] = []
    top20_means: List[float] = []
    top50_means: List[float] = []
    grad_norms: List[float] = []

    # 액션 분포 (stone/power) 집계
    power0_ratio = power1_ratio = power2_ratio = 0.0
    stone0_ratio = stone1_ratio = stone2_ratio = 0.0
    if N > 0:
        with torch.no_grad():
            per_stone = N_ANGLES * N_POWERS
            actions_all = actions_t.clone()
            stone_ids = actions_all // per_stone
            rem = actions_all % per_stone
            power_ids = rem % N_POWERS

            stone0_ratio = (stone_ids == 0).float().mean().item()
            stone1_ratio = (stone_ids == 1).float().mean().item()
            stone2_ratio = (stone_ids == 2).float().mean().item()

            power0_ratio = (power_ids == 0).float().mean().item()
            if N_POWERS >= 2:
                power1_ratio = (power_ids == 1).float().mean().item()
            if N_POWERS >= 3:
                power2_ratio = (power_ids == 2).float().mean().item()

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

            logits = policy.actor(mb_actor_states)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # ratio, KL, clip_frac
            logratio = new_logprobs - mb_old_logprobs
            ratio = logratio.exp()
            approx_kl = ((ratio - 1.0) - logratio).mean()

            clip_mask = (ratio - 1.0).abs() > config.clip_coef
            clip_frac = clip_mask.float().mean()

            # top-k 확률
            probs = dist.probs.detach()  # (B, A)
            sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
            top1 = sorted_probs[:, 0]
            k5 = min(5, sorted_probs.size(1))
            k20 = min(20, sorted_probs.size(1))
            k50 = min(50, sorted_probs.size(1))
            top5 = sorted_probs[:, :k5].sum(dim=-1)
            top20 = sorted_probs[:, :k20].sum(dim=-1)
            top50 = sorted_probs[:, :k50].sum(dim=-1)

            top1_means.append(top1.mean().item())
            top5_means.append(top5.mean().item())
            top20_means.append(top20.mean().item())
            top50_means.append(top50.mean().item())

            # PPO objective
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(
                ratio,
                1.0 - config.clip_coef,
                1.0 + config.clip_coef,
            ) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            values = policy.critic(mb_critic_states)
            value_loss = F.mse_loss(values, mb_returns)

            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

            policy.optimizer.zero_grad()
            loss.backward()
            total_params = list(policy.actor.parameters()) + list(policy.critic.parameters())
            grad_norm = nn.utils.clip_grad_norm_(total_params, config.max_grad_norm)
            if isinstance(grad_norm, torch.Tensor):
                grad_norm_val = float(grad_norm.item())
            else:
                grad_norm_val = float(grad_norm)
            grad_norms.append(grad_norm_val)

            policy.optimizer.step()

            # 로그용 누적
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            approx_kls.append(approx_kl.item())
            clip_fracs.append(clip_frac.item())

    metrics: Dict[str, float] = {}
    metrics["policy_loss"] = float(np.mean(policy_losses)) if policy_losses else 0.0
    metrics["value_loss"] = float(np.mean(value_losses)) if value_losses else 0.0
    metrics["entropy_mean"] = float(np.mean(entropies)) if entropies else 0.0
    metrics["entropy_std"] = float(np.std(entropies)) if entropies else 0.0
    metrics["approx_kl"] = float(np.mean(approx_kls)) if approx_kls else 0.0
    metrics["clip_frac"] = float(np.mean(clip_fracs)) if clip_fracs else 0.0
    metrics["top1_prob_mean"] = float(np.mean(top1_means)) if top1_means else 0.0
    metrics["top5_prob_mean"] = float(np.mean(top5_means)) if top5_means else 0.0
    metrics["top20_prob_mean"] = float(np.mean(top20_means)) if top20_means else 0.0
    metrics["top50_prob_mean"] = float(np.mean(top50_means)) if top50_means else 0.0
    metrics["grad_norm_mean"] = float(np.mean(grad_norms)) if grad_norms else 0.0
    metrics["lr"] = float(policy.optimizer.param_groups[0]["lr"])
    metrics["power0_ratio"] = power0_ratio
    metrics["power1_ratio"] = power1_ratio
    metrics["power2_ratio"] = power2_ratio
    metrics["stone0_ratio"] = stone0_ratio
    metrics["stone1_ratio"] = stone1_ratio
    metrics["stone2_ratio"] = stone2_ratio

    return metrics


# ------------------------------------------------
# ELO / 리그
# ------------------------------------------------
@dataclass
class RatedPolicy:
    id: str
    policy: PPOPolicy
    rating: float = 1500.0
    games: int = 0


def elo_expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def elo_update(
    ra: float,
    rb: float,
    score_a: float,
    k: float = 32.0,
) -> Tuple[float, float]:
    ea = elo_expected(ra, rb)
    eb = 1.0 - ea
    ra_new = ra + k * (score_a - ea)
    rb_new = rb + k * ((1.0 - score_a) - eb)
    return ra_new, rb_new


class EloLeague:
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
        li = self.find_index(learner_id)
        learner = self.players[li]

        candidates = [p for p in self.players if p.id != learner_id]
        if not candidates:
            raise RuntimeError("no opponent candidates")

        diffs = np.array([abs(p.rating - learner.rating) for p in candidates], dtype=np.float32)
        weights = np.exp(-diffs / self.temperature)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights /= weights.sum()

        idx = np.random.choice(len(candidates), p=weights)
        return candidates[idx]

    def update_result(self, a_id: str, b_id: str, score_a: float) -> None:
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
# 체크포인트 / 로그 (1: performance, 2: reward, 3: PPO diagnostics)
# ------------------------------------------------
def save_checkpoint_policy(
    policy: PPOPolicy,
    epoch: int,
    checkpoint_dir: str = "checkpoints",
    max_keep: int = 20,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch:06d}.pt")
    policy.save(ckpt_path)

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
    avg_alive_diff: float,
    avg_steps: float,
    steps_p50: float,
    steps_p90: float,
    steps_max: float,
    learner_rating: float,
    num_players: int,
    early_my_alive_avg: float,
    early_opp_alive_avg: float,
    early_alive_drop_ratio: float,
    early_avg_power: float,
    log_path: str = "training_metrics.csv",
) -> None:
    """
    2번 CSV: 성능/리그 관련 메트릭 (승률, 스텝, ELO 등 + early behaviour)
    """
    file_exists = os.path.exists(log_path)

    if not file_exists:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "epoch,episodes,wins,draws,losses,win_rate,"
                "avg_alive_diff,avg_steps,steps_p50,steps_p90,steps_max,"
                "learner_rating,num_players,"
                "early_my_alive_avg,early_opp_alive_avg,early_alive_drop_ratio,early_avg_power\n"
            )

    win_rate = wins / episodes if episodes > 0 else 0.0

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch},{episodes},{wins},{draws},{losses},"
            f"{win_rate:.6f},{avg_alive_diff:.6f},{avg_steps:.6f},"
            f"{steps_p50:.6f},{steps_p90:.6f},{steps_max:.6f},"
            f"{learner_rating:.6f},{num_players},"
            f"{early_my_alive_avg:.6f},{early_opp_alive_avg:.6f},"
            f"{early_alive_drop_ratio:.6f},{early_avg_power:.6f}\n"
        )


def append_reward_breakdown_log(
    epoch: int,
    episodes: int,
    avg_step_penalty: float,
    avg_trade: float,
    avg_shaping: float,
    avg_game_reward: float,
    avg_phi_abs: float,
    log_path: str = "reward_breakdown_metrics.csv",
) -> None:
    """
    1번 CSV: 에포크 단위 리워드 구성요소 평균 로그
      - avg_step_penalty : episode당 step_penalty 합 (보통 음수)
      - avg_trade        : episode당 alive_trade 합
      - avg_shaping      : episode당 potential shaping 합
      - avg_game_reward  : episode당 최종 WIN_REWARD 합
      - total_avg_reward : 위 네 항목 합
      - avg_phi_abs      : segment당 |Φ(s)| 평균 (포지셔닝 shaping 스케일)
    """
    file_exists = os.path.exists(log_path)

    if not file_exists:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "epoch,episodes,avg_step_penalty,avg_alive_trade,"
                "avg_shaping,avg_game_reward,total_avg_reward,avg_phi_abs\n"
            )

    total_avg_reward = avg_step_penalty + avg_trade + avg_shaping + avg_game_reward

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch},{episodes},"
            f"{avg_step_penalty:.6f},{avg_trade:.6f},{avg_shaping:.6f},"
            f"{avg_game_reward:.6f},{total_avg_reward:.6f},{avg_phi_abs:.6f}\n"
        )


def append_ppo_metrics_log(
    epoch: int,
    num_samples: int,
    metrics: Dict[str, float],
    log_path: str = "ppo_diagnostics_metrics.csv",
) -> None:
    """
    3번 CSV: PPO 최적화/정책 형태 진단용 메트릭 로그
      - policy_loss, value_loss
      - entropy_mean, entropy_std
      - approx_kl, clip_frac
      - top1_prob_mean, top5_prob_mean, top20_prob_mean, top50_prob_mean
      - grad_norm_mean, lr
      - power0_ratio, power1_ratio, power2_ratio
      - stone0_ratio, stone1_ratio, stone2_ratio
    """
    file_exists = os.path.exists(log_path)

    if not file_exists:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "epoch,num_samples,policy_loss,value_loss,"
                "entropy_mean,entropy_std,approx_kl,clip_frac,"
                "top1_prob_mean,top5_prob_mean,top20_prob_mean,top50_prob_mean,"
                "grad_norm_mean,lr,"
                "power0_ratio,power1_ratio,power2_ratio,"
                "stone0_ratio,stone1_ratio,stone2_ratio\n"
            )

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch},{num_samples},"
            f"{metrics.get('policy_loss', 0.0):.6f},"
            f"{metrics.get('value_loss', 0.0):.6f},"
            f"{metrics.get('entropy_mean', 0.0):.6f},"
            f"{metrics.get('entropy_std', 0.0):.6f},"
            f"{metrics.get('approx_kl', 0.0):.6f},"
            f"{metrics.get('clip_frac', 0.0):.6f},"
            f"{metrics.get('top1_prob_mean', 0.0):.6f},"
            f"{metrics.get('top5_prob_mean', 0.0):.6f},"
            f"{metrics.get('top20_prob_mean', 0.0):.6f},"
            f"{metrics.get('top50_prob_mean', 0.0):.6f},"
            f"{metrics.get('grad_norm_mean', 0.0):.6f},"
            f"{metrics.get('lr', 0.0):.8f},"
            f"{metrics.get('power0_ratio', 0.0):.6f},"
            f"{metrics.get('power1_ratio', 0.0):.6f},"
            f"{metrics.get('power2_ratio', 0.0):.6f},"
            f"{metrics.get('stone0_ratio', 0.0):.6f},"
            f"{metrics.get('stone1_ratio', 0.0):.6f},"
            f"{metrics.get('stone2_ratio', 0.0):.6f}\n"
        )


def find_latest_checkpoint(
    checkpoint_dir: str = "checkpoints",
) -> Tuple[str | None, int]:
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpts = sorted(Path(checkpoint_dir).glob("policy_epoch_*.pt"))
    if not ckpts:
        return None, 0

    latest = ckpts[-1]
    stem = latest.stem
    epoch = 0
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        epoch = int(parts[-1])
    else:
        try:
            epoch = int(stem[-6:])
        except ValueError:
            epoch = 0

    return str(latest), epoch


# ------------------------------------------------
# 리그 self-play 학습 루프
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
    existing_learner: PPOPolicy | None = None,
    start_epoch: int = 0,
) -> Tuple[PPOPolicy, EloLeague]:
    """
    ELO 리그 기반 self-play 학습 루프.

    RL 타임스텝(학습 기준):
      - 상태 s_t: learner_color의 턴에서 보이는 보드 상태
      - 행동: learner가 쏜 샷
      - 다음 상태 s_{t+1}: 다음에 learner_color에게 턴이 돌아왔을 때 보드 상태
        (그 사이에 env.step은 여러 번 발생할 수 있음: 내 샷 + 상대 응수들)

    reward:
      - base_step:
          * env step당 STEP_PENALTY 누적
          * segment 전체에서 alive_diff 변화에 ALIVE_TRADE_WEIGHT 반영
      - shaping:
          * potential-based: γ Φ(s_{t+1}) - Φ(s_t)
      - episode 마지막 스텝:
          * 최종 alive_diff에 따른 WIN_REWARD (판정승 포함)
          * truncated도 포함해서 "여기서 게임 끝"으로 간주 (GAE bootstrap 없음)
    """
    if config is None:
        config = PPOConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(render_mode=None, bgm=False)

    # learner 초기화/이어학습
    if existing_learner is None:
        learner = PPOPolicy(device=device, lr=config.learning_rate)
        print("[TRAIN] Initialize new learner.")
    else:
        learner = existing_learner
        learner.device = device
        learner.actor.to(device)
        learner.critic.to(device)
        print(f"[TRAIN] Continue from existing learner, start_epoch={start_epoch}")

    # 첫 스냅샷
    snap0 = PPOPolicy(device=device, lr=config.learning_rate)
    snap0.actor.load_state_dict(learner.actor.state_dict())
    snap0.critic.load_state_dict(learner.critic.state_dict())

    league = EloLeague(
        players=[
            RatedPolicy(id="learner", policy=learner, rating=1500.0),
            RatedPolicy(id="snap_000", policy=snap0, rating=1500.0),
        ],
        k=32.0,
        temperature=200.0,
    )
    snapshot_counter = 1

    for epoch_idx in range(start_epoch + 1, start_epoch + num_epochs + 1):
        all_actor_states: List[torch.Tensor] = []
        all_critic_states: List[torch.Tensor] = []
        all_actions: List[int] = []
        all_logprobs: List[float] = []
        all_advantages: List[float] = []
        all_returns: List[float] = []

        ep_wins = ep_draws = ep_losses = 0
        ep_alive_diff_sum = 0.0
        ep_step_sum = 0
        episodes_this_epoch = 0
        episode_steps_list: List[int] = []

        # 리워드 breakdown 누적 (epoch 단위)
        epoch_step_penalty_sum = 0.0
        epoch_trade_sum = 0.0
        epoch_shaping_sum = 0.0
        epoch_game_reward_sum = 0.0

        # potential magnitude / segment 카운트
        epoch_phi_abs_sum = 0.0
        epoch_segment_count = 0

        # early behaviour (learner 첫 3턴 기준)
        epoch_early_my_alive_sum = 0.0
        epoch_early_opp_alive_sum = 0.0
        epoch_early_alive_episodes = 0
        epoch_early_alive_drop_episodes = 0
        epoch_early_power_sum = 0.0
        epoch_early_power_episodes = 0

        for ep in range(episodes_per_epoch):
            opponent = league.choose_opponent("learner")

            # learner_color: 0(흑) 또는 1(백)
            learner_color = int(np.random.randint(0, 2))

            global_seed = (epoch_idx - 1) * episodes_per_epoch + ep
            obs, info = env.reset(seed=global_seed)

            done = False
            step = 0
            truncated = False
            terminated = False

            ep_actor_states: List[torch.Tensor] = []
            ep_critic_states: List[torch.Tensor] = []
            ep_actions: List[int] = []
            ep_logprobs: List[float] = []
            ep_rewards: List[float] = []

            # 리워드 breakdown (episode 단위)
            ep_step_penalty_sum = 0.0
            ep_trade_sum = 0.0
            ep_shaping_sum = 0.0
            ep_game_reward_sum = 0.0

            # potential magnitude / segment (episode 단위)
            ep_phi_abs_sum = 0.0
            ep_segment_count = 0

            # early behaviour (episode 단위)
            early_my_alive_list: List[float] = []
            early_opp_alive_list: List[float] = []
            early_powers: List[float] = []
            learner_turn_count = 0

            # --- 처음 learner에게 턴이 올 때까지 opponent만 두게 함 ---
            while not done and int(obs["turn"]) != learner_color:
                opp_turn_color = int(obs["turn"])
                opp_action = opponent.policy.act_eval(
                    observation=obs,
                    my_color=opp_turn_color,
                    greedy=False,
                )
                obs, _, terminated, truncated, info = env.step(opp_action)
                done = terminated or truncated
                step += 1

            # 에피소드 내 메인 루프: "내 턴 기준" 타임스텝
            while not done:
                # s_t: learner 턴 시작 상태
                turn = int(obs["turn"])
                assert turn == learner_color, "내 턴일 때만 이 블록에 들어와야 함."

                # early behaviour용 alive 카운트
                my_alive, opp_alive = compute_alive_counts(obs, my_color=learner_color)
                learner_turn_count += 1
                if learner_turn_count <= 3:
                    early_my_alive_list.append(my_alive)
                    early_opp_alive_list.append(opp_alive)

                # actor/critic 입력 상태 저장 + 행동 샘플링
                (
                    action,
                    action_idx,
                    logprob,
                    actor_state_vec,
                    critic_state_vec,
                ) = learner.act_train(
                    observation=obs,
                    my_color=learner_color,
                )

                power_used = float(action["power"])
                if learner_turn_count <= 3:
                    early_powers.append(power_used)

                ep_actor_states.append(actor_state_vec.cpu())
                ep_critic_states.append(critic_state_vec.cpu())
                ep_actions.append(action_idx)
                ep_logprobs.append(logprob)

                # base/alive 기준 상태
                alive_before = my_alive - opp_alive
                phi_before = potential(obs, my_color=learner_color)

                # potential magnitude 추적
                ep_phi_abs_sum += abs(phi_before)
                ep_segment_count += 1

                # --- learner 샷 ---
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

                # env step당 패널티: 이번 segment의 첫 step
                segment_steps = 1
                step_penalty_accum = -STEP_PENALTY

                # --- 다음 learner 턴이 오거나 에피소드가 끝날 때까지 opponent 차례 ---
                while not done and int(obs["turn"]) != learner_color:
                    opp_turn_color = int(obs["turn"])
                    opp_action = opponent.policy.act_eval(
                        observation=obs,
                        my_color=opp_turn_color,
                        greedy=False,
                    )
                    obs, _, terminated, truncated, info = env.step(opp_action)
                    done = terminated or truncated
                    step += 1

                    segment_steps += 1
                    step_penalty_accum -= STEP_PENALTY

                # segment 끝: s_{t+1} (다음 learner 턴 시작 상태 또는 terminal)
                alive_after = compute_alive_diff(obs, my_color=learner_color)
                delta_alive = alive_after - alive_before
                r_trade = ALIVE_TRADE_WEIGHT * delta_alive

                base_step = step_penalty_accum + r_trade

                # terminal에서는 Φ(s_{t+1}) = 0으로 처리 (potential-based shaping 이론 정합성 보강)
                if done:
                    phi_after = 0.0
                else:
                    phi_after = potential(obs, my_color=learner_color)

                shaping_term = config.gamma * phi_after - phi_before
                shaped_r = base_step + shaping_term

                ep_rewards.append(shaped_r)

                # breakdown 누적
                ep_step_penalty_sum += step_penalty_accum
                ep_trade_sum += r_trade
                ep_shaping_sum += shaping_term

                # done이면 while 탈출, 아니면 다음 learner 턴으로 계속
                # obs는 이미 "다음 learner 턴 시작 상태" (또는 terminal)

            # 에피소드 종료 후 alive_diff 기준 승/무/패 판단
            R_learner = compute_alive_diff(obs, my_color=learner_color)
            ep_alive_diff_sum += R_learner
            ep_step_sum += step
            episodes_this_epoch += 1
            episode_steps_list.append(step)

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

            # 최종 승/패 보상: truncated 여부와 관계없이 alive_diff 판정으로 반영
            if R_learner > 0:
                game_reward = WIN_REWARD
            elif R_learner < 0:
                game_reward = -WIN_REWARD
            else:
                game_reward = 0.0

            if len(ep_rewards) > 0:
                ep_rewards[-1] += game_reward
                ep_game_reward_sum += game_reward
            # 만약 learner가 한 번도 두지 못하고 끝났다면(ep_rewards 비어있음) 그 에피소드는 학습에 쓰지 않음.

            # GAE
            if len(ep_rewards) > 0:
                rewards_np = np.array(ep_rewards, dtype=np.float32)

                critic_states_ep = torch.stack(ep_critic_states).to(device)
                with torch.no_grad():
                    values_t = learner.critic(critic_states_ep).cpu().numpy()

                # 여기서는 truncated도 포함해서 "판정 종료된 terminal"로 취급하므로
                # 더 이상 미래 value를 bootstrap 하지 않는다.
                bootstrap_value_last = 0.0

                adv_np, ret_np = compute_gae_returns(
                    rewards=rewards_np,
                    values=values_t,
                    gamma=config.gamma,
                    lam=config.gae_lambda,
                    bootstrap_value_last=bootstrap_value_last,
                )

                all_actor_states.extend(ep_actor_states)
                all_critic_states.extend(ep_critic_states)
                all_actions.extend(ep_actions)
                all_logprobs.extend(ep_logprobs)
                all_advantages.extend(adv_np.tolist())
                all_returns.extend(ret_np.tolist())

            # 에피소드 리워드 breakdown을 epoch 누적
            epoch_step_penalty_sum += ep_step_penalty_sum
            epoch_trade_sum += ep_trade_sum
            epoch_shaping_sum += ep_shaping_sum
            epoch_game_reward_sum += ep_game_reward_sum

            # potential magnitude / segment count 누적
            epoch_phi_abs_sum += ep_phi_abs_sum
            epoch_segment_count += ep_segment_count

            # early behaviour 누적
            if len(early_my_alive_list) > 0:
                mean_my_alive = float(np.mean(early_my_alive_list))
                mean_opp_alive = float(np.mean(early_opp_alive_list))
                epoch_early_my_alive_sum += mean_my_alive
                epoch_early_opp_alive_sum += mean_opp_alive
                epoch_early_alive_episodes += 1
                if min(early_my_alive_list) <= 1.0:
                    epoch_early_alive_drop_episodes += 1

            if len(early_powers) > 0:
                mean_power_ep = float(np.mean(early_powers))
                epoch_early_power_sum += mean_power_ep
                epoch_early_power_episodes += 1

        # --- epoch 통계/로그 ---
        if episodes_this_epoch > 0:
            avg_alive_diff = ep_alive_diff_sum / episodes_this_epoch
            avg_steps = ep_step_sum / episodes_this_epoch

            avg_step_penalty = epoch_step_penalty_sum / episodes_this_epoch
            avg_trade = epoch_trade_sum / episodes_this_epoch
            avg_shaping = epoch_shaping_sum / episodes_this_epoch
            avg_game_reward = epoch_game_reward_sum / episodes_this_epoch

            steps_arr = np.array(episode_steps_list, dtype=np.float32)
            steps_p50 = float(np.percentile(steps_arr, 50))
            steps_p90 = float(np.percentile(steps_arr, 90))
            steps_max = float(steps_arr.max())
        else:
            avg_alive_diff = 0.0
            avg_steps = 0.0
            avg_step_penalty = 0.0
            avg_trade = 0.0
            avg_shaping = 0.0
            avg_game_reward = 0.0
            steps_p50 = 0.0
            steps_p90 = 0.0
            steps_max = 0.0

        if epoch_segment_count > 0:
            avg_phi_abs = epoch_phi_abs_sum / epoch_segment_count
        else:
            avg_phi_abs = 0.0

        if epoch_early_alive_episodes > 0:
            early_my_alive_avg = epoch_early_my_alive_sum / epoch_early_alive_episodes
            early_opp_alive_avg = epoch_early_opp_alive_sum / epoch_early_alive_episodes
            early_alive_drop_ratio = epoch_early_alive_drop_episodes / epoch_early_alive_episodes
        else:
            early_my_alive_avg = 0.0
            early_opp_alive_avg = 0.0
            early_alive_drop_ratio = 0.0

        if epoch_early_power_episodes > 0:
            early_avg_power = epoch_early_power_sum / epoch_early_power_episodes
        else:
            early_avg_power = 0.0

        learner_rating = league.players[league.find_index("learner")].rating

        # 2번 CSV: 성능/리그 + early behaviour
        append_epoch_log(
            epoch=epoch_idx,
            episodes=episodes_this_epoch,
            wins=ep_wins,
            draws=ep_draws,
            losses=ep_losses,
            avg_alive_diff=avg_alive_diff,
            avg_steps=avg_steps,
            steps_p50=steps_p50,
            steps_p90=steps_p90,
            steps_max=steps_max,
            learner_rating=learner_rating,
            num_players=len(league.players),
            early_my_alive_avg=early_my_alive_avg,
            early_opp_alive_avg=early_opp_alive_avg,
            early_alive_drop_ratio=early_alive_drop_ratio,
            early_avg_power=early_avg_power,
            log_path=log_path,
        )

        # 1번 CSV: 리워드 breakdown
        append_reward_breakdown_log(
            epoch=epoch_idx,
            episodes=episodes_this_epoch,
            avg_step_penalty=avg_step_penalty,
            avg_trade=avg_trade,
            avg_shaping=avg_shaping,
            avg_game_reward=avg_game_reward,
            avg_phi_abs=avg_phi_abs,
            log_path="reward_breakdown_metrics.csv",
        )

        # --- PPO 업데이트 + 3번 CSV(PPO diagnostics) ---
        if len(all_actor_states) > 0:
            ppo_metrics = ppo_update(
                policy=learner,
                actor_states=all_actor_states,
                critic_states=all_critic_states,
                actions=all_actions,
                old_logprobs=all_logprobs,
                advantages=all_advantages,
                returns=all_returns,
                config=config,
            )
            append_ppo_metrics_log(
                epoch=epoch_idx,
                num_samples=len(all_actor_states),
                metrics=ppo_metrics,
                log_path="ppo_diagnostics_metrics.csv",
            )

        # --- 체크포인트 저장 ---
        if epoch_idx % checkpoint_interval == 0:
            save_checkpoint_policy(
                learner,
                epoch=epoch_idx,
                checkpoint_dir=checkpoint_dir,
                max_keep=max_checkpoints,
            )

        # --- snapshot 추가 & 오래된 플레이어 정리 ---
        if epoch_idx % snapshot_interval == 0:
            snap_policy = PPOPolicy(device=device, lr=config.learning_rate)
            snap_policy.actor.load_state_dict(learner.actor.state_dict())
            snap_policy.critic.load_state_dict(learner.critic.state_dict())

            snap_id = f"snap_{snapshot_counter:03d}"
            snapshot_counter += 1

            league.players.append(
                RatedPolicy(
                    id=snap_id,
                    policy=snap_policy,
                    rating=learner_rating,
                    games=0,
                )
            )

            while len(league.players) > max_players:
                remove_idx = None
                for i, p in enumerate(league.players):
                    if p.id != "learner":
                        remove_idx = i
                        break
                if remove_idx is not None:
                    del league.players[remove_idx]
                else:
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
        return self.policy.act_eval(observation, my_color=0, greedy=True)

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
        return self.policy.act_eval(observation, my_color=1, greedy=True)

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
    - env reward는 0, 여기서 정의한 base + shaping으로만 학습.
    - base:
        * env step당 STEP_PENALTY (segment 단위로 누적)
        * segment 단위 Δalive_diff * ALIVE_TRADE_WEIGHT
        * 에피소드 종료시 WIN_REWARD (alive_diff 판정)
    - shaping:
        * potential-based Φ_safe + Φ_center
          (RL 타임스텝: 내 턴 s_t → 다음 내 턴 s_{t+1} 기준, γ Φ(s') - Φ(s))
    - ELO 리그 self-play
    - truncated도 포함해서 "alive_diff로 판정 끝난 시점"을 terminal로 사용
    - CSV:
        * training_metrics.csv            : 승률/스텝/ELO 등 + early behaviour (2번)
        * reward_breakdown_metrics.csv    : 리워드 구성 요소 + potential magnitude (1번)
        * ppo_diagnostics_metrics.csv     : PPO/정책 형태 진단 + 액션 분포 (3번)
    """
    config = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,   # 엔트로피 계수 상향
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=1e-4,
        update_epochs=3,
        batch_size=64,
    )

    checkpoint_dir = "checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        num_epochs=100000000,
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

    weight_path = "shared_policy.pt"
    learner.save(weight_path)

    for p in league.players:
        pass


if __name__ == "__main__":
    main_train()
