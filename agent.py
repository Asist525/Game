# agent.py
from __future__ import annotations
import gymnasium as gym
import kymnasium as kym  # env 등록용
import numpy as np
import torch
from typing import Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import torch.optim as optim
import os
from pathlib import Path



# ------------------------------------------------
# 기본 설정
# ------------------------------------------------
N_STONES = 3
N_OBS = 3

BOARD_W = 600
BOARD_H = 600

# --- 여기부터 추가: 액션 디스크리타이제이션 ---
N_ANGLES = 16   # 각도 16칸 (22.5도 step)
N_POWERS = 4    # 파워 4단계
N_ACTIONS = N_STONES * N_ANGLES * N_POWERS  # 전체 디스크리트 액션 수

# ------------------------------------------------
# 환경 생성
# ------------------------------------------------
def make_env(render_mode=None, bgm: bool = False):
    env = gym.make(
        "kymnasium/AlKkaGi-3x3-v0",
        obs_type="custom",
        render_mode=render_mode,
        bgm=bgm,
    )
    return env


# ------------------------------------------------
# 1) 공통 전처리 함수들
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
# 2) Baseline 인코더 (31차원)
# ------------------------------------------------
def encode_state_basic_alkkagi(
    obs,
    my_color: int,
    board_w: float,
    board_h: float,
) -> np.ndarray:
    """
    Baseline 관측 인코더
    - turn (1)
    - me_norm  : 3 * (x, y, alive) = 9
    - opp_norm : 3 * (x, y, alive) = 9
    - obs_norm : 3 * (x, y, w, h) = 12

    => 최종 shape: (31,)
    """
    me, opp, obstacles, turn = split_me_opp(obs, my_color)

    me_norm  = normalize_stones(me,  board_w, board_h)           # (3, 3)
    opp_norm = normalize_stones(opp, board_w, board_h)           # (3, 3)
    obs_norm = normalize_obstacles(obstacles, board_w, board_h)  # (3, 4)

    feat = np.concatenate([
        np.array([turn], dtype=np.float32),  # (1,)
        me_norm.flatten(),                   # (9,)
        opp_norm.flatten(),                  # (9,)
        obs_norm.flatten(),                  # (12,)
    ]).astype(np.float32)

    # assert feat.shape == (31,)
    return feat


# ------------------------------------------------
# 3) Feature Engineering용 helper들
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
# 4) Feature Engineering 인코더 (52차원)
# ------------------------------------------------
def encode_state_fe_alkkagi(
    obs,
    my_color: int,
    board_w: float,
    board_h: float,
) -> np.ndarray:
    """
    Feature engineering 버전 state encoder.

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

    # --- 2) 정규화된 돌/장애물 다시 구하기 ---
    me, opp, obstacles, turn_raw = split_me_opp(obs, my_color)
    me_norm  = normalize_stones(me,  board_w, board_h)
    opp_norm = normalize_stones(opp, board_w, board_h)
    obs_norm = normalize_obstacles(obstacles, board_w, board_h)

    # --- 3) scalar feature들 ---
    my_alive_cnt  = float((me_norm[:, 2] > 0.5).sum())
    opp_alive_cnt = float((opp_norm[:, 2] > 0.5).sum())
    alive_diff    = my_alive_cnt - opp_alive_cnt
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
    min_my_op   = min_pairwise_dist(me_norm, opp_norm)

    relation_feats = np.array(
        [my_min_edge, op_min_edge, min_my_op],
        dtype=np.float32,
    )

    # --- 6) 장애물 요약 ---
    obs_stats = obstacle_summary(obs_norm)  # (5,)

    # --- 7) 전부 concat ---
    extra_feats = np.concatenate([
        scalar_feats,   # 5
        my_stats,       # 4
        op_stats,       # 4
        relation_feats, # 3
        obs_stats,      # 5
    ]).astype(np.float32)

    feat = np.concatenate([base_feat, extra_feats]).astype(np.float32)

    # assert feat.shape == (52,)
    return feat


# ------------------------------------------------
# 5) torch 텐서 래퍼
# ------------------------------------------------
def encode_state_basic_tensor(
    obs,
    my_color: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    basic encoder → torch.Tensor
    shape: (31,)  또는 배치 고려하면 (1, 31)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_np = encode_state_basic_alkkagi(obs, my_color, BOARD_W, BOARD_H)
    feat_t = torch.from_numpy(feat_np).to(device=device, dtype=torch.float32)
    return feat_t  # 네트워크 입력에 따라 unsqueeze(0) 해서 씀


def encode_state_fe_tensor(
    obs,
    my_color: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    feature engineering encoder → torch.Tensor
    shape: (52,)  또는 배치 고려하면 (1, 52)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feat_np = encode_state_fe_alkkagi(obs, my_color, BOARD_W, BOARD_H)
    feat_t = torch.from_numpy(feat_np).to(device=device, dtype=torch.float32)
    return feat_t


def decode_action_index(action_idx: int):
    """
    디스크리트 액션 인덱스 -> (stone_id, angle, power)로 변환.

    - stone_id ∈ {0,1,2}
    - angle ∈ [-180, 180] 근처의 균일 그리드
    - power ∈ [500, 2500] 범위에서 균일 그리드
    """
    # 총 하나의 인덱스를 (stone, angle, power) 3중 인덱스로 분해
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
    파일이 없으면 헤더를 먼저 쓰고, 있으면 뒤에 append.
    """
    file_exists = os.path.exists(log_path)

    # 헤더 없으면 생성
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





# ------------------------------------------------
# 6) Policy / Value Network
# ------------------------------------------------
class PolicyValueNet(nn.Module):
    """
    52차원 state -> 정책(logits) + 가치(value) 내는 기본 MLP.
    나중에 PPO 업데이트에서 그대로 사용 가능.
    """

    def __init__(self, state_dim: int = 52, n_actions: int = N_ACTIONS):
        super().__init__()
        hidden = 128
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, state_dim)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logits = self.policy_head(h)              # (B, n_actions)
        value = self.value_head(h).squeeze(-1)    # (B,)
        return logits, value





@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    update_epochs: int = 4
    batch_size: int = 64


def compute_terminal_reward(obs, my_color: int) -> float:
    """
    에피소드가 끝났을 때, 최종 상태에서
    my_color(0=흑,1=백) 입장에서 alive_diff 리워드 계산.
    """
    me, opp, _, _ = split_me_opp(obs, my_color)
    me = np.array(me, dtype=np.float32)
    opp = np.array(opp, dtype=np.float32)

    my_alive = float((me[:, 2] > 0.5).sum())
    opp_alive = float((opp[:, 2] > 0.5).sum())
    alive_diff = my_alive - opp_alive  # [-3, +3] 범위

    return alive_diff


def compute_gae_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    단일 에피소드에 대해 GAE + return 계산.
    rewards: (T,)
    values : (T,)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)

    gae = 0.0
    next_value = 0.0  # terminal에서 bootstrap 없음

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
        returns[t] = advantages[t] + values[t]

    return advantages, returns



def clone_policy(src: PPOPolicy, new_lr: float | None = None) -> PPOPolicy:
    """
    학습 중인 learner policy를 그대로 복사해서
    opponent snapshot으로 쓰기 위한 helper.

    - 파라미터(state_dict)만 복사
    - optimizer는 새로 생성
    """
    device = src.device
    if new_lr is None:
        # 기존 optimizer에서 lr 가져오기 (param_group 0 기준)
        lr = src.optimizer.param_groups[0]["lr"]
    else:
        lr = new_lr

    new_policy = PPOPolicy(device=device, lr=lr)
    new_policy.model.load_state_dict(src.model.state_dict())
    return new_policy



class PPOPolicy:
    """
    하나의 PolicyValueNet을 흑/백 모두가 공유하는 PPO 정책.
    - act_eval : 평가용 (no-grad)
    - act_train: 학습용 (logprob, value, state 텐서까지 반환)
    """

    def __init__(self, device: torch.device | None = None, lr: float = 3e-4):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = PolicyValueNet(state_dim=52, n_actions=N_ACTIONS).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)


    # ===== 여기부터 추가 =====
    def save(self, path: str) -> None:
        """
        모델 파라미터만 저장.
        """
        ckpt = {
            "state_dict": self.model.state_dict(),
        }
        torch.save(ckpt, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device | None = None,
        lr: float = 3e-4,
    ) -> "PPOPolicy":
        """
        저장된 파라미터를 불러와서 PPOPolicy 인스턴스를 다시 만든다.
        """
        policy = cls(device=device, lr=lr)
        ckpt = torch.load(path, map_location=policy.device)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        policy.model.load_state_dict(state_dict)
        return policy


    @torch.no_grad()
    def act_eval(self, observation: Dict[str, Any], my_color: int) -> Dict[str, Any]:
        """
        대회용/평가용 act (그리디 or 샘플링은 나중에 옵션으로 조절).
        """
        self.model.eval()

        state_t = encode_state_fe_tensor(
            observation,
            my_color=my_color,
            device=self.device,
        ).unsqueeze(0)  # (1, 52)

        logits, value = self.model(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample().item()

        stone_id, angle, power = decode_action_index(action_idx)

        # 현재 턴 기준으로 사용할 돌 리스트 선택
        if my_color == 0:
            stones = observation["black"]
        else:
            stones = observation["white"]

        n_stones = len(stones)
        if n_stones > 0:
            stone_id = stone_id % n_stones
        else:
            stone_id = 0

        action = {
            "turn": int(observation["turn"]),
            "index": int(stone_id),
            "power": float(power),
            "angle": float(angle),
        }
        return action

    def act_train(
        self,
        observation: Dict[str, Any],
        my_color: int,
    ) -> tuple[Dict[str, Any], int, float, float, torch.Tensor]:
        """
        학습용 act.
        반환:
          - action dict
          - action_idx (int)
          - logprob (float)
          - value (float)
          - state_tensor (52,)  ← rollout 저장용
        """
        self.model.train()

        state_t = encode_state_fe_tensor(
            observation,
            my_color=my_color,
            device=self.device,
        ).unsqueeze(0)  # (1, 52)

        logits, value_t = self.model(state_t)          # logits: (1, A), value_t: (1,1)
        dist = torch.distributions.Categorical(logits=logits)
        action_idx_t = dist.sample()                   # (1,)
        logprob_t = dist.log_prob(action_idx_t)        # (1,)

        action_idx = int(action_idx_t.item())
        logprob = float(logprob_t.item())
        value = float(value_t.squeeze(0).item())

        stone_id, angle, power = decode_action_index(action_idx)

        if my_color == 0:
            stones = observation["black"]
        else:
            stones = observation["white"]

        n_stones = len(stones)
        if n_stones > 0:
            stone_id = stone_id % n_stones
        else:
            stone_id = 0

        action = {
            "turn": int(observation["turn"]),
            "index": int(stone_id),
            "power": float(power),
            "angle": float(angle),
        }

        # (52,) 텐서로 저장 (배치 차원 제거)
        state_vec = state_t.squeeze(0).detach()  # (52,)

        return action, action_idx, logprob, value, state_vec


def ppo_update(
    policy: PPOPolicy,
    states: list[torch.Tensor],
    actions: list[int],
    old_logprobs: list[float],
    advantages: list[float],
    returns: list[float],
    config: PPOConfig,
):
    device = policy.device
    policy.model.train()

    states_t = torch.stack(states).to(device)  # (N, 52)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)             # (N,)
    old_logprobs_t = torch.tensor(old_logprobs, dtype=torch.float32, device=device)  # (N,)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)      # (N,)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)            # (N,)

    # advantage 정규화
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    N = states_t.size(0)
    batch_size = min(config.batch_size, N)

    for _ in range(config.update_epochs):
        idx = torch.randperm(N, device=device)

        for start in range(0, N, batch_size):
            mb_idx = idx[start:start + batch_size]

            mb_states = states_t[mb_idx]
            mb_actions = actions_t[mb_idx]
            mb_old_logprobs = old_logprobs_t[mb_idx]
            mb_adv = advantages_t[mb_idx]
            mb_returns = returns_t[mb_idx]

            logits, values = policy.model(mb_states)      # logits: (B, A), values: (B,)
            dist = torch.distributions.Categorical(logits=logits)
            new_logprobs = dist.log_prob(mb_actions)      # (B,)
            entropy = dist.entropy().mean()

            ratio = (new_logprobs - mb_old_logprobs).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(
                ratio,
                1.0 - config.clip_coef,
                1.0 + config.clip_coef,
            ) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, mb_returns)

            loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy

            policy.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.model.parameters(), config.max_grad_norm)
            policy.optimizer.step()




def train_selfplay_shared_ppo(
    num_episodes: int = 50,
    config: PPOConfig | None = None,
) -> PPOPolicy:
    """
    흑/백 모두 같은 PPOPolicy를 사용하는 self-play 뼈대.
    - 에피소드마다 env reset
    - obs['turn'] == 0 → 흑 관점으로 act_train
    - obs['turn'] == 1 → 백 관점으로 act_train
    - 터미널에서:
        R_black = compute_terminal_reward(obs_final, my_color=0)
        R_white = compute_terminal_reward(obs_final, my_color=1)
      를 각 색 마지막 타임스텝에 부여
    - 흑/백 rollout을 합쳐서 한 번에 PPO 업데이트
    """
    if config is None:
        config = PPOConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(render_mode=None, bgm=False)

    policy = PPOPolicy(device=device, lr=config.learning_rate)

    all_states: list[torch.Tensor] = []
    all_actions: list[int] = []
    all_logprobs: list[float] = []
    all_advantages: list[float] = []
    all_returns: list[float] = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        step = 0

        # 색별 rollout 저장
        states_black: list[torch.Tensor] = []
        actions_black: list[int] = []
        logprobs_black: list[float] = []
        values_black: list[float] = []
        rewards_black: list[float] = []

        states_white: list[torch.Tensor] = []
        actions_white: list[int] = []
        logprobs_white: list[float] = []
        values_white: list[float] = []
        rewards_white: list[float] = []

        while not done:
            turn = int(obs["turn"])  # 0=흑, 1=백

            # 현재 턴 색 기준으로 관측 인코딩 + 액션 샘플
            action, action_idx, logprob, value, state_vec = policy.act_train(
                observation=obs,
                my_color=turn,
            )

            # 색별로 rollout 쌓기
            if turn == 0:
                states_black.append(state_vec.cpu())
                actions_black.append(action_idx)
                logprobs_black.append(logprob)
                values_black.append(value)
                rewards_black.append(0.0)  # 중간은 0, 터미널에 한 번에 줌
            else:
                states_white.append(state_vec.cpu())
                actions_white.append(action_idx)
                logprobs_white.append(logprob)
                values_white.append(value)
                rewards_white.append(0.0)

            # 환경 진행
            obs, reward_env, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        # 에피소드 종료 후, 최종 상태 기준 터미널 리워드 계산
        R_black = compute_terminal_reward(obs, my_color=0)
        R_white = compute_terminal_reward(obs, my_color=1)  # 자동으로 -R_black

        if len(rewards_black) > 0:
            rewards_black[-1] += R_black
        if len(rewards_white) > 0:
            rewards_white[-1] += R_white

        # 색별로 GAE + return 계산 후, 전체 버퍼에 합치기
        if len(rewards_black) > 0:
            r_b = np.array(rewards_black, dtype=np.float32)
            v_b = np.array(values_black, dtype=np.float32)
            adv_b, ret_b = compute_gae_returns(
                rewards=r_b,
                values=v_b,
                gamma=config.gamma,
                lam=config.gae_lambda,
            )

            all_states.extend(states_black)
            all_actions.extend(actions_black)
            all_logprobs.extend(logprobs_black)
            all_advantages.extend(adv_b.tolist())
            all_returns.extend(ret_b.tolist())

        if len(rewards_white) > 0:
            r_w = np.array(rewards_white, dtype=np.float32)
            v_w = np.array(values_white, dtype=np.float32)
            adv_w, ret_w = compute_gae_returns(
                rewards=r_w,
                values=v_w,
                gamma=config.gamma,
                lam=config.gae_lambda,
            )

            all_states.extend(states_white)
            all_actions.extend(actions_white)
            all_logprobs.extend(logprobs_white)
            all_advantages.extend(adv_w.tolist())
            all_returns.extend(ret_w.tolist())

        # print(
        #     f"[Ep {ep+1:03d}/{num_episodes:03d}] "
        #     f"steps={step}, R_black={R_black:.2f}, R_white={R_white:.2f}, "
        #     f"transitions_B={len(rewards_black)}, W={len(rewards_white)}"
        # )

    env.close()

    # 수집된 전체 transition으로 PPO 한 번(or 여러 번) 업데이트
    if len(all_states) > 0:
        # print(f"\n[ PPO UPDATE ] total transitions = {len(all_states)}")
        ppo_update(
            policy=policy,
            states=all_states,
            actions=all_actions,
            old_logprobs=all_logprobs,
            advantages=all_advantages,
            returns=all_returns,
            config=config,
        )
    else:
        # print("수집된 transition이 없습니다.")
        pass

    return policy




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
        # 흑 기준 my_color=0
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


@dataclass
class RatedPolicy:
    """
    ELO 리그에서 하나의 정책 엔트리.
    - id    : 'main_v1', 'snapshot_005' 같은 이름
    - policy: PPOPolicy 인스턴스 (또는 나중에 path만 들고 있어도 됨)
    - rating: 현재 ELO 점수
    - games : 총 경기 수
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
    B의 score는 1-score_a로 본다.
    """
    ea = elo_expected(ra, rb)
    eb = 1.0 - ea

    ra_new = ra + k * (score_a - ea)
    rb_new = rb + k * ((1.0 - score_a) - eb)
    return ra_new, rb_new


class EloLeague:
    """
    ELO 기반 self-play 리그 구조 뼈대.
    - players 리스트에서 learner와 opponent를 고르고
    - 경기 결과에 따라 레이팅 업데이트.
    """

    def __init__(self, players: List[RatedPolicy], k: float = 32.0):
        self.players = players
        self.k = k

    def find_index(self, player_id: str) -> int:
        for i, p in enumerate(self.players):
            if p.id == player_id:
                return i
        raise ValueError(f"player_id '{player_id}' not found")

    def choose_opponent(self, learner_id: str) -> RatedPolicy:
        """
        가장 간단한 버전:
        - learner와 rating 차이가 가장 작은 상대를 고른다.
        나중에 랜덤 샘플링/탐험 비율 추가 가능.
        """
        li = self.find_index(learner_id)
        learner = self.players[li]

        candidates = [
            p for p in self.players
            if p.id != learner_id
        ]
        if not candidates:
            raise RuntimeError("opponent candidates가 없습니다.")

        # rating 차이 최소인 상대 선택
        opponent = min(candidates, key=lambda p: abs(p.rating - learner.rating))
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




def train_league_selfplay(
    num_epochs: int = 5,
    episodes_per_epoch: int = 20,
    snapshot_interval: int = 2,
    config: PPOConfig | None = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 5000,
    max_checkpoints: int = 20,
    log_path: str = "training_metrics.csv",
) -> tuple[PPOPolicy, EloLeague]:
    """
    ELO 리그 기반 self-play 학습 루프.

    - learner : 현재 학습 중인 최신 정책 (ID: "learner")
    - opponents: 과거 스냅샷들 (ID: "snap_000", "snap_001", ...)

    매 epoch:
      1) 여러 에피소드 동안 learner vs opponent self-play
      2) learner의 transition으로 PPO 업데이트
      3) snapshot_interval마다 learner 스냅샷 생성
      4) checkpoint_interval마다 checkpoints/에 ckpt 저장 (최근 max_checkpoints개 유지)
      5) training_metrics.csv에 에폭별 요약 지표 기록
    """
    if config is None:
        config = PPOConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(render_mode=None, bgm=False)

    # 1) learner 초기화
    learner = PPOPolicy(device=device, lr=config.learning_rate)
    # 2) 초기 opponent 스냅샷 하나 복제
    snap0 = clone_policy(learner)

    # 3) ELO 리그 초기화
    league = EloLeague(
        players=[
            RatedPolicy(id="learner", policy=learner, rating=1500.0),
            RatedPolicy(id="snap_000", policy=snap0, rating=1500.0),
        ],
        k=32.0,
    )
    snapshot_counter = 1

    for epoch in range(num_epochs):
        all_states: list[torch.Tensor] = []
        all_actions: list[int] = []
        all_logprobs: list[float] = []
        all_advantages: list[float] = []
        all_returns: list[float] = []

        # ---- 에폭 집계용 지표 ----
        ep_wins = 0
        ep_draws = 0
        ep_losses = 0
        ep_reward_sum = 0.0
        ep_step_sum = 0
        episodes_this_epoch = 0

        # print(f"\n===== [Epoch {epoch+1}/{num_epochs}] =====")
        for ep in range(episodes_per_epoch):
            # --- opponent 선택 (learner와 rating 가까운 상대) ---
            opponent = league.choose_opponent("learner")

            # learner 색을 랜덤으로 배치: 0=흑, 1=백
            learner_color = int(np.random.randint(0, 2))

            # env reset
            global_seed = epoch * episodes_per_epoch + ep
            obs, info = env.reset(seed=global_seed)
            done = False
            step = 0

            # learner rollout 버퍼
            ep_states: list[torch.Tensor] = []
            ep_actions: list[int] = []
            ep_logprobs: list[float] = []
            ep_values: list[float] = []
            ep_rewards: list[float] = []

            while not done:
                turn = int(obs["turn"])  # 현재 턴 색 (0 or 1)

                if turn == learner_color:
                    # ----- learner 턴 (학습 대상) -----
                    action, action_idx, logprob, value, state_vec = learner.act_train(
                        observation=obs,
                        my_color=turn,
                    )

                    ep_states.append(state_vec.cpu())
                    ep_actions.append(action_idx)
                    ep_logprobs.append(logprob)
                    ep_values.append(value)
                    ep_rewards.append(0.0)  # 터미널에서 한 번에 줌
                else:
                    # ----- opponent 턴 (gradient X) -----
                    action = opponent.policy.act_eval(
                        observation=obs,
                        my_color=turn,
                    )

                obs, reward_env, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step += 1

            # ---- 에피소드 종료: learner 입장에서 최종 리워드 & 승패 계산 ----
            R_learner = compute_terminal_reward(obs, my_color=learner_color)
            if len(ep_rewards) > 0:
                ep_rewards[-1] += R_learner

            # 승패 → ELO score (1/0.5/0)
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

            # 에폭 집계용
            episodes_this_epoch += 1
            ep_reward_sum += R_learner
            ep_step_sum += step

            # ---- learner rollout에 대해 GAE + return 계산 ----
            if len(ep_rewards) > 0:
                rewards_np = np.array(ep_rewards, dtype=np.float32)
                values_np = np.array(ep_values, dtype=np.float32)
                adv_np, ret_np = compute_gae_returns(
                    rewards=rewards_np,
                    values=values_np,
                    gamma=config.gamma,
                    lam=config.gae_lambda,
                )

                all_states.extend(ep_states)
                all_actions.extend(ep_actions)
                all_logprobs.extend(ep_logprobs)
                all_advantages.extend(adv_np.tolist())
                all_returns.extend(ret_np.tolist())

            learner_idx = league.find_index("learner")
            learner_rating = league.players[learner_idx].rating

            # print(
            #     f"[Ep {epoch+1:02d}-{ep+1:03d}] "
            #     f"steps={step}, learner_color={learner_color}, "
            #     f"R_learner={R_learner:.2f}, score={score_a:.1f}, "
            #     f"opp_id={opponent.id}, "
            #     f"learner_rating={learner_rating:.1f}, "
            #     f"opp_rating={opponent.rating:.1f}"
            # )

        # ---- 에폭 요약 지표 계산 ----
        if episodes_this_epoch > 0:
            avg_R = ep_reward_sum / episodes_this_epoch
            avg_steps = ep_step_sum / episodes_this_epoch
            win_rate = ep_wins / episodes_this_epoch
        else:
            avg_R = 0.0
            avg_steps = 0.0
            win_rate = 0.0

        learner_idx = league.find_index("learner")
        learner_rating = league.players[learner_idx].rating
        num_players = len(league.players)

        # 콘솔 요약 출력
        # print(
        #     f"[Epoch {epoch+1}] SUMMARY: "
        #     f"W/D/L={ep_wins}/{ep_draws}/{ep_losses} "
        #     f"(win_rate={win_rate:.3f}), "
        #     f"avg_R={avg_R:.2f}, avg_steps={avg_steps:.2f}, "
        #     f"learner_rating={learner_rating:.1f}, "
        #     f"num_players={num_players}"
        # )

        # CSV에 기록
        append_epoch_log(
            epoch=epoch + 1,
            episodes=episodes_this_epoch,
            wins=ep_wins,
            draws=ep_draws,
            losses=ep_losses,
            avg_reward=avg_R,
            avg_steps=avg_steps,
            learner_rating=learner_rating,
            num_players=num_players,
            log_path=log_path,
        )

        # ---- epoch 끝: PPO 업데이트 ----
        if len(all_states) > 0:
            # print(f"[Epoch {epoch+1}] PPO UPDATE: total transitions = {len(all_states)}")
            ppo_update(
                policy=learner,
                states=all_states,
                actions=all_actions,
                old_logprobs=all_logprobs,
                advantages=all_advantages,
                returns=all_returns,
                config=config,
            )
        else:
            # print(f"[Epoch {epoch+1}] 수집된 transition이 없습니다.")
            pass

        # ---- 체크포인트 저장 (예: 5000 epoch마다) ----
        if (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = save_checkpoint_policy(
                learner,
                epoch=epoch + 1,
                checkpoint_dir=checkpoint_dir,
                max_keep=max_checkpoints,
            )
            # print(f"[Epoch {epoch+1}] Checkpoint saved to {ckpt_path}")

        # ---- snapshot 저장 (interval마다) ----
        if (epoch + 1) % snapshot_interval == 0:
            learner_idx = league.find_index("learner")
            learner_rating = league.players[learner_idx].rating

            snap_policy = clone_policy(learner)
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
            # print(f"[Epoch {epoch+1}] New snapshot added: {snap_id} (rating={learner_rating:.1f})")

    env.close()
    return learner, league


def main_train():
    """
    학습 전용 엔트리포인트.
    - ELO 리그 self-play로 learner 학습
    - 최종 learner policy를 'shared_policy.pt'로 저장
    """
    config = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        update_epochs=2,     # 테스트/디버그 땐 작게, 진짜 학습할 땐 키우면 됨
        batch_size=64,
    )

    learner, league = train_league_selfplay(
        num_epochs=2000,
        episodes_per_epoch=100,
        snapshot_interval=20,
        config=config,
        checkpoint_dir="checkpoints",
        checkpoint_interval=200,
        max_checkpoints=20,
        log_path="training_metrics.csv",
    )

    # 최종 learner 정책 저장
    weight_path = "shared_policy.pt"
    learner.save(weight_path)
    # print(f"[TRAIN DONE] Saved learner policy to '{weight_path}'")

    # print("\n=== Final League Ratings ===")
    for p in league.players:
        # print(f"id={p.id}, rating={p.rating:.1f}, games={p.games}")
        pass

if __name__ == "__main__":
    # 이 파일을 직접 실행하면 학습 모드로 돌도록
    main_train()












