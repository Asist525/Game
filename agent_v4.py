from __future__ import annotations

import gymnasium as gym
import kymnasium as kym  # env 등록용
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import csv
import torch.optim as optim  
# ================================================================
# 0. 기본 상수
# ================================================================

BOARD_W = 600.0
BOARD_H = 600.0

N_STONES = 3
N_OBS = 3  # obstacles 개수 (env 기본이 3개)

# 디스크리트 액션 설정
N_ANGLES = 24   # 각도 24bins -> 15도 간격
N_POWERS = 6    # 파워 6단계
N_ACTIONS = N_STONES * N_ANGLES * N_POWERS

# 연속 파워/각도 범위 (env 스펙)
MIN_POWER = 1.0
MAX_POWER = 2500.0
ANGLE_LOW = -180.0
ANGLE_HIGH = 180.0

STATE_DIM = 31  # encode_state_basic 출력 차원


# ================================================================
# 1. 관측/액션 타입 정의
# ================================================================

@dataclass
class AlkkagiAction:
    """알까기 환경에 넘길 액션 dict를 타입 안전하게 다루기 위한 구조체."""
    turn: int      # 0: black, 1: white
    index: int     # 돌 인덱스 (0,1,2)
    power: float   # [1, 2500]
    angle: float   # [-180, 180]


@dataclass
class AlkkagiObservation:
    """원시 obs dict를 조금 더 타입 분명하게 감싸는 래퍼."""
    raw: Dict[str, Any]

    @property
    def turn(self) -> int:
        # 0: 흑 차례, 1: 백 차례
        return int(self.raw["turn"])

    @property
    def black(self) -> np.ndarray:
        # shape: (3, 3) -> [ [x, y, alive], ... ]
        return np.array(self.raw["black"], dtype=np.float32)

    @property
    def white(self) -> np.ndarray:
        return np.array(self.raw["white"], dtype=np.float32)

    @property
    def obstacles(self) -> np.ndarray:
        # shape: (3, 4) -> [x, y, w, h]
        return np.array(self.raw["obstacles"], dtype=np.float32)


# ================================================================
# 2. 환경 래퍼 + 리워드 셰이핑
# ================================================================

# 리워드 셰이핑 상수
WIN_REWARD       = 5.0    # 또는 5.0
KILL_BONUS       = 1.0    # 상대 돌 한 개 → +1
SELF_LOSS_PENAL  = 0    # 내 돌 한 개 → -1 => 일단 없애기
STEP_PENALTY     = 0.001   # 그대로 두거나, 체감되게 0.001 정도(실제로는 -0.001)



def count_alive_stones(obs: AlkkagiObservation, color: int) -> int:
    """color: 0=black, 1=white 기준으로 살아있는 돌 개수."""
    assert color in (0, 1)
    stones = obs.black if color == 0 else obs.white
    return int((stones[:, 2] > 0.5).sum())


def compute_shaped_reward(
    prev_obs: AlkkagiObservation,
    next_obs: AlkkagiObservation,
    acting_color: int,
    terminated: bool,
    truncated: bool,
) -> float:
    """
    한 스텝 동안의 shaped reward를 계산.
    - acting_color: 이번 스텝에서 행동한 쪽(0=흑, 1=백)
    """
    assert acting_color in (0, 1)

    my_color  = acting_color
    opp_color = 1 - acting_color

    my_alive_prev  = count_alive_stones(prev_obs, my_color)
    opp_alive_prev = count_alive_stones(prev_obs, opp_color)
    my_alive_next  = count_alive_stones(next_obs, my_color)
    opp_alive_next = count_alive_stones(next_obs, opp_color)

    kill_diff      = opp_alive_prev - opp_alive_next   # 내가 죽인 상대 돌 수
    self_loss_diff = my_alive_prev - my_alive_next     # 내가 잃은 내 돌 수

    reward = 0.0
    reward += KILL_BONUS      * kill_diff
    reward -= SELF_LOSS_PENAL * self_loss_diff
    reward -= STEP_PENALTY

    # 에피소드 종료 시 승/패 보너스
    if terminated or truncated:
        if opp_alive_next == 0 and my_alive_next > 0:
            # 내가 이김
            reward += WIN_REWARD
        elif my_alive_next == 0 and opp_alive_next > 0:
            # 내가 짐
            reward -= WIN_REWARD
        # 둘 다 0 또는 둘 다 >0 이면 무승부 -> 추가 보상 없음

    return float(reward)


class AlkkagiEnvWrapper:
    """
    kymnasium/AlKkaGi-3x3-v0 래퍼.
    - reset / step 인터페이스 정리
    - reward 셰이핑 적용
    """
    def __init__(
        self,
        render_mode: Optional[str] = None,  # "human" / "rgb_array" / None
        bgm: bool = False,
    ):
        self.env = gym.make(
            "kymnasium/AlKkaGi-3x3-v0",
            obs_type="custom",
            render_mode=render_mode,
            bgm=bgm,
        )
        self.last_obs: Optional[AlkkagiObservation] = None
        self.last_info: Optional[Dict[str, Any]] = None

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> Tuple[AlkkagiObservation, Dict[str, Any]]:
        obs_raw, info = self.env.reset(seed=seed)
        obs = AlkkagiObservation(obs_raw)
        self.last_obs = obs
        self.last_info = info
        return obs, info

    def step(
        self,
        action: AlkkagiAction,
    ) -> Tuple[AlkkagiObservation, float, bool, bool, Dict[str, Any]]:
        act_dict = {
            "turn": int(action.turn),
            "index": int(action.index),
            "power": float(action.power),
            "angle": float(action.angle),
        }

        prev_obs = self.last_obs
        obs_raw, env_reward, terminated, truncated, info = self.env.step(act_dict)
        next_obs = AlkkagiObservation(obs_raw)

        if prev_obs is None:
            shaped_reward = 0.0
        else:
            shaped_reward = compute_shaped_reward(
                prev_obs=prev_obs,
                next_obs=next_obs,
                acting_color=action.turn,
                terminated=terminated,
                truncated=truncated,
            )

        self.last_obs = next_obs
        self.last_info = info

        return next_obs, shaped_reward, bool(terminated), bool(truncated), info

    def close(self):
        self.env.close()


# ================================================================
# 3. 관측 인코더 (31차원)
# ================================================================

def encode_state_basic(
    obs: AlkkagiObservation,
    my_color: int,
) -> np.ndarray:
    """
    알까기 관측을 31차원 실수 벡터로 인코딩.
    - 항상 '내 관점(me vs opp)'으로 정렬
    - 좌표/크기는 [0,1] 범위로 정규화
    """
    assert my_color in (0, 1), "my_color는 0(흑) 또는 1(백)이어야 합니다."

    # 현재 턴 정보
    is_my_turn = 1.0 if obs.turn == my_color else 0.0

    # 내 돌 / 상대 돌 분리
    if my_color == 0:  # 내가 흑
        me = obs.black   # shape (3,3)
        opp = obs.white
    else:               # 내가 백
        me = obs.white
        opp = obs.black

    obstacles = obs.obstacles  # shape (3,4)

    features = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    # 1) 내 턴인지 여부
    features[idx] = is_my_turn
    idx += 1

    # 2) 나(me)의 돌 3개: 각 [x_norm, y_norm, alive]
    for i in range(N_STONES):
        x, y, alive = me[i]
        features[idx + 0] = x / BOARD_W
        features[idx + 1] = y / BOARD_H
        features[idx + 2] = alive
        idx += 3

    # 3) 상대(opp)의 돌 3개
    for i in range(N_STONES):
        x, y, alive = opp[i]
        features[idx + 0] = x / BOARD_W
        features[idx + 1] = y / BOARD_H
        features[idx + 2] = alive
        idx += 3

    # 4) 장애물 3개: 각 [x_norm, y_norm, w_norm, h_norm]
    for i in range(N_OBS):
        x, y, w, h = obstacles[i]
        features[idx + 0] = x / BOARD_W
        features[idx + 1] = y / BOARD_H
        features[idx + 2] = w / BOARD_W
        features[idx + 3] = h / BOARD_H
        idx += 4

    assert idx == STATE_DIM, f"feature index mismatch: idx={idx}"
    return features


def encode_state_basic_tensor(
    obs: AlkkagiObservation,
    my_color: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    feat_np = encode_state_basic(obs, my_color)
    return torch.from_numpy(feat_np).to(device=device, dtype=torch.float32)


# ================================================================
# 4. 디스크리트 액션 + 마스크
# ================================================================

def decode_discrete_action(action_idx: int) -> tuple[int, int, int]:
    """
    디스크리트 액션 인덱스 -> (stone_idx, angle_idx, power_idx)
    """
    assert 0 <= action_idx < N_ACTIONS, f"action_idx 범위 오류: {action_idx}"

    stones_block = N_ANGLES * N_POWERS
    stone_idx = action_idx // stones_block
    rem = action_idx % stones_block
    angle_idx = rem // N_POWERS
    power_idx = rem % N_POWERS
    return stone_idx, angle_idx, power_idx


def bins_to_angle(angle_idx: int) -> float:
    """
    angle_idx (0..N_ANGLES-1) -> 실제 각도 값 (도 단위)
    """
    assert 0 <= angle_idx < N_ANGLES
    delta = (ANGLE_HIGH - ANGLE_LOW) / N_ANGLES  # 360 / N_ANGLES
    angle = ANGLE_LOW + (angle_idx + 0.5) * delta
    return float(angle)


def bins_to_power(power_idx: int) -> float:
    """
    power_idx (0..N_POWERS-1) -> 실제 파워 값
    """
    assert 0 <= power_idx < N_POWERS
    delta = (MAX_POWER - MIN_POWER) / N_POWERS
    power = MIN_POWER + (power_idx + 0.5) * delta
    return float(power)


def discrete_to_env_action(
    action_idx: int,
    obs: AlkkagiObservation,
) -> AlkkagiAction:
    """
    디스크리트 액션 인덱스를 env에 넣을 수 있는 AlkkagiAction으로 변환.
    - turn은 항상 obs.turn에 맞춘다 (지금 누구 차례인지).
    """
    stone_idx, angle_idx, power_idx = decode_discrete_action(action_idx)

    angle = bins_to_angle(angle_idx)
    power = bins_to_power(power_idx)
    turn = obs.turn

    return AlkkagiAction(
        turn=turn,
        index=stone_idx,
        power=power,
        angle=angle,
    )


def get_valid_action_mask(obs: AlkkagiObservation) -> np.ndarray:
    """
    현재 obs 기준으로 유효한 디스크리트 액션 마스크 반환.
    - 기준: 현재 턴(obs.turn)의 살아있는 돌만 선택 가능
    - 반환: shape (N_ACTIONS,), 값은 {0.0, 1.0}
    """
    my_color = obs.turn
    stones = obs.black if my_color == 0 else obs.white
    alive_indices = [i for i, s in enumerate(stones) if s[2] > 0.5]

    mask = np.zeros(N_ACTIONS, dtype=np.float32)

    for a in range(N_ACTIONS):
        stone_idx, _, _ = decode_discrete_action(a)
        if stone_idx in alive_indices:
            mask[a] = 1.0

    # 전부 0이면 fallback
    if mask.sum() == 0:
        mask[:] = 1.0

    return mask


# ================================================================
# 5. Policy + Value 네트워크
# ================================================================

class PolicyValueNet(nn.Module):
    """
    알까기용 정책+가치 네트워크.
    - 입력: 31차원 상태 (내 관점)
    - 출력: policy_logits (N_ACTIONS), state_value (1,)
    """
    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions

        hidden_size = 256

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.policy_head = nn.Linear(hidden_size, n_actions)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, state_dim) 또는 (state_dim,)
        return:
          - logits: (B, n_actions)
          - value:  (B,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (state_dim,) -> (1, state_dim)

        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        logits = self.policy_head(h)              # (B,N_ACTIONS)
        value = self.value_head(h).squeeze(-1)    # (B,)

        return logits, value


# ================================================================
# 6. Rollout Buffer + GAE
# ================================================================

class RolloutBuffer:
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.masks: List[np.ndarray] = []  # valid action mask

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.masks.clear()
        self.advantages = None
        self.returns = None

    def add(
        self,
        state_np: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        mask_np: np.ndarray,
    ):
        self.states.append(state_np.astype(np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.masks.append(mask_np.astype(np.float32))

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        T = len(self.rewards)
        self.advantages = np.zeros(T, dtype=np.float32)
        self.returns = np.zeros(T, dtype=np.float32)

        last_adv = 0.0
        next_value = 0.0  # 마지막 다음 상태의 V(s')는 0으로

        for t in reversed(range(T)):
            done = self.dones[t]
            mask = 0.0 if done else 1.0

            delta = (
                self.rewards[t]
                + gamma * next_value * mask
                - self.values[t]
            )
            last_adv = delta + gamma * gae_lambda * mask * last_adv

            self.advantages[t] = last_adv
            next_value = self.values[t]

        self.returns = self.advantages + np.array(self.values, dtype=np.float32)


# ================================================================
# 7. PPO 업데이트
# ================================================================

def ppo_update(
    net: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device | str = "cpu",
    batch_size: int = 512,
    ppo_epochs: int = 4,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
):
    net.train()

    device = torch.device(device)

    # numpy 리스트를 먼저 스택해서 텐서로 변환 (경고 제거)
    states_np = np.array(buffer.states, dtype=np.float32)   # (T,31)
    masks_np = np.array(buffer.masks, dtype=np.float32)     # (T,N_ACTIONS)

    states = torch.from_numpy(states_np).to(device)         # (T,31)
    masks = torch.from_numpy(masks_np).to(device)           # (T,N_ACTIONS)

    actions = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32, device=device)
    returns = torch.tensor(buffer.returns, dtype=torch.float32, device=device)
    advantages = torch.tensor(buffer.advantages, dtype=torch.float32, device=device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T = states.size(0)
    indices = np.arange(T)

    for epoch in range(ppo_epochs):
        np.random.shuffle(indices)

        for start in range(0, T, batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]

            mb_states = states[mb_idx]      # (B,31)
            mb_masks = masks[mb_idx]        # (B,N_ACTIONS)
            mb_actions = actions[mb_idx]    # (B,)
            mb_old_log_probs = old_log_probs[mb_idx]
            mb_returns = returns[mb_idx]
            mb_advantages = advantages[mb_idx]

            logits, values = net(mb_states)  # logits: (B,N_ACTIONS), values: (B,)

            # 유효 액션 마스크 적용
            invalid = mb_masks < 0.5
            logits = logits.masked_fill(invalid, -1e9)

            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)  # (B,)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - mb_old_log_probs).exp()
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, mb_returns)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()


# ================================================================
# 8. self-play rollout + 전체 학습 루프
# ================================================================

def collect_rollout(
    envw: AlkkagiEnvWrapper,
    net: PolicyValueNet,
    buffer: RolloutBuffer,
    device: torch.device | str,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[int], int, int, int]:
    """
    반환:
      - episode_returns: 각 에피소드 총 리턴 리스트
      - episode_lengths: 각 에피소드 스텝 수
      - wins_black: 이번 rollout에서 흑 승 수
      - wins_white: 이번 rollout에서 백 승 수
      - draws: 무승부 수
    """
    net.eval()
    device = torch.device(device)

    buffer.clear()
    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    wins_black = 0
    wins_white = 0
    draws = 0

    obs, info = envw.reset()
    ep_return = 0.0
    ep_len = 0

    for step in range(rollout_steps):
        ep_len += 1

        my_color = obs.turn  # 현재 턴 기준 인코딩

        state_np = encode_state_basic(obs, my_color)  # (31,)
        state_tensor = torch.from_numpy(state_np).to(device=device, dtype=torch.float32)
        state_tensor = state_tensor.unsqueeze(0)      # (1,31)

        mask_np = get_valid_action_mask(obs)          # (N_ACTIONS,)
        mask_tensor = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
        mask_tensor = mask_tensor.unsqueeze(0)        # (1,N_ACTIONS)

        with torch.no_grad():
            logits, value = net(state_tensor)         # logits: (1,N_ACTIONS)
            invalid = mask_tensor < 0.5               # (1,N_ACTIONS)
            logits_masked = logits.masked_fill(invalid, -1e9)

            dist = Categorical(logits=logits_masked)
            action_idx_tensor = dist.sample()         # (1,)
            log_prob_tensor = dist.log_prob(action_idx_tensor)

        action_idx = int(action_idx_tensor.item())
        log_prob = float(log_prob_tensor.item())
        value_scalar = float(value.squeeze(0).item())

        env_action = discrete_to_env_action(action_idx, obs)

        next_obs, reward, terminated, truncated, info = envw.step(env_action)
        done = bool(terminated or truncated)

        ep_return += float(reward)

        # 🔻 step별 리워드 + 현재 에피소드 누적 리턴 출력
        print(
            f"[ROLL] step={step:04d}, ep_step={ep_len:03d}, "
            f"reward={reward:.3f}, ep_return={ep_return:.3f}"
        )

        buffer.add(
            state_np=state_np,
            action=action_idx,
            log_prob=log_prob,
            reward=float(reward),
            done=done,
            value=value_scalar,
            mask_np=mask_np,
        )

        if done:
            # 에피소드 종료: 리턴/길이 기록
            episode_returns.append(ep_return)
            episode_lengths.append(ep_len)

            black_alive = count_alive_stones(next_obs, 0)
            white_alive = count_alive_stones(next_obs, 1)

            # 🔻 에피소드 요약 로그
            print(
                f"[EP DONE] ep_return={ep_return:.3f}, ep_steps={ep_len}, "
                f"black_alive={black_alive}, white_alive={white_alive}"
            )

            # 승/패/무 판정 (마지막 next_obs 기준)
            if black_alive > 0 and white_alive == 0:
                wins_black += 1
            elif white_alive > 0 and black_alive == 0:
                wins_white += 1
            else:
                draws += 1

            ep_return = 0.0
            ep_len = 0
            obs, info = envw.reset()
        else:
            obs = next_obs

    buffer.compute_returns_and_advantages(gamma=gamma, gae_lambda=gae_lambda)
    return episode_returns, episode_lengths, wins_black, wins_white, draws


def train_ppo_selfplay(
    num_updates: int = 1500,
    rollout_steps: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    lr: float = 3e-4,
    batch_size: int = 512,
    ppo_epochs: int = 4,
    clip_coef: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    device: torch.device | str = "cpu",
    save_path: str | None = "alkkagi_ppo.pt",
    log_csv_path: str | None = "training_log.csv",
    checkpoint_every_episodes: int = 1000,
    checkpoint_dir: str = "checkpoints",
) -> PolicyValueNet:
    """
    간단한 single-env self-play PPO 학습 루프 + 체크포인트/CSV 로깅.
    """
    device = torch.device(device)

    envw = AlkkagiEnvWrapper(render_mode='human', bgm=False)
    net = PolicyValueNet().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    buffer = RolloutBuffer()

    total_black = 0
    total_white = 0
    total_draws = 0

    total_episodes = 0
    next_ckpt_ep = checkpoint_every_episodes

    learner_rating = 1500.0
    num_players = 2  # self-play 흑/백

    # CSV 헤더 초기화
    if log_csv_path is not None and (not os.path.exists(log_csv_path)):
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "episodes",
                "wins",
                "draws",
                "losses",
                "win_rate",
                "avg_reward",
                "avg_steps",
                "learner_rating",
                "num_players",
            ])

    for update in range(1, num_updates + 1):
        ep_returns, ep_lengths, wins_black, wins_white, draws = collect_rollout(
            envw=envw,
            net=net,
            buffer=buffer,
            device=device,
            rollout_steps=rollout_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        episodes = len(ep_returns)
        total_episodes += episodes

        total_black += wins_black
        total_white += wins_white
        total_draws += draws

        ppo_update(
            net=net,
            optimizer=optimizer,
            buffer=buffer,
            device=device,
            batch_size=batch_size,
            ppo_epochs=ppo_epochs,
            clip_coef=clip_coef,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
        )

        if episodes > 0:
            avg_return = float(np.mean(ep_returns))
            avg_steps = float(np.mean(ep_lengths))
        else:
            avg_return = 0.0
            avg_steps = 0.0

        wins = wins_black
        losses = wins_white

        if episodes > 0:
            win_rate = wins / episodes
        else:
            win_rate = 0.0

        # 간단한 Elo 스타일 rating 업데이트 (상대도 1500 가정)
        if wins + losses > 0:
            score = wins / (wins + losses)      # 실제 점수
            expected = 0.5                      # 동급이라고 가정
            K = 32.0
            learner_rating += K * (score - expected)

        # 콘솔 로그
        print(
            f"[Update {update:04d}] "
            f"ep_total={total_episodes}, ep_this={episodes}, "
            f"wins={wins}, draws={draws}, losses={losses}, "
            f"win_rate={win_rate:.3f}, "
            f"avg_return={avg_return:.3f}, avg_steps={avg_steps:.2f}, "
            f"rating={learner_rating:.2f}"
        )

        # CSV 로그 기록
        if log_csv_path is not None:
            with open(log_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    update,            # epoch
                    episodes,
                    wins,
                    draws,
                    losses,
                    f"{win_rate:.6f}",
                    f"{avg_return:.6f}",
                    f"{avg_steps:.6f}",
                    f"{learner_rating:.6f}",
                    num_players,
                ])

        # 에피소드 기준 체크포인트 저장
        if checkpoint_every_episodes > 0:
            while total_episodes >= next_ckpt_ep:
                os.makedirs(checkpoint_dir, exist_ok=True)
                ckpt_path = os.path.join(
                    checkpoint_dir,
                    f"alkkagi_ep_{next_ckpt_ep:06d}.pt",
                )
                torch.save(net.state_dict(), ckpt_path)
                print(f"[CKPT] Saved checkpoint at {ckpt_path} (episodes={next_ckpt_ep})")
                next_ckpt_ep += checkpoint_every_episodes

    envw.close()

    # 최종 모델 저장
    if save_path is not None:
        torch.save(net.state_dict(), save_path)
        print(f"[SAVE] Saved final model to {save_path}")

    return net


# ================================================================
# 10. 제출용 에이전트: YourBlackAgent / YourWhiteAgent
# ================================================================
# 위쪽에 이미 있는 import들:
# import kymnasium as kym
# import torch
# from typing import Any, Dict

class YourAlkkagiAgentBase(kym.Agent):
    """
    대회 제출용 기본 Agent.
    - color: 0(흑), 1(백)
    - PolicyValueNet을 내부에 들고 있고,
      obs -> (31차원 인코딩) -> 네트워크 -> 디스크리트 액션 -> env action dict
    """
    def __init__(
        self,
        net: PolicyValueNet,
        color: int,
        device: torch.device | str = "cpu",
    ):
        super().__init__()
        assert color in (0, 1)
        self.color = color
        self.device = torch.device(device)
        self.net = net.to(self.device)
        self.net.eval()

    def act(self, observation: Any, info: Dict) -> Dict[str, float]:
        """
        env가 매 스텝마다 호출하는 함수.
        - observation: env에서 넘어오는 raw dict
        - 반환: {"turn", "index", "power", "angle"}
        """
        obs = AlkkagiObservation(observation)

        # 방어용: 내 차례가 아닌데 호출되면 무시되는 액션 리턴
        if obs.turn != self.color:
            return {
                "turn": obs.turn,
                "index": 0,
                "power": 0.0,
                "angle": 0.0,
            }

        my_color = obs.turn  # canonical: 항상 현재 턴 기준 인코딩

        # 1) 상태 인코딩 (31차원)
        state_np = encode_state_basic(obs, my_color)  # (31,)
        state_tensor = (
            torch.from_numpy(state_np)
            .to(self.device, dtype=torch.float32)
            .unsqueeze(0)  # (1,31)
        )

        # 2) 유효 액션 마스크
        mask_np = get_valid_action_mask(obs)          # (N_ACTIONS,)
        mask_tensor = (
            torch.from_numpy(mask_np)
            .to(self.device, dtype=torch.float32)
            .unsqueeze(0)  # (1, N_ACTIONS)
        )

        # 3) 정책 네트워크 forward + 마스크 적용 + argmax로 액션 선택
        with torch.no_grad():
            logits, _ = self.net(state_tensor)        # logits: (1,N_ACTIONS)
            invalid = mask_tensor < 0.5
            logits_masked = logits.masked_fill(invalid, -1e9)

            # 평가 시에는 deterministic하게 argmax 사용
            action_idx_tensor = torch.argmax(logits_masked, dim=-1)  # (1,)
            action_idx = int(action_idx_tensor.item())

        # 4) 디스크리트 인덱스를 env 액션으로 변환
        env_action = discrete_to_env_action(action_idx, obs)

        return {
            "turn": int(env_action.turn),
            "index": int(env_action.index),
            "power": float(env_action.power),
            "angle": float(env_action.angle),
        }

    def save(self, path: str):
        """
        현재 네트워크 weight를 path에 저장.
        - 학습 스크립트에서 바로 쓸 수 있음.
        """
        torch.save(self.net.state_dict(), path)

    @staticmethod
    def _load_net_from_path(path: str, device: torch.device | str):
        """
        공통: PolicyValueNet 생성 + state_dict 로드
        """
        device = torch.device(device)
        net = PolicyValueNet()
        state_dict = torch.load(path, map_location=device)
        net.load_state_dict(state_dict)
        return net.to(device)


class YourBlackAgent(YourAlkkagiAgentBase):
    """
    흑 에이전트.
    - 평가 서버에서는 보통 YourBlackAgent.load(path)로 불러서 사용.
    """
    def __init__(self, net: PolicyValueNet, device: torch.device | str = "cpu"):
        super().__init__(net=net, color=0, device=device)

    @classmethod
    def load(cls, path: str) -> "kym.Agent":
        """
        path에 저장된 weight로부터 흑 에이전트 하나 생성.
        - 예: black_agent = YourBlackAgent.load("alkkagi_ppo.pt")
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = YourAlkkagiAgentBase._load_net_from_path(path, device)
        return cls(net=net, device=device)


class YourWhiteAgent(YourAlkkagiAgentBase):
    """
    백 에이전트.
    - 평가 서버에서는 YourWhiteAgent.load(path)로 불러서 사용.
    """
    def __init__(self, net: PolicyValueNet, device: torch.device | str = "cpu"):
        super().__init__(net=net, color=1, device=device)

    @classmethod
    def load(cls, path: str) -> "kym.Agent":
        """
        path에 저장된 weight로부터 백 에이전트 하나 생성.
        - 예: white_agent = YourWhiteAgent.load("alkkagi_ppo.pt")
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = YourAlkkagiAgentBase._load_net_from_path(path, device)
        return cls(net=net, device=device)



# ================================================================
# 9. 간단 테스트용 엔트리포인트
# ================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained_net = train_ppo_selfplay(
        num_updates=1000,          # 대략 5만 에피 정도
        rollout_steps=2048,
        batch_size=512,
        device=device,
        save_path="alkkagi_ppo.pt",
        log_csv_path="training_metrics_v4.csv",
        checkpoint_every_episodes=1000,
        checkpoint_dir="checkpoints_v4",
    )

