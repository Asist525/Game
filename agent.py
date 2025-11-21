import gymnasium as gym
import kymnasium as kym  # env 등록용
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Dict
import os

# ============================================================
# 0. 기본 상수 / 환경
# ============================================================
N_STONES = 3
N_OBS = 3

BOARD_W = 600.0
BOARD_H = 600.0

# ============================================================
# PPO / Reward Shaping 하이퍼파라미터
# ============================================================
GAMMA = 0.99          # PPO에서 쓰는 gamma와 shaping gamma를 일치

WIN_REWARD = 100.0    # 승/패 보상 스케일 (이기면 +100, 지면 -100)

# Φ_alive(s) = α * alive_diff - β * total_alive
ALIVE_ALPHA = 1.0
ALIVE_BETA  = 0.3
ALIVE_LAMBDA = 1.0    # r_total에 곱해줄 λ_alive

# Ψ_pos(s) = -γ_d * min_my_op_dist(s)
POS_GAMMA_D  = 1.0
# 초기에는 0.0으로 두고, 필요할 때 0.05~0.1까지 올려보기
POS_LAMBDA   = 0.0    # r_total에 곱해줄 λ_pos


# 디스크리트 액션 설정 (self-play, PPO에서 사용할 수 있도록)
N_ANGLES = 32   # 방향 개수
N_POWERS = 6    # 파워 단계
N_ACTIONS = N_STONES * N_ANGLES * N_POWERS


def make_env():
    env = gym.make(
        "kymnasium/AlKkaGi-3x3-v0",
        obs_type="custom",
        render_mode=None,  # 학습 중엔 None
        bgm=False,
    )
    return env


# ============================================================
# 1. 관측 전처리 / 인코더
# ============================================================
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

    - count (정규화: /3)
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

    cnt_norm = float(obs_norm.shape[0]) / 3.0

    return np.array(
        [cnt_norm, cx.mean(), cy.mean(), w.mean(), h.mean()],
        dtype=np.float32,
    )


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
    base_feat = encode_state_basic_alkkagi(obs, my_color, board_w, board_h)

    me, opp, obstacles, turn_raw = split_me_opp(obs, my_color)
    me_norm  = normalize_stones(me,  board_w, board_h)
    opp_norm = normalize_stones(opp, board_w, board_h)
    obs_norm = normalize_obstacles(obstacles, board_w, board_h)

    my_alive_cnt  = float((me_norm[:, 2] > 0.5).sum())
    opp_alive_cnt = float((opp_norm[:, 2] > 0.5).sum())
    alive_diff    = my_alive_cnt - opp_alive_cnt
    denom = my_alive_cnt + opp_alive_cnt
    alive_ratio = my_alive_cnt / denom if denom > 0 else 0.0

    # env 기준 턴(0=흑,1=백)과 my_color 비교
    turn_is_me = 1.0 if int(turn_raw) == int(my_color) else 0.0

    my_alive_norm = my_alive_cnt / 3.0
    opp_alive_norm = opp_alive_cnt / 3.0
    alive_diff_norm = (my_alive_cnt - opp_alive_cnt) / 3.0

    scalar_feats = np.array(
        [
            turn_is_me,
            my_alive_norm,
            opp_alive_norm,
            alive_diff_norm,
            alive_ratio,
        ],
        dtype=np.float32,
    )

    my_stats = group_stats(me_norm)   # (4,)
    op_stats = group_stats(opp_norm)  # (4,)

    my_min_edge = min_edge_dist(me_norm)
    op_min_edge = min_edge_dist(opp_norm)
    min_my_op   = min_pairwise_dist(me_norm, opp_norm)

    relation_feats = np.array(
        [my_min_edge, op_min_edge, min_my_op],
        dtype=np.float32,
    )

    obs_stats = obstacle_summary(obs_norm)  # (5,)

    extra_feats = np.concatenate([
        scalar_feats,   # 5
        my_stats,       # 4
        op_stats,       # 4
        relation_feats, # 3
        obs_stats,      # 5
    ]).astype(np.float32)

    feat = np.concatenate([base_feat, extra_feats]).astype(np.float32)
    return feat   # (52,)


STATE_DIM = 52


# ============================================================
# 2. Reward 설계: 승/패 + potential-based shaping
# ============================================================
def compute_win_loss_reward(
    obs,
    my_color: int,
    win_reward: float = WIN_REWARD,
) -> float:
    """
    obs: 종료 시점 obs
    my_color: 학습 중인 에이전트의 색 (0=흑,1=백)

    이기면 +win_reward, 지면 -win_reward, 동점이면 0
    """
    me, opp, obstacles, _ = split_me_opp(obs, my_color)
    my_alive  = int((me[:, 2] > 0.5).sum())
    opp_alive = int((opp[:, 2] > 0.5).sum())

    if my_alive > opp_alive:
        return win_reward
    elif my_alive < opp_alive:
        return -win_reward
    else:
        return 0.0


def compute_alive_potential(
    obs,
    my_color: int,
    alpha: float = ALIVE_ALPHA,
    beta: float = ALIVE_BETA,
) -> float:
    """
    Φ_alive(s) = α * (my_alive - opp_alive) - β * (my_alive + opp_alive)
    """
    me, opp, _, _ = split_me_opp(obs, my_color)
    my_alive  = float((me[:, 2] > 0.5).sum())
    opp_alive = float((opp[:, 2] > 0.5).sum())

    alive_diff  = my_alive - opp_alive
    total_alive = my_alive + opp_alive

    return alpha * alive_diff - beta * total_alive


def compute_pos_potential(
    obs,
    my_color: int,
    board_w: float = BOARD_W,
    board_h: float = BOARD_H,
    gamma_d: float = POS_GAMMA_D,
) -> float:
    """
    Ψ_pos(s) = -γ_d * min_my_op_dist(s)
    (내 돌과 상대 돌 간 최소 거리 기반 포지션 잠재함수)
    """
    me, opp, _, _ = split_me_opp(obs, my_color)
    me_norm  = normalize_stones(me,  board_w, board_h)
    opp_norm = normalize_stones(opp, board_w, board_h)

    min_d = min_pairwise_dist(me_norm, opp_norm)
    return -gamma_d * float(min_d)


def compute_shaping_rewards(
    prev_obs,
    curr_obs,
    my_color: int,
    gamma: float = GAMMA,
) -> tuple[float, float]:
    """
    s=prev_obs, s'=curr_obs 에 대해
      r_alive = γ Φ_alive(s') - Φ_alive(s)
      r_pos   = γ Ψ_pos(s')   - Ψ_pos(s)
    를 계산해서 반환.
    """
    phi_prev = compute_alive_potential(prev_obs, my_color)
    phi_curr = compute_alive_potential(curr_obs, my_color)

    psi_prev = compute_pos_potential(prev_obs, my_color)
    psi_curr = compute_pos_potential(curr_obs, my_color)

    r_alive = gamma * phi_curr - phi_prev
    r_pos   = gamma * psi_curr - psi_prev
    return r_alive, r_pos


# ============================================================
# 3. 디스크리트 액션 → 실제 Env 액션 매핑
# ============================================================
def decode_action_index(action_idx: int):
    stone_id = action_idx // (N_ANGLES * N_POWERS)
    rem = action_idx % (N_ANGLES * N_POWERS)
    angle_id = rem // N_POWERS
    power_id = rem % N_POWERS

    angle_step = 360.0 / N_ANGLES
    angle = -180.0 + angle_id * angle_step

    power_min, power_max = 500.0, 2500.0  # 예: 500 이상만 허용
    if N_POWERS == 1:
        power = (power_min + power_max) / 2.0
    else:
        power = power_min + (power_id / (N_POWERS - 1)) * (power_max - power_min)

    return stone_id, angle, power


def idx_to_action(action_idx: int, obs) -> Dict[str, Any]:
    """
    Env.step()에 넣을 딕셔너리 액션 생성
      {
        "turn": 0 or 1,
        "index": 0~2 (실제로는 살아있는 돌 인덱스로 매핑),
        "power": float in [1,2500],
        "angle": float in [-180,180]
      }
    """
    # 현재 턴(0=흑, 1=백)
    turn = int(obs["turn"])

    # 디스크리트 -> (stone_id 후보, angle, power)
    stone_id_raw, angle, power = decode_action_index(action_idx)

    # 현재 턴에 해당하는 돌들 가져오기
    if turn == 0:
        stones = np.array(obs["black"], dtype=np.float32)  # (3,3) [x,y,alive]
    else:
        stones = np.array(obs["white"], dtype=np.float32)

    # 살아있는 돌 인덱스만 모으기
    alive_mask = stones[:, 2] > 0.5
    alive_indices = np.nonzero(alive_mask)[0]

    if len(alive_indices) == 0:
        # 이 상황이면 사실 게임이 거의 끝난 상태라서 어느 인덱스를 줘도 의미 없음
        real_index = 0
    else:
        # stone_id_raw (0~N_STONES-1)를 alive 돌들 인덱스로 매핑
        real_index = int(alive_indices[stone_id_raw % len(alive_indices)])

    return {
        "turn": turn,
        "index": real_index,
        "power": float(power),
        "angle": float(angle),
    }


# ============================================================
# 4. PPO Actor-Critic 네트워크
# ============================================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim=STATE_DIM, n_actions=N_ACTIONS):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.pi_head = nn.Linear(128, n_actions)
        self.v_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        logits = self.pi_head(h)
        value = self.v_head(h).squeeze(-1)
        return logits, value

    def act(self, state: torch.Tensor, deterministic: bool = False):
        """
        state: (state_dim,) or (batch, state_dim)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)

        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def evaluate_actions(self, states, actions):
        """
        PPO 업데이트용
        states: (B, state_dim)
        actions: (B,)
        """
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values


# ============================================================
# 5. PPO Rollout 버퍼 / GAE / 업데이트
# ============================================================
class PPORolloutBuffer:
    def __init__(self, state_dim: int, size: int):
        # self.size = size  # 아직 안 씀
        self.state_dim = state_dim
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.is_learner = []   # <-- 추가

    def __len__(self):
        return len(self.states)

    def add(self, state, action, logprob, reward, done, value, is_learner: bool):
        self.states.append(state.astype(np.float32))
        self.actions.append(int(action))
        self.logprobs.append(float(logprob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.is_learner.append(bool(is_learner))   # <-- 추가

    def to_tensors(self, device):
        states = torch.from_numpy(np.stack(self.states)).float().to(device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=device)
        logprobs = torch.tensor(self.logprobs, dtype=torch.float32, device=device)
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        values = torch.tensor(self.values, dtype=torch.float32, device=device)
        is_learner = torch.tensor(self.is_learner, dtype=torch.float32, device=device)
        return states, actions, logprobs, rewards, dones, values, is_learner


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float = GAMMA,
    lam: float = 0.95,
):
    T = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            # 버퍼 마지막 스텝: 다음 상태의 value는 0, done[t] 기준으로 끊어줌
            next_value = 0.0
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_non_terminal = 1.0 - dones[t]

        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return returns, advantages


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    buffer: PPORolloutBuffer,
    device: torch.device,
    epochs: int = 4,
    batch_size: int = 64,
    gamma: float = GAMMA,
    lam: float = 0.95,
    clip_coef: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    max_grad_norm: float = 0.5,
):
    states, actions, old_logprobs, rewards, dones, values, is_learner = buffer.to_tensors(device)

    returns, advantages = compute_gae(rewards, dones, values, gamma, lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = states.size(0)
    indices = np.arange(dataset_size)

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            b_states = states[batch_idx]
            b_actions = actions[batch_idx]
            b_old_logprob = old_logprobs[batch_idx]
            b_returns = returns[batch_idx]
            b_advantages = advantages[batch_idx]
            b_is_learner = is_learner[batch_idx]

            logprobs, entropy, values_pred = model.evaluate_actions(b_states, b_actions)

            # --- Value loss: 전체 스텝 사용 ---
            value_loss = F.mse_loss(values_pred, b_returns)

            # --- Policy / Entropy: learner 턴만 사용 ---
            learner_mask = b_is_learner > 0.5
            if learner_mask.any():
                lp_l = logprobs[learner_mask]
                old_lp_l = b_old_logprob[learner_mask]
                adv_l = b_advantages[learner_mask]
                ent_l = entropy[learner_mask]

                ratio = torch.exp(lp_l - old_lp_l)
                surr1 = ratio * adv_l
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv_l
                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -ent_l.mean()
            else:
                # 이 배치 안에 learner 턴이 없으면 policy/entropy는 0으로
                policy_loss = torch.tensor(0.0, device=device)
                entropy_loss = torch.tensor(0.0, device=device)

            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()


# ============================================================
# 6. Self-play 훈련 루프
# ============================================================
def train_selfplay_ppo(
    num_episodes: int = 5000,
    rollout_size: int = 2048,
    opponent_update_interval: int = 500,
    save_interval_episodes: int = 10000,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    env = make_env()

    # 체크포인트 저장 폴더
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    model = ActorCritic(STATE_DIM, N_ACTIONS).to(device)
    opponent_model = ActorCritic(STATE_DIM, N_ACTIONS).to(device)
    opponent_model.load_state_dict(model.state_dict())
    opponent_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    buffer = PPORolloutBuffer(state_dim=STATE_DIM, size=rollout_size)

    global_step = 0
    update_count = 0

    # ====== ★ 전역 평가 지표 누적용 변수들 ★ ======
    total_episodes = 0
    total_wins = 0
    total_loses = 0
    total_draws = 0
    sum_alive_diff = 0.0

    # ---- (1) 전역 액션 히스토그램 ----
    global_action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    global_action_total = 0

    # ---- (2) stone / angle / power 마진 분포 ----
    global_stone_counts = np.zeros(N_STONES, dtype=np.int64)
    global_angle_counts = np.zeros(N_ANGLES, dtype=np.int64)
    global_power_counts = np.zeros(N_POWERS, dtype=np.int64)
    # ==========================================

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False

        learner_color = np.random.randint(0, 2)
        opponent_color = 1 - learner_color

        ep_states = []
        ep_actions = []
        ep_logprobs = []
        ep_values = []
        ep_rewards = []
        ep_dones = []
        ep_is_learner = []

        # 로그용 컴포넌트 누적
        ep_alive_sum = 0.0
        ep_pos_sum = 0.0
        ep_env_sum = 0.0

        # learner 턴 관련 로그용
        learner_actions = []
        learner_entropy_sum = 0.0
        learner_entropy_steps = 0

        while not done:
            turn_color = int(obs["turn"])

            # learner 관점 state 인코딩 (항상 learner_color 기준)
            state_np_learner = encode_state_fe_alkkagi(
                obs, learner_color, BOARD_W, BOARD_H
            )
            state_t_learner = torch.from_numpy(state_np_learner).float().to(device)

            if turn_color == learner_color:
                # ================= learner 턴 =================
                prev_obs = obs  # shaping용 이전 상태

                model.train()
                action_t, logprob_t, value_t = model.act(
                    state_t_learner, deterministic=False
                )
                action_idx = int(action_t.item())
                env_action = idx_to_action(action_idx, obs)

                # ---- (1)(2) 전역 액션 통계: learner 턴에서만 업데이트 ----
                global_action_counts[action_idx] += 1
                global_action_total += 1

                stone_id = action_idx // (N_ANGLES * N_POWERS)
                rem = action_idx % (N_ANGLES * N_POWERS)
                angle_id = rem // N_POWERS
                power_id = rem % N_POWERS

                global_stone_counts[stone_id] += 1
                global_angle_counts[angle_id] += 1
                global_power_counts[power_id] += 1
                # -------------------------------------------------------

                next_obs, _, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated

                # --- reward shaping ---
                r_alive, r_pos = compute_shaping_rewards(
                    prev_obs,
                    next_obs,
                    my_color=learner_color,
                    gamma=GAMMA,
                )
                shaping_reward = ALIVE_LAMBDA * r_alive + POS_LAMBDA * r_pos
                step_reward = shaping_reward

                ep_alive_sum += ALIVE_LAMBDA * r_alive
                ep_pos_sum += POS_LAMBDA * r_pos

                # --- terminal win/lose reward ---
                if done:
                    r_env = compute_win_loss_reward(
                        next_obs,
                        learner_color,
                        win_reward=WIN_REWARD,
                    )
                    step_reward += r_env
                    ep_env_sum += r_env

                # ----- 정책 엔트로피 & 액션 로그 -----
                with torch.no_grad():
                    logits_ent, _ = model.forward(state_t_learner.unsqueeze(0))
                    dist_ent = torch.distributions.Categorical(logits=logits_ent)
                    entropy_step = dist_ent.entropy().mean().item()
                learner_entropy_sum += entropy_step
                learner_entropy_steps += 1
                learner_actions.append(action_idx)
                # -----------------------------------

                # rollout 저장
                ep_states.append(state_np_learner)
                ep_actions.append(action_idx)
                ep_logprobs.append(float(logprob_t.item()))
                ep_values.append(float(value_t.item()))
                ep_rewards.append(float(step_reward))
                ep_dones.append(bool(done))
                ep_is_learner.append(True)

                obs = next_obs

            else:
                # ================= opponent 턴 =================
                # learner 관점 value만 뽑아서 trajectory에 넣음
                with torch.no_grad():
                    value_t = model.forward(state_t_learner)[1]

                # 이 스텝의 state(learner 관점)를 먼저 저장
                ep_states.append(state_np_learner)
                ep_actions.append(0)           # dummy
                ep_logprobs.append(0.0)        # dummy
                ep_values.append(float(value_t.item()))
                ep_is_learner.append(False)

                prev_obs = obs  # opponent 액션 이전 상태

                # opponent state 인코딩 후 액션
                state_np_opp = encode_state_fe_alkkagi(
                    obs, opponent_color, BOARD_W, BOARD_H
                )
                state_opp_t = torch.from_numpy(state_np_opp).float().to(device)

                opponent_model.eval()
                with torch.no_grad():
                    action_opp_t, _, _ = opponent_model.act(
                        state_opp_t, deterministic=False
                    )
                action_idx_opp = int(action_opp_t.item())
                env_action_opp = idx_to_action(action_idx_opp, obs)

                next_obs, _, terminated, truncated, info = env.step(env_action_opp)
                done = terminated or truncated

                # --- reward shaping (여전히 learner 관점) ---
                r_alive, r_pos = compute_shaping_rewards(
                    prev_obs,
                    next_obs,
                    my_color=learner_color,
                    gamma=GAMMA,
                )
                shaping_reward = ALIVE_LAMBDA * r_alive + POS_LAMBDA * r_pos
                step_reward = shaping_reward

                ep_alive_sum += ALIVE_LAMBDA * r_alive
                ep_pos_sum += POS_LAMBDA * r_pos

                if done:
                    r_env = compute_win_loss_reward(
                        next_obs,
                        learner_color,
                        win_reward=WIN_REWARD,
                    )
                    step_reward += r_env
                    ep_env_sum += r_env

                ep_rewards.append(float(step_reward))
                ep_dones.append(bool(done))

                obs = next_obs

        # ===== 에피소드 종료 후: 버퍼 적재 =====
        if len(ep_states) > 0:
            # 안전빵: 마지막 done이 혹시 False로 남아있으면 True로 강제
            ep_dones[-1] = True

            for s, a, lp, v, r, d, is_l in zip(
                ep_states, ep_actions, ep_logprobs,
                ep_values, ep_rewards, ep_dones, ep_is_learner
            ):
                buffer.add(
                    state=np.asarray(s, dtype=np.float32),
                    action=a,
                    logprob=lp,
                    reward=r,
                    done=d,
                    value=v,
                    is_learner=is_l,
                )
                global_step += 1

        # ===== 전역 평가 지표 업데이트 =====
        total_episodes += 1

        # 승/패/무
        if ep_env_sum > 0:
            total_wins += 1
        elif ep_env_sum < 0:
            total_loses += 1
        else:
            total_draws += 1

        win_rate = total_wins / max(1, total_episodes)

        # 최종 alive 차이
        me_final, opp_final, _, _ = split_me_opp(obs, learner_color)
        my_alive_final  = int((me_final[:, 2] > 0.5).sum())
        opp_alive_final = int((opp_final[:, 2] > 0.5).sum())
        alive_diff = my_alive_final - opp_alive_final  # -3 ~ +3

        sum_alive_diff += alive_diff
        avg_alive_diff = sum_alive_diff / max(1, total_episodes)

        # 에피소드 길이
        ep_len = len(ep_states)

        # 정책 엔트로피 / 액션 다양성
        if learner_entropy_steps > 0:
            entropy_mean = learner_entropy_sum / learner_entropy_steps
        else:
            entropy_mean = 0.0

        uniq_learner_actions = len(set(learner_actions)) if learner_actions else 0

        # ===== PPO 업데이트 =====
        if len(buffer) >= rollout_size:
            ppo_update(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                device=device,
                epochs=4,
                batch_size=64,
                gamma=GAMMA,
                lam=0.95,
                clip_coef=0.2,
                vf_coef=0.5,
                ent_coef=0.01,
                max_grad_norm=0.5,
            )
            buffer.reset()
            update_count += 1

            if update_count % opponent_update_interval == 0:
                opponent_model.load_state_dict(model.state_dict())

        ep_return = sum(ep_rewards) if len(ep_rewards) > 0 else 0.0

        # === 전역 액션 분포 기반 지표 계산 ===
        if global_action_total > 0:
            p = global_action_counts / global_action_total
            p_nonzero = p[p > 0]
            H = -np.sum(p_nonzero * np.log(p_nonzero))
            eff_act = float(np.exp(H))              # 유효 액션 개수
            top1 = float(p.max())
            top5 = float(np.sort(p)[-5:].sum())

            stone_mode = int(np.argmax(global_stone_counts))
            angle_mode = int(np.argmax(global_angle_counts))
            power_mode = int(np.argmax(global_power_counts))

            stone_mode_p = global_stone_counts[stone_mode] / global_action_total
            angle_mode_p = global_angle_counts[angle_mode] / global_action_total
            power_mode_p = global_power_counts[power_mode] / global_action_total
        else:
            eff_act = 0.0
            top1 = 0.0
            top5 = 0.0
            stone_mode = angle_mode = power_mode = -1
            stone_mode_p = angle_mode_p = power_mode_p = 0.0

        # ===== 로그 출력 =====
        if (ep + 1) % 100 == 0:
            print(
                f"[Ep {ep+1:05d}] "
                f"color={learner_color}, "
                f"W/L/D={total_wins}/{total_loses}/{total_draws} "
                f"(win_rate={win_rate:.3f}), "
                f"ep_len={ep_len:3d}, "
                f"alive_diff={alive_diff:+d} (avg={avg_alive_diff:+.2f}), "
                f"entropy={entropy_mean:.3f}, "
                f"uniq_act={uniq_learner_actions:3d}, "
                f"ret_total={ep_return:7.2f}, "
                f"ret_env={ep_env_sum:7.2f}, "
                f"ret_alive={ep_alive_sum:7.2f}, "
                f"ret_pos={ep_pos_sum:7.2f}, "
                f"eff_act={eff_act:6.1f}, "
                f"top1={top1:.3f}, top5={top5:.3f}, "
                f"stone_mode={stone_mode}({stone_mode_p:.2f}), "
                f"angle_mode={angle_mode}({angle_mode_p:.2f}), "
                f"power_mode={power_mode}({power_mode_p:.2f}), "
                f"buffer_len={len(buffer)}"
            )

        # =============== ★ 체크포인트 저장 ★ ===============
        if (ep + 1) % save_interval_episodes == 0:
            ckpt_path = os.path.join(ckpt_dir, f"alkkagi_ppo_ep_{ep+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model to {ckpt_path}")
        # ===================================================

    env.close()

    # 마지막 저장
    final_ckpt_path = os.path.join(ckpt_dir, "alkkagi_ppo_final.pt")
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"Training finished. Saved final model to {final_ckpt_path}")


# ============================================================
# 7. 제출용 Agent 래퍼 (YourBlackAgent / YourWhiteAgent)
# ============================================================
# kym.Agent가 없을 경우를 위한 fallback
try:
    BaseAgent = kym.Agent
except AttributeError:
    class BaseAgent:
        pass


class BaseAlkkagiAgent(BaseAgent):
    def __init__(self, color: int, ckpt_path: str | None = None, device=None):
        """
        color: 0=흑, 1=백
        ckpt_path: PPO ActorCritic state_dict 저장된 .pt 경로
        """
        super().__init__()
        self.color = int(color)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # 학습할 때 쓴 것과 같은 네트워크 구조
        self.model = ActorCritic(STATE_DIM, N_ACTIONS).to(self.device)

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()

    @torch.no_grad()
    def act(self, observation, info=None, **kwargs):
        """
        env.step()에 바로 넣을 action(dict)을 반환.
        observation: env에서 주는 custom obs (dict)
        info: dict 또는 None
        """
        obs = observation

        state_np = encode_state_fe_alkkagi(
            obs,
            my_color=self.color,
            board_w=BOARD_W,
            board_h=BOARD_H,
        )
        state_t = torch.from_numpy(state_np).float().to(self.device)

        # 평가 시엔 deterministic=True 로 그리디 정책 사용
        action_t, _, _ = self.model.act(state_t, deterministic=True)
        action_idx = int(action_t.item())

        env_action = idx_to_action(action_idx, obs)
        return env_action


class YourBlackAgent(BaseAlkkagiAgent):
    def __init__(self, ckpt_path: str | None = None, device=None):
        super().__init__(color=0, ckpt_path=ckpt_path, device=device)

    @classmethod
    def load(cls, ckpt_path: str, device=None):
        """
        evalute.py 에서 쓰는 인터페이스:
          black = YourBlackAgent.load(black_ckpt)
        """
        return cls(ckpt_path=ckpt_path, device=device)

    def save(self, ckpt_path: str):
        """
        kym.Agent의 abstract method 구현.
        필요하면 학습 중에 agent.save(path)로 체크포인트 저장 가능.
        """
        torch.save(self.model.state_dict(), ckpt_path)


class YourWhiteAgent(BaseAlkkagiAgent):
    def __init__(self, ckpt_path: str | None = None, device=None):
        super().__init__(color=1, ckpt_path=ckpt_path, device=device)

    @classmethod
    def load(cls, ckpt_path: str, device=None):
        """
        evalute.py 에서 쓰는 인터페이스:
          white = YourWhiteAgent.load(white_ckpt)
        """
        return cls(ckpt_path=ckpt_path, device=device)

    def save(self, ckpt_path: str):
        """
        kym.Agent의 abstract method 구현.
        필요하면 학습 중에 agent.save(path)로 체크포인트 저장 가능.
        """
        torch.save(self.model.state_dict(), ckpt_path)


if __name__ == "__main__":
    train_selfplay_ppo(
        num_episodes=100000000,
        rollout_size=2048,
        opponent_update_interval=10,
        save_interval_episodes=10000,
    )
