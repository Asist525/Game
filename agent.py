import gymnasium as gym
import kymnasium as kym  # env 등록용
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Any, Dict

# ============================================================
# 0. 기본 상수 / 환경
# ============================================================
N_STONES = 3
N_OBS = 3

BOARD_W = 600.0
BOARD_H = 600.0

# 디스크리트 액션 설정 (self-play, PPO에서 사용할 수 있도록)
N_ANGLES = 32   # 방향 개수
N_POWERS = 6   # 파워 단계
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
# 1. 관측 전처리 / 인코더 (네가 만든 코드 기반)
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
# 2. 승패 기반 sparse 보상 (끝에서만 ±1 / 0)
# ============================================================
def compute_win_loss_reward(
    obs,
    my_color: int,
) -> float:
    """
    obs: 종료 시점 obs
    my_color: 학습 중인 에이전트의 색 (0=흑,1=백)
    """
    me, opp, obstacles, _ = split_me_opp(obs, my_color)
    my_alive  = int((me[:, 2] > 0.5).sum())
    opp_alive = int((opp[:, 2] > 0.5).sum())

    if my_alive > opp_alive:
        return 1.0
    elif my_alive < opp_alive:
        return -1.0
    else:
        return 0.0


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

    # 여기 수정
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
    gamma: float = 0.99,
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
    gamma: float = 0.99,
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
#    - 하나의 PPO policy (model)
#    - 상대는 opponent_model (frozen copy)
#    - 에피소드마다 learner_color를 랜덤(0 or 1)으로 선택
# ============================================================
def train_selfplay_ppo(
    num_episodes: int = 5000,
    rollout_size: int = 2048,
    opponent_update_interval: int = 500,
    save_interval_episodes: int = 10000,   # <-- 추가
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    env = make_env()

    model = ActorCritic(STATE_DIM, N_ACTIONS).to(device)
    opponent_model = ActorCritic(STATE_DIM, N_ACTIONS).to(device)
    opponent_model.load_state_dict(model.state_dict())
    opponent_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    buffer = PPORolloutBuffer(state_dim=STATE_DIM, size=rollout_size)

    global_step = 0
    update_count = 0

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

        while not done:
            turn_color = int(obs["turn"])

            state_np_learner = encode_state_fe_alkkagi(
                obs, learner_color, BOARD_W, BOARD_H
            )
            state_t_learner = torch.from_numpy(state_np_learner).float().to(device)

            if turn_color == learner_color:
                model.train()
                action_t, logprob_t, value_t = model.act(
                    state_t_learner, deterministic=False
                )
                action_idx = int(action_t.item())
                env_action = idx_to_action(action_idx, obs)

                next_obs, _, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated

                ep_states.append(state_np_learner)
                ep_actions.append(action_idx)
                ep_logprobs.append(float(logprob_t.item()))
                ep_values.append(float(value_t.item()))
                ep_rewards.append(0.0)
                ep_dones.append(False)
                ep_is_learner.append(True)

                obs = next_obs

            else:
                with torch.no_grad():
                    value_t = model.forward(state_t_learner)[1]

                ep_states.append(state_np_learner)
                ep_actions.append(0)
                ep_logprobs.append(0.0)
                ep_values.append(float(value_t.item()))
                ep_rewards.append(0.0)
                ep_dones.append(False)
                ep_is_learner.append(False)

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
                obs = next_obs

        if len(ep_states) > 0:
            final_reward = compute_win_loss_reward(obs, learner_color)
            ep_rewards[-1] = final_reward
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

        if len(buffer) >= rollout_size:
            ppo_update(
                model=model,
                optimizer=optimizer,
                buffer=buffer,
                device=device,
                epochs=4,
                batch_size=64,
                gamma=0.99,
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
        if (ep + 1) % 100 == 0:
            print(
                f"[Ep {ep+1:05d}] learner_color={learner_color}, "
                f"ep_steps={len(ep_states)}, ep_return={ep_return:.2f}, "
                f"buffer_len={len(buffer)}"
            )

        # =============== ★ 추가된 부분 ★ ===============
        if (ep + 1) % save_interval_episodes == 0:
            ckpt_path = f"alkkagi_ppo_ep_{ep+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[Checkpoint] Saved model to {ckpt_path}")
        # ===============================================

    env.close()

    # 마지막 저장
    torch.save(model.state_dict(), "alkkagi_ppo_final.pt")
    print("Training finished. Saved final model to alkkagi_ppo_final.pt")


# ============================================================
# 7. 제출용 Agent 래퍼 (YourBlackAgent / YourWhiteAgent)
#    - evalute.py: from agent import YourBlackAgent, YourWhiteAgent
#    - black = YourBlackAgent.load(ckpt_path)
#      white = YourWhiteAgent.load(ckpt_path)
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
        save_interval_episodes=10000,   # ← 명시적으로 넣기
    )
