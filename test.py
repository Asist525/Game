# test.py
import os
import numpy as np
import torch

from agent import (
    make_env,
    ActorCritic,
    encode_state_fe_alkkagi,
    idx_to_action,
    STATE_DIM,
    N_ACTIONS,
    BOARD_W,
    BOARD_H,
    compute_shaping_rewards,
    compute_win_loss_reward,
    GAMMA,
    WIN_REWARD,
    ALIVE_LAMBDA,
    POS_LAMBDA,
)


def make_model(ckpt_path: str | None = None):
    """
    ckpt_path가 주어지고 실제로 존재하면 로드,
    아니면 랜덤 초기화된 ActorCritic을 반환.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(STATE_DIM, N_ACTIONS).to(device)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"[make_model] Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print("[make_model] No checkpoint. Using randomly initialized model.")

    model.eval()
    return model, device


def check_reward_scale(
    model,
    device,
    num_episodes: int = 10,
    max_steps_per_ep: int = 200,
):
    """
    self-play 비슷하게 돌리면서,
    에피소드별로
      - ret_total : 전체 리턴
      - ret_env   : 승/패 보상 합 (±WIN_REWARD)
      - ret_alive : λ_alive * r_alive 합
      - ret_pos   : λ_pos   * r_pos   합
    을 출력해서, shaping이 승/패 보상을 먹고 있는지 스케일 확인.
    """

    import numpy as np  # np.random 쓰려고

    env = make_env()

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        # 학습 코드랑 맞추기 위해 매 에피소드 learner_color 랜덤
        learner_color = np.random.randint(0, 2)
        opponent_color = 1 - learner_color

        ep_total_return = 0.0
        ep_env_sum = 0.0
        ep_alive_sum = 0.0
        ep_pos_sum = 0.0

        while not done and steps < max_steps_per_ep:
            steps += 1
            turn_color = int(obs["turn"])

            # learner 관점 state
            state_np_learner = encode_state_fe_alkkagi(
                obs, learner_color, BOARD_W, BOARD_H
            )
            state_t_learner = torch.from_numpy(state_np_learner).float().to(device)

            # ------------------------------------------------
            # learner 턴
            # ------------------------------------------------
            if turn_color == learner_color:
                prev_obs = obs

                with torch.no_grad():
                    logits, _ = model(state_t_learner.unsqueeze(0))
                    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                # deterministic 평가 (그리디)
                action_idx = int(probs.argmax())
                env_action = idx_to_action(action_idx, obs)

                obs, _, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated

                # shaping (alive / pos)
                r_alive, r_pos = compute_shaping_rewards(
                    prev_obs,
                    obs,
                    my_color=learner_color,
                    gamma=GAMMA,
                )
                alive_contrib = ALIVE_LAMBDA * r_alive
                pos_contrib = POS_LAMBDA * r_pos
                step_reward = alive_contrib + pos_contrib

                ep_alive_sum += alive_contrib
                ep_pos_sum += pos_contrib

                # 끝났으면 env 승/패 보상 추가
                if done:
                    r_env = compute_win_loss_reward(
                        obs,
                        my_color=learner_color,
                        win_reward=WIN_REWARD,
                    )
                    step_reward += r_env
                    ep_env_sum += r_env

                ep_total_return += step_reward

            # ------------------------------------------------
            # opponent 턴 (같은 model로 self-play)
            # ------------------------------------------------
            else:
                prev_obs = obs

                # opponent 관점 state
                state_np_opp = encode_state_fe_alkkagi(
                    obs, opponent_color, BOARD_W, BOARD_H
                )
                state_opp_t = torch.from_numpy(state_np_opp).float().to(device)

                with torch.no_grad():
                    logits_opp, _ = model(state_opp_t.unsqueeze(0))
                    probs_opp = torch.softmax(logits_opp, dim=-1)[0].cpu().numpy()

                action_idx_opp = int(probs_opp.argmax())
                env_action_opp = idx_to_action(action_idx_opp, obs)

                obs, _, terminated, truncated, info = env.step(env_action_opp)
                done = terminated or truncated

                # reward는 여전히 learner 관점에서 계산
                r_alive, r_pos = compute_shaping_rewards(
                    prev_obs,
                    obs,
                    my_color=learner_color,
                    gamma=GAMMA,
                )
                alive_contrib = ALIVE_LAMBDA * r_alive
                pos_contrib = POS_LAMBDA * r_pos
                step_reward = alive_contrib + pos_contrib

                ep_alive_sum += alive_contrib
                ep_pos_sum += pos_contrib

                if done:
                    r_env = compute_win_loss_reward(
                        obs,
                        my_color=learner_color,
                        win_reward=WIN_REWARD,
                    )
                    step_reward += r_env
                    ep_env_sum += r_env

                ep_total_return += step_reward

        print(
            f"[Scale Ep {ep+1:03d}] "
            f"steps={steps:3d}, "
            f"ret_total={ep_total_return:7.2f}, "
            f"ret_env={ep_env_sum:7.2f}, "
            f"ret_alive={ep_alive_sum:7.2f}, "
            f"ret_pos={ep_pos_sum:7.2f}"
        )

    env.close()


def check_action_distribution(
    model,
    device,
    num_episodes: int = 30,
    max_steps_per_ep: int = 200,
    deterministic_eval: bool = True,
):
    """
    학습된(또는 랜덤) policy로 self-play를 돌리면서
    - 액션 히스토그램
    - 평균 엔트로피
    를 찍어서 collapse 여부 감.

    여기서는 흑/백 모두 같은 모델을 사용.
    """

    env = make_env()

    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    entropies = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps_per_ep:
            steps += 1

            my_color = int(obs["turn"])
            state_np = encode_state_fe_alkkagi(
                obs, my_color, BOARD_W, BOARD_H
            )
            state_t = torch.from_numpy(state_np).float().to(device)

            with torch.no_grad():
                logits, _ = model(state_t.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            # 엔트로피 (stochastic 기준)
            entropy = -(probs * np.log(probs + 1e-8)).sum()
            entropies.append(entropy)

            if deterministic_eval:
                action_idx = int(probs.argmax())
            else:
                action_idx = int(np.random.choice(N_ACTIONS, p=probs))

            action_counts[action_idx] += 1

            env_action = idx_to_action(action_idx, obs)
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

    env.close()

    total_actions = action_counts.sum()
    print("\n==== Action Distribution Debug ====")
    print(f"episodes      : {num_episodes}")
    print(f"total actions : {total_actions}")
    print(f"mean entropy  : {np.mean(entropies):.4f}")
    print(f"min entropy   : {np.min(entropies):.4f}")
    print(f"max entropy   : {np.max(entropies):.4f}")

    top_k = 10
    top_indices = np.argsort(-action_counts)[:top_k]
    print(f"\nTop {top_k} actions by freq:")
    for idx in top_indices:
        freq = action_counts[idx]
        if freq == 0:
            continue
        ratio = freq / total_actions
        print(f"  action {idx:4d}: count={freq:6d}, ratio={ratio*100:5.2f}%")


if __name__ == "__main__":
    # ✅ ckpt 없는 환경 가정 → None
    # 나중에 있으면 경로 넣고 돌리면 됨.
    CKPT_PATH = None

    model, device = make_model(CKPT_PATH)

    print("=== 1) Reward scale check (env vs shaping) ===")
    check_reward_scale(
        model=model,
        device=device,
        num_episodes=5,      # 필요하면 늘리기
        max_steps_per_ep=200,
    )

    print("\n=== 2) Action distribution / entropy (collapse 여부) ===")
    check_action_distribution(
        model=model,
        device=device,
        num_episodes=10,
        max_steps_per_ep=200,
        deterministic_eval=True,
    )
