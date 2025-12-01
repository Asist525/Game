# evalute.py
"""
알까기 자가대전 / 평가 스크립트 (agent 기반, 좌표 로그 버전)

- 흑/백 에이전트 weight를 로드해서 여러 에피소드 플레이
- 각 스텝마다:
    * alive_diff (흑/백 관점)
    * policy entropy
    * 상위 5개 액션 (디스크리트 인덱스, 확률)
    * 실제 선택된 액션 (돌 index, angle, power)
    * 흑/백 돌 좌표 및 alive 상태
    * 슈팅한 돌의 위치 변화 (before → after)

- 전체 에피소드가 끝나면:
    * top1~top5 확률 전체 평균
    * 평균 policy entropy
    * 전체 액션 분포 요약 (action별 평균 확률)
    * 전체 probs에 대한 히스토그램 / rank-plot까지 바로 plt로 출력
"""

import sys

import gymnasium as gym
import kymnasium as kym  # env 등록용

import torch
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt  # ★ 추가

from agent import (
    YourBlackAgent,
    YourWhiteAgent,
    compute_alive_diff,
    encode_state_fe_tensor,
)

DEBUG = True  # 디버그 로그 on/off


# ------------------------------------------------
# 좌표 출력 helper
# ------------------------------------------------
def print_stones(obs):
    black = obs["black"]
    white = obs["white"]

    print("  black stones (x, y, alive):")
    for i, (x, y, alive) in enumerate(black):
        print(f"    B{i}: ({x:7.2f}, {y:7.2f}), alive={alive:4.1f}")

    print("  white stones (x, y, alive):")
    for i, (x, y, alive) in enumerate(white):
        print(f"    W{i}: ({x:7.2f}, {y:7.2f}), alive={alive:4.1f}")


# ------------------------------------------------
# 메인 평가 루프
# ------------------------------------------------
def main(
    render_mode: str = "human",
    black_weight: str = "",
    white_weight: str = "",
    num_episodes: int = 50,
):
    env = gym.make(
        id="kymnasium/AlKkaGi-3x3-v0",
        render_mode=render_mode,
        obs_type="custom",
        bgm=True,
    )

    if black_weight == "" or white_weight == "":
        raise ValueError("black_weight와 white_weight 경로를 반드시 지정해야 합니다.")

    if DEBUG:
        print("======================================")
        print(" EVALUATION START")
        print("======================================")
        print(f"[EVAL] black_weight path: {black_weight}")
        print(f"[EVAL] white_weight path: {white_weight}")
        print(f"[EVAL] num_episodes = {num_episodes}")
        print("======================================")

    # 각각 별도 weight 로드
    agent_black = YourBlackAgent.load(black_weight)
    agent_white = YourWhiteAgent.load(white_weight)

    if DEBUG:
        print("[EVAL] Agents loaded successfully.")
        print("       BlackAgent : YourBlackAgent")
        print("       WhiteAgent : YourWhiteAgent")

    # 전체 에피소드 기준 누적 리워드
    total_reward_black = 0.0
    total_reward_white = 0.0

    # ★ top1~top5 확률 평균 계산용 전체 누적 변수
    topk_prob_sums = [0.0, 0.0, 0.0, 0.0, 0.0]
    topk_prob_counts = [0, 0, 0, 0, 0]

    # ★ 평균 entropy 계산용 (전체 에피소드 기준)
    entropy_sum = 0.0
    entropy_count = 0

    # ★ 전체 분포용 누적 (액션별 평균 / 전체 히스토그램)
    action_prob_sums = None   # shape: (n_actions,)
    action_prob_counts = 0
    all_probs_flat = []       # 모든 스텝의 probs를 1D로 쌓기

    # 통계용
    total_steps_all_episodes = 0
    black_wins = 0
    white_wins = 0
    draws = 0

    # ===========================================================
    #                   여러 에피소드 반복
    # ===========================================================
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0

        if DEBUG:
            print(f"\n========== EPISODE {ep+1}/{num_episodes} ==========")
            print("\n[EVAL] Initial observation")
            print(f"  turn         : {obs['turn']} (0=black, 1=white)")
            print_stones(obs)
            diff_b0 = compute_alive_diff(obs, my_color=0)
            diff_w0 = compute_alive_diff(obs, my_color=1)
            print(f"  initial alive_diff: black POV={diff_b0}, white POV={diff_w0}")
            print("--------------------------------------")

        # 에피소드 진행
        while not done:
            turn = int(obs["turn"])  # 0=black,1=white
            current_player = "black" if turn == 0 else "white"

            if DEBUG:
                print(f"[EP {ep+1} | STEP {step}] turn = {turn} ({current_player})")
                diff_b = compute_alive_diff(obs, my_color=0)
                diff_w = compute_alive_diff(obs, my_color=1)
                print(f"  alive_diff(before action): black POV={diff_b}, white POV={diff_w}")
                print_stones(obs)

            # 현재 턴의 PPOPolicy 선택
            if turn == 0:
                ppo = agent_black.policy
                actor_id = "BlackAgent"
            else:
                ppo = agent_white.policy
                actor_id = "WhiteAgent"

            # --- policy 분포 / top-5 / entropy 계산 & 전체 누적 ---
            with torch.no_grad():
                state_t = encode_state_fe_tensor(
                    obs,
                    my_color=turn,
                    device=ppo.device,
                ).unsqueeze(0)  # (1, state_dim)

                logits = ppo.actor(state_t)
                dist = Categorical(logits=logits)
                probs = dist.probs.squeeze(0).cpu().numpy()  # (n_actions,)
                entropy = dist.entropy().item()

                # 확률 내림차순 상위 5개 인덱스
                topk_idx = probs.argsort()[::-1][:5]

            # ★ 전체 액션 분포 누적 (액션별 평균용)
            if action_prob_sums is None:
                action_prob_sums = np.zeros_like(probs, dtype=np.float64)
            action_prob_sums += probs
            action_prob_counts += 1

            # ★ 전체 probs 플랫하게 모으기 (히스토그램/분포용)
            all_probs_flat.extend(probs.tolist())

            # entropy 전체 누적
            entropy_sum += entropy
            entropy_count += 1

            # top1~top5 확률 각각 전체 누적
            for rank, idx_a in enumerate(topk_idx):
                if rank >= 5:
                    break
                p = float(probs[idx_a])
                topk_prob_sums[rank] += p
                topk_prob_counts[rank] += 1

            if DEBUG:
                print(f"  policy entropy = {entropy:.4f}")
                print("  top-5 actions (idx, prob):")
                for rank, idx_a in enumerate(topk_idx):
                    p = float(probs[idx_a])
                    print(f"    idx={int(idx_a):4d}, p={p:.4f}")

            # 에이전트 액션 (act_eval → 기본 greedy 정책)
            if turn == 0:
                action = agent_black.act(obs, info)
            else:
                action = agent_white.act(obs, info)

            shooter_idx = int(action.get("index", 0))
            shooter_angle = float(action.get("angle", 0.0))
            shooter_power = float(action.get("power", 0.0))

            # 슈팅 돌의 좌표 (before)
            if turn == 0:
                stones = obs["black"]
            else:
                stones = obs["white"]

            if 0 <= shooter_idx < len(stones):
                x0, y0, alive0 = stones[shooter_idx]
            else:
                x0, y0, alive0 = float("nan"), float("nan"), 0.0

            if DEBUG:
                print(
                    f"  {actor_id} action -> "
                    f"stone_idx={shooter_idx}, angle={shooter_angle:.2f}, power={shooter_power:.1f}"
                )
                print(
                    f"  shooter pos BEFORE: "
                    f"({x0:7.2f}, {y0:7.2f}), alive={alive0:4.1f}"
                )

            # 환경 진행
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # reward는 "현재 턴 플레이어" 기준이라고 가정
            if turn == 0:
                total_reward_black += reward
            else:
                total_reward_white += reward

            # 슈팅 돌의 좌표 (after)
            if turn == 0:
                stones_after = obs_next["black"]
            else:
                stones_after = obs_next["white"]

            if 0 <= shooter_idx < len(stones_after):
                x1, y1, alive1 = stones_after[shooter_idx]
            else:
                x1, y1, alive1 = float("nan"), float("nan"), 0.0

            if DEBUG:
                print(
                    f"  env.step -> reward={float(reward):.3f}, "
                    f"terminated={terminated}, truncated={truncated}"
                )
                print(
                    f"  shooter pos AFTER : "
                    f"({x1:7.2f}, {y1:7.2f}), alive={alive1:4.1f}"
                )

                diff_b_after = compute_alive_diff(obs_next, my_color=0)
                diff_w_after = compute_alive_diff(obs_next, my_color=1)
                print(
                    f"  alive_diff(after action): "
                    f"black POV={diff_b_after}, white POV={diff_w_after}"
                )
                print(f"  next turn : {obs_next['turn']} (0=black,1=white)")
                print("--------------------------------------")

            obs = obs_next
            step += 1

        # 한 에피소드 종료
        total_steps_all_episodes += step

        if DEBUG:
            print("\n[EVAL] Final observation (episode end)")
            print(f"  final turn   : {obs['turn']} (0=black, 1=white)")
            print_stones(obs)

        black_alive_diff = compute_alive_diff(obs, my_color=0)
        white_alive_diff = compute_alive_diff(obs, my_color=1)

        if black_alive_diff > 0:
            winner = "Black"
            black_wins += 1
        elif black_alive_diff < 0:
            winner = "White"
            white_wins += 1
        else:
            winner = "Draw"
            draws += 1

        if DEBUG:
            print("\n[EVAL] Episode Result summary")
            print(f"  steps in this episode : {step}")
            print(f"  alive_diff (black POV) = {black_alive_diff}")
            print(
                f"  alive_diff (white POV) = {white_alive_diff}  "
                "# 항상 -black_diff일 것"
            )
            print(f"  Winner = {winner}")
            print("======================================")

    # ===========================================================
    #        모든 에피소드 종료 후, 전체 통계 + plt 출력
    # ===========================================================
    if DEBUG:
        print("\n\n=========== EVAL SUMMARY (ALL EPISODES) ===========")
        print(f"  num_episodes        : {num_episodes}")
        print(f"  total_steps         : {total_steps_all_episodes}")
        if num_episodes > 0:
            print(f"  avg steps/episode   : {total_steps_all_episodes / num_episodes:.2f}")
        print(f"  total_reward_black  : {total_reward_black:.3f}")
        print(f"  total_reward_white  : {total_reward_white:.3f}")
        print(f"  black_wins          : {black_wins}")
        print(f"  white_wins          : {white_wins}")
        print(f"  draws               : {draws}")

        # ★ top1~top5 확률 전체 평균 출력
        print("\n[EVAL] Average top-k probs over all steps (k=1..5)")
        for rank in range(5):
            cnt = topk_prob_counts[rank]
            if cnt > 0:
                avg_p = topk_prob_sums[rank] / cnt
                print(f"  avg top{rank+1} prob = {avg_p:.4f}  (count={cnt})")
            else:
                print(f"  avg top{rank+1} prob = n/a (no data)")

        # ★ 평균 entropy 출력
        if entropy_count > 0:
            avg_entropy = entropy_sum / entropy_count
            print(f"\n[EVAL] avg policy entropy (all steps) = {avg_entropy:.4f}")
        else:
            print("\n[EVAL] avg policy entropy (all steps) = n/a (no data)")

        # ★ 전체 액션 분포 요약 + rank-plot 재료
        mean_probs = None
        if action_prob_sums is not None and action_prob_counts > 0:
            mean_probs = action_prob_sums / action_prob_counts  # (n_actions,)

            print("\n[EVAL] Mean action probs summary (over all steps):")
            print(f"  n_actions      = {mean_probs.shape[0]}")
            print(f"  min(prob)      = {mean_probs.min():.6f}")
            print(f"  max(prob)      = {mean_probs.max():.6f}")
            print(f"  mean(prob)     = {mean_probs.mean():.6f}")
            print(f"  std(prob)      = {mean_probs.std():.6f}")

            # 상위 10개 액션 (평균 확률 기준)
            sorted_idx = np.argsort(-mean_probs)
            print("\n  Top-10 actions by mean prob:")
            for i in range(min(10, mean_probs.shape[0])):
                a_idx = int(sorted_idx[i])
                p = float(mean_probs[a_idx])
                print(f"    rank={i+1:2d}, action_idx={a_idx:4d}, mean_prob={p:.6f}")

            # CSV로 저장 (action_idx, mean_prob)
            out_csv = "eval_action_mean_probs.csv"
            data = np.stack(
                [np.arange(mean_probs.shape[0], dtype=np.int32), mean_probs],
                axis=1,
            )
            np.savetxt(
                out_csv,
                data,
                delimiter=",",
                header="action_idx,mean_prob",
                comments="",
            )
            print(f"\n[EVAL] saved mean action probs to: {out_csv}")

        # ★ 전체 probs (flatten) 저장
        all_probs_arr = None
        if len(all_probs_flat) > 0:
            all_probs_arr = np.array(all_probs_flat, dtype=np.float32)
            out_npy = "eval_all_probs.npy"
            np.save(out_npy, all_probs_arr)
            print(f"[EVAL] saved all step probs (flatten) to: {out_npy}")
            print(f"       shape={all_probs_arr.shape}, "
                  f"min={all_probs_arr.min():.6f}, max={all_probs_arr.max():.6f}")

        # =======================================================
        #                      matplotlib 플롯
        # =======================================================
        # 1) 전체 확률 히스토그램
        # if all_probs_arr is not None and all_probs_arr.size > 0:
        #     plt.figure(figsize=(6, 4))
        #     plt.hist(all_probs_arr, bins=50)
        #     plt.xlabel("Action probability")
        #     plt.ylabel("Count")
        #     plt.title("Histogram of action probabilities (all steps)")
        #     plt.tight_layout()
        #     plt.savefig("eval_probs_hist.png", dpi=150)
        #     print("[EVAL] saved histogram plot to: eval_probs_hist.png")

        # # 2) mean_probs rank-plot (내림차순 정렬)
        # if mean_probs is not None:
        #     sorted_idx = np.argsort(-mean_probs)
        #     sorted_probs = mean_probs[sorted_idx]
        #     ranks = np.arange(1, len(sorted_probs) + 1)

        #     plt.figure(figsize=(6, 4))
        #     plt.plot(ranks, sorted_probs)
        #     plt.xlabel("Action rank (1=highest mean prob)")
        #     plt.ylabel("Mean probability")
        #     plt.title("Rank plot of mean action probabilities")
        #     plt.xscale("log")  # 액션 많으면 로그 스케일이 보기 편함
        #     plt.tight_layout()
        #     plt.savefig("eval_mean_rank_plot.png", dpi=150)
        #     print("[EVAL] saved rank plot to: eval_mean_rank_plot.png")

        # # 3) 화면에도 한 번에 띄우고 싶으면
        # if (all_probs_arr is not None and all_probs_arr.size > 0) or (mean_probs is not None):
        #     plt.show()

    env.close()
    print(
        f"[EVAL] finished {num_episodes} episodes, "
        f"total_steps={total_steps_all_episodes}, "
        f"black_wins={black_wins}, white_wins={white_wins}, draws={draws}"
    )


# ------------------------------------------------
# CLI 엔트리포인트
# ------------------------------------------------
if __name__ == "__main__":
    # 기본 weight 경로 (필요에 따라 수정)
    black_path = r"/home/ubuntu/alkhagi/Game/checkpoints_v2/policy_epoch_000520.pt"
    white_path = r"/home/ubuntu/alkhagi/Game/checkpoints_v2/policy_epoch_000520.pt"

    # 기본 에피소드 수
    num_episodes = 50

    # CLI 인자 파싱: black=..., white=..., episodes=N
    for arg in sys.argv[1:]:
        if arg.startswith("black="):
            black_path = arg.split("=", 1)[1]
        elif arg.startswith("white="):
            white_path = arg.split("=", 1)[1]
        elif arg.startswith("episodes="):
            num_episodes = int(arg.split("=", 1)[1])

    main(
        render_mode="rgb_array",
        black_weight=black_path,
        white_weight=white_path,
        num_episodes=num_episodes,
    )
