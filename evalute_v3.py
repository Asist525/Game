# evalute.py
"""
알까기 자가대전 / 평가 스크립트 (agent_v3 기반, 좌표 로그 버전)

- 흑/백 에이전트 weight를 로드해서 한 에피소드 플레이
- 각 스텝마다:
    * alive_diff (흑/백 관점)
    * policy entropy
    * 상위 5개 액션 (디스크리트 인덱스, 확률)
    * 실제 선택된 액션 (돌 index, angle, power)
    * 흑/백 돌 좌표 및 alive 상태
    * 슈팅한 돌의 위치 변화 (before → after)

실행 예시:
    (venv) python evalute.py \
        black=/home/ubuntu/alkhagi/Game/checkpoints_v3/policy_epoch_000200_v3.pt \
        white=/home/ubuntu/alkhagi/Game/checkpoints_v3/policy_epoch_000105_v3.pt
"""

import sys

import gymnasium as gym
import kymnasium as kym  # env 등록용

import torch
from torch.distributions import Categorical

from agent_v3 import (
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

    # 각각 별도 weight 로드
    agent_black = YourBlackAgent.load(black_weight)
    agent_white = YourWhiteAgent.load(white_weight)

    if DEBUG:
        print("[EVAL] Agents loaded successfully.")
        print("       BlackAgent : YourBlackAgent")
        print("       WhiteAgent : YourWhiteAgent")

    obs, info = env.reset()
    done = False
    step = 0

    total_reward_black = 0.0
    total_reward_white = 0.0

    if DEBUG:
        print("\n[EVAL] Initial observation")
        print(f"  turn         : {obs['turn']} (0=black, 1=white)")
        print_stones(obs)
        diff_b0 = compute_alive_diff(obs, my_color=0)
        diff_w0 = compute_alive_diff(obs, my_color=1)
        print(f"  initial alive_diff: black POV={diff_b0}, white POV={diff_w0}")
        print("--------------------------------------")

    while not done:
        turn = int(obs["turn"])  # 0=black,1=white
        current_player = "black" if turn == 0 else "white"

        if DEBUG:
            print(f"[STEP {step}] turn = {turn} ({current_player})")
            diff_b = compute_alive_diff(obs, my_color=0)
            diff_w = compute_alive_diff(obs, my_color=1)
            print(f"  alive_diff(before action): black POV={diff_b}, white POV={diff_w}")
            print_stones(obs)

        # 현재 턴의 PPOPolicy 가져와서 policy 분포/entropy 확인
        if turn == 0:
            ppo = agent_black.policy
            actor_id = "BlackAgent"
        else:
            ppo = agent_white.policy
            actor_id = "WhiteAgent"

        if DEBUG:
            with torch.no_grad():
                state_t = encode_state_fe_tensor(
                    obs,
                    my_color=turn,
                    device=ppo.device,
                ).unsqueeze(0)  # (1, state_dim)

                logits = ppo.actor(state_t)
                dist = Categorical(logits=logits)
                probs = dist.probs.squeeze(0).cpu().numpy()
                entropy = dist.entropy().item()

                topk_idx = probs.argsort()[::-1][:5]

                print(f"  policy entropy = {entropy:.4f}")
                print("  top-5 actions (idx, prob):")
                for i in topk_idx:
                    print(f"    idx={int(i):4d}, p={probs[i]:.4f}")

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

    # 에피소드 종료 후 최종 상태 출력
    if DEBUG:
        print("\n[EVAL] Final observation")
        print(f"  final turn   : {obs['turn']} (0=black, 1=white)")
        print_stones(obs)

    black_alive_diff = compute_alive_diff(obs, my_color=0)
    white_alive_diff = compute_alive_diff(obs, my_color=1)

    if black_alive_diff > 0:
        winner = "Black"
    elif black_alive_diff < 0:
        winner = "White"
    else:
        winner = "Draw"

    if DEBUG:
        print("\n[EVAL] Result summary")
        print(f"  total_reward_black : {total_reward_black:.3f}")
        print(f"  total_reward_white : {total_reward_white:.3f}")
        print(f"  alive_diff (black POV) = {black_alive_diff}")
        print(
            f"  alive_diff (white POV) = {white_alive_diff}  "
            "# 항상 -black_diff일 것"
        )
        print(f"  Winner = {winner}")

    env.close()
    print(f"[EVAL] episode finished in {step} steps, winner={winner}")


# ------------------------------------------------
# CLI 엔트리포인트
# ------------------------------------------------
if __name__ == "__main__":
    # 기본 weight 경로 (필요에 따라 수정)
    black_path = r"/home/ubuntu/alkhagi/Game/checkpoints_v3/policy_epoch_003000_v3.pt"
    white_path = r"/home/ubuntu/alkhagi/Game/checkpoints_v3/policy_epoch_003150_v3.pt"

    # CLI 인자 파싱: black=..., white=...
    for arg in sys.argv[1:]:
        if arg.startswith("black="):
            black_path = arg.split("=", 1)[1]
        elif arg.startswith("white="):
            white_path = arg.split("=", 1)[1]

    main(
        render_mode="human",
        black_weight=black_path,
        white_weight=white_path,
    )
