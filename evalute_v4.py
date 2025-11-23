# evalute.py
"""
알까기 자가대전 / 평가 스크립트 (PPO + baseline 인코더 버전)

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
        black=/home/ubuntu/alkhagi/Game/alkkagi_ppo_black.pt \
        white=/home/ubuntu/alkhagi/Game/alkkagi_ppo_white.pt
"""

import sys
from typing import Any, Dict

import gymnasium as gym
import kymnasium as kym  # env 등록용

import numpy as np
import torch
from torch.distributions import Categorical

# === 여기 파일 이름을 네가 쓰는 에이전트 파일명으로 바꿔줘 ===
# 예: from agent_ppo import ...
from agent_v4 import (
    YourBlackAgent,
    YourWhiteAgent,
    AlkkagiObservation,
    encode_state_basic_tensor,
    get_valid_action_mask,
    N_ACTIONS,
)

DEBUG = True  # 디버그 로그 on/off


# ------------------------------------------------
# alive diff helper (흑/백 돌 개수 차이)
# ------------------------------------------------
def compute_alive_diff(obs: Dict[str, Any], my_color: int) -> int:
    """
    my_color 관점에서 alive_diff = (내 alive 수) - (상대 alive 수)
    obs: env에서 주는 raw dict
    """
    assert my_color in (0, 1)
    if my_color == 0:
        me = np.array(obs["black"], dtype=np.float32)
        opp = np.array(obs["white"], dtype=np.float32)
    else:
        me = np.array(obs["white"], dtype=np.float32)
        opp = np.array(obs["black"], dtype=np.float32)

    my_alive = int((me[:, 2] > 0.5).sum())
    opp_alive = int((opp[:, 2] > 0.5).sum())
    return my_alive - opp_alive


# ------------------------------------------------
# 좌표 출력 helper
# ------------------------------------------------
def print_stones(obs: Dict[str, Any]):
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

    # 각각 별도 weight 로드 (Your*Agent.load 안에서 PolicyValueNet 생성 + state_dict 로드)
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

        # 현재 턴의 정책 네트워크 가져오기
        if turn == 0:
            agent = agent_black
            actor_id = "BlackAgent"
        else:
            agent = agent_white
            actor_id = "WhiteAgent"

        # YourAlkkagiAgentBase 에서 self.net, self.device를 들고 있음
        net = agent.net
        device = agent.device

        # ---- policy 분포 / entropy / top-5 액션 디버그 ----
        if DEBUG:
            with torch.no_grad():
                obs_wrapped = AlkkagiObservation(obs)
                state_t = encode_state_basic_tensor(
                    obs_wrapped,
                    my_color=turn,
                    device=device,
                ).unsqueeze(0)  # (1, state_dim)

                logits, _ = net(state_t)  # logits: (1, N_ACTIONS)

                # 유효 액션 마스크 적용 (죽은 돌 제거)
                mask_np = get_valid_action_mask(obs_wrapped)  # (N_ACTIONS,)
                mask_t = torch.from_numpy(mask_np).to(device=device, dtype=torch.float32)
                mask_t = mask_t.unsqueeze(0)  # (1, N_ACTIONS)

                invalid = mask_t < 0.5
                logits_masked = logits.masked_fill(invalid, -1e9)

                dist = Categorical(logits=logits_masked)
                probs = dist.probs.squeeze(0).cpu().numpy()
                entropy = dist.entropy().item()

                topk_idx = probs.argsort()[::-1][:5]

                print(f"  policy entropy = {entropy:.4f}")
                print("  top-5 actions (idx, prob):")
                for i in topk_idx:
                    print(f"    idx={int(i):4d}, p={probs[i]:.4f}")

        # ---- 에이전트 액션 (Your*Agent.act: greedy argmax 정책) ----
        action = agent.act(obs, info)

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

        # 환경 진행 (env.reward는 기본적으로 0이지만, alive_diff로도 충분히 분석 가능)
        obs_next, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # reward는 "현재 턴 플레이어" 기준이라고 가정 (지금 env는 0이라 크게 의미는 없음)
        if turn == 0:
            total_reward_black += float(reward)
        else:
            total_reward_white += float(reward)

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
    black_path = r"/home/ubuntu/alkhagi/Game/alkkagi_ppo.pt"
    white_path = r"/home/ubuntu/alkhagi/Game/checkpoints_v4/alkkagi_ep_003000.pt"

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
