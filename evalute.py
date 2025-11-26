# evalute_human.py
"""
알까기 자가대전 / 평가 스크립트 (human 렌더링 전용, 최소 로그)

- 흑/백 에이전트 weight를 로드해서 사람 눈으로 보면서 플레이만 확인
- 스텝 단위 디버그 로그, 엔트로피, top-k 등은 전부 제거
- 에피소드마다 최종 승패만 한 줄로 출력

실행 예시:
    (venv) python evalute_human.py \
        black=/home/ubuntu/alkhagi/Game/checkpoints/policy_epoch_000200.pt \
        white=/home/ubuntu/alkhagi/Game/checkpoints/policy_epoch_000200.pt \
        episodes=10
"""

import sys

import gymnasium as gym
import kymnasium as kym  # env 등록용

from agent import (
    YourBlackAgent,
    YourWhiteAgent,
    compute_alive_diff,
)


def evaluate_human(
    black_weight: str,
    white_weight: str,
    num_episodes: int = 1,
    render_mode: str = "human",
):
    """
    human 렌더링으로 num_episodes 만큼 자가대전 평가.
    스텝별 로그 없이, 에피소드마다 최종 승패만 출력.
    """
    if not black_weight or not white_weight:
        raise ValueError("black_weight와 white_weight 경로를 반드시 지정해야 합니다.")

    # 환경 생성 (human 렌더링)
    env = gym.make(
        id="kymnasium/AlKkaGi-3x3-v0",
        render_mode=render_mode,
        obs_type="custom",
        bgm=True,
    )

    # 에이전트 로드
    agent_black = YourBlackAgent.load(black_weight)
    agent_white = YourWhiteAgent.load(white_weight)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            turn = int(obs["turn"])  # 0 = black, 1 = white

            if turn == 0:
                action = agent_black.act(obs, info)
            else:
                action = agent_white.act(obs, info)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # 에피소드 종료 후 최종 승패 계산
        black_alive_diff = compute_alive_diff(obs, my_color=0)

        if black_alive_diff > 0:
            winner = "Black"
        elif black_alive_diff < 0:
            winner = "White"
        else:
            winner = "Draw"

        print(f"[EP {ep+1}/{num_episodes}] episode finished, winner = {winner}")

    env.close()


if __name__ == "__main__":
    # 기본 weight 경로 (필요에 따라 수정)
    black_path = r"/home/ubuntu/alkhagi/Game/checkpoints/policy_epoch_002300.pt"
    white_path = r"/home/ubuntu/alkhagi/Game/checkpoints/policy_epoch_002300.pt"
    episodes = 1

    # CLI 인자 파싱: black=..., white=..., episodes=...
    for arg in sys.argv[1:]:
        if arg.startswith("black="):
            black_path = arg.split("=", 1)[1]
        elif arg.startswith("white="):
            white_path = arg.split("=", 1)[1]
        elif arg.startswith("episodes="):
            try:
                episodes = int(arg.split("=", 1)[1])
            except ValueError:
                pass

    evaluate_human(
        black_weight=black_path,
        white_weight=white_path,
        num_episodes=episodes,
        render_mode="human",
    )
