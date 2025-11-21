# evaluate.py
import gymnasium as gym
import kymnasium as kym  # env 등록용

from agent import YourBlackAgent, YourWhiteAgent


def main(render_mode: str = "human"):
    env = gym.make(
        id="kymnasium/AlKkaGi-3x3-v0",
        render_mode=render_mode,
        obs_type="custom",
        bgm=True,
    )

    # 학습 때 agent.py에서 저장한 weight 파일
    weight_path = "shared_policy.pt"

    agent_black = YourBlackAgent.load(weight_path)
    agent_white = YourWhiteAgent.load(weight_path)

    obs, info = env.reset()
    done = False
    step = 0

    while not done:
        if obs["turn"] == 0:
            action = agent_black.act(obs, info)
        else:
            action = agent_white.act(obs, info)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    env.close()
    print(f"[EVAL] episode finished in {step} steps")


if __name__ == "__main__":
    main(render_mode="human")  # 학습 중 디버그면 None으로 돌려도 됨
