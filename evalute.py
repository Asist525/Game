import gymnasium as gym
import kymnasium as kym

from agent import YourBlackAgent, YourWhiteAgent

# checkpoint paths
black_ckpt = r"/home/ubuntu/alkhagi/Game/alkkagi_ppo_selfplay.pt"
white_ckpt = r"/home/ubuntu/alkhagi/Game/alkkagi_ppo_selfplay.pt"

black = YourBlackAgent.load(black_ckpt)
white = YourWhiteAgent.load(white_ckpt)

env = gym.make(
    id='kymnasium/AlKkaGi-3x3-v0',
    render_mode='human',
    obs_type='custom',
    bgm=False,
)

obs, info = env.reset()
done = False

while not done:
    print("turn:", obs["turn"])
    if obs["turn"] == 0:
        print("  >> Black turn")
        action = black.act(obs, info)
    else:
        print("  >> White turn")
        action = white.act(obs, info)

    print("  action:", action)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
