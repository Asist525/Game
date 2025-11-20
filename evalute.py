import gymnasium as gym
import kymnasium as kym

from agent_submission import YourBlackAgent, YourWhiteAgent

# checkpoint paths
black_ckpt = r"C:\Users\Samsung\Desktop\avoid\checkpoints_alkkagi\episode0007_champion_team3.pt"
white_ckpt = r"C:\Users\Samsung\Desktop\avoid\checkpoints_alkkagi\episode0006_champion_team6.pt"

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
    if obs["turn"] == 0:
        action = black.act(obs, info)
    else:
        action = white.act(obs, info)

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
