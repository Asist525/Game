import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import softmax

from agent import (
    encode_state_fe_alkkagi, ActorCritic, idx_to_action,
    BOARD_W, BOARD_H, STATE_DIM,
    N_STONES, N_ANGLES, N_POWERS, N_ACTIONS
)


# -----------------------------
# Utils for full policy analysis
# -----------------------------
def decode_triplet(action_idx, N_STONES, N_ANGLES, N_POWERS):
    stone_id = action_idx // (N_ANGLES * N_POWERS)
    rem = action_idx % (N_ANGLES * N_POWERS)
    angle_id = rem // N_POWERS
    power_id = rem % N_POWERS
    return stone_id, angle_id, power_id


def build_table_from_logits(logits):
    """logits → stone x angle x power 3D 테이블"""
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    table = np.zeros((N_STONES, N_ANGLES, N_POWERS), dtype=np.float32)

    for idx, p in enumerate(probs):
        s, a, pw = decode_triplet(idx, N_STONES, N_ANGLES, N_POWERS)
        table[s, a, pw] = p

    return table


def print_axis_preferences(table):
    stone_pref = table.sum(axis=(1, 2))
    angle_pref = table.sum(axis=(0, 2))
    power_pref = table.sum(axis=(0, 1))

    print("\n[Stone preference]", stone_pref)
    print("[Angle preference]", angle_pref)
    print("[Power preference]", power_pref)


def plot_angle_power_heatmap(table, stone_id=None, title_prefix=""):
    if stone_id is None:
        mat = table.sum(axis=0)
        title = f"{title_prefix} Heatmap (ALL Stones)"
    else:
        mat = table[stone_id]
        title = f"{title_prefix} Heatmap (Stone {stone_id})"

    plt.figure(figsize=(6, 5))
    plt.imshow(mat, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Power ID")
    plt.ylabel("Angle ID")
    plt.tight_layout()
    plt.show()


# -----------------------------
#   FULL VERIFY FUNCTION
# -----------------------------
def verify_policy_full(model_path: str, steps: int = 20, analyze_every: int = 1):
    """
    analyze_every: STEP마다 분석하면 1, 5로 하면 5스텝마다 분석 출력
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ActorCritic(state_dim=STATE_DIM, n_actions=N_ACTIONS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = gym.make(
        "kymnasium/AlKkaGi-3x3-v0",
        obs_type="custom",
        render_mode=None,
        bgm=False,
    )

    obs, info = env.reset()
    my_color = 0

    print("\n==== FULL POLICY TEST START ====\n")

    for t in range(steps):
        print(f"\n================")
        print(f"----- STEP {t} -----")

        # -------- Encode state -----------
        state_np = encode_state_fe_alkkagi(obs, my_color, BOARD_W, BOARD_H)
        state_t = torch.from_numpy(state_np).float().to(device)

        # -------- Forward pass ------------
        with torch.no_grad():
            logits, value = model(state_t.unsqueeze(0))
            logits = logits.squeeze(0)

        probs = softmax(logits, dim=-1)

        # -------- Basic info print --------
        argmax_idx = int(torch.argmax(probs).item())
        argmax_prob = float(probs[argmax_idx].item())
        dist = torch.distributions.Categorical(logits=logits)
        sampled_idx = int(dist.sample().item())

        print(f"[Value] {value.item():.4f}")
        print(f"[Argmax] idx={argmax_idx}, prob={argmax_prob:.4f}")
        print(f"[Sample] idx={sampled_idx}")

        topk = torch.topk(probs, 5)
        print("Top-5 idx:", topk.indices.tolist())
        print("Top-5 prob:", [float(p) for p in topk.values.tolist()])

        # -------- FULL DISTRIBUTION ANALYSIS --------
        if t % analyze_every == 0:
            print("\n=== FULL ACTION DISTRIBUTION ANALYSIS ===")

            # build 3D table (stone × angle × power)
            table = build_table_from_logits(logits)

            # axis-wise preferences
            print_axis_preferences(table)

            # heatmap (ALL stones)
            plot_angle_power_heatmap(table, None, f"STEP {t}")

            # per-stone heatmap
            for s in range(N_STONES):
                plot_angle_power_heatmap(table, s, f"STEP {t}")

        # -------- environment step --------
        action_dict = idx_to_action(sampled_idx, obs)
        obs, _, terminated, truncated, info = env.step(action_dict)

        if terminated or truncated:
            print("Episode ended -> reset.")
            obs, info = env.reset()

    env.close()
    print("\n==== FULL TEST END ====\n")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    verify_policy_full("alkkagi_ppo_selfplay.pt", steps=30, analyze_every=1)
