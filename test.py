# test.py
# 같은 디렉토리에 있는 agent.py 를 import 해서 전체 검증

import os
import tempfile
from pprint import pprint
from pathlib import Path

import numpy as np
import torch

import agent


def header(msg: str):
    print("\n" + "=" * 100)
    print(f"### {msg}")
    print("=" * 100)


# ============================================================
# 더미 obs 생성
# ============================================================
def make_dummy_obs(turn: int = 0):
    BW, BH = agent.BOARD_W, agent.BOARD_H
    return {
        "black": np.array(
            [
                [0.0, 0.0, 1.0],
                [BW / 2, BH / 2, 1.0],
                [BW * 1.2, BH * 1.2, 0.0],
            ],
            dtype=np.float32,
        ),
        "white": np.array(
            [
                [BW * 0.8, BH * 0.2, 1.0],
                [BW * 0.9, BH * 0.9, 0.0],
                [BW * -0.1, BH * 1.1, 1.0],
            ],
            dtype=np.float32,
        ),
        "obstacles": np.array(
            [
                [0.0, 0.0, BW * 0.2, BH * 0.2],
                [BW * 0.5, BH * 0.3, BW * 0.1, BH * 0.1],
                [BW * 1.1, BH * 1.1, BW * 0.3, BH * 0.3],
            ],
            dtype=np.float32,
        ),
        "turn": int(turn),
    }


# ============================================================
# 1) 인코딩 / 정규화
# ============================================================
def test_full_encoding():
    header("1) Encoding / Normalization Diagnostics")

    obs = make_dummy_obs(turn=1)
    print("[RAW OBS]")
    pprint(obs)

    me, opp, obs_n, t = agent.split_me_opp(obs, my_color=0)
    print("\nSplit (my_color=0):")
    pprint({"me": me, "opp": opp, "turn": t})

    n_black = agent.normalize_stones(obs["black"], agent.BOARD_W, agent.BOARD_H)
    print("\nNormalized black stones:")
    print(n_black)
    print("  x range:", n_black[:, 0].min(), "→", n_black[:, 0].max())
    print("  y range:", n_black[:, 1].min(), "→", n_black[:, 1].max())

    n_obs = agent.normalize_obstacles(
        obs["obstacles"], agent.BOARD_W, agent.BOARD_H
    )
    print("\nNormalized obstacles:")
    print(n_obs)

    feat31 = agent.encode_state_basic_alkkagi(
        obs, my_color=0, board_w=agent.BOARD_W, board_h=agent.BOARD_H
    )
    print("\n31D feature:")
    print(feat31)
    print(
        "  shape:", feat31.shape,
        "  stats: min=%.5f max=%.5f mean=%.5f"
        % (feat31.min(), feat31.max(), feat31.mean())
    )

    feat52 = agent.encode_state_fe_alkkagi(
        obs, my_color=0, board_w=agent.BOARD_W, board_h=agent.BOARD_H
    )
    print("\n52D feature:")
    print(feat52)
    print(
        "  shape:", feat52.shape,
        "  stats: min=%.5f max=%.5f mean=%.5f"
        % (feat52.min(), feat52.max(), feat52.mean())
    )

    # 31차원 부분 일치 확인
    assert np.allclose(feat31, feat52[:31]), "31D 부분이 FE encoder와 불일치"

    print("[OK] Encoding diagnostics done")


# ============================================================
# 2) Action Discretization
# ============================================================
def test_full_action_space():
    header("2) Action Discretization Diagnostics")

    print("N_STONES =", agent.N_STONES)
    print("N_ANGLES =", agent.N_ANGLES)
    print("N_POWERS =", agent.N_POWERS)
    print("N_ACTIONS =", agent.N_ACTIONS)

    stones = []
    angles = []
    powers = []

    for idx in range(agent.N_ACTIONS):
        s, a, p = agent.decode_action_index(idx)
        stones.append(s)
        angles.append(a)
        powers.append(p)

    stones = np.array(stones)
    angles = np.array(angles, dtype=np.float32)
    powers = np.array(powers, dtype=np.float32)

    uniq_stones, cnt_stones = np.unique(stones, return_counts=True)
    print("\nStone distribution:")
    for sid, c in zip(uniq_stones, cnt_stones):
        print(f"  stone {sid}: {c}")

    print("\nAngle range: min=%.2f max=%.2f" % (angles.min(), angles.max()))
    print("Unique angles:", sorted(set(round(v, 2) for v in angles.tolist())))

    print("\nPower range: min=%.2f max=%.2f" % (powers.min(), powers.max()))
    print("Unique powers:", sorted(set(round(v, 2) for v in powers.tolist())))

    # 유일 조합 검증
    seen = set()
    for s, a, p in zip(stones, angles, powers):
        seen.add((int(s), round(float(a), 4), round(float(p), 4)))
    assert len(seen) == agent.N_ACTIONS, "decode_action_index 조합이 중복됨"

    print("[OK] Action space diagnostics done")


# ============================================================
# 3) PolicyValueNet
# ============================================================
def test_network_diagnostics():
    header("3) PolicyValueNet Diagnostics")

    net = agent.PolicyValueNet(state_dim=52, n_actions=agent.N_ACTIONS)
    x = torch.randn(4, 52)
    logits, values = net(x)

    print("logits shape:", logits.shape)
    print("values shape:", values.shape)
    print(
        "logits stats: min=%.4f max=%.4f mean=%.4f"
        % (logits.min().item(), logits.max().item(), logits.mean().item())
    )
    print(
        "values stats: min=%.4f max=%.4f mean=%.4f"
        % (values.min().item(), values.max().item(), values.mean().item())
    )

    loss = (logits ** 2).mean() + values.mean()
    loss.backward()

    print("\nGradient norms:")
    for name, p in net.named_parameters():
        if p.grad is not None:
            print(f"  {name}: {p.grad.norm().item():.6f}")

    print("[OK] PolicyValueNet diagnostics done")


# ============================================================
# 4) PPOPolicy
# ============================================================
def test_policy_diagnostics():
    header("4) PPOPolicy Diagnostics")

    device = torch.device("cpu")
    policy = agent.PPOPolicy(device=device)

    obs = make_dummy_obs(turn=0)

    print("[act_eval 결과]")
    act_eval = policy.act_eval(observation=obs, my_color=0)
    pprint(act_eval)

    print("\n[act_train 결과]")
    action, action_idx, logprob, value, state_vec = policy.act_train(
        observation=obs, my_color=0
    )
    print("action_idx:", action_idx)
    print("logprob:", logprob)
    print("value:", value)
    print(
        "state_vec: shape", state_vec.shape,
        "mean=%.4f std=%.4f" % (state_vec.mean().item(), state_vec.std().item())
    )

    # save / load roundtrip
    print("\n[Save / Load roundtrip]")
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name

    try:
        policy.save(path)
        loaded = agent.PPOPolicy.load(path, device=device)
    finally:
        if os.path.exists(path):
            os.remove(path)

    max_diff = 0.0
    for k, v in policy.model.state_dict().items():
        diff = (v - loaded.model.state_dict()[k]).abs().max().item()
        max_diff = max(max_diff, diff)
    print("max weight diff after load:", max_diff)

    # PPO update 영향도
    print("\n[PPO update 영향도]")
    before = {k: v.clone() for k, v in policy.model.state_dict().items()}

    n = 32
    states = [torch.randn(52) for _ in range(n)]
    actions = [int(np.random.randint(0, agent.N_ACTIONS)) for _ in range(n)]
    old_logprobs = [0.0 for _ in range(n)]
    advantages = np.random.randn(n).astype(np.float32).tolist()
    returns = np.random.randn(n).astype(np.float32).tolist()

    cfg = agent.PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        update_epochs=1,  # 테스트용
        batch_size=16,
    )

    agent.ppo_update(
        policy=policy,
        states=states,
        actions=actions,
        old_logprobs=old_logprobs,
        advantages=advantages,
        returns=returns,
        config=cfg,
    )

    movements = []
    for k in before:
        movements.append(
            (before[k] - policy.model.state_dict()[k]).abs().mean().item()
        )
    print("avg weight movement after one PPO update:", float(np.mean(movements)))

    print("[OK] PPOPolicy diagnostics done")


# ============================================================
# 5) GAE / Terminal Reward
# ============================================================
def test_gae_reward_diagnostics():
    header("5) GAE / Terminal Reward Diagnostics")

    obs = make_dummy_obs()
    obs["black"][:, 2] = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    obs["white"][:, 2] = np.array([1.0, 0.0, 1.0], dtype=np.float32)

    r_black = agent.compute_terminal_reward(obs, my_color=0)
    r_white = agent.compute_terminal_reward(obs, my_color=1)
    print("R_black:", r_black, "R_white:", r_white, "sum:", r_black + r_white)
    assert abs(r_black + r_white) < 1e-6

    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.zeros_like(rewards)
    adv, ret = agent.compute_gae_returns(
        rewards=rewards, values=values, gamma=0.99, lam=0.95
    )
    print("\nGAE advantages:", adv)
    print("GAE returns:", ret)

    print("[OK] GAE / reward diagnostics done")


# ============================================================
# 6) ELO / League
# ============================================================
def test_elo_diagnostics():
    header("6) ELO / League Diagnostics")

    p1 = agent.RatedPolicy(id="A", policy=agent.PPOPolicy(), rating=1500.0)
    p2 = agent.RatedPolicy(id="B", policy=agent.PPOPolicy(), rating=1400.0)
    league = agent.EloLeague(players=[p1, p2], k=32.0)

    print("Before ratings:", [(p.id, p.rating) for p in league.players])
    opp = league.choose_opponent("A")
    print("Opponent chosen for A:", opp.id)

    league.update_result("A", "B", score_a=1.0)
    print("After ratings:", [(p.id, p.rating, p.games) for p in league.players])

    print("[OK] ELO / League diagnostics done")


# ============================================================
# 7) 체크포인트 / 로그 파일
# ============================================================
def test_checkpoint_and_logging():
    header("7) Checkpoint / CSV Logging Diagnostics")

    device = torch.device("cpu")
    policy = agent.PPOPolicy(device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "ckpts")
        log_path = os.path.join(tmpdir, "metrics.csv")

        print("\n[Checkpoint Rotation Test]")
        for ep in range(1, 6):
            path = agent.save_checkpoint_policy(
                policy,
                epoch=ep,
                checkpoint_dir=ckpt_dir,
                max_keep=3,
            )
            print(f"  saved: {path}")

        ckpts = sorted(Path(ckpt_dir).glob("policy_epoch_*.pt"))
        print("  remaining checkpoints:", [p.name for p in ckpts])
        assert len(ckpts) <= 3

        print("\n[Training Metrics CSV Test]")
        agent.append_epoch_log(
            epoch=1,
            episodes=10,
            wins=6,
            draws=2,
            losses=2,
            avg_reward=0.4,
            avg_steps=12.3,
            learner_rating=1510.0,
            num_players=2,
            log_path=log_path,
        )
        agent.append_epoch_log(
            epoch=2,
            episodes=8,
            wins=4,
            draws=1,
            losses=3,
            avg_reward=0.125,
            avg_steps=11.0,
            learner_rating=1520.0,
            num_players=3,
            log_path=log_path,
        )

        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
        print("CSV content:\n", content)

    print("[OK] Checkpoint / logging diagnostics done")


# ============================================================
# 8) Env step
# ============================================================
def test_env_diagnostics():
    header("8) Environment Step Diagnostics")

    env = agent.make_env(render_mode=None, bgm=False)
    device = torch.device("cpu")
    shared_policy = agent.PPOPolicy(device=device)
    black_agent = agent.YourBlackAgent(policy=shared_policy, device=device)
    white_agent = agent.YourWhiteAgent(policy=shared_policy, device=device)

    obs, info = env.reset(seed=0)
    print("Initial turn:", obs["turn"])
    print("Initial black stones:\n", obs["black"])
    print("Initial white stones:\n", obs["white"])

    for step in range(10):
        if obs["turn"] == 0:
            action = black_agent.act(obs, info)
        else:
            action = white_agent.act(obs, info)

        print(f"\n[Step {step}] action:", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"  reward={reward}, terminated={terminated}, truncated={truncated}"
        )
        if terminated or truncated:
            print("  -> episode finished")
            break

    env.close()
    print("[OK] Env step diagnostics done")


# ============================================================
# 9) Self-play (shared + league)
# ============================================================
def test_selfplay_full():
    header("9) Self-play PPO / League Full Diagnostics")

    cfg = agent.PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        update_epochs=1,   # 테스트용 작게
        batch_size=32,
    )

    print("\n[Shared PPO Self-play]")
    learner_shared = agent.train_selfplay_shared_ppo(
        num_episodes=3,
        config=cfg,
    )
    print("Shared PPO training finished. type:", type(learner_shared))

    print("\n[League Self-play]")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = os.path.join(tmpdir, "ckpts")
        log_path = os.path.join(tmpdir, "metrics.csv")

        learner, league = agent.train_league_selfplay(
            num_epochs=1,
            episodes_per_epoch=3,
            snapshot_interval=1,
            config=cfg,
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=1000,  # 1 epoch에서는 안 찍힘
            max_checkpoints=3,
            log_path=log_path,
        )

    print("\nFinal league players:")
    for p in league.players:
        print(f"  id={p.id}, rating={p.rating:.1f}, games={p.games}")

    print("[OK] Self-play full diagnostics done")


# ============================================================
# MAIN
# ============================================================
def main():
    print("======== FULL DIAGNOSTICS START ========")

    test_full_encoding()
    test_full_action_space()
    test_network_diagnostics()
    test_policy_diagnostics()
    test_gae_reward_diagnostics()
    test_elo_diagnostics()
    test_checkpoint_and_logging()
    test_env_diagnostics()
    test_selfplay_full()

    print("======== FULL DIAGNOSTICS DONE ========")


if __name__ == "__main__":
    main()
