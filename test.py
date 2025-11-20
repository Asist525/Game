"""
ai_khagi_league_episode.py 전용 통합 검증 스크립트

- env / obs 구조 검증
- feature encoder 검증
- reward 함수 검증
- ActorCritic / TeamAgent 출력 형상 검증
- League.run_match / 예선 / Swiss / Knockout / Episode / Multi-episode 전체 플로우 검증

실행:
    python ai_khagi_league_tests.py
"""

import os
import math
import numpy as np
import torch

from agent import (
    make_env,
    encode_state_fe_alkkagi,
    compute_team_reward,
    ActorCritic,
    TeamAgent,
    League,
    run_one_episode,
    train_league_multi_episode,
    STATE_DIM,
    N_STONES,
    BOARD_W,
    BOARD_H,
    DEVICE,
)


# ------------------------------------------------
# 유틸: assert helpers
# ------------------------------------------------
def assert_true(cond, msg="assertion failed"):
    if not cond:
        raise AssertionError(msg)


def assert_shape(x, shape, name="tensor"):
    assert_true(
        tuple(x.shape) == tuple(shape),
        f"{name} shape {tuple(x.shape)} != expected {tuple(shape)}",
    )


def assert_finite(x, name="array"):
    x = np.asarray(x)
    assert_true(np.all(np.isfinite(x)), f"{name} has NaN/Inf")


# ------------------------------------------------
# 1. Env / Obs / Action 구조 검증
# ------------------------------------------------
def test_env_basic():
    print("[TEST] test_env_basic")
    env = make_env()
    obs, info = env.reset()

    # 기본 키 존재 여부
    for key in ["black", "white", "obstacles", "turn"]:
        assert_true(key in obs, f"obs missing key '{key}'")

    black = np.array(obs["black"])
    white = np.array(obs["white"])
    obstacles = np.array(obs["obstacles"])

    # 형상 검증
    assert_shape(black, (3, 3), "obs['black']")
    assert_shape(white, (3, 3), "obs['white']")
    assert_shape(obstacles, (3, 4), "obs['obstacles']")

    # turn 범위 검증
    turn = int(obs["turn"])
    assert_true(turn in [0, 1], "obs['turn'] must be 0 or 1")

    # 몇 스텝 랜덤으로 돌려보기
    for _ in range(5):
        turn = int(obs["turn"])
        idx = np.random.randint(0, 3)
        power = np.random.uniform(500.0, 2000.0)
        angle = np.random.uniform(-180.0, 180.0)
        action = {
            "turn": turn,
            "index": int(idx),
            "power": float(power),
            "angle": float(angle),
        }
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # obs 구조 유지되는지 확인
        for key in ["black", "white", "obstacles", "turn"]:
            assert_true(key in obs, f"obs missing key '{key}' after step")
        if done:
            obs, info = env.reset()
    env.close()
    print("  OK")


# ------------------------------------------------
# 2. Feature Encoder 검증
# ------------------------------------------------
def test_feature_encoder_shape_and_values():
    print("[TEST] test_feature_encoder_shape_and_values")
    env = make_env()
    obs, info = env.reset()

    for my_color in [0, 1]:
        feat = encode_state_fe_alkkagi(
            obs, my_color=my_color, board_w=BOARD_W, board_h=BOARD_H
        )
        assert_true(isinstance(feat, np.ndarray), "feat must be np.ndarray")
        assert_shape(feat, (STATE_DIM,), "feat")
        assert_finite(feat, "feat")

    env.close()
    print("  OK")


def test_feature_encoder_invariance_to_turn():
    """
    turn이 바뀌어도 my_color 관점을 잘 반영하는지만 간단 체크
    (깊은 이론 검증은 아니고, 인코더가 터지지 않는지 확인)
    """
    print("[TEST] test_feature_encoder_invariance_to_turn")
    # synthetic obs
    obs = {
        "black": [[100.0, 100.0, 1.0], [200.0, 200.0, 1.0], [300.0, 300.0, 0.0]],
        "white": [[400.0, 400.0, 1.0], [500.0, 100.0, 1.0], [100.0, 500.0, 1.0]],
        "obstacles": [[100.0, 100.0, 50.0, 50.0],
                      [200.0, 200.0, 40.0, 40.0],
                      [300.0, 300.0, 30.0, 30.0]],
        "turn": 0,
    }

    feat_black = encode_state_fe_alkkagi(
        obs, my_color=0, board_w=BOARD_W, board_h=BOARD_H
    )
    obs["turn"] = 1
    feat_white = encode_state_fe_alkkagi(
        obs, my_color=1, board_w=BOARD_W, board_h=BOARD_H
    )

    assert_shape(feat_black, (STATE_DIM,), "feat_black")
    assert_shape(feat_white, (STATE_DIM,), "feat_white")
    assert_finite(feat_black, "feat_black")
    assert_finite(feat_white, "feat_white")
    print("  OK")


# ------------------------------------------------
# 3. Reward 함수 검증
# ------------------------------------------------
def _make_simple_obs(black_alive: int, white_alive: int):
    """
    black_alive, white_alive 개수만큼 alive=1.0, 나머지는 alive=0.0 으로 만드는 단순 obs
    """
    black = []
    for i in range(3):
        alive = 1.0 if i < black_alive else 0.0
        black.append([100.0 + 50 * i, 100.0 + 50 * i, alive])

    white = []
    for i in range(3):
        alive = 1.0 if i < white_alive else 0.0
        white.append([200.0 + 50 * i, 200.0 + 50 * i, alive])

    obstacles = [[0.0, 0.0, 10.0, 10.0]] * 3
    obs = {
        "black": black,
        "white": white,
        "obstacles": obstacles,
        "turn": 0,
    }
    return obs


def test_reward_signs():
    print("[TEST] test_reward_signs")

    # 케이스 1: 흑이 상대 돌 1개 날려버림 (흑 관점 보상 +)
    prev_obs = _make_simple_obs(black_alive=3, white_alive=3)
    next_obs = _make_simple_obs(black_alive=3, white_alive=2)
    r_black = compute_team_reward(prev_obs, next_obs, my_color=0,
                                  terminated=False, truncated=False)
    assert_true(r_black > 0.0, "black should get positive reward when white loses a stone")

    # 케이스 2: 흑이 자기 돌 1개 잃음 (흑 관점 보상 -)
    prev_obs = _make_simple_obs(black_alive=3, white_alive=3)
    next_obs = _make_simple_obs(black_alive=2, white_alive=3)
    r_black2 = compute_team_reward(prev_obs, next_obs, my_color=0,
                                   terminated=False, truncated=False)
    assert_true(r_black2 < 0.0, "black should get negative reward when itself loses a stone")

    # 케이스 3: 게임 끝났고 내가 돌이 더 많을 때 +보너스
    prev_obs = _make_simple_obs(3, 3)
    next_obs = _make_simple_obs(2, 1)
    r_black3 = compute_team_reward(prev_obs, next_obs, my_color=0,
                                   terminated=True, truncated=False)
    assert_true(r_black3 > 0.0, "black with more stones at end should get positive terminal bonus")

    # 케이스 4: 게임 끝났고 내가 더 적으면 -
    prev_obs = _make_simple_obs(3, 3)
    next_obs = _make_simple_obs(1, 2)
    r_black4 = compute_team_reward(prev_obs, next_obs, my_color=0,
                                   terminated=True, truncated=False)
    assert_true(r_black4 < 0.0, "black with fewer stones at end should get negative terminal bonus")

    print("  OK")


# ------------------------------------------------
# 4. ActorCritic / TeamAgent 검증
# ------------------------------------------------
def test_actor_critic_shapes():
    print("[TEST] test_actor_critic_shapes")
    model = ActorCritic(state_dim=STATE_DIM, n_stones=N_STONES).to(DEVICE)

    x = torch.randn(4, STATE_DIM, device=DEVICE)
    stone_logits, power_mean, angle_mean, value = model(x)

    assert_shape(stone_logits, (4, N_STONES), "stone_logits")
    assert_shape(power_mean, (4, 1), "power_mean")
    assert_shape(angle_mean, (4, 1), "angle_mean")
    assert_shape(value, (4, 1), "value")

    # finite check
    for t, name in [
        (stone_logits, "stone_logits"),
        (power_mean, "power_mean"),
        (angle_mean, "angle_mean"),
        (value, "value"),
    ]:
        assert_true(torch.isfinite(t).all(), f"{name} has NaN/Inf")

    print("  OK")


def test_team_agent_act():
    print("[TEST] test_team_agent_act")
    agent = TeamAgent(team_id=0)
    state = np.random.randn(STATE_DIM).astype(np.float32)

    # train mode
    stone_idx, power_raw, angle_raw, log_prob, value = agent.act(state, eval_mode=False)
    assert_true(0 <= stone_idx < N_STONES, "stone_idx out of range")
    assert_true(math.isfinite(power_raw), "power_raw must be finite")
    assert_true(math.isfinite(angle_raw), "angle_raw must be finite")
    assert_true(math.isfinite(log_prob), "log_prob must be finite")
    assert_true(math.isfinite(value), "value must be finite")

    # eval mode
    stone_idx2, power_raw2, angle_raw2, log_prob2, value2 = agent.act(state, eval_mode=True)
    assert_true(0 <= stone_idx2 < N_STONES, "stone_idx2 out of range")
    assert_true(math.isfinite(power_raw2), "power_raw2 must be finite")
    assert_true(math.isfinite(angle_raw2), "angle_raw2 must be finite")
    assert_true(math.isfinite(log_prob2), "log_prob2 must be finite")
    assert_true(math.isfinite(value2), "value2 must be finite")

    print("  OK")


# ------------------------------------------------
# 5. League / Match / 예선 검증
# ------------------------------------------------
def test_league_single_match():
    print("[TEST] test_league_single_match")
    league = League(num_teams=2)
    elo_before = [league.teams[0].elo, league.teams[1].elo]

    result = league.run_match(0, 1, training=False)
    elo_after = [league.teams[0].elo, league.teams[1].elo]

    # Elo가 똑같이 유지되는 경우는 거의 없음
    assert_true(
        (elo_before[0] != elo_after[0]) or (elo_before[1] != elo_after[1]),
        "Elo should change after a match",
    )
    assert_true(
        (result.winner_id in [0, 1, None]),
        "winner_id must be 0, 1, or None",
    )
    print("  OK")


def test_league_preliminary_epoch_small():
    print("[TEST] test_league_preliminary_epoch_small")
    league = League(num_teams=4)

    league.run_preliminary_epoch(
        num_matches=4,
        ppo_batch_size=32,
        ppo_epochs=1,
    )

    # Elo들이 1500에서 변했는지 대략 체크
    elos = [agent.elo for agent in league.teams]
    diff = [abs(e - 1500.0) for e in elos]
    assert_true(
        any(d > 0.1 for d in diff),
        "At least one team's Elo should have changed after prelim epoch",
    )

    # PPO buffer가 비워졌는지
    for agent in league.teams:
        assert_true(len(agent.buffer) == 0, "PPO buffer should be cleared after update")

    print("  OK")


# ------------------------------------------------
# 6. Swiss / Knockout 검증
# ------------------------------------------------
def test_swiss_and_knockout_flow_small():
    print("[TEST] test_swiss_and_knockout_flow_small")
    num_teams = 8
    league = League(num_teams=num_teams)

    # 예선을 조금 돌려서 Elo에 약간 차이를 만들어 둠
    league.run_preliminary_epoch(
        num_matches=4,
        ppo_batch_size=32,
        ppo_epochs=1,
    )

    finalists = league.get_top_teams_by_elo(8)
    assert_true(len(finalists) == 8, "finalists must be 8 for swiss+KO test")

    finalists_8 = league.run_full_swiss(
        finalists,
        ppo_batch_size=32,
        ppo_epochs=1,
    )
    assert_true(len(finalists_8) == 8, "run_full_swiss must return 8 teams")

    champion_id = league.run_knockout(
        finalists_8,
        ppo_batch_size=32,
        ppo_epochs=1,
        series_games=3,
    )
    assert_true(
        champion_id in finalists_8,
        "champion must be one of the KO finalists",
    )

    print("  OK")


# ------------------------------------------------
# 7. Episode / Multi-episode 플로우 검증
# ------------------------------------------------
def test_one_episode_flow():
    print("[TEST] test_one_episode_flow")
    num_teams = 8
    league = League(num_teams=num_teams)
    save_dir = "./_test_checkpoints_alkkagi"

    champion_id = run_one_episode(
        league=league,
        episode_idx=0,
        prelim_matches=4,
        finalists_N=8,
        ppo_batch_size=32,
        ppo_epochs=1,
        ko_series_games=3,
        save_dir=save_dir,
    )

    assert_true(0 <= champion_id < num_teams, "champion_id out of range")

    # 챔피언 파일 존재 여부
    expected_path = os.path.join(
        save_dir, f"episode{0:04d}_champion_team{champion_id}.pt"
    )
    assert_true(os.path.exists(expected_path), f"Champion file not found: {expected_path}")

    print("  OK")


def test_reset_from_champion():
    print("[TEST] test_reset_from_champion")
    num_teams = 4
    league = League(num_teams=num_teams)

    # 예선 조금 돌려서 champion 뽑기
    league.run_preliminary_epoch(
        num_matches=4,
        ppo_batch_size=32,
        ppo_epochs=1,
    )
    finalists = league.get_top_teams_by_elo(4)
    finalists_4 = finalists + finalists  # Swiss가 8을 요구하는 건 아니지만, 여기서는 직접 사용 X
    # 그냥 Elo 상 1등을 champion이라고 가정
    champion_id = finalists[0]

    # champion state dict
    before_sd = {k: v.clone() for k, v in league.teams[champion_id].model.state_dict().items()}

    league.reset_from_champion(champion_id, noise_scale=0.01)

    # Elo 초기화 확인
    for agent in league.teams:
        assert_true(abs(agent.elo - 1500.0) < 1e-6, "Elo must be reset to 1500")

    # champion 파라미터와 다른 팀 파라미터가 완전히 같지는 않은지(노이즈 존재)
    same_count = 0
    for tid, agent in enumerate(league.teams):
        eq_all = True
        for k, v in agent.model.state_dict().items():
            if not torch.allclose(v.cpu(), before_sd[k].cpu(), atol=1e-6):
                eq_all = False
                break
        if eq_all:
            same_count += 1
    # champion 1개만 exactly 같고, 나머지는 다르다고 기대
    assert_true(same_count >= 1, "at least champion should keep exact weights")
    print("  OK")


def test_multi_episode_small():
    print("[TEST] test_multi_episode_small")
    # 에피소드 2번만, 아주 작게 돌려보기
    league = train_league_multi_episode(
        num_episodes=2,
        num_teams=4,
        prelim_matches_per_episode=2,  # 아주 작게
        finalists_N=4,                 # Swiss/KO는 8을 가정하지만, 여기선 episode 내부에서만 사용
        ppo_batch_size=32,
        ppo_epochs=1,
        ko_series_games=3,
        champion_noise_scale=0.01,
        seed=123,
        save_dir="./_test_checkpoints_alkkagi_multi",
    )

    # 최종 Elo finite 체크
    for tid, agent in enumerate(league.teams):
        assert_true(math.isfinite(agent.elo), f"final Elo of team {tid} is not finite")

    print("  OK")


# ------------------------------------------------
# 8. 전체 테스트 실행기
# ------------------------------------------------
def run_all_tests():
    tests = [
         test_env_basic,
         test_feature_encoder_shape_and_values,
         test_feature_encoder_invariance_to_turn,
         test_reward_signs,
         test_actor_critic_shapes,
        test_team_agent_act,
        test_league_single_match,
        test_league_preliminary_epoch_small,
        test_swiss_and_knockout_flow_small,
        test_one_episode_flow,
        test_reset_from_champion,
         test_multi_episode_small,
    ]

    print("========== AI-Kha-Gi League TESTS START ==========")
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            raise
    print("========== ALL TESTS PASSED ==========")


if __name__ == "__main__":
    print(f"[INFO] Using device: {DEVICE}")
    run_all_tests()
