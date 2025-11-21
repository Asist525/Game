#!/usr/bin/env python
"""
training_metrics.csv를 실시간으로 읽어서
- win_rate (%)
- learner_rating
- avg_reward
를 epoch 기준으로 계속 그래프로 보여주는 모니터.

GREEN / YELLOW / RED 상태를 제목에 같이 표시하고,
각 지표에는 단기 이평선(MA)을 같이 그린다.
"""

import os
import time
import csv
from typing import List, Dict, Any

import matplotlib.pyplot as plt

LOG_PATH = "training_metrics.csv"
REFRESH_SEC = 2          # 몇 초마다 그래프 갱신할지
LAST_N_EPOCHS = 200      # 최근 N 에폭만 그리기
MA_WINDOW = 20           # 이동 평균 윈도우 (epoch 기준)


def load_metrics(log_path: str) -> List[Dict[str, Any]]:
    """
    training_metrics.csv 읽어서
    각 row를 dict로 반환.
    숫자 컬럼은 float/int로 캐스팅.
    """
    if not os.path.exists(log_path):
        return []

    rows: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["epoch"] = int(row["epoch"])
                row["episodes"] = int(row["episodes"])
                row["wins"] = int(row["wins"])
                row["draws"] = int(row["draws"])
                row["losses"] = int(row["losses"])
                row["win_rate"] = float(row["win_rate"])
                row["avg_reward"] = float(row["avg_reward"])
                row["avg_steps"] = float(row["avg_steps"])
                row["learner_rating"] = float(row["learner_rating"])
                row["num_players"] = int(row["num_players"])
            except (KeyError, ValueError):
                # 포맷이 깨진 라인은 스킵
                continue
            rows.append(row)
    return rows


def summarize(metrics: List[Dict[str, Any]]):
    """
    plot.py와 동일한 로직으로 최근 구간 요약 + 상태 계산에 사용.
    """
    if not metrics:
        return None

    total_epochs = len(metrics)
    if total_epochs <= LAST_N_EPOCHS:
        recent = metrics
    else:
        recent = metrics[-LAST_N_EPOCHS:]

    def avg(key: str) -> float:
        if not recent:
            return 0.0
        return sum(r[key] for r in recent) / len(recent)

    recent_win_rate = avg("win_rate")
    recent_avg_reward = avg("avg_reward")
    recent_avg_steps = avg("avg_steps")
    recent_rating = recent[-1]["learner_rating"]
    best_rating = max(r["learner_rating"] for r in metrics)
    total_episodes = sum(r["episodes"] for r in metrics)
    recent_episodes = sum(r["episodes"] for r in recent)

    total_wins = sum(r["wins"] for r in metrics)
    total_draws = sum(r["draws"] for r in metrics)
    total_losses = sum(r["losses"] for r in metrics)
    overall_win_rate = (
        total_wins / (total_wins + total_draws + total_losses)
        if (total_wins + total_draws + total_losses) > 0
        else 0.0
    )

    return {
        "total_epochs": total_epochs,
        "recent_win_rate": recent_win_rate,
        "recent_avg_reward": recent_avg_reward,
        "recent_avg_steps": recent_avg_steps,
        "recent_rating": recent_rating,
        "best_rating": best_rating,
        "total_episodes": total_episodes,
        "recent_episodes": recent_episodes,
        "overall_win_rate": overall_win_rate,
        "num_players": recent[-1]["num_players"],
    }


def compute_status(info: Dict[str, Any]) -> tuple[str, str]:
    """
    GREEN / YELLOW / RED 상태 판정 + 간단한 이유 텍스트.
    (plot.py와 동일 기준)
    """
    recent_win = info["recent_win_rate"]
    recent_rew = info["recent_avg_reward"]
    recent_rating = info["recent_rating"]
    best_rating = info["best_rating"]
    rating_drop = best_rating - recent_rating

    status = "GREEN"
    reasons: list[str] = []

    # RED 조건
    if recent_win < 0.40:
        status = "RED"
        reasons.append(f"win_rate<{40}% (현재 {recent_win*100:.1f}%)")
    if recent_rew < -0.2:
        status = "RED"
        reasons.append(f"avg_reward<-0.2 (현재 {recent_rew:.3f})")
    if rating_drop > 80:
        status = "RED"
        reasons.append(f"rating {rating_drop:.1f}점 하락 (best={best_rating:.1f} → now={recent_rating:.1f})")

    # YELLOW 조건
    if status != "RED":
        if 0.40 <= recent_win < 0.50:
            status = "YELLOW"
            reasons.append(f"win_rate 40~50% 구간 (현재 {recent_win*100:.1f}%)")
        if -0.2 <= recent_rew < 0.05:
            status = "YELLOW"
            reasons.append(f"avg_reward -0.2~0.05 구간 (현재 {recent_rew:.3f})")
        if 40 < rating_drop <= 80:
            status = "YELLOW"
            reasons.append(f"rating best 대비 {rating_drop:.1f}점 하락")

    if not reasons:
        reasons.append(
            f"win_rate={recent_win*100:.1f}%, avg_reward={recent_rew:.3f}, "
            f"rating={recent_rating:.1f} (best={best_rating:.1f})"
        )

    return status, "; ".join(reasons)


def moving_average(values: List[float], window: int) -> tuple[List[float], List[float]]:
    """
    단순 이동 평균.
    returns:
      - ma_x_indices: window-1, window, ..., len(values)-1
      - ma_values
    (epoch 축과 align 시에 사용)
    """
    n = len(values)
    if n < window:
        return [], []

    ma_vals: List[float] = []
    s = sum(values[:window])
    ma_vals.append(s / window)
    for i in range(window, n):
        s += values[i] - values[i - window]
        ma_vals.append(s / window)

    indices = list(range(window - 1, n))
    return indices, ma_vals


def main():
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax_win, ax_rating, ax_reward = axes

    last_mtime = None

    while True:
        try:
            if os.path.exists(LOG_PATH):
                mtime = os.path.getmtime(LOG_PATH)

                # 파일이 바뀐 경우에만 다시 읽기
                if last_mtime is None or mtime != last_mtime:
                    metrics = load_metrics(LOG_PATH)
                    last_mtime = mtime
                else:
                    metrics = globals().get("_last_metrics", [])
            else:
                metrics = []

            globals()["_last_metrics"] = metrics

            # 그래프 초기화
            ax_win.cla()
            ax_rating.cla()
            ax_reward.cla()

            if metrics:
                # 최근 N 에폭만 사용
                if len(metrics) > LAST_N_EPOCHS:
                    recent = metrics[-LAST_N_EPOCHS:]
                else:
                    recent = metrics

                epochs = [r["epoch"] for r in recent]
                win_rates = [r["win_rate"] * 100.0 for r in recent]  # %
                ratings = [r["learner_rating"] for r in recent]
                avg_rewards = [r["avg_reward"] for r in recent]

                # 상태 요약
                info = summarize(metrics)
                status_str = ""
                if info is not None:
                    status, reason = compute_status(info)
                    status_str = f" | Status: {status}"

                # 1) win_rate + MA
                ax_win.plot(epochs, win_rates, label="win_rate (%)")
                idx_ma, ma_win = moving_average(win_rates, MA_WINDOW)
                if idx_ma:
                    ma_epochs = [epochs[i] for i in idx_ma]
                    ax_win.plot(ma_epochs, ma_win, label=f"MA{MA_WINDOW}")
                ax_win.set_ylabel("Win Rate (%)")
                ax_win.grid(True)
                ax_win.legend()

                # 2) rating + MA
                ax_rating.plot(epochs, ratings, label="rating")
                idx_ma_r, ma_rating = moving_average(ratings, MA_WINDOW)
                if idx_ma_r:
                    ma_epochs_r = [epochs[i] for i in idx_ma_r]
                    ax_rating.plot(ma_epochs_r, ma_rating, label=f"MA{MA_WINDOW}")
                ax_rating.set_ylabel("ELO Rating")
                ax_rating.grid(True)
                ax_rating.legend()

                # 3) avg_reward + MA
                ax_reward.plot(epochs, avg_rewards, label="avg_reward")
                idx_ma_rew, ma_rew = moving_average(avg_rewards, MA_WINDOW)
                if idx_ma_rew:
                    ma_epochs_rew = [epochs[i] for i in idx_ma_rew]
                    ax_reward.plot(ma_epochs_rew, ma_rew, label=f"MA{MA_WINDOW}")
                ax_reward.set_xlabel("Epoch")
                ax_reward.set_ylabel("Avg Reward")
                ax_reward.grid(True)
                ax_reward.legend()

                fig.suptitle(
                    f"RL League Training (last {len(recent)} epochs, total={len(metrics)}){status_str}"
                )
            else:
                # 데이터 없으면 안내만
                ax_win.text(
                    0.5,
                    0.5,
                    f"'{LOG_PATH}' 없음 또는 내용 비어 있음\n학습 시작 후 다시 보세요.",
                    ha="center",
                    va="center",
                    transform=ax_win.transAxes,
                )
                ax_win.set_axis_off()
                ax_rating.set_axis_off()
                ax_reward.set_axis_off()

            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()

            time.sleep(REFRESH_SEC)

        except KeyboardInterrupt:
            print("그래프 모니터 종료.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    main()
