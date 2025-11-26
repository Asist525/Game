#!/usr/bin/env python
"""
training_metrics.csv를 실시간으로 읽어서
- win_rate (%)
- learner_rating
- avg_reward
를 epoch 기준으로 계속 그래프로 보여주는 모니터.

• GREEN / YELLOW / RED 상태를 제목에 같이 표시
• 각 지표에 이동평균(MA) 추가
• 전체기간(all) / 최근 N 에폭 토글 (--window) 지원
"""

import os
import time
import csv
import argparse
from typing import List, Dict, Any

import matplotlib.pyplot as plt

# -------------------
# CLI
# -------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="training_metrics.csv", help="로그 CSV 경로")
    p.add_argument("--refresh", type=float, default=2.0, help="갱신 주기(sec)")
    p.add_argument("--window", default="all", help="'all' 또는 최근 N(정수)")
    p.add_argument("--ma", type=int, default=20, help="이동평균 윈도우(epochs)")
    return p.parse_args()

# -------------------
# IO
# -------------------
def load_metrics(log_path: str) -> List[Dict[str, Any]]:
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
                continue
            rows.append(row)
    return rows

# -------------------
# Stats / Status
# -------------------
def summarize(metrics: List[Dict[str, Any]], last_n: int | None):
    if not metrics:
        return None

    total_epochs = len(metrics)
    recent = metrics if (last_n is None or last_n <= 0 or total_epochs <= last_n) else metrics[-last_n:]

    def avg(key: str) -> float:
        return sum(r[key] for r in recent) / len(recent) if recent else 0.0

    recent_win_rate = avg("win_rate")
    recent_avg_reward = avg("avg_reward")
    recent_avg_steps = avg("avg_steps")
    recent_rating = recent[-1]["learner_rating"] if recent else 0.0
    best_rating = max(r["learner_rating"] for r in metrics) if metrics else 0.0

    total_episodes = sum(r["episodes"] for r in metrics)
    recent_episodes = sum(r["episodes"] for r in recent) if recent else 0

    total_wins = sum(r["wins"] for r in metrics)
    total_draws = sum(r["draws"] for r in metrics)
    total_losses = sum(r["losses"] for r in metrics)
    overall_win_rate = (
        total_wins / (total_wins + total_draws + total_losses)
        if (total_wins + total_draws + total_losses) > 0 else 0.0
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
        "num_players": recent[-1]["num_players"] if recent else metrics[-1]["num_players"],
    }

def compute_status(info: Dict[str, Any]) -> tuple[str, str]:
    recent_win = info["recent_win_rate"]
    recent_rew = info["recent_avg_reward"]
    recent_rating = info["recent_rating"]
    best_rating = info["best_rating"]
    rating_drop = best_rating - recent_rating

    status = "GREEN"
    reasons: list[str] = []

    if recent_win < 0.40:
        status = "RED"; reasons.append(f"win_rate<{40}% (현재 {recent_win*100:.1f}%)")
    if recent_rew < -0.2:
        status = "RED"; reasons.append(f"avg_reward<-0.2 (현재 {recent_rew:.3f})")
    if rating_drop > 80:
        status = "RED"; reasons.append(f"rating {rating_drop:.1f}점 하락 (best={best_rating:.1f} → now={recent_rating:.1f})")

    if status != "RED":
        if 0.40 <= recent_win < 0.50:
            status = "YELLOW"; reasons.append(f"win_rate 40~50% (현재 {recent_win*100:.1f}%)")
        if -0.2 <= recent_rew < 0.05:
            status = "YELLOW"; reasons.append(f"avg_reward -0.2~0.05 (현재 {recent_rew:.3f})")
        if 40 < rating_drop <= 80:
            status = "YELLOW"; reasons.append(f"rating best 대비 {rating_drop:.1f} 하락")

    if not reasons:
        reasons.append(
            f"win_rate={recent_win*100:.1f}%, avg_reward={recent_rew:.3f}, "
            f"rating={recent_rating:.1f} (best={best_rating:.1f})"
        )
    return status, "; ".join(reasons)

def moving_average(values: List[float], window: int) -> tuple[List[float], List[float]]:
    n = len(values)
    if n < window or window <= 1:
        return list(range(n)), values[:]  # 데이터가 적으면 그대로 반환
    ma_vals: List[float] = []
    s = sum(values[:window])
    ma_vals.append(s / window)
    for i in range(window, n):
        s += values[i] - values[i - window]
        ma_vals.append(s / window)
    indices = list(range(window - 1, n))
    return indices, ma_vals

# -------------------
# Main (plot loop)
# -------------------
def main():
    args = parse_args()

    # window 해석: 'all' 이면 전체, 정수면 최근 N
    if str(args.window).lower() == "all":
        last_n = None  # 전체기간
        window_label = "ENTIRE period"
    else:
        try:
            n = int(args.window)
            last_n = max(1, n)
            window_label = f"last {last_n} epochs"
        except ValueError:
            last_n = None
            window_label = "ENTIRE period"

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax_win, ax_rating, ax_reward = axes

    last_mtime = None
    cached_metrics: List[Dict[str, Any]] = []

    while True:
        try:
            if os.path.exists(args.log):
                mtime = os.path.getmtime(args.log)
                if last_mtime is None or mtime != last_mtime:
                    cached_metrics = load_metrics(args.log)
                    last_mtime = mtime
            else:
                cached_metrics = []

            # 그래프 초기화
            for ax in axes:
                ax.cla()

            if cached_metrics:
                metrics = cached_metrics

                # 최근 N 또는 전체기간 선택
                if last_n is None or len(metrics) <= (last_n or 0):
                    recent = metrics
                else:
                    recent = metrics[-last_n:]

                epochs = [r["epoch"] for r in recent]
                win_rates = [r["win_rate"] * 100.0 for r in recent]
                ratings = [r["learner_rating"] for r in recent]
                avg_rewards = [r["avg_reward"] for r in recent]

                # 상태
                info = summarize(metrics, last_n)
                status_str = ""
                if info is not None:
                    status, reason = compute_status(info)
                    status_str = f" | Status: {status}"

                # 1) win_rate + MA
                ax_win.plot(epochs, win_rates, label="win_rate (%)")
                idx_ma, ma_win = moving_average(win_rates, args.ma)
                if ma_win:
                    ma_epochs = [epochs[i] for i in idx_ma]
                    ax_win.plot(ma_epochs, ma_win, label=f"MA{args.ma}")
                ax_win.set_ylabel("Win Rate (%)")
                ax_win.grid(True)
                ax_win.legend()

                # 2) rating + MA
                ax_rating.plot(epochs, ratings, label="rating")
                idx_ma_r, ma_rating = moving_average(ratings, args.ma)
                if ma_rating:
                    ma_epochs_r = [epochs[i] for i in idx_ma_r]
                    ax_rating.plot(ma_epochs_r, ma_rating, label=f"MA{args.ma}")
                ax_rating.set_ylabel("ELO Rating")
                ax_rating.grid(True)
                ax_rating.legend()

                # 3) avg_reward + MA
                ax_reward.plot(epochs, avg_rewards, label="avg_reward")
                idx_ma_rew, ma_rew = moving_average(avg_rewards, args.ma)
                if ma_rew:
                    ma_epochs_rew = [epochs[i] for i in idx_ma_rew]
                    ax_reward.plot(ma_epochs_rew, ma_rew, label=f"MA{args.ma}")
                ax_reward.set_xlabel("Epoch")
                ax_reward.set_ylabel("Avg Reward")
                ax_reward.grid(True)
                ax_reward.legend()

                fig.suptitle(
                    f"RL League Training ({window_label}, total={len(metrics)} epochs){status_str}"
                )
            else:
                ax_win.text(
                    0.5, 0.5,
                    f"'{args.log}' 없음 또는 내용 비어 있음\n학습 시작 후 다시 보세요.",
                    ha="center", va="center", transform=ax_win.transAxes,
                )
                for ax in axes:
                    ax.set_axis_off()

            fig.tight_layout()
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(args.refresh)

        except KeyboardInterrupt:
            print("그래프 모니터 종료.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(args.refresh)

if __name__ == "__main__":
    main()
