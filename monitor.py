#!/usr/bin/env python
import os
import time
import csv
from typing import List, Dict, Any

LOG_PATH = "training_metrics.csv"
REFRESH_SEC = 5          # 몇 초마다 갱신할지
LAST_N_EPOCHS = 100      # 최근 N 에폭 기준으로 요약 통계


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
            # 캐스팅
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


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def summarize(metrics: List[Dict[str, Any]], last_n: int):
    """
    전체 + 최근 last_n 에폭에 대한 요약 통계를 계산.
    """
    if not metrics:
        return None

    total_epochs = len(metrics)
    last_row = metrics[-1]

    if total_epochs <= last_n:
        recent = metrics
    else:
        recent = metrics[-last_n:]

    # 최근 N 에폭 기준 평균
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

    # 전체 승/무/패
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
        "last": last_row,
        "recent_win_rate": recent_win_rate,
        "recent_avg_reward": recent_avg_reward,
        "recent_avg_steps": recent_avg_steps,
        "recent_rating": recent_rating,
        "best_rating": best_rating,
        "total_episodes": total_episodes,
        "recent_episodes": recent_episodes,
        "total_wins": total_wins,
        "total_draws": total_draws,
        "total_losses": total_losses,
        "overall_win_rate": overall_win_rate,
        "num_players": last_row["num_players"],
    }


def compute_status(info: Dict[str, Any]) -> tuple[str, str]:
    """
    GREEN / YELLOW / RED 상태 판정 + 간단한 이유 텍스트.

    기준(최근 LAST_N_EPOCHS):
      - win_rate
      - avg_reward
      - rating vs best_rating
    """
    recent_win = info["recent_win_rate"]      # 0~1
    recent_rew = info["recent_avg_reward"]    # 대략 [-3,3]
    recent_rating = info["recent_rating"]
    best_rating = info["best_rating"]
    rating_drop = best_rating - recent_rating

    status = "GREEN"
    reasons: list[str] = []

    # --- RED 조건 ---
    if recent_win < 0.40:
        status = "RED"
        reasons.append(f"win_rate<{40}% (현재 {recent_win*100:.1f}%)")
    if recent_rew < -0.2:
        status = "RED"
        reasons.append(f"avg_reward<-0.2 (현재 {recent_rew:.3f})")
    if rating_drop > 80:
        status = "RED"
        reasons.append(f"rating {rating_drop:.1f}점 하락 (best={best_rating:.1f} → now={recent_rating:.1f})")

    # --- YELLOW 조건 (RED가 아닐 때만) ---
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


def print_dashboard(metrics: List[Dict[str, Any]]):
    clear_screen()

    if not metrics:
        print(f"[WAIT] '{LOG_PATH}' 파일을 찾을 수 없거나 아직 비어 있습니다.")
        print("      agent.py 학습이 시작되면 training_metrics.csv가 생성될 거야.")
        return

    info = summarize(metrics, LAST_N_EPOCHS)
    if info is None:
        print("[WARN] metrics 요약 계산 실패")
        return

    last = info["last"]
    status, status_reason = compute_status(info)
    recent_window = min(LAST_N_EPOCHS, info["total_epochs"])

    print("====================================")
    print(" RL League Self-Play Training Monitor")
    print("====================================")
    print(f"Log file          : {LOG_PATH}")
    print(f"Total epochs      : {info['total_epochs']}")
    print(f"Total episodes    : {info['total_episodes']}")
    print(f"Num league players: {info['num_players']}")
    print(f"Recent window     : last {recent_window} epochs")
    print("------------------------------------\n")

    # 마지막 에폭 상세
    print("[Last Epoch]")
    print(
        "Epoch  Ep(s)   W   D   L   Win%   AvgR     Steps   Rating  Players"
    )
    print(
        f"{last['epoch']:5d}  "
        f"{last['episodes']:5d}  "
        f"{last['wins']:3d} "
        f"{last['draws']:3d} "
        f"{last['losses']:3d} "
        f"{last['win_rate']*100:5.1f}% "
        f"{last['avg_reward']:7.3f} "
        f"{last['avg_steps']:7.2f} "
        f"{last['learner_rating']:7.1f} "
        f"{last['num_players']:7d}"
    )
    print()

    # 최근 N 에폭 요약
    print(f"[Recent {recent_window} Epochs Summary]")
    print(f"Recent episodes   : {info['recent_episodes']}")
    print(f"Recent win_rate   : {info['recent_win_rate']*100:6.2f}%")
    print(f"Recent avg_reward : {info['recent_avg_reward']:8.4f}")
    print(f"Recent avg_steps  : {info['recent_avg_steps']:8.2f}")
    print(f"Recent rating     : {info['recent_rating']:8.2f}")
    print()

    # 전체 누적 요약
    print("[Overall Summary]")
    print(f"Total W/D/L       : {info['total_wins']}/{info['total_draws']}/{info['total_losses']}")
    print(f"Overall win_rate  : {info['overall_win_rate']*100:6.2f}%")
    print(f"Best rating       : {info['best_rating']:8.2f}")
    print()

    # 상태 진단
    print("[Status Assessment]")
    print(f"Status (last {recent_window} epochs): {status}")
    print(f"Reason: {status_reason}")
    print("\n(CTRL+C 로 종료)\n")


def main():
    last_mtime = None

    while True:
        try:
            if os.path.exists(LOG_PATH):
                mtime = os.path.getmtime(LOG_PATH)
                # 파일이 바뀐 경우에만 다시 읽기
                if last_mtime is None or mtime != last_mtime:
                    metrics = load_metrics(LOG_PATH)
                    last_mtime = mtime
                    print_dashboard(metrics)
                # 파일이 안 바뀌었으면 그냥 화면 유지
            else:
                # 파일이 없으면 대기 메시지
                print_dashboard([])
            time.sleep(REFRESH_SEC)
        except KeyboardInterrupt:
            clear_screen()
            print("모니터링 종료.")
            break
        except Exception as e:
            clear_screen()
            print(f"[ERROR] {e}")
            time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    main()
