#!/usr/bin/env python3
import csv
import os
import time
from pathlib import Path
from typing import List, Tuple

REFRESH_SEC = 1.0  # 갱신 주기 (초)


def read_header_and_last_row(path: Path) -> Tuple[List[str], List[str]]:
    """
    CSV에서 컬럼명(첫 줄)과 마지막 데이터 줄을 읽어서 (header, row)로 반환.
    파일이 없거나 데이터가 없으면 빈 리스트 반환.
    """
    if not path.exists():
        return [], []

    with path.open("r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    if not reader:
        return [], []

    header = reader[0]
    if len(reader) == 1:
        # 헤더만 있고 데이터 없음
        return header, []

    last_row = reader[-1]
    return header, last_row


def format_table(title: str, header: List[str], row: List[str]) -> str:
    """
    간단한 텍스트 테이블로 예쁘게 정렬해서 문자열로 반환.
    """
    if not header:
        return f"{title}\n  (파일 없음 또는 비어 있음)\n"

    if not row:
        return f"{title}\n  (헤더만 있고 데이터 없음)\n"

    # 각 컬럼 폭 계산
    widths = []
    for h, v in zip(header, row):
        widths.append(max(len(h), len(v)))

    def fmt_line(cols: List[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cols, widths))

    sep = "-+-".join("-" * w for w in widths)

    lines = [
        title,
        fmt_line(header),
        sep,
        fmt_line(row),
    ]
    return "\n".join(lines) + "\n"


def clear_screen() -> None:
    # 터미널 화면 지우기
    print("\033[H\033[J", end="")


def main():
    base_dir = Path(".")  # 필요하면 여기 경로 고정 가능

    files = [
        ("[2] training_metrics.csv (리그/승률)", base_dir / "training_metrics.csv"),
        ("[1] reward_breakdown_metrics.csv (리워드 구성)", base_dir / "reward_breakdown_metrics.csv"),
        ("[3] ppo_diagnostics_metrics.csv (PPO 진단)", base_dir / "ppo_diagnostics_metrics.csv"),
    ]

    try:
        while True:
            clear_screen()

            print("=== AlKkaGi PPO Self-Play 모니터링 ===")
            print("(agent.py 는 학습/로그만, 이 스크립트는 CSV 읽기 전용)\n")

            for title, path in files:
                header, row = read_header_and_last_row(path)
                print(format_table(title, header, row))

            print(f"(refresh: {REFRESH_SEC:.1f}s, Ctrl+C 로 종료)\n")
            time.sleep(REFRESH_SEC)
    except KeyboardInterrupt:
        print("\n모니터링 종료.")


if __name__ == "__main__":
    main()
