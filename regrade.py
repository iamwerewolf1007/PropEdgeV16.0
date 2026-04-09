"""
regrade.py — PropEdge V16.0
──────────────────────────────────────────────────────────────────────────────
Clears grading for a specific date in both season_2025_26.json and today.json,
then re-runs batch0_grade.py with the updated rolling recompute logic.

Usage:
    python3 regrade.py                     # re-grade yesterday
    python3 regrade.py 2026-04-05          # re-grade specific date
    python3 regrade.py 2026-04-05 --dry-run  # show what would be cleared, don't run

What it does:
    1. Clears result/actualPts/delta/lossType/postMatchReason for the date
       in season_2025_26.json and today.json
    2. Runs batch0_grade.py which:
       - Fetches box scores via ScoreboardV3 + BoxScoreTraditionalV3
       - Grades all cleared plays
       - Appends new game rows to nba_gamelogs_2025_26.csv
       - Recomputes rolling stats (L3/L5/L10/L20/L30) for all players who played
       - Rebuilds DVP + H2H
       - Pushes to GitHub
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import FILE_SEASON_2526, FILE_TODAY, clean_json, uk_now


# ── Fields that get written during grading — reset to ungraded state ─────────
GRADING_FIELDS = {
    "result":           "",
    "actualPts":        None,
    "delta":            None,
    "lossType":         None,
    "postMatchReason":  None,
}


def clear_grading_for_date(date_str: str, dry_run: bool = False) -> dict:
    """
    Clear grading fields for all plays on date_str in both JSON files.
    Returns summary of what was cleared.
    """
    summary = {
        "season_cleared": 0,
        "today_cleared":  0,
        "season_total":   0,
        "today_total":    0,
    }

    # ── season_2025_26.json ───────────────────────────────────────────────────
    if FILE_SEASON_2526.exists():
        try:
            with open(FILE_SEASON_2526) as f:
                season = json.load(f)

            cleared = 0
            for p in season:
                if p.get("date") != date_str:
                    continue
                summary["season_total"] += 1
                if p.get("result") in ("WIN", "LOSS", "DNP", "PUSH"):
                    cleared += 1
                    if not dry_run:
                        p.update(GRADING_FIELDS)

            summary["season_cleared"] = cleared

            if not dry_run and cleared > 0:
                with open(FILE_SEASON_2526, "w") as f:
                    json.dump(clean_json(season), f, indent=2)
                print(f"  ✓ season_2025_26.json — cleared {cleared} graded plays for {date_str}")
            elif dry_run:
                print(f"  [DRY RUN] season_2025_26.json — would clear {cleared} graded plays for {date_str}")
            else:
                print(f"  season_2025_26.json — no graded plays found for {date_str}")

        except Exception as e:
            print(f"  ✗ season_2025_26.json error: {e}")

    else:
        print(f"  ⚠ season_2025_26.json not found at {FILE_SEASON_2526}")

    # ── today.json ────────────────────────────────────────────────────────────
    if FILE_TODAY.exists():
        try:
            with open(FILE_TODAY) as f:
                today = json.load(f)

            cleared = 0
            for p in today:
                if p.get("date") != date_str:
                    continue
                summary["today_total"] += 1
                if p.get("result") in ("WIN", "LOSS", "DNP", "PUSH"):
                    cleared += 1
                    if not dry_run:
                        p.update(GRADING_FIELDS)

            summary["today_cleared"] = cleared

            if not dry_run and cleared > 0:
                with open(FILE_TODAY, "w") as f:
                    json.dump(clean_json(today), f, indent=2)
                print(f"  ✓ today.json          — cleared {cleared} graded plays for {date_str}")
            elif dry_run:
                print(f"  [DRY RUN] today.json          — would clear {cleared} graded plays for {date_str}")
            else:
                print(f"  today.json          — no graded plays found for {date_str}")

        except Exception as e:
            print(f"  ✗ today.json error: {e}")

    else:
        print(f"  ⚠ today.json not found — will be created during grading")

    return summary


def run_grade(date_str: str) -> None:
    """Run batch0_grade.py for the given date."""
    import subprocess
    print(f"\n  Running batch0_grade.py for {date_str}...")
    r = subprocess.run(
        [sys.executable, str(ROOT / "batch0_grade.py"), date_str],
        cwd=ROOT,
    )
    if r.returncode != 0:
        print(f"  ✗ batch0_grade.py failed (rc={r.returncode})")
    else:
        print(f"  ✓ batch0_grade.py complete")


def main() -> None:
    # Parse args
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if not a.startswith("--")]

    # Determine date
    if args:
        date_str = args[0]
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print(f"  ✗ Invalid date format '{date_str}' — use YYYY-MM-DD")
            sys.exit(1)
    else:
        date_str = (uk_now() - __import__('datetime').timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"\n  PropEdge V16.0 — Re-grade {date_str}{' [DRY RUN]' if dry_run else ''}")
    print("  " + "─" * 56)

    # Step 1: Show what exists before clearing
    print(f"\n[1/3] Checking current grading state for {date_str}...")
    season_plays = []
    today_plays  = []
    if FILE_SEASON_2526.exists():
        with open(FILE_SEASON_2526) as f:
            all_plays = json.load(f)
        season_plays = [p for p in all_plays if p.get("date") == date_str]
    if FILE_TODAY.exists():
        with open(FILE_TODAY) as f:
            all_today = json.load(f)
        today_plays = [p for p in all_today if p.get("date") == date_str]

    s_graded   = [p for p in season_plays if p.get("result") in ("WIN","LOSS","DNP","PUSH")]
    s_ungraded = [p for p in season_plays if p.get("result") not in ("WIN","LOSS","DNP","PUSH")]
    t_graded   = [p for p in today_plays  if p.get("result") in ("WIN","LOSS","DNP","PUSH")]
    t_ungraded = [p for p in today_plays  if p.get("result") not in ("WIN","LOSS","DNP","PUSH")]

    print(f"  season_2025_26.json : {len(season_plays)} plays for {date_str}")
    print(f"    Graded:   {len(s_graded)} ({sum(1 for p in s_graded if p.get('result')=='WIN')}W / {sum(1 for p in s_graded if p.get('result')=='LOSS')}L / {sum(1 for p in s_graded if p.get('result')=='DNP')} DNP)")
    print(f"    Ungraded: {len(s_ungraded)}")
    print(f"  today.json          : {len(today_plays)} plays for {date_str}")
    print(f"    Graded:   {len(t_graded)}")
    print(f"    Ungraded: {len(t_ungraded)}")

    total_to_clear = len(s_graded) + len(t_graded)
    if total_to_clear == 0 and not season_plays and not today_plays:
        print(f"\n  ⚠ No plays found for {date_str} in either file.")
        print(f"  Run: python3 run.py predict  to generate predictions first.")
        sys.exit(0)

    # Step 2: Clear grading
    print(f"\n[2/3] Clearing grading for {date_str}...")
    summary = clear_grading_for_date(date_str, dry_run=dry_run)
    print(f"  Total cleared: {summary['season_cleared']} (season) + {summary['today_cleared']} (today)")

    if dry_run:
        print(f"\n  [DRY RUN] No changes made. Remove --dry-run to execute.")
        sys.exit(0)

    # Step 3: Re-grade
    print(f"\n[3/3] Re-grading {date_str} with updated rolling recompute...")
    run_grade(date_str)

    print(f"\n  ✓ Re-grade complete for {date_str}")
    print(f"  Rolling stats recomputed for all players who played {date_str}.")
    print(f"  Run python3 run.py predict to generate fresh predictions.\n")


if __name__ == "__main__":
    main()
