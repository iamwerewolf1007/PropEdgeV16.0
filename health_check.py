"""
PropEdge V16.0 — health_check.py
==================================
Self-healing system health checker.

Runs a battery of checks across every data store, detects issues,
and auto-fixes what it can. Logs all findings to data/audit_log.csv.

Usage:
    python3 health_check.py            # full check + auto-fix
    python3 health_check.py --dry-run  # report only, no fixes
    python3 health_check.py --quick    # skip slow Excel checks
    python3 run.py check               # calls this via run.py

Check categories:
    [FILE]   Required files present and non-empty
    [GRADE]  Stuck ungraded plays older than 1 day
    [DEDUP]  Duplicate plays in season JSON and ML dataset
    [TRUST]  Trust scores stale (>3 days old) or missing players
    [MODEL]  Elite pkl missing, stale (>35 days), or not loadable
    [LOG]    Game log gaps, suspicious row counts
    [SYNC]   today.json vs season JSON consistency
    [MISS]   Missing box scores for graded plays (0 actual pts)

Auto-fix actions:
    grade --date YYYY-MM-DD   Re-run grading for stuck/missing dates
    dedup season JSON         Remove duplicate plays in season JSON
    dedup ML dataset          Run ml_dataset.write_ml_dataset() to rebuild
    update trust scores       Call model_trainer.update_trust_scores()
"""

from __future__ import annotations

import csv
from monthly_split import (
    get_monthly_index, verify_monthly_integrity,
    load_monthly_split, list_monthly_files
)
import json
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    VERSION, FILE_SEASON_2526, FILE_SEASON_2425, FILE_TODAY,
    FILE_GL_2526, FILE_H2H, FILE_PROPS, FILE_PROPS_2425,
    FILE_ELITE_MODEL, FILE_V12_TRUST, FILE_V14_TRUST,
    FILE_DVP, FILE_AUDIT, uk_now,
)

DRY_RUN = "--dry-run" in sys.argv
QUICK   = "--quick"   in sys.argv

# ─────────────────────────────────────────────────────────────────────────────
# RESULT TRACKING
# ─────────────────────────────────────────────────────────────────────────────

class CheckResult:
    def __init__(self):
        self.passed:  list[str] = []
        self.warnings: list[str] = []
        self.failures: list[str] = []
        self.fixes_applied: list[str] = []
        self.fixes_failed:  list[str] = []

    def ok(self,   msg: str): self.passed.append(msg);   print(f"  ✓ {msg}")
    def warn(self, msg: str): self.warnings.append(msg); print(f"  ⚠ {msg}")
    def fail(self, msg: str): self.failures.append(msg); print(f"  ✗ {msg}")
    def fixed(self, msg: str): self.fixes_applied.append(msg); print(f"  🔧 {msg}")
    def fix_fail(self, msg: str): self.fixes_failed.append(msg); print(f"  ✗ FIX FAILED: {msg}")

    @property
    def score(self) -> str:
        t = len(self.passed) + len(self.warnings) + len(self.failures)
        return f"{len(self.passed)}/{t} passed, {len(self.warnings)} warnings, {len(self.failures)} failures"


R = CheckResult()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _run_grade(date_str: str) -> bool:
    """Run batch0_grade.py for a specific date. Returns True on success."""
    if DRY_RUN:
        R.warn(f"[DRY RUN] Would run: python3 run.py grade --date {date_str}")
        return False
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ROOT / "batch0_grade.py"), date_str],
        cwd=ROOT, timeout=600,
    )
    return r.returncode == 0


def _load_json(path: Path) -> list[dict] | None:
    if not path.exists(): return None
    try:
        with open(path) as f: return json.load(f)
    except Exception as e:
        R.fail(f"Cannot load {path.name}: {e}")
        return None


def _log_to_audit(checks: list[dict]) -> None:
    FILE_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "check", "status", "detail"]
    exists = FILE_AUDIT.exists()
    with open(FILE_AUDIT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists: w.writeheader()
        for c in checks:
            w.writerow({
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                **c,
            })


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Required files
# ─────────────────────────────────────────────────────────────────────────────

def check_required_files() -> None:
    print("\n[FILE] Required files")
    required = {
        "Game log 2025-26":  FILE_GL_2526,
        "Game log 2024-25":  (ROOT / "source-files" / "nba_gamelogs_2024_25.csv"),
        "Props Excel 25-26": FILE_PROPS,
        "Props Excel 24-25": FILE_PROPS_2425,
        "Season JSON 25-26": FILE_SEASON_2526,
        "H2H database":      FILE_H2H,
        "DVP rankings":      FILE_DVP,
        "Elite model pkl":   FILE_ELITE_MODEL,
        "V12 trust scores":  FILE_V12_TRUST,
    }
    for label, path in required.items():
        if not path.exists():
            R.fail(f"{label} MISSING: {path.name}")
        elif path.stat().st_size < 100:
            R.warn(f"{label} exists but suspiciously small ({path.stat().st_size} bytes)")
        else:
            size = f"{path.stat().st_size / 1024:.0f}KB"
            R.ok(f"{label} ({size})")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Stuck ungraded plays
# ─────────────────────────────────────────────────────────────────────────────

def check_stuck_grading() -> None:
    print("\n[GRADE] Stuck ungraded plays")
    plays_2526 = _load_json(FILE_SEASON_2526) or []
    plays_2425 = _load_json(FILE_SEASON_2425) or []

    today_str = uk_now().strftime("%Y-%m-%d")

    for season_label, plays in [("2025-26", plays_2526), ("2024-25", plays_2425)]:
        from collections import defaultdict
        ungraded_by_date: dict[str, list] = defaultdict(list)
        for p in plays:
            date   = p.get("date", "")
            result = p.get("result", "")
            if result not in ("WIN", "LOSS", "DNP", "PUSH") and date < today_str:
                ungraded_by_date[date].append(p.get("player", "?"))

        if not ungraded_by_date:
            R.ok(f"{season_label}: No stuck ungraded plays")
            continue

        total_plays  = sum(len(v) for v in ungraded_by_date.values())
        total_dates  = len(ungraded_by_date)
        dates_sorted = sorted(ungraded_by_date.keys())
        oldest_date  = dates_sorted[0]
        newest_date  = dates_sorted[-1]

        # Separate actionable (1-3 days) from historical (>3 days)
        actionable = {d: v for d, v in ungraded_by_date.items()
                      if (datetime.strptime(today_str,"%Y-%m-%d") -
                          datetime.strptime(d,"%Y-%m-%d")).days <= 3}
        historical = {d: v for d, v in ungraded_by_date.items()
                      if d not in actionable}

        if historical:
            R.warn(f"{season_label}: {sum(len(v) for v in historical.values()):,} ungraded plays "
                   f"across {len(historical)} dates ({oldest_date} → {newest_date}) "
                   f"— too old for box score API, run: python3 run.py generate")

        for date, players in sorted(actionable.items()):
            age_days = (datetime.strptime(today_str, "%Y-%m-%d") -
                        datetime.strptime(date, "%Y-%m-%d")).days
            msg = f"{season_label}: {len(players)} ungraded plays on {date} ({age_days}d old)"
            R.fail(msg)
            if not DRY_RUN:
                print(f"     Auto-fixing: grading {date}...")
                success = _run_grade(date)
                if success:
                    R.fixed(f"Graded {date} successfully")
                else:
                    R.fix_fail(f"Grade failed for {date} — run: python3 run.py grade --date {date}")
            else:
                R.warn(f"[DRY RUN] Would run: python3 run.py grade --date {date}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Duplicate plays
# ─────────────────────────────────────────────────────────────────────────────

def check_duplicates() -> None:
    print("\n[DEDUP] Duplicate plays in season JSON")
    plays = _load_json(FILE_SEASON_2526)
    if plays is None: return

    # Check for (player, date) duplicates — same player same day two entries
    from collections import Counter
    keys_player_date = Counter(
        (p.get("player", ""), p.get("date", "")) for p in plays
    )
    dupes_player_date = {k: v for k, v in keys_player_date.items() if v > 1}

    if dupes_player_date:
        total = sum(v - 1 for v in dupes_player_date.values())
        R.fail(f"Found {total} duplicate plays ({len(dupes_player_date)} player/date pairs)")
        for (player, date), count in sorted(dupes_player_date.items())[:5]:
            R.warn(f"  → {player} on {date}: {count} entries")

        if not DRY_RUN:
            # Auto-fix: keep the graded version, or highest elite_prob if both graded
            GRADED = {"WIN", "LOSS", "DNP", "PUSH"}
            seen: dict[tuple, dict] = {}
            for p in plays:
                k = (p.get("player", ""), p.get("date", ""))
                existing = seen.get(k)
                if existing is None:
                    seen[k] = p
                else:
                    # Prefer graded over ungraded, then highest prob
                    ex_graded = existing.get("result", "") in GRADED
                    p_graded  = p.get("result", "") in GRADED
                    if p_graded and not ex_graded:
                        seen[k] = p
                    elif p_graded == ex_graded:
                        if float(p.get("elite_prob", 0)) > float(existing.get("elite_prob", 0)):
                            seen[k] = p

            deduped = sorted(seen.values(), key=lambda x: (x.get("date", ""), x.get("player", "")))
            from config import clean_json
            with open(FILE_SEASON_2526, "w") as f:
                json.dump(clean_json(deduped), f, indent=2)
            R.fixed(f"Removed {len(plays) - len(deduped)} duplicate plays from season_2025_26.json")
    else:
        R.ok(f"No duplicates in season JSON ({len(plays):,} plays, all unique player/date)")

    # Check (player, date, line) — same player same day same line
    keys_full = Counter(
        (p.get("player", ""), p.get("date", ""), str(p.get("line", "")))
        for p in plays
    )
    dupes_full = {k: v for k, v in keys_full.items() if v > 1}
    if dupes_full:
        R.warn(f"Found {len(dupes_full)} player/date/line duplicates (line changes) — kept best version above")
    else:
        R.ok("No player/date/line duplicates")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — ML dataset integrity
# ─────────────────────────────────────────────────────────────────────────────

def check_ml_dataset() -> None:
    if QUICK:
        R.warn("[ML Dataset] Skipped (--quick mode)")
        return

    print("\n[DEDUP] ML dataset integrity")
    ml_path = ROOT / "data" / "propedge_ml_dataset.xlsx"

    if not ml_path.exists():
        R.warn("ML dataset not yet created — will be built on next generate or grade run")
        return

    try:
        import pandas as pd
        df = pd.read_excel(ml_path, sheet_name="Plays")
    except Exception as e:
        R.fail(f"ML dataset unreadable: {e}")
        return

    R.ok(f"ML dataset loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Check duplicates on (Player, Date, Direction)
    key_cols = [c for c in ["Player", "Date", "Direction"] if c in df.columns]
    if len(key_cols) >= 2:
        dupes = df[df.duplicated(subset=key_cols, keep=False)]
        if not dupes.empty:
            R.fail(f"ML dataset: {len(dupes)} duplicate rows on {key_cols}")
            if not DRY_RUN:
                # Rebuild from season JSONs
                try:
                    from ml_dataset import write_ml_dataset
                    plays_2425 = _load_json(FILE_SEASON_2425) or []
                    plays_2526 = _load_json(FILE_SEASON_2526) or []
                    write_ml_dataset(plays_2425 + plays_2526)
                    R.fixed("ML dataset rebuilt from season JSONs — duplicates removed")
                except Exception as e:
                    R.fix_fail(f"ML dataset rebuild failed: {e}")
        else:
            R.ok(f"No duplicates in ML dataset")

    # Check graded row count vs BOTH season JSONs combined
    plays_2526 = _load_json(FILE_SEASON_2526) or []
    plays_2425 = _load_json(FILE_SEASON_2425) or []
    all_json_graded = [p for p in (plays_2526 + plays_2425)
                       if p.get("result") in ("WIN", "LOSS")]
    if "Result" in df.columns:
        graded_ml = len(df[df["Result"].isin(["WIN", "LOSS"])])
        diff = abs(graded_ml - len(all_json_graded))
        tolerance = max(50, len(all_json_graded) * 0.02)  # 2% tolerance
        if diff > tolerance:
            R.warn(f"ML dataset has {graded_ml:,} graded rows, "
                   f"season JSONs (both) have {len(all_json_graded):,} — "
                   f"gap of {diff:,} (run: python3 run.py generate to rebuild ML dataset)")
        else:
            R.ok(f"ML graded count matches season JSONs ({graded_ml:,} vs {len(all_json_graded):,})")

    # Check for NULL player/date
    if "Player" in df.columns:
        nulls = df["Player"].isna().sum()
        if nulls > 0:
            R.warn(f"ML dataset: {nulls} rows with null Player")

    # Check hit rate sanity
    if "Result" in df.columns:
        graded = df[df["Result"].isin(["WIN", "LOSS"])]
        if len(graded) > 50:
            hr = (graded["Result"] == "WIN").mean()
            if hr < 0.35:
                R.warn(f"ML dataset hit rate is very low: {hr:.1%} — check grading logic")
            elif hr > 0.75:
                R.warn(f"ML dataset hit rate is suspiciously high: {hr:.1%} — check grading logic")
            else:
                R.ok(f"ML dataset hit rate: {hr:.1%} ({len(graded):,} graded plays)")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 5 — Trust score freshness
# ─────────────────────────────────────────────────────────────────────────────

def check_trust_scores() -> None:
    print("\n[TRUST] Player trust scores")
    if not FILE_V12_TRUST.exists():
        R.fail("V12 trust scores missing")
        if not DRY_RUN:
            try:
                from model_trainer import update_trust_scores
                update_trust_scores()
                R.fixed("Trust scores rebuilt")
            except Exception as e:
                R.fix_fail(f"Trust rebuild failed: {e}")
        return

    try:
        trust = json.loads(FILE_V12_TRUST.read_text())
    except Exception as e:
        R.fail(f"Trust scores unreadable: {e}")
        return

    # Check freshness via file mtime
    mtime = datetime.fromtimestamp(FILE_V12_TRUST.stat().st_mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - mtime).total_seconds() / 3600

    if age_hours > 72:
        R.warn(f"Trust scores are {age_hours:.0f}h old (>72h) — updating now")
        if not DRY_RUN:
            try:
                from model_trainer import update_trust_scores
                update_trust_scores()
                R.fixed("Trust scores refreshed")
            except Exception as e:
                R.fix_fail(f"Trust refresh failed: {e}")
    else:
        R.ok(f"Trust scores are {age_hours:.0f}h old")

    # Stats
    values = list(trust.values())
    below  = sum(1 for v in values if v < 0.42)
    avg    = sum(values) / len(values) if values else 0

    # Cross-check: are major current players in trust table?
    plays = _load_json(FILE_SEASON_2526) or []
    recent = {p.get("player") for p in plays
              if p.get("date", "") >= (uk_now() - timedelta(days=30)).strftime("%Y-%m-%d")}
    missing = [p for p in recent if p and p not in trust]
    if missing:
        R.warn(f"{len(missing)} active players missing from trust table: {missing[:5]}")
    else:
        R.ok(f"Trust table covers all recent players ({len(trust)} total, avg={avg:.3f}, {below} below threshold)")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 6 — Elite model
# ─────────────────────────────────────────────────────────────────────────────

def check_elite_model() -> None:
    print("\n[MODEL] Elite V2 meta-model")
    if not FILE_ELITE_MODEL.exists():
        R.fail("propedge_elite_v2.pkl MISSING — run: python3 run.py generate")
        return

    try:
        import pickle
        with open(FILE_ELITE_MODEL, "rb") as f:
            pkg = pickle.load(f)
    except Exception as e:
        R.fail(f"Elite model unloadable: {e} — run: python3 run.py retrain")
        return

    trained_at = pkg.get("trained_at", "")
    n_plays    = pkg.get("n_plays", 0)
    features   = pkg.get("features", [])

    R.ok(f"Elite model loads — {n_plays:,} training plays, {len(features)} features")

    # Check age
    if trained_at:
        try:
            dt = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - dt).days
            if age_days > 35:
                R.warn(f"Elite model is {age_days} days old (>35 days) — consider retraining")
            else:
                R.ok(f"Model trained {age_days} days ago ({trained_at[:10]})")
        except Exception:
            R.warn(f"Cannot parse trained_at: {trained_at}")

    # Check feature count
    if len(features) < 70:
        R.warn(f"Only {len(features)} features — expected ~82. Model may be outdated.")

    # Smoke test: can it produce a prediction?
    try:
        import numpy as np
        import pandas as pd
        X = pd.DataFrame([{k: 0.5 for k in features}])[features].fillna(0).values
        prob = pkg["model"].predict_proba(pkg["scaler"].transform(X))[0, 1]
        R.ok(f"Model smoke test passed (test pred: {prob:.3f})")
    except Exception as e:
        R.fail(f"Model smoke test failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 7 — Game log integrity
# ─────────────────────────────────────────────────────────────────────────────

def check_game_log() -> None:
    print("\n[LOG] Game log integrity")
    if not FILE_GL_2526.exists():
        R.fail("2025-26 game log missing")
        return

    try:
        import pandas as pd
        df = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"], low_memory=False)
    except Exception as e:
        R.fail(f"Game log unreadable: {e}")
        return

    R.ok(f"Game log: {len(df):,} rows, {df['PLAYER_NAME'].nunique()} players")

    # Check for duplicates
    dupes = df[df.duplicated(subset=["PLAYER_NAME", "GAME_DATE"], keep=False)]
    if not dupes.empty:
        R.warn(f"Game log: {len(dupes)} duplicate PLAYER_NAME+GAME_DATE rows")
        if not DRY_RUN:
            df = df.sort_values(["PLAYER_NAME", "GAME_DATE"]).drop_duplicates(
                subset=["PLAYER_NAME", "GAME_DATE"], keep="last"
            )
            df.to_csv(FILE_GL_2526, index=False)
            R.fixed(f"Game log deduplicated ({len(dupes)} rows removed)")
    else:
        R.ok("Game log: no duplicate player/date rows")

    # Check recency — latest game should be within last 5 days (NBA season)
    latest = df["GAME_DATE"].max()
    age_days = (datetime.now() - latest).days
    if age_days > 5:
        R.warn(f"Latest game log entry is {age_days} days old ({latest.date()}) — run B0 to update")
    else:
        R.ok(f"Game log is current — latest game: {latest.date()}")

    # Check for NaN PTS on played rows (MIN_NUM > 0)
    played = df[df["MIN_NUM"].fillna(0) > 0]
    null_pts = played["PTS"].isna().sum()
    if null_pts > 10:
        R.warn(f"Game log: {null_pts} played rows with null PTS")
    else:
        R.ok(f"PTS populated for all played rows ({null_pts} nulls)")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 8 — today.json vs season JSON sync
# ─────────────────────────────────────────────────────────────────────────────

def check_json_sync() -> None:
    print("\n[SYNC] today.json vs season JSON")
    today_plays  = _load_json(FILE_TODAY)    or []
    season_plays = _load_json(FILE_SEASON_2526) or []

    if not today_plays:
        R.warn("today.json is empty or missing — normal if no predictions today")
        return

    today_date = today_plays[0].get("date", "") if today_plays else ""
    R.ok(f"today.json: {len(today_plays)} plays for {today_date}")

    # Check that graded plays in today.json are also in season JSON
    season_keys = {
        (p.get("player"), p.get("date"), str(p.get("line")))
        for p in season_plays
    }
    today_graded = [p for p in today_plays
                    if p.get("result") in ("WIN", "LOSS", "DNP", "PUSH")]

    missing_in_season = [
        p for p in today_graded
        if (p.get("player"), p.get("date"), str(p.get("line"))) not in season_keys
    ]

    if missing_in_season:
        R.fail(f"{len(missing_in_season)} graded plays in today.json missing from season JSON")
        for p in missing_in_season[:3]:
            R.warn(f"  → {p.get('player')} {p.get('date')} line={p.get('line')} result={p.get('result')}")
        if not DRY_RUN:
            try:
                from batch0_grade import update_season_json
                date_str = today_date
                update_season_json(today_graded, date_str)
                R.fixed(f"Season JSON updated with {len(today_graded)} graded plays from today.json")
            except Exception as e:
                R.fix_fail(f"Season JSON sync failed: {e}")
    else:
        R.ok(f"All {len(today_graded)} graded plays synced to season JSON")

    # Check for ungraded plays older than today
    stuck = [p for p in season_plays
             if p.get("result") not in ("WIN", "LOSS", "DNP", "PUSH")
             and p.get("date", "") < today_date]
    if stuck:
        dates = sorted({p.get("date") for p in stuck})
        date_range = f"{dates[0]} → {dates[-1]}" if len(dates) > 2 else str(dates)
        R.warn(f"{len(stuck)} ungraded plays before {today_date} "
               f"across {len(dates)} dates ({date_range}) "
               f"— run: python3 run.py generate to rebuild with correct DNP marking")
    else:
        R.ok("No historical ungraded plays in season JSON")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 9 — Missing actual scores (graded but actualPts is None)
# ─────────────────────────────────────────────────────────────────────────────

def check_missing_box_scores() -> None:
    print("\n[MISS] Missing box scores in graded plays")
    plays = _load_json(FILE_SEASON_2526)
    if plays is None: return

    graded = [p for p in plays if p.get("result") in ("WIN", "LOSS")]
    missing_pts = [p for p in graded if p.get("actualPts") is None]

    if missing_pts:
        R.fail(f"{len(missing_pts)} graded plays have result but no actualPts")
        # Group by date to show which dates affected
        from collections import Counter
        by_date = Counter(p.get("date", "?") for p in missing_pts)
        for date, count in sorted(by_date.items())[:5]:
            R.warn(f"  → {date}: {count} plays missing actualPts")
    else:
        R.ok(f"All {len(graded):,} graded plays have actualPts populated")

    # Check for implausible scores (0 pts graded as LOSS for OVER)
    zero_pts = [p for p in graded
                if p.get("actualPts") == 0 and
                "OVER" in str(p.get("direction", "")) and
                p.get("result") == "LOSS"]
    if zero_pts:
        R.warn(f"{len(zero_pts)} OVER plays graded LOSS with 0 actual pts — possible DNP mis-grade")
        for p in zero_pts[:3]:
            R.warn(f"  → {p.get('player')} {p.get('date')} line={p.get('line')}")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 10 — predGap sign integrity
# ─────────────────────────────────────────────────────────────────────────────

def check_pred_gap_signs() -> None:
    print("\n[SIGN] predGap sign integrity")
    plays = _load_json(FILE_SEASON_2526)
    if plays is None: return

    graded = [p for p in plays
              if p.get("result") in ("WIN", "LOSS")
              and p.get("actualPts") is not None
              and p.get("predPts") is not None]

    if len(graded) < 20:
        R.warn("Not enough graded plays to check predGap signs")
        return

    # predGap = predPts - line. If OVER and predGap > 0 → model agreed with direction
    import random
    sample = random.sample(graded, min(200, len(graded)))

    all_positive = all(p.get("predGap", 0) >= 0 for p in sample)
    all_negative = all(p.get("predGap", 0) <= 0 for p in sample)

    if all_positive:
        R.fail("predGap is always non-negative — stored as abs() not signed. Run generate to rebuild.")
    elif all_negative:
        R.fail("predGap is always non-positive — sign issue. Run generate to rebuild.")
    else:
        neg_count = sum(1 for p in sample if p.get("predGap", 0) < 0)
        pos_count = len(sample) - neg_count
        R.ok(f"predGap signs look correct — {pos_count} positive (over), {neg_count} negative (under) in sample")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def check_stale_data() -> None:
    """Alert if game log hasn't been updated in >36h — B0 may be broken."""
    print("\n[STALE] Data freshness")
    if not FILE_GL_2526.exists():
        R.warn("Game log 2025-26 missing"); return
    import stat as _stat
    age_hours = (datetime.now() - datetime.fromtimestamp(FILE_GL_2526.stat().st_mtime)).total_seconds() / 3600
    if age_hours > 36:
        R.fail(f"Game log last updated {age_hours:.0f}h ago — B0 may not be running (check launchd)")
    elif age_hours > 24:
        R.warn(f"Game log last updated {age_hours:.1f}h ago — expected daily update")
    else:
        R.ok(f"Game log is fresh ({age_hours:.1f}h old)")


def check_model_performance() -> None:
    """Alert if recent hit rate drops >5% vs season baseline — possible model drift."""
    print("\n[PERF] Model performance check")
    plays = _load_json(FILE_SEASON_2526)
    if not plays:
        R.warn("Cannot check — season_2025_26.json missing"); return
    graded = [p for p in plays if p.get("result") in ("WIN","LOSS")]
    if len(graded) < 100:
        R.warn(f"Only {len(graded)} graded plays — not enough for reliable HR check"); return

    # Season baseline
    wins_all = sum(1 for p in graded if p.get("result") == "WIN")
    hr_season = wins_all / len(graded)

    # Last 30 days
    today = datetime.now()
    recent = [p for p in graded if p.get("date","") >= (today - __import__("datetime").timedelta(days=30)).strftime("%Y-%m-%d")]
    if len(recent) >= 30:
        wins_recent = sum(1 for p in recent if p.get("result") == "WIN")
        hr_recent = wins_recent / len(recent)
        drift = hr_season - hr_recent
        if drift > 0.05:
            R.fail(f"Recent HR {hr_recent:.1%} is {drift:.1%} below season avg {hr_season:.1%} — possible model drift")
        elif drift < -0.05:
            R.ok(f"Recent HR {hr_recent:.1%} trending ABOVE season avg {hr_season:.1%} — model performing well")
        else:
            R.ok(f"Recent HR {hr_recent:.1%} vs season avg {hr_season:.1%} — stable")
    else:
        R.ok(f"Season HR: {hr_season:.1%} ({len(graded):,} graded plays)")

    # Elite tier specific check
    apex = [p for p in graded if p.get("elite_tier") == "APEX"]
    if len(apex) >= 20:
        apex_hr = sum(1 for p in apex if p.get("result") == "WIN") / len(apex)
        if apex_hr < 0.80:
            R.warn(f"APEX tier HR {apex_hr:.1%} below 80% target ({len(apex)} plays)")
        else:
            R.ok(f"APEX tier HR: {apex_hr:.1%} ({len(apex)} plays)")


def check_monthly_files() -> None:
    """Verify monthly split files exist, are readable, and counts match full JSONs."""
    print("\n[MONTHLY] Monthly file integrity")

    for season_key, season_file, label in [
        ("2025_26", FILE_SEASON_2526, "2025-26"),
        ("2024_25", FILE_SEASON_2425, "2024-25"),
    ]:
        month_files = list_monthly_files(season_key)
        if not month_files:
            R.warn(f"{label}: No monthly files found — run: python3 run.py generate")
            continue

        index = get_monthly_index(season_key)
        if not index:
            R.warn(f"{label}: Monthly index missing — run: python3 run.py generate")
            continue

        # Load full JSON for cross-check
        full_plays = _load_json(season_file) or []
        if not full_plays:
            R.warn(f"{label}: Full season JSON not found — cannot verify monthly counts")
            continue

        ok, msg = verify_monthly_integrity(season_key, full_plays)
        if ok:
            R.ok(f"{label}: {msg}")
            # Per-month breakdown (skip in quick mode)
            if not QUICK:
                from pathlib import Path as _P
                root = _P(__file__).resolve().parent
                for month, count in sorted(index.get("counts", {}).items()):
                    mpath = root / "data" / "monthly" / season_key / f"{month}.json"
                    month_plays = _load_json(mpath) or []
                    graded = sum(1 for p in month_plays if p.get("result") in ("WIN","LOSS"))
                    hr = f"{graded/count*100:.0f}%" if count else "—"
                    print(f"    {month}: {count:>5} plays | {graded:>5} graded ({hr})")
        else:
            R.fail(msg)
            if not DRY_RUN:
                R.warn("Run: python3 run.py generate  to rebuild monthly files")


def main() -> None:
    mode = "DRY RUN" if DRY_RUN else ("QUICK" if QUICK else "FULL")
    print(f"\n  PropEdge {VERSION} — Health Check [{mode}]")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("  " + "─" * 56)

    check_required_files()
    check_stuck_grading()
    check_duplicates()
    check_ml_dataset()
    check_trust_scores()
    check_elite_model()
    check_game_log()
    check_json_sync()
    check_missing_box_scores()
    check_pred_gap_signs()
    check_stale_data()
    check_model_performance()
    check_monthly_files()

    # Summary
    print(f"\n  {'─'*56}")
    print(f"  Result: {R.score}")
    if R.fixes_applied:
        print(f"  Auto-fixed ({len(R.fixes_applied)}):")
        for f in R.fixes_applied: print(f"    🔧 {f}")
    if R.fixes_failed:
        print(f"  Failed fixes ({len(R.fixes_failed)}):")
        for f in R.fixes_failed: print(f"    ✗ {f}")
    if R.failures:
        print(f"\n  Outstanding failures ({len(R.failures)}):")
        for f in R.failures: print(f"    ✗ {f}")
    if not R.failures and not R.fixes_failed:
        print(f"  ✓ System healthy")
    print()

    # Log to audit
    audit_rows = (
        [{"check": "PASS",     "status": "pass",    "detail": m} for m in R.passed] +
        [{"check": "WARNING",  "status": "warning", "detail": m} for m in R.warnings] +
        [{"check": "FAILURE",  "status": "fail",    "detail": m} for m in R.failures] +
        [{"check": "FIXED",    "status": "fixed",   "detail": m} for m in R.fixes_applied] +
        [{"check": "FIX_FAIL", "status": "error",   "detail": m} for m in R.fixes_failed]
    )
    try:
        _log_to_audit(audit_rows)
    except Exception:
        pass

    sys.exit(0 if not R.failures else 1)


if __name__ == "__main__":
    main()
