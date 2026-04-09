"""
PropEdge V16.0 — monthly_split.py
===================================
Splits season JSON files into per-month files for GitHub Pages hosting.

WHY: Each full season JSON is ~84MB which causes GitHub Desktop to disconnect
     and the REST API to time out. Monthly files are 4-16MB each — well within
     GitHub's 100MB limit and fast to push/fetch.

FILE LAYOUT:
  data/monthly/2025_26/index.json       ← list of months + counts
  data/monthly/2025_26/2025-10.json     ← Oct 2025 plays
  data/monthly/2025_26/2025-11.json     ← Nov 2025 plays
  ...
  data/monthly/2024_25/index.json
  data/monthly/2024_25/2024-10.json
  ...

ATOMIC WRITES: Every file is written to a .tmp file then renamed.
  This means a crashed process never leaves a half-written file.

BACKUP: Before overwriting, the previous file is saved as .bak.
  One backup per file — enough to rollback a bad generate.

IDEMPOTENT: Writing the same data twice produces identical output.

PUBLIC API:
  write_monthly_split(plays, season_key)      called by generate_season_json
  update_month(plays, season_key, month_str)  called by batch0_grade
  load_monthly_split(season_key)              called by health_check + tests
  get_monthly_index(season_key)               returns {month: count} dict
  list_monthly_files(season_key)              returns list of Path objects
"""

from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "monthly"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _monthly_dir(season_key: str) -> Path:
    """Return the directory for a season's monthly files.
    season_key: '2025_26' or '2024_25'
    """
    return DATA_DIR / season_key


def _month_path(season_key: str, month_str: str) -> Path:
    """Return path to a monthly file.  month_str: '2025-10'"""
    return _monthly_dir(season_key) / f"{month_str}.json"


def _index_path(season_key: str) -> Path:
    return _monthly_dir(season_key) / "index.json"


def _atomic_write(path: Path, data: Any) -> None:
    """Write JSON atomically: write to .tmp, backup existing, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
        # Backup existing file before overwriting
        if path.exists():
            bak = path.with_suffix(".bak")
            shutil.copy2(path, bak)
        # Atomic rename
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


def _season_key_from_file(season_file: str | Path) -> str:
    """Derive '2025_26' from 'season_2025_26.json'."""
    stem = Path(season_file).stem  # e.g. 'season_2025_26'
    return stem.replace("season_", "")


# ─────────────────────────────────────────────────────────────────────────────
# CORE SPLIT LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _group_by_month(plays: list[dict]) -> dict[str, list[dict]]:
    """Group plays by YYYY-MM month string."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in plays:
        month = str(p.get("date", ""))[:7]  # '2025-10'
        if month and len(month) == 7:
            groups[month].append(p)
    return dict(groups)


def write_monthly_split(plays: list[dict], season_key: str) -> dict[str, int]:
    """
    Write all monthly files for a season.
    Called by generate_season_json after the full season JSON is written.

    Returns {month: play_count} for verification.
    """
    by_month = _group_by_month(plays)
    counts: dict[str, int] = {}

    for month in sorted(by_month):
        month_plays = sorted(by_month[month],
                             key=lambda p: (p.get("date", ""), p.get("player", "")))
        _atomic_write(_month_path(season_key, month), month_plays)
        counts[month] = len(month_plays)

    # Write the index file
    index = {
        "season":       season_key,
        "months":       sorted(counts.keys()),
        "counts":       counts,
        "total_plays":  sum(counts.values()),
        "updated_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _atomic_write(_index_path(season_key), index)

    # Verify: sum of monthly counts must equal input plays
    total_monthly = sum(counts.values())
    if total_monthly != len(plays):
        raise ValueError(
            f"Monthly split verification failed for {season_key}: "
            f"split total {total_monthly} ≠ input {len(plays)}"
        )

    return counts


def update_month(plays_for_month: list[dict], season_key: str, month_str: str) -> None:
    """
    Update a single month's file after daily grading.
    Called by batch0_grade when graded_today belongs to month_str.

    The file is read, new plays replace existing plays for the same
    player+date keys, then written back atomically.
    """
    path = _month_path(season_key, month_str)

    # Load existing month file if present
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = []

    # Build lookup of existing plays
    existing_by_key = {
        (p.get("player", ""), p.get("date", "")): p
        for p in existing
    }

    # Merge: new plays override existing for same player+date
    for p in plays_for_month:
        k = (p.get("player", ""), p.get("date", ""))
        existing_by_key[k] = p

    merged = sorted(existing_by_key.values(),
                    key=lambda p: (p.get("date", ""), p.get("player", "")))
    _atomic_write(path, merged)

    # Update index counts
    _refresh_index(season_key)


def _refresh_index(season_key: str) -> None:
    """Recompute and rewrite the index after a single-month update."""
    monthly_dir = _monthly_dir(season_key)
    if not monthly_dir.exists():
        return

    counts: dict[str, int] = {}
    for f in sorted(monthly_dir.glob("????-??.json")):
        month = f.stem  # '2025-10'
        try:
            plays = json.loads(f.read_text())
            counts[month] = len(plays)
        except Exception:
            pass

    index = {
        "season":       season_key,
        "months":       sorted(counts.keys()),
        "counts":       counts,
        "total_plays":  sum(counts.values()),
        "updated_at":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    _atomic_write(_index_path(season_key), index)


# ─────────────────────────────────────────────────────────────────────────────
# READ / QUERY
# ─────────────────────────────────────────────────────────────────────────────

def load_monthly_split(season_key: str) -> list[dict]:
    """Load all monthly files for a season and return as a flat list."""
    monthly_dir = _monthly_dir(season_key)
    if not monthly_dir.exists():
        return []
    plays: list[dict] = []
    for f in sorted(monthly_dir.glob("????-??.json")):
        try:
            plays.extend(json.loads(f.read_text()))
        except Exception:
            pass
    return plays


def get_monthly_index(season_key: str) -> dict:
    """Return the index dict: {months, counts, total_plays, updated_at}."""
    path = _index_path(season_key)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def list_monthly_files(season_key: str) -> list[Path]:
    """Return sorted list of monthly file Paths for a season."""
    monthly_dir = _monthly_dir(season_key)
    if not monthly_dir.exists():
        return []
    return sorted(monthly_dir.glob("????-??.json"))


def verify_monthly_integrity(season_key: str, full_plays: list[dict]) -> tuple[bool, str]:
    """
    Cross-check: sum of all monthly file counts == len(full_plays).
    Returns (ok: bool, message: str).
    Called by health_check.
    """
    index = get_monthly_index(season_key)
    if not index:
        return False, f"No monthly index found for {season_key}"

    monthly_total = index.get("total_plays", 0)
    expected = len(full_plays)

    if monthly_total != expected:
        return False, (
            f"{season_key}: monthly total {monthly_total:,} ≠ "
            f"full JSON {expected:,} (diff={abs(monthly_total-expected):,})"
        )

    # Also verify each month file is readable
    for month_file in list_monthly_files(season_key):
        try:
            data = json.loads(month_file.read_text())
            expected_count = index["counts"].get(month_file.stem, -1)
            if len(data) != expected_count:
                return False, (
                    f"{season_key}/{month_file.stem}: file has {len(data)} plays "
                    f"but index says {expected_count}"
                )
        except Exception as e:
            return False, f"{season_key}/{month_file.stem}: unreadable: {e}"

    return True, (
        f"{season_key}: {len(index['months'])} months, "
        f"{monthly_total:,} plays — all verified"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUSH FILE LIST (for git_push.py)
# ─────────────────────────────────────────────────────────────────────────────

def get_push_paths(season_key: str, only_current_month: bool = False) -> list[str]:
    """
    Return relative paths for git_push.py to push.

    only_current_month=True: just the current month + index (for daily grade push)
    only_current_month=False: all months + index (for generate push)
    """
    from datetime import date
    paths: list[str] = []

    if only_current_month:
        current = date.today().strftime("%Y-%m")
        month_path = f"data/monthly/{season_key}/{current}.json"
        index_path = f"data/monthly/{season_key}/index.json"
        paths = [month_path, index_path]
    else:
        monthly_dir = _monthly_dir(season_key)
        if monthly_dir.exists():
            for f in sorted(monthly_dir.glob("*.json")):
                rel = f.relative_to(ROOT).as_posix()
                paths.append(rel)

    return paths


if __name__ == "__main__":
    print("monthly_split.py — import and use write_monthly_split(plays, season_key)")
    print("  season_key: '2025_26' or '2024_25'")
    print("  Data written to: data/monthly/{season_key}/{YYYY-MM}.json")
