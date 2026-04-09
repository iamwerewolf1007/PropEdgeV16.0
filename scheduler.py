"""
PropEdge V16.0 — scheduler.py
Six-batch macOS launchd scheduler with weekend tip-relative offsets.

Weekday schedule (UK local):
  B0  07:00 — Grade yesterday + monthly Elite retrain
  B1  08:30 — Morning scan  (overnight lines posted)
  B2  11:00 — Mid-morning refresh (line movement; before injury report)
  B3  16:00 — Afternoon sweep (most US props live by now)
  B4  18:30 — Pre-game final (1.5 hr before 7 pm ET tip)
  B5  21:00 — Late West Coast top-up

Daily recalculator runs at 05:55 UK.  On Sat/Sun it fetches the first
NBA tip-off from The Odds API and rewrites B1-B5 plists to tip-relative
times within floor/ceiling bounds.  B0 is always fixed.

Commands:
  python3 scheduler.py install          — write + load all launchd agents
  python3 scheduler.py uninstall        — unload + delete all agents
  python3 scheduler.py reinstall        — uninstall then install
  python3 scheduler.py status           — show agent states
  python3 scheduler.py next             — print next run times
  python3 scheduler.py daily-recalc     — daily schedule recalculator (called by launchd)
  python3 scheduler.py weekend-check [YYYY-MM-DD]
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT      = Path(__file__).parent.resolve()
PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
PYTHON    = sys.executable
LOG_DIR   = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

_UK = ZoneInfo("Europe/London")
_ET = ZoneInfo("America/New_York")

# ─────────────────────────────────────────────────────────────────────────────
# AGENT LABELS
# ─────────────────────────────────────────────────────────────────────────────
AGENTS = {
    "b0": "com.propedge.v16.batch0",
    "b1": "com.propedge.v16.batch1",
    "b2": "com.propedge.v16.batch2",
    "b3": "com.propedge.v16.batch3",
    "b4": "com.propedge.v16.batch4",
    "b5": "com.propedge.v16.batch5",
    "db": "com.propedge.v16.daily",
}

# ─────────────────────────────────────────────────────────────────────────────
# WEEKDAY SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
WEEKDAY_TIMES: dict[str, tuple[int, int]] = {
    "b0": (7,  0),    # 07:00 — grade + retrain
    "b1": (8,  30),   # 08:30 — morning scan
    "b2": (11, 0),    # 11:00 — mid-morning refresh
    "b3": (16, 0),    # 16:00 — afternoon sweep
    "b4": (18, 30),   # 18:30 — pre-game final
    "b5": (21, 0),    # 21:00 — late West Coast top-up
}

# ─────────────────────────────────────────────────────────────────────────────
# WEEKEND OFFSETS  (minutes relative to first tip-off in UK time)
# ─────────────────────────────────────────────────────────────────────────────
WEEKEND_OFFSETS_MINS: dict[str, int] = {
    "b1": -120,   # 2 hr before first tip  (earliest morning scan)
    "b2": -90,    # 90 min before tip       (pre-injury refresh)
    "b3": -60,    # 60 min before tip       (final pre-tip sweep)
    "b4": -15,    # 15 min before tip       (very last look)
    "b5": +150,   # 2.5 hr after first tip  (late West Coast games)
}

WEEKEND_FLOOR: dict[str, tuple[int, int]] = {
    "b1": (11, 0),
    "b2": (13, 0),
    "b3": (16, 0),
    "b4": (17, 30),
    "b5": (20, 30),
}

WEEKEND_CEIL: dict[str, tuple[int, int]] = {
    "b1": (15, 0),
    "b2": (17, 0),
    "b3": (19, 0),
    "b4": (20, 30),
    "b5": (23, 59),
}

# ─────────────────────────────────────────────────────────────────────────────
# TIP-OFF DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def _get_odds_key() -> str:
    try:
        sys.path.insert(0, str(ROOT))
        from config import ODDS_API_KEY
        return ODDS_API_KEY
    except Exception:
        return ""


def fetch_first_tip_et(date_str: str) -> datetime | None:
    """Query Odds API for first NBA tip on date_str; return ET datetime or None."""
    import requests
    key = _get_odds_key()
    if not key:
        return None
    try:
        sys.path.insert(0, str(ROOT))
        from config import et_window
        fr_utc, to_utc = et_window(date_str)
        r = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_nba/events",
            params={
                "apiKey": key,
                "commenceTimeFrom": fr_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "commenceTimeTo":   to_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            timeout=12,
        )
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        print(f"  [scheduler] Tip-off API error: {e}")
        return None

    earliest = None
    from datetime import timezone as _tz
    for ev in events:
        ts = ev.get("commence_time", "")
        if not ts:
            continue
        try:
            dt_utc = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_tz.utc)
            dt_et  = dt_utc.astimezone(_ET)
            if earliest is None or dt_et < earliest:
                earliest = dt_et
        except Exception:
            continue
    return earliest


def compute_weekend_times(date_str: str) -> dict[str, tuple[int, int]]:
    """
    Return B1-B5 UK (hour, minute) tuples for a weekend date.
    Falls back to weekday times if tip detection fails.
    """
    first_tip = fetch_first_tip_et(date_str)
    if first_tip is None:
        print("  [scheduler] Tip detection failed — using weekday fallback.")
        return {k: WEEKDAY_TIMES[k] for k in ("b1", "b2", "b3", "b4", "b5")}

    first_tip_uk = first_tip.astimezone(_UK)
    print(f"  [scheduler] First tip: {first_tip.strftime('%H:%M ET')} "
          f"= {first_tip_uk.strftime('%H:%M UK')}")

    result: dict[str, tuple[int, int]] = {}
    for batch, offset in WEEKEND_OFFSETS_MINS.items():
        target = first_tip_uk + timedelta(minutes=offset)
        h, m   = target.hour, target.minute

        fl_h, fl_m = WEEKEND_FLOOR[batch]
        if (h, m) < (fl_h, fl_m):
            h, m = fl_h, fl_m

        ce_h, ce_m = WEEKEND_CEIL[batch]
        if (h, m) > (ce_h, ce_m):
            h, m = ce_h, ce_m

        result[batch] = (h, m)
        print(f"    {batch.upper()}: {h:02d}:{m:02d} UK (offset {offset:+d} min from tip)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# PLIST GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def _plist(label: str, script: str, hour: int, minute: int,
           log_name: str, args: list[str] | None = None) -> str:
    prog_args = f"        <string>{PYTHON}</string>\n        <string>{ROOT / script}</string>"
    if args:
        for a in args:
            prog_args += f"\n        <string>{a}</string>"

    path_val = f"/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{Path(PYTHON).parent}"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
{prog_args}
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>   <integer>{hour}</integer>
        <key>Minute</key> <integer>{minute}</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{LOG_DIR / log_name}.log</string>
    <key>StandardErrorPath</key>
    <string>{LOG_DIR / log_name}_err.log</string>

    <key>RunAtLoad</key>  <false/>
    <key>WorkingDirectory</key>
    <string>{ROOT}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>  <string>{path_val}</string>
        <key>HOME</key>  <string>{Path.home()}</string>
    </dict>
</dict>
</plist>
"""


def _daily_plist(label: str) -> str:
    """05:55 UK daily schedule recalculator."""
    path_val = f"/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:{Path(PYTHON).parent}"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{PYTHON}</string>
        <string>{ROOT / "scheduler.py"}</string>
        <string>daily-recalc</string>
    </array>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>   <integer>5</integer>
        <key>Minute</key> <integer>55</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>{LOG_DIR / "scheduler"}.log</string>
    <key>StandardErrorPath</key>
    <string>{LOG_DIR / "scheduler"}_err.log</string>

    <key>RunAtLoad</key>  <false/>
    <key>WorkingDirectory</key>
    <string>{ROOT}</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>  <string>{path_val}</string>
        <key>HOME</key>  <string>{Path.home()}</string>
    </dict>
</dict>
</plist>
"""


# ─────────────────────────────────────────────────────────────────────────────
# LAUNCHCTL HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _lctl(cmd: list[str]) -> bool:
    return subprocess.run(cmd, capture_output=True).returncode == 0


def _load(path: Path) -> None:
    _lctl(["launchctl", "unload", str(path)])
    if _lctl(["launchctl", "load", str(path)]):
        print(f"  ✓ Loaded:   {path.name}")
    else:
        print(f"  ✗ Failed:   {path.name}")


def _unload(path: Path) -> None:
    _lctl(["launchctl", "unload", str(path)])
    if path.exists():
        path.unlink()
        print(f"  ✓ Removed:  {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# INSTALL / UNINSTALL
# ─────────────────────────────────────────────────────────────────────────────
def install(times: dict[str, tuple[int, int]] | None = None) -> None:
    """Write all plists with current schedule and load via launchctl."""
    if times is None:
        times = WEEKDAY_TIMES
    PLIST_DIR.mkdir(parents=True, exist_ok=True)

    # B0 — always fixed
    p = PLIST_DIR / f"{AGENTS['b0']}.plist"
    p.write_text(_plist(AGENTS["b0"], "batch0_grade.py", *WEEKDAY_TIMES["b0"], "batch0"))
    _load(p)

    # B1-B5
    for bk in ("b1", "b2", "b3", "b4", "b5"):
        arg = bk[1]   # "1" .. "5"
        p = PLIST_DIR / f"{AGENTS[bk]}.plist"
        p.write_text(_plist(AGENTS[bk], "batch_predict.py", *times[bk], bk, args=[arg]))
        _load(p)

    # Daily recalculator
    p = PLIST_DIR / f"{AGENTS['db']}.plist"
    p.write_text(_daily_plist(AGENTS["db"]))
    _load(p)

    print("\n  Schedule installed:")
    print(f"    B0 Grade:       {WEEKDAY_TIMES['b0'][0]:02d}:{WEEKDAY_TIMES['b0'][1]:02d} UK (fixed)")
    for bk in ("b1", "b2", "b3", "b4", "b5"):
        h, m = times[bk]
        names = {"b1":"Morning scan","b2":"Mid-morning","b3":"Afternoon",
                 "b4":"Pre-game","b5":"Late/WestCoast"}
        print(f"    {bk.upper()} {names[bk]:12s}  {h:02d}:{m:02d} UK")
    print("    Daily recalc:   05:55 UK")


def uninstall() -> None:
    for label in AGENTS.values():
        path = PLIST_DIR / f"{label}.plist"
        if path.exists():
            _unload(path)
    print("  All V16 agents removed.")


def _reinstall_predict_plists(times: dict[str, tuple[int, int]]) -> None:
    for bk in ("b1", "b2", "b3", "b4", "b5"):
        path = PLIST_DIR / f"{AGENTS[bk]}.plist"
        path.write_text(_plist(AGENTS[bk], "batch_predict.py", *times[bk], bk, args=[bk[1]]))
        _lctl(["launchctl", "unload", str(path)])
        _load(path)


# ─────────────────────────────────────────────────────────────────────────────
# STATUS & NEXT
# ─────────────────────────────────────────────────────────────────────────────
def status() -> None:
    print(f"\n  {'Agent':<42} {'Status':>12}")
    print(f"  {'─'*56}")
    for key, label in AGENTS.items():
        path = PLIST_DIR / f"{label}.plist"
        r = subprocess.run(["launchctl", "list", label], capture_output=True, text=True)
        if r.returncode == 0:
            state = "LOADED ✓"
        elif path.exists():
            state = "NOT LOADED"
        else:
            state = "NOT INSTALLED"
        print(f"  {label:<42} {state:>12}")


def show_next() -> None:
    print(f"\n  {'Agent':<42} {'Next run (UK)':>22}")
    print(f"  {'─'*66}")
    now_uk = datetime.now(_UK)
    for key, label in AGENTS.items():
        path = PLIST_DIR / f"{label}.plist"
        if not path.exists():
            print(f"  {label:<42} {'NOT INSTALLED':>22}")
            continue
        try:
            with open(path, "rb") as f:
                pl = plistlib.load(f)
            sci = pl.get("StartCalendarInterval", {})
            h, m = sci.get("Hour", 0), sci.get("Minute", 0)
            candidate = now_uk.replace(hour=h, minute=m, second=0, microsecond=0)
            if candidate <= now_uk:
                candidate += timedelta(days=1)
            print(f"  {label:<42} {candidate.strftime('%a %d %b  %H:%M UK'):>22}")
        except Exception as e:
            print(f"  {label:<42} {'ERROR: '+str(e):>22}")


# ─────────────────────────────────────────────────────────────────────────────
# DAILY RECALCULATOR  (called at 05:55 UK by launchd every day)
# ─────────────────────────────────────────────────────────────────────────────
def daily_recalc() -> None:
    now_uk  = datetime.now(_UK)
    weekday = now_uk.weekday()   # 0=Mon … 5=Sat, 6=Sun
    date_str = now_uk.strftime("%Y-%m-%d")
    print(f"[daily-recalc] {date_str}  weekday={weekday}")

    if weekday not in (5, 6):
        print("  Weekday — restoring fixed schedule.")
        _reinstall_predict_plists(WEEKDAY_TIMES)
        return

    day_name = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][weekday]
    print(f"  {day_name} — computing tip-relative schedule...")
    wt = compute_weekend_times(date_str)
    _reinstall_predict_plists(wt)
    print(f"  Weekend schedule applied for {date_str}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd == "install":
        print("\n  Installing PropEdge V16 launchd agents (weekday schedule)...")
        install()
    elif cmd == "uninstall":
        print("\n  Uninstalling PropEdge V16 agents...")
        uninstall()
    elif cmd == "reinstall":
        uninstall()
        install()
    elif cmd == "status":
        status()
    elif cmd == "next":
        show_next()
    elif cmd == "daily-recalc":
        daily_recalc()
    elif cmd == "weekend-check":
        ds = sys.argv[2] if len(sys.argv) > 2 else datetime.now(_UK).strftime("%Y-%m-%d")
        print(f"\n  Weekend preview for {ds}:")
        wt = compute_weekend_times(ds)
        for bk, (h, m) in wt.items():
            print(f"    {bk.upper()}: {h:02d}:{m:02d} UK")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
