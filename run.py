#!/usr/bin/env python3
"""
PropEdge V16.0 — run.py
Master CLI orchestrator.

IMPORTANT: Always run from the LOCAL folder:
  cd ~/Documents/GitHub/PropEdgeV16.0-Local

Commands:
  python3 run.py setup                       — First-time: git config + install launchd
  python3 run.py generate                    — Build season JSONs + train Elite model (~15 min)
  python3 run.py grade                       — B0: grade yesterday
  python3 run.py grade --date 2026-04-05     — Grade specific date
  python3 run.py grade --date 2026-04-05 --no-retrain
  python3 run.py predict                     — B2 predict (default)
  python3 run.py predict 1                   — B1 08:30 UK
  python3 run.py predict 2                   — B2 11:00 UK
  python3 run.py predict 3                   — B3 16:00 UK
  python3 run.py predict 4                   — B4 18:30 UK
  python3 run.py predict 5                   — B5 21:00 UK
  python3 run.py 0                           — Alias for grade
  python3 run.py 1-5                         — Alias for predict 1-5
  python3 run.py retrain                     — Retrain Elite V2 meta-model only
  python3 run.py h2h                         — Rebuild H2H database from game logs
  python3 run.py dvp                         — Rebuild DVP rankings from game log
  python3 run.py install                     — Install launchd scheduler agents
  python3 run.py uninstall                   — Remove launchd scheduler agents
  python3 run.py status                      — Scheduler + model status + data check
  python3 run.py weekend [YYYY-MM-DD]        — Preview weekend batch schedule
  python3 run.py check                       — Data integrity check (all files)
  python3 run.py diagnose                    — Full system diagnostic
  python3 run.py audit                       — Show recent audit log entries
  python3 run.py regrade [YYYY-MM-DD]        — Clear + re-run grade for a date
  python3 run.py health                      — Full system health check + auto-fix
  python3 run.py health --dry-run            — Report issues only, no auto-fix
  python3 run.py health --quick             — Skip slow Excel checks
  python3 run.py sync                        — Sync code changes to GitHub folder (run before committing)
  python3 run.py rollback                    — Restore season JSONs + monthly files from last backup
  python3 run.py git-cleanup                 — Remove large files from git tracking (run once)
  python3 run.py token-check                 — Diagnose GitHub token + test connection

Batch schedule (launchd, UK time):
  B0  07:00 — grade + H2H + game log + trust update + monthly retrain
  B1  08:30 — morning scan
  B2  11:00 — mid-morning refresh
  B3  16:00 — afternoon sweep
  B4  18:30 — pre-game final
  B5  21:00 — late West Coast top-up
  Daily recalc: 05:55 — adjusts weekend schedule to tip-off times
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import VERSION, GIT_REMOTE, REPO_DIR, uk_now

# Auto-clear stale bytecode — prevents old .pyc files from shadowing replaced .py files
import shutil as _shutil
_cache = ROOT / "__pycache__"
if _cache.exists():
    try:
        _shutil.rmtree(_cache)
        _cache.mkdir(exist_ok=True)
    except Exception:
        pass  # non-fatal — Python will recompile on next run


def _run(script: str, *args: str) -> int:
    cmd = [sys.executable, str(ROOT / script)] + list(args)
    r   = subprocess.run(cmd, cwd=ROOT)
    return r.returncode


# ─────────────────────────────────────────────────────────────────────────────
# SETUP  (run once — git config + launchd)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_setup() -> None:
    print(f"\n  PropEdge {VERSION} — Setup")
    REPO_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=REPO_DIR, capture_output=True)
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=REPO_DIR, capture_output=True,
    )
    if result.returncode != 0:
        subprocess.run(["git", "remote", "add", "origin", GIT_REMOTE], cwd=REPO_DIR)
        print(f"  ✓ Git remote set: {GIT_REMOTE}")
    else:
        print(f"  ✓ Git remote already configured")

    token_file = ROOT / ".github_token"
    if not token_file.exists():
        print(f"\n  ⚠ GitHub token not set. Run:")
        print(f"    echo 'YOUR_TOKEN' > {token_file}")
        print(f"    chmod 600 {token_file}")

    print("\n  Installing launchd agents...")
    from scheduler import install
    install()
    (ROOT / "logs").mkdir(exist_ok=True)
    print(f"\n  ✓ Setup complete.")
    print(f"  Next: python3 run.py generate   (builds season JSONs + trains Elite model)")


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE  (full rebuild of season JSONs + Elite model training)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_generate() -> None:
    print(f"\n  PropEdge {VERSION} — Generate")
    from config import FILE_GL_2425, FILE_GL_2526, FILE_PROPS, FILE_PROPS_2425
    missing = [f for f in (FILE_GL_2425, FILE_GL_2526, FILE_PROPS, FILE_PROPS_2425)
               if not f.exists()]
    if missing:
        print("  ✗ Missing source files:")
        for f in missing:
            print(f"    {f}")
        print("  Add to source-files/ and retry.")
        return

    print("\n  [1/2] Generating season JSONs...")
    rc = _run("generate_season_json.py")
    if rc != 0:
        print("  ✗ generate_season_json failed — fix errors above before training.")
        return

    print("\n  [2/2] Training Elite V2 meta-model...")
    _run("model_trainer.py")
    print(f"\n  ✓ Generate complete.")


# ─────────────────────────────────────────────────────────────────────────────
# GRADE  (B0 — grade yesterday, update game log, trust scores, monthly retrain)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_grade(date_str: str | None = None, no_retrain: bool = False) -> None:
    args = []
    if date_str:
        args.append(date_str)
    _run("batch0_grade.py", *args)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT  (B1-B5 — score today's props)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_predict(batch_n: str = "2") -> None:
    _run("batch_predict.py", batch_n)


# ─────────────────────────────────────────────────────────────────────────────
# RETRAIN  (manual trigger — retrain Elite V2 meta-model only)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_retrain() -> None:
    print(f"\n  PropEdge {VERSION} — Retrain Elite V2")
    _run("model_trainer.py")


# ─────────────────────────────────────────────────────────────────────────────
# H2H  (rebuild head-to-head database from game logs)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_h2h() -> None:
    print(f"\n  PropEdge {VERSION} — Rebuild H2H")
    from h2h_builder import build_h2h
    build_h2h()


# ─────────────────────────────────────────────────────────────────────────────
# DVP  (rebuild defence vs position rankings from game log)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_dvp() -> None:
    print(f"\n  PropEdge {VERSION} — Rebuild DVP")
    from dvp_updater import compute_and_save_dvp
    from config import FILE_GL_2526, FILE_DVP
    compute_and_save_dvp(FILE_GL_2526, FILE_DVP)


# ─────────────────────────────────────────────────────────────────────────────
# INSTALL / UNINSTALL launchd
# ─────────────────────────────────────────────────────────────────────────────
def cmd_install() -> None:
    from scheduler import install
    install()


def cmd_uninstall() -> None:
    from scheduler import uninstall
    uninstall()


# ─────────────────────────────────────────────────────────────────────────────
# STATUS  (scheduler state + model info + data check)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_status() -> None:
    print(f"\n  PropEdge {VERSION} — Status")
    from scheduler import status, show_next
    status()
    show_next()
    cmd_check()


# ─────────────────────────────────────────────────────────────────────────────
# WEEKEND  (preview weekend batch schedule)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_weekend(date_str: str | None = None) -> None:
    from scheduler import compute_weekend_times
    ds = date_str or uk_now().strftime("%Y-%m-%d")
    print(f"\n  Weekend schedule preview for {ds}:")
    times = compute_weekend_times(ds)
    names = {"b1": "Morning scan", "b2": "Mid-morning",
             "b3": "Afternoon",    "b4": "Pre-game", "b5": "Late/WestCoast"}
    for bk, (h, m) in times.items():
        print(f"    {bk.upper()} {names.get(bk,''):12s}: {h:02d}:{m:02d} UK")


# ─────────────────────────────────────────────────────────────────────────────
# CHECK  (data integrity — all required files)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_check() -> None:
    print(f"\n  PropEdge {VERSION} — Data Integrity Check")
    from config import (FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_PROPS,
                        FILE_PROPS_2425, FILE_SEASON_2526, FILE_SEASON_2425,
                        FILE_ELITE_MODEL)
    checks = {
        "Game log 2024-25":     FILE_GL_2425,
        "Game log 2025-26":     FILE_GL_2526,
        "H2H database":         FILE_H2H,
        "Props 2025-26 Excel":  FILE_PROPS,
        "Props 2024-25 Excel":  FILE_PROPS_2425,
        "Season 2024-25 JSON":  FILE_SEASON_2425,
        "Season 2025-26 JSON":  FILE_SEASON_2526,
        "Elite model pkl":      FILE_ELITE_MODEL,
    }
    all_ok = True
    for label, path in checks.items():
        exists = path.exists()
        size   = f"{path.stat().st_size/1024:.0f} KB" if exists else "—"
        sym    = "✓" if exists else "✗"
        if not exists:
            all_ok = False
        print(f"  {sym} {label:<26} {size:>10}   {path.name}")

    if FILE_ELITE_MODEL.exists():
        try:
            import pickle
            with open(FILE_ELITE_MODEL, "rb") as f:
                pkg = pickle.load(f)
            trained  = pkg.get("trained_at", "?")[:19]
            n_plays  = pkg.get("n_plays", "?")
            seasons  = pkg.get("seasons_used", ["?"])
            print(f"\n  Elite model: trained {trained} | {n_plays:,} plays | "
                  f"seasons: {', '.join(seasons)}")
        except Exception as e:
            print(f"\n  ✗ Elite model load error: {e}")

    for sf, label in ((FILE_SEASON_2526, "2025-26"), (FILE_SEASON_2425, "2024-25")):
        if sf.exists():
            try:
                import json
                with open(sf) as f:
                    plays = json.load(f)
                graded = [p for p in plays if p.get("result") in ("WIN", "LOSS")]
                wins   = sum(1 for p in graded if p.get("result") == "WIN")
                hr     = f"{wins/len(graded)*100:.1f}%" if graded else "—"
                print(f"  Season {label}: {len(plays):,} plays | "
                      f"{len(graded):,} graded | HR={hr}")
            except Exception:
                pass

    from config import FILE_V12_TRUST
    if FILE_V12_TRUST.exists():
        try:
            import json
            trust = json.loads(FILE_V12_TRUST.read_text())
            below = sum(1 for v in trust.values() if v < 0.42)
            avg   = sum(trust.values()) / len(trust) if trust else 0
            print(f"  Trust scores: {len(trust):,} players | "
                  f"avg={avg:.3f} | {below} below threshold (0.42)")
        except Exception:
            pass

    print(f"\n  {'✓ All files present' if all_ok else '✗ Some files missing — run: python3 run.py generate'}")


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSE  (full system diagnostic)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_diagnose() -> None:
    import diagnose
    diagnose.main()


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT  (show recent audit log entries)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_audit() -> None:
    from config import FILE_AUDIT
    if FILE_AUDIT.exists():
        import pandas as pd
        df = pd.read_csv(FILE_AUDIT)
        print(df.tail(20).to_string())
    else:
        print("  No audit log yet.")


# ─────────────────────────────────────────────────────────────────────────────
# REGRADE  (clear + re-run grade for a specific date)
# ─────────────────────────────────────────────────────────────────────────────
def cmd_rollback() -> None:
    """
    Restore season JSONs and monthly files from the last backup (.bak files).
    Use after a generate that produced bad data.
    """
    import shutil
    from config import FILE_SEASON_2425, FILE_SEASON_2526

    print(f"\n  PropEdge {VERSION} — Rollback")
    rolled_back = []

    for f in (FILE_SEASON_2425, FILE_SEASON_2526):
        bak = f.with_suffix(".bak")
        if bak.exists():
            shutil.copy2(bak, f)
            rolled_back.append(f.name)
            print(f"  ✓ Restored {f.name} from {bak.name}")
        else:
            print(f"  ⚠ No backup found for {f.name}")

    # Also restore monthly file backups
    monthly_base = FILE_SEASON_2526.parent / "monthly"
    restored_monthly = 0
    if monthly_base.exists():
        for bak in monthly_base.rglob("*.bak"):
            orig = bak.with_suffix(".json")
            shutil.copy2(bak, orig)
            restored_monthly += 1
    if restored_monthly:
        print(f"  ✓ Restored {restored_monthly} monthly .bak files")

    if rolled_back:
        print(f"\n  ✓ Rollback complete. Files restored: {rolled_back}")
        print(f"  Run: python3 run.py health  to verify data integrity")
    else:
        print(f"  ✗ Nothing to roll back — no .bak files found")
        print(f"  Backups are created automatically each time generate runs")


def cmd_regrade(date_str: str | None = None) -> None:
    args = []
    if date_str:
        args.append(date_str)
    _run("regrade.py", *args)


# ─────────────────────────────────────────────────────────────────────────────
# SYNC — copy code changes from Local to GitHub folder
# ─────────────────────────────────────────────────────────────────────────────
def cmd_sync() -> None:
    """
    Copy all Python scripts and index.html from the Local folder to the
    GitHub folder (PropEdgeV16.0). Run this before committing via GitHub Desktop.

    Only syncs code files — data files (season JSONs, today.json etc.) are
    pushed to GitHub automatically by git_push.py via the REST API.
    """
    import shutil
    from config import REPO_DIR

    if not REPO_DIR.exists():
        print(f"  ✗ GitHub folder not found: {REPO_DIR}")
        print(f"  Extract PropEdgeV16_GitHub.zip to {REPO_DIR.parent} first.")
        return

    # Files to sync: all Python scripts + index.html + README + trust JSONs
    to_sync = list(ROOT.glob("*.py")) + [
        ROOT / "index.html",
        ROOT / "README.md",
        ROOT / ".gitignore",
        ROOT / "models" / "V12" / "player_trust.json",
        ROOT / "models" / "V14" / "player_trust.json",
    ]

    synced = 0
    for src in to_sync:
        if not src.exists():
            continue
        # Build destination path (same relative structure)
        rel = src.relative_to(ROOT)
        dst = REPO_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        synced += 1

    print(f"\n  ✓ Synced {synced} files from Local → GitHub folder")
    print(f"  Local:  {ROOT}")
    print(f"  GitHub: {REPO_DIR}")
    print(f"\n  Now open GitHub Desktop, review changes, commit and push.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"

    if cmd in ("setup",):
        cmd_setup()

    elif cmd == "generate":
        cmd_generate()

    elif cmd in ("grade", "0"):
        date_arg   = next((a for a in sys.argv[2:] if a.startswith("20")), None)
        no_retrain = "--no-retrain" in sys.argv
        cmd_grade(date_arg, no_retrain)

    elif cmd in ("predict",) or (cmd.isdigit() and cmd in "12345"):
        n = sys.argv[2] if cmd == "predict" and len(sys.argv) > 2 and sys.argv[2].isdigit() else cmd if cmd.isdigit() else "2"
        cmd_predict(n)

    elif cmd == "retrain":
        cmd_retrain()

    elif cmd == "h2h":
        cmd_h2h()

    elif cmd == "dvp":
        cmd_dvp()

    elif cmd == "install":
        cmd_install()

    elif cmd == "uninstall":
        cmd_uninstall()

    elif cmd == "status":
        cmd_status()

    elif cmd == "weekend":
        ds = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_weekend(ds)

    elif cmd == "check":
        cmd_check()

    elif cmd == "diagnose":
        cmd_diagnose()

    elif cmd == "audit":
        cmd_audit()

    elif cmd == "regrade":
        ds = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_regrade(ds)

    elif cmd == "sync":
        cmd_sync()

    elif cmd == "rollback":
        cmd_rollback()

    elif cmd in ("git-cleanup", "git-untrack"):
        # Remove large files from git tracking (keeps files on disk, just stops git tracking them)
        # Run this once after adding .gitignore to clean up already-tracked large files
        import subprocess
        large_files = [
            "data/season_2024_25.json",
            "data/season_2025_26.json",
            "data/today.json",
            "data/dvp_rankings.json",
            "data/propedge_ml_dataset.xlsx",
            "data/audit_log.csv",
            "source-files/nba_gamelogs_2024_25.csv",
            "source-files/nba_gamelogs_2025_26.csv",
            "source-files/h2h_database.csv",
            "source-files/PropEdge_-_Match_and_Player_Prop_lines_.xlsx",
            "source-files/PropEdge_2024_25_Match_and_Player_Prop_lines.xlsx",
        ]
        print(f"\n  PropEdge {VERSION} — Git Cleanup")
        print("  Removing large data files from git tracking (files stay on disk)...")
        for f in large_files:
            r = subprocess.run(
                ["git", "rm", "--cached", "--ignore-unmatch", f],
                cwd=ROOT, capture_output=True, text=True
            )
            if "rm " in r.stdout:
                print(f"  ✓ Untracked: {f}")
        r2 = subprocess.run(
            ["git", "commit", "-m", "Remove large data files from git tracking (.gitignore)"],
            cwd=ROOT, capture_output=True, text=True
        )
        if r2.returncode == 0:
            print("  ✓ Committed. Now push with: git push origin main")
        else:
            print(f"  (No changes to commit or: {r2.stderr.strip()[:100]})")
        print("  ✓ GitHub Desktop will no longer try to push large files.")

    elif cmd in ("health", "health-check"):
        import subprocess
        args = ["--dry-run"] if "--dry-run" in sys.argv else []
        args += ["--quick"] if "--quick" in sys.argv else []
        subprocess.run([sys.executable, str(ROOT / "health_check.py")] + args, cwd=ROOT)

    elif cmd in ("token-check", "token", "auth"):
        from git_push import token_check
        token_check()

    elif cmd == "all":
        cmd_grade()
        cmd_predict("2")

    else:
        print(__doc__)


if __name__ == "__main__":
    main()
