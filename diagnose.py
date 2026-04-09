"""
PropEdge V16.0 — diagnose.py
Diagnostic checks: data integrity, model file presence, CSV row counts,
season JSON stats, today.json summary.

Usage: python3 diagnose.py
"""
from __future__ import annotations
import json, pickle, sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    FILE_GL_2425, FILE_GL_2526, FILE_H2H, FILE_PROPS,
    FILE_TODAY, FILE_SEASON_2526, FILE_SEASON_2425, FILE_AUDIT, FILE_DVP,
    FILE_V92_REG, FILE_V10_REG, FILE_V11_REG,
    FILE_V12_REG, FILE_V12_CLF, FILE_V12_CAL, FILE_V12_SEG, FILE_V12_Q,
    FILE_V12_TRUST, FILE_V14_REG, FILE_V14_CLF, FILE_V14_CAL, FILE_V14_TRUST,
    FILE_ELITE_MODEL, uk_now,
)

def section(title: str) -> None:
    print(f"\n  {'─'*56}")
    print(f"  {title}")
    print(f"  {'─'*56}")

def check_file(label: str, path: Path, expected_min: int = 0) -> bool:
    if not path.exists():
        print(f"  ✗ MISSING  {label:<35} {path.name}")
        return False
    size = path.stat().st_size
    print(f"  ✓ OK       {label:<35} {size/1024:.0f} KB")
    return True

def main() -> None:
    print(f"\n  PropEdge V16.0 — Diagnostics  ({uk_now().strftime('%Y-%m-%d %H:%M UK')})")

    section("Source Files")
    check_file("Game log 2024-25",  FILE_GL_2425)
    check_file("Game log 2025-26",  FILE_GL_2526)
    check_file("H2H database",      FILE_H2H)
    check_file("Props Excel",       FILE_PROPS)

    section("Model Files")
    for label, path in [
        ("V9.2 GBR",         FILE_V92_REG),
        ("V10 GBR",          FILE_V10_REG),
        ("V11 GBR",          FILE_V11_REG),
        ("V12 GBR",          FILE_V12_REG),
        ("V12 LightGBM clf", FILE_V12_CLF),
        ("V12 Calibrator",   FILE_V12_CAL),
        ("V12 Segment",      FILE_V12_SEG),
        ("V12 Quantile",     FILE_V12_Q),
        ("V12 Trust",        FILE_V12_TRUST),
        ("V14 GBR",          FILE_V14_REG),
        ("V14 GBC clf",      FILE_V14_CLF),
        ("V14 Calibrator",   FILE_V14_CAL),
        ("V14 Trust",        FILE_V14_TRUST),
        ("Elite V2 GBT",     FILE_ELITE_MODEL),
    ]:
        check_file(label, path)

    section("Data Files")
    check_file("season_2024_25.json", FILE_SEASON_2425)
    check_file("season_2025_26.json", FILE_SEASON_2526)
    check_file("today.json",          FILE_TODAY)
    check_file("dvp_rankings.json",   FILE_DVP)
    check_file("audit_log.csv",       FILE_AUDIT)

    section("Elite Model Info")
    if FILE_ELITE_MODEL.exists():
        try:
            with open(FILE_ELITE_MODEL, "rb") as f:
                pkg = pickle.load(f)
            print(f"  Trained: {pkg.get('trained_at','?')[:19]}")
            print(f"  Plays:   {pkg.get('n_plays','?'):,}")
            print(f"  Features:{len(pkg.get('features',[]))}")
            oof = pkg.get("oof_summary", {})
            for tier, stats in oof.items():
                print(f"  {tier:8s}: {stats.get('acc',0):.1%} OOF | n={stats.get('n',0):,}")
        except Exception as e:
            print(f"  ✗ Elite model load error: {e}")
    else:
        print("  ⚠ Elite model not trained yet. Run: python3 run.py retrain")

    section("Season JSON Summary")
    for path, label in [(FILE_SEASON_2526,"2025-26"),(FILE_SEASON_2425,"2024-25")]:
        if path.exists():
            try:
                with open(path) as f:
                    plays = json.load(f)
                graded = [p for p in plays if p.get("result") in ("WIN","LOSS")]
                wins   = sum(1 for p in graded if p.get("result")=="WIN")
                hr     = f"{wins/len(graded)*100:.1f}%" if graded else "—"
                print(f"  {label}: {len(plays):,} plays | {len(graded):,} graded | HR: {hr}")
            except Exception as e:
                print(f"  {label}: error — {e}")

    section("Today.json Summary")
    if FILE_TODAY.exists():
        try:
            with open(FILE_TODAY) as f:
                today = json.load(f)
            dates = sorted(set(p.get("date","") for p in today))
            print(f"  Dates:   {dates}")
            print(f"  Plays:   {len(today):,}")
            for tier in ["APEX","ULTRA","ELITE","STRONG","PLAY+"]:
                n = sum(1 for p in today if p.get("elite_tier")==tier)
                if n: print(f"  {tier}: {n}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  today.json not found (run a batch to generate)")

    section("Audit Log")
    if FILE_AUDIT.exists():
        try:
            import pandas as pd
            audit = pd.read_csv(FILE_AUDIT)
            alerts = audit[audit["event"].str.contains("FAIL|ALERT|ABORT", na=False)]
            print(f"  Events:  {len(audit):,}")
            print(f"  Alerts:  {len(alerts)}")
            if len(alerts):
                print("  Recent alerts:")
                for _, row in alerts.tail(3).iterrows():
                    print(f"    {row.get('ts','')} {row.get('event','')} {row.get('detail','')}")
        except Exception as e:
            print(f"  Error: {e}")
    else:
        print("  No audit log yet.")

    print()

if __name__ == "__main__":
    main()
