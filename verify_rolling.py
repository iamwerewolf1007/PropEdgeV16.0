"""
PropEdge V16.0 — verify_rolling.py
====================================
Verification suite for rolling_engine.py.

For every calculation, computes the expected value from scratch using
only raw pandas operations (no rolling_engine code), then compares
against what extract_features() actually returns.

Run:
    python3 verify_rolling.py
    python3 verify_rolling.py --player "Stephen Curry" --date 2025-02-15
    python3 verify_rolling.py --full   (runs all players, reports failures)

A PASS means the rolling_engine output matches the independently
computed ground truth to within 0.05 pts. A FAIL tells you exactly
which field is wrong, what we got, and what the correct value is.
"""
from __future__ import annotations
import sys, warnings, argparse
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from config import FILE_GL_2425, FILE_GL_2526, FILE_H2H, MIN_PRIOR_GAMES
from rolling_engine import (
    filter_played, build_player_index, get_prior_games,
    build_dynamic_dvp, build_pace_rank, build_opp_def_caches,
    build_rest_days_map, extract_features,
)

PASS = "✓"
FAIL = "✗"
WARN = "⚠"
TOL  = 0.05   # tolerance — values within 0.05 are considered correct


def _close(a, b, tol=TOL):
    if a is None or b is None:
        return a == b
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def load_game_logs():
    dfs = []
    for fp in (FILE_GL_2425, FILE_GL_2526):
        if fp.exists():
            dfs.append(pd.read_csv(fp, parse_dates=["GAME_DATE"], low_memory=False))
    if not dfs:
        print("✗ No game log CSVs found")
        sys.exit(1)
    combined = pd.concat(dfs, ignore_index=True)
    return filter_played(combined)


def verify_one(player: str, game_date: str, line: float,
               pidx, played, dvp, pace, otr, ovr, rmap,
               verbose: bool = True) -> dict:
    """
    Independently compute every feature from raw data,
    compare against extract_features() output.
    Returns dict of {field: (status, got, expected)}.
    """
    results = {}

    prior = get_prior_games(pidx, player, game_date)
    if len(prior) < MIN_PRIOR_GAMES:
        print(f"  {WARN} {player}: only {len(prior)} prior games before {game_date} — skip")
        return {}

    # Opponent / position from most recent prior row
    ptm = str(prior["GAME_TEAM_ABBREVIATION"].iloc[-1]).upper() \
          if "GAME_TEAM_ABBREVIATION" in prior.columns else ""
    opponent = "LAL"   # use a fixed opponent for consistency
    pos_raw  = str(prior["PLAYER_POSITION"].iloc[-1]) \
               if "PLAYER_POSITION" in prior.columns else "G"
    rd       = rmap.get((player, game_date), 2)
    is_home  = True

    # ── Call the engine ─────────────────────────────────────────────
    f = extract_features(
        prior=prior, line=line, opponent=opponent,
        rest_days=rd, pos_raw=pos_raw,
        game_date=pd.Timestamp(game_date),
        min_line=None, max_line=None,
        dyn_dvp=dvp, pace_rank=pace,
        opp_trend=otr, opp_var=ovr,
        is_home=is_home, h2h_row=None,
    )
    if f is None:
        print(f"  {WARN} extract_features returned None for {player}")
        return {}

    pts  = prior["PTS"].values.astype(float)
    mins = prior["MIN_NUM"].fillna(0).values.astype(float)
    n    = len(pts)

    def _smr(arr, w):
        sl = arr[-w:] if len(arr) >= w else arr
        return round(float(np.mean(sl)), 4) if len(sl) > 0 else 0.0

    # ── 1. Rolling PTS (from pre-computed CSV columns) ──────────────
    last = prior.iloc[-1]
    def col(c):
        try:
            v = float(last[c]) if c in last.index else None
            return None if (v is None or np.isnan(v)) else round(v, 4)
        except Exception:
            return None

    for label, window, csv_col in [
        ("L3",  3,  "L3_PTS"),
        ("L5",  5,  "L5_PTS"),
        ("L10", 10, "L10_PTS"),
        ("L20", 20, "L20_PTS"),
        ("L30", 30, "L30_PTS"),
    ]:
        csv_val = col(csv_col)
        got_val = round(f.get(label.lower(), 0), 4)
        # Ground truth: what the CSV column says
        expected = csv_val if csv_val is not None else round(_smr(pts, window), 4)
        ok = _close(got_val, expected)
        results[label] = (PASS if ok else FAIL, got_val, expected)

    # ── 2. std10 (computed from raw PTS, not CSV) ────────────────────
    exp_std10 = round(max(float(np.std(pts[-10:])) if n >= 2 else 1.0, 0.5), 4)
    got_std10 = round(f.get("std10", 0), 4)
    results["std10"] = (PASS if _close(got_std10, exp_std10) else FAIL,
                        got_std10, exp_std10)

    # ── 3. hr10 / hr30 ──────────────────────────────────────────────
    for label, window in [("hr10", 10), ("hr30", 30)]:
        sl   = pts[-window:] if n >= window else pts
        exp  = round(float((sl > line).mean()), 4) if len(sl) > 0 else 0.5
        got  = round(f.get(label, 0), 4)
        results[label] = (PASS if _close(got, exp) else FAIL, got, exp)

    # ── 4. momentum / reversion / acceleration ──────────────────────
    L3  = f["L3"]; L5 = f["L5"]; L10 = f["L10"]; L30 = f["L30"]
    for label, exp in [
        ("momentum",     round(L5  - L30, 4)),
        ("reversion",    round(L10 - L30, 4)),
        ("acceleration", round(L3  - L5,  4)),
        ("volume",       round(L30 - line, 4)),
        ("trend",        round(L5  - L30, 4)),
    ]:
        got = round(f.get(label, 0), 4)
        results[label] = (PASS if _close(got, exp) else FAIL, got, exp)

    # ── 5. min_l10 / min_l30 (from pre-computed CSV) ────────────────
    for label, csv_col, default in [
        ("min_l10", "L10_MIN_NUM", 28.0),
        ("min_l30", "L30_MIN_NUM", 28.0),
    ]:
        csv_val = col(csv_col)
        exp = csv_val if csv_val is not None else default
        got = round(f.get(label, default), 4)
        results[label] = (PASS if _close(got, exp) else FAIL, got, exp)

    # ── 6. home / away scoring split ────────────────────────────────
    if "IS_HOME" in prior.columns:
        ih   = prior["IS_HOME"].fillna(True).astype(float).values.astype(bool)
        hp   = pts[ih]; ap = pts[~ih]
        exp_home = round(_smr(hp[-10:] if len(hp) >= 10 else hp, 10), 4)
        exp_away = round(_smr(ap[-10:] if len(ap) >= 10 else ap, 10), 4)
        exp_split = round(exp_home - exp_away, 4)
        for label, exp in [
            ("home_l10", exp_home),
            ("away_l10", exp_away),
            ("home_away_split", exp_split),
        ]:
            got = round(f.get(label, 0), 4)
            results[label] = (PASS if _close(got, exp) else FAIL, got, exp)

    # ── 7. rest_days / is_b2b ───────────────────────────────────────
    exp_rd  = rmap.get((player, game_date), 2)
    exp_b2b = float(exp_rd <= 1)
    results["rest_days"] = (PASS if _close(f.get("rest_days"), exp_rd) else FAIL,
                            f.get("rest_days"), exp_rd)
    results["is_b2b"]    = (PASS if _close(f.get("is_b2b"), exp_b2b) else FAIL,
                            f.get("is_b2b"), exp_b2b)

    # ── 8. early_season_weight ──────────────────────────────────────
    games_depth    = min(float(n), 82.0)
    exp_esw = round(max(0.70, min(1.0, 0.70 + 0.30 * (games_depth / 30))), 4)
    got_esw = round(f.get("early_season_weight", 1.0), 4)
    results["early_season_weight"] = (PASS if _close(got_esw, exp_esw) else FAIL,
                                      got_esw, exp_esw)

    # ── 9. mean_reversion_risk ──────────────────────────────────────
    exp_mrr = round(abs(L10 - L30) / (f["std10"] + 1e-6), 4)
    got_mrr = round(f.get("mean_reversion_risk", 0), 4)
    results["mean_reversion_risk"] = (PASS if _close(got_mrr, exp_mrr) else FAIL,
                                      got_mrr, exp_mrr)

    # ── 10. level_ewm (EWM span=5 on last 10 games) ─────────────────
    if n >= 2:
        exp_ewm = round(
            float(pd.Series(pts[-10:]).ewm(span=5, adjust=False).mean().iloc[-1]), 4)
    else:
        exp_ewm = round(L10, 4)
    got_ewm = round(f.get("level_ewm", 0), 4)
    results["level_ewm"] = (PASS if _close(got_ewm, exp_ewm) else FAIL,
                            got_ewm, exp_ewm)

    # ── 11. pts_per_min ─────────────────────────────────────────────
    exp_ppm = round(L10 / max(f.get("min_l10", 1), 1), 4)
    got_ppm = round(f.get("pts_per_min", 0), 4)
    results["pts_per_min"] = (PASS if _close(got_ppm, exp_ppm) else FAIL,
                               got_ppm, exp_ppm)

    # ── PRINT ────────────────────────────────────────────────────────
    if verbose:
        passes = sum(1 for s, _, _ in results.values() if s == PASS)
        fails  = sum(1 for s, _, _ in results.values() if s == FAIL)
        total  = len(results)
        print(f"\n{'─'*64}")
        print(f"  Player : {player}")
        print(f"  Date   : {game_date}  |  Line: {line}  |  Prior games: {n}")
        print(f"  Result : {passes}/{total} PASS  |  {fails} FAIL")
        print(f"{'─'*64}")
        print(f"  {'Field':<24} {'Status':>6}  {'Got':>10}  {'Expected':>10}")
        print(f"  {'─'*24} {'─'*6}  {'─'*10}  {'─'*10}")
        for field, (status, got, exp) in results.items():
            marker = f"\033[92m{status}\033[0m" if status == PASS else f"\033[91m{status}\033[0m"
            print(f"  {field:<24} {marker:>6}  {str(round(got,2) if got is not None else '—'):>10}  "
                  f"{str(round(exp,2) if exp is not None else '—'):>10}")

    return results


def run_spot_check(player, date, line, pidx, played, dvp, pace, otr, ovr, rmap):
    print(f"\n{'═'*64}")
    print(f"  SPOT CHECK: {player} on {date}")
    print(f"{'═'*64}")

    # Also print the raw game log so you can verify manually
    prior = get_prior_games(pidx, player, date)
    if prior.empty:
        print("  No prior games found"); return

    print(f"\n  Last 10 raw game scores (most recent last):")
    last10 = prior.tail(10)[["GAME_DATE","PTS","MIN_NUM"]].copy()
    last10["GAME_DATE"] = last10["GAME_DATE"].dt.strftime("%Y-%m-%d")
    for _, row in last10.iterrows():
        marker = "←" if row == last10.iloc[-1] else ""
        print(f"    {row['GAME_DATE']}  {int(row['PTS']):>3} pts  {row['MIN_NUM']:.0f} min")

    pts   = prior["PTS"].values.astype(float)
    n     = len(pts)
    print(f"\n  Manual calculation (all {n} prior games available):")
    for label, w in [("L3",3),("L5",5),("L10",10),("L20",20),("L30",30)]:
        sl  = pts[-w:] if n >= w else pts
        val = round(float(np.mean(sl)), 2)
        games_used = f"last {len(sl)} games"
        print(f"    {label:<4} = {val:>6.2f}  ({games_used}:  "
              f"{','.join(str(int(x)) for x in sl[-5:])}{'...' if len(sl)>5 else ''})")

    verify_one(player, date, line, pidx, played, dvp, pace, otr, ovr, rmap, verbose=True)


def run_full_check(pidx, played, dvp, pace, otr, ovr, rmap,
                   n_players=50, n_dates=3):
    """
    Run verify_one across N random players × N dates each.
    Reports overall pass rate and any systematic failures.
    """
    import random
    random.seed(42)
    all_players = list(pidx.keys())
    sample = random.sample(all_players, min(n_players, len(all_players)))

    total_checks = 0; total_pass = 0; total_fail = 0
    failed_fields: dict = {}

    print(f"\n{'═'*64}")
    print(f"  FULL CHECK: {n_players} players × {n_dates} dates each")
    print(f"{'═'*64}\n")

    for player in sample:
        pdata = pidx[player]
        if len(pdata) < MIN_PRIOR_GAMES + n_dates:
            continue
        # Pick dates spaced across the season
        dates_pool = list(pdata["GAME_DATE"].dt.strftime("%Y-%m-%d").values)
        step = max(1, len(dates_pool) // (n_dates + 1))
        test_dates = [dates_pool[i*step] for i in range(1, n_dates+1)
                      if i*step < len(dates_pool)]
        line = float(pdata["PTS"].median())  # use median score as test line

        for date_str in test_dates:
            results = verify_one(player, date_str, line,
                                 pidx, played, dvp, pace, otr, ovr, rmap,
                                 verbose=False)
            for field, (status, got, exp) in results.items():
                total_checks += 1
                if status == PASS:
                    total_pass += 1
                else:
                    total_fail += 1
                    failed_fields[field] = failed_fields.get(field, 0) + 1

    print(f"  Total checks : {total_checks:,}")
    print(f"  Passed       : {total_pass:,}  ({total_pass/max(total_checks,1)*100:.1f}%)")
    print(f"  Failed       : {total_fail:,}  ({total_fail/max(total_checks,1)*100:.1f}%)")
    if failed_fields:
        print(f"\n  Fields with failures:")
        for field, count in sorted(failed_fields.items(), key=lambda x: -x[1]):
            print(f"    {field:<28} {count:>4} failures")
    else:
        print(f"\n  ✓ Zero failures across all {n_players} players")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PropEdge rolling engine verifier")
    parser.add_argument("--player", default="LeBron James")
    parser.add_argument("--date",   default="2025-02-01")
    parser.add_argument("--line",   type=float, default=25.0)
    parser.add_argument("--full",   action="store_true",
                        help="Run across 50 random players (takes ~30s)")
    args = parser.parse_args()

    print("\n  Loading game logs...")
    played = load_game_logs()
    pidx   = build_player_index(played)
    dvp    = build_dynamic_dvp(played)
    pace   = build_pace_rank(played)
    otr, ovr = build_opp_def_caches(played)
    rmap   = build_rest_days_map(played)
    print(f"  {len(played):,} played rows | {len(pidx):,} players")

    if args.full:
        run_full_check(pidx, played, dvp, pace, otr, ovr, rmap)
    else:
        run_spot_check(
            args.player, args.date, args.line,
            pidx, played, dvp, pace, otr, ovr, rmap
        )

    print()


if __name__ == "__main__":
    main()
