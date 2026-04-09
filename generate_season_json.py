"""
PropEdge V16.0 — generate_season_json.py
Builds season_2024_25.json and season_2025_26.json from REAL prop lines.

  2024-25 → FILE_PROPS_2425  (PropEdge_2024_25_Match_and_Player_Prop_lines.xlsx)
  2025-26 → FILE_PROPS       (PropEdge_-_Match_and_Player_Prop_lines_.xlsx)

Player names from both Excel files are normalised through player_name_aliases
before matching to the NBA game log CSVs.  No synthetic lines are used.

Run once (or after adding new source files):
  python3 generate_season_json.py
"""
from __future__ import annotations
import json
import time, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    VERSION, FILE_GL_2425, FILE_GL_2526, FILE_H2H,
    FILE_PROPS, FILE_PROPS_2425,
    FILE_SEASON_2425, FILE_SEASON_2526,
    MIN_PRIOR_GAMES, get_pos_group, clean_json,
)
from rolling_engine import (
    filter_played, build_player_index, get_prior_games,
    build_dynamic_dvp, build_pace_rank, build_opp_def_caches,
    build_rest_days_map, extract_features,
)
from batch_predict import (
    sv92, sv10, sv11, sv12, sv14, build_ev, score_elite,
    flag_details, M,
)
from reasoning_engine import generate_pre_match_reason, generate_post_match_reason
from player_name_aliases import resolve_name, _norm
import hashlib
from ml_dataset import write_ml_dataset
from monthly_split import write_monthly_split, verify_monthly_integrity


# ── Load real props ────────────────────────────────────────────────────────────
def load_props_for_season(
    source_file: Path,
    season_start: str,
    season_end: str,
    season_label: str,
) -> list[dict]:
    """
    Load prop lines from the correct PropEdge Excel file for the given season.
    Player names are kept as-is; normalisation happens in score_and_grade via
    resolve_name().

    Returns list of prop dicts, or empty list on error / no rows found.
    No synthetic fallback — if the file is missing or empty, the caller logs
    an error and skips writing that season's JSON.
    """
    if not source_file.exists():
        print(f"  ✗ Source file not found: {source_file}")
        print(f"    Place the {season_label} PropEdge Excel in source-files/ and re-run.")
        return []

    s = pd.Timestamp(season_start)
    e = pd.Timestamp(season_end)

    try:
        xl = pd.read_excel(source_file, sheet_name="Player_Points_Props")
        xl["Date"] = pd.to_datetime(xl["Date"], errors="coerce")
        xl = xl[(xl["Date"] >= s) & (xl["Date"] <= e)]

        if xl.empty:
            print(f"  ✗ No rows in date range {season_start} → {season_end} "
                  f"in {source_file.name}")
            return []

        props = []
        for _, r in xl.dropna(subset=["Line"]).iterrows():
            try:
                props.append({
                    "player":       str(r["Player"]).strip(),
                    "date":         str(r["Date"].date()),
                    "line":         float(r["Line"]),
                    "over_odds":    float(r.get("Over Odds")  or -110),
                    "under_odds":   float(r.get("Under Odds") or -110),
                    "books":        int(r.get("Books") or 0),
                    "min_line":     float(r["Min Line"]) if pd.notna(r.get("Min Line")) else None,
                    "max_line":     float(r["Max Line"]) if pd.notna(r.get("Max Line")) else None,
                    "game":         str(r.get("Game")         or ""),
                    "home":         str(r.get("Home")         or ""),
                    "away":         str(r.get("Away")         or ""),
                    "game_time_et": str(r.get("Game_Time_ET") or ""),
                    "source":       "excel",
                })
            except Exception:
                continue

        print(f"  Props loaded ({season_label}): {len(props):,} rows from {source_file.name}")
        return props

    except Exception as e:
        print(f"  ✗ Excel read error ({source_file.name}): {e}")
        return []


# ── Score + grade ──────────────────────────────────────────────────────────────
def score_and_grade(
    props: list[dict],
    pidx: dict,
    played: pd.DataFrame,
    combined_df: pd.DataFrame,
    h2h_lkp: dict,
    dvp: dict, pace: dict, otr: dict, ovr: dict, rmap: dict,
    season: str,
    prev_directions: dict | None = None,
) -> list[dict]:
    """
    Score and grade props against the game log.

    prev_directions: {(player, date): "OVER"|"UNDER"}
        If provided, already-graded plays use their stored direction instead of
        re-deriving from the current model. This ensures historical WIN/LOSS
        records don't change when the game log is updated and prb12 shifts.
    """
    from config import (
        assign_elite_tier, ELITE_STAKES, ELITE_TIER_NUM,
        ELITE_TIER_LABEL, ELITE_THRESHOLDS, TRUST_THRESHOLD,
    )

    # nmap is built from the game log CSV names — resolve_name() uses it
    nmap = {_norm(k): k for k in pidx}
    tv12 = M.tv12
    tv14 = M.tv14

    scored = []; skipped = 0; total = len(props)

    for pi, prop in enumerate(props):
        if pi % 2000 == 0 and pi > 0:
            print(f"  {pi:,}/{total:,} | scored={len(scored)} skipped={skipped}")

        line       = float(prop["line"])
        player_raw = prop["player"]
        date_str   = prop["date"]
        opp        = str(prop.get("opponent", "")).upper()
        home_team  = str(prop.get("home", "")).upper()
        away_team  = str(prop.get("away", "")).upper()
        game       = str(prop.get("game", ""))
        gtime      = str(prop.get("game_time_et", ""))
        source     = prop.get("source", "excel")

        # Normalise player name through alias table + fuzzy fallback
        player = resolve_name(player_raw, nmap)
        if player is None:
            skipped += 1
            continue

        prior = get_prior_games(pidx, player, date_str)
        if len(prior) < MIN_PRIOR_GAMES:
            skipped += 1
            continue

        ptm = (str(prior["GAME_TEAM_ABBREVIATION"].iloc[-1]).upper()
               if "GAME_TEAM_ABBREVIATION" in prior.columns else "")
        if not opp and home_team and away_team:
            opp = home_team if ptm == away_team else away_team
        is_home  = (ptm == home_team) if ptm and home_team else None
        pos_raw  = (str(prior["PLAYER_POSITION"].iloc[-1])
                    if "PLAYER_POSITION" in prior.columns else "G")
        rd       = rmap.get((player, date_str), 2)
        h2h_row  = h2h_lkp.get((_norm(player), opp))

        f = extract_features(
            prior=prior, line=line, opponent=opp, rest_days=rd, pos_raw=pos_raw,
            game_date=pd.Timestamp(date_str), min_line=prop.get("min_line"),
            max_line=prop.get("max_line"), dyn_dvp=dvp, pace_rank=pace,
            opp_trend=otr, opp_var=ovr, is_home=is_home, h2h_row=h2h_row,
        )
        if f is None:
            skipped += 1
            continue

        f["pos_grp_str"] = get_pos_group(pos_raw)
        f["books"]       = prop.get("books", 0)
        f["line_spread"] = float(
            (prop.get("max_line") or line) - (prop.get("min_line") or line)
        )

        p92, g92 = sv92(f, line)
        p10, g10 = sv10(f, line)
        p11, g11 = sv11(f, line)
        p12, g12, prb12, q25, q75 = sv12(f, line)
        p14, g14, prb14 = sv14(f, line)

        t12  = float(tv12.get(player, 0.68))
        t14v = float(tv14.get(player, 0.67))
        hG   = 0.0; havg = None
        if h2h_row:
            hG = float(h2h_row.get("H2H_GAMES") or 0)
            try:
                v = h2h_row.get("H2H_AVG_PTS")
                havg = float(v) if v is not None else None
            except Exception:
                pass

        ev = build_ev(
            f, line, p92, g92, p10, g10, p11, g11,
            p12, g12, prb12, q25, q75, p14, g14, prb14,
            hG, havg, t12, t14v, is_home,
        )
        ep = score_elite(ev)
        et = assign_elite_tier(ep)

        if rd >= 4 and et in ("APEX", "ULTRA", "ELITE", "STRONG"):
            et = "PLAY+" if ep >= ELITE_THRESHOLDS["PLAY+"] else "SKIP"
        tm = (t12 + t14v) / 2
        if tm < TRUST_THRESHOLD and et in ("APEX", "ULTRA", "ELITE"):
            et = "STRONG"

        est   = ELITE_STAKES.get(et, 0.0)
        pmean = float(0.30*p92 + 0.25*p10 + 0.10*p11 + 0.25*p12 + 0.10*p14)
        gmean = round(pmean - line, 2)
        cv12  = abs(prb12 - 0.5) * 2
        # Use locked direction if this play was already graded, otherwise model direction
        _locked_dir = (prev_directions or {}).get((player, date_str))
        dl = _locked_dir if _locked_dir else ("OVER" if prb12 >= 0.5 else "UNDER")

        nf = sum([
            int(ev.get("v92_v12clf_agree", 0)), int(ev.get("v92_v14clf_agree", 0)),
            int(ev.get("all_clf_agree", 0)),    int(ev.get("reg_consensus", 0)),
            int(ev.get("_vex", 0)),             int(ev.get("_hal", 0)),
            int(not ev.get("_rru", 0)),         int(ev.get("_lv", 0)),
            int(ev.get("_tm", 0)),              int(tm >= 0.65),
        ])

        # Grade against actual game result (if available in game logs)
        #
        # GRADING LOGIC: WIN = model's direction prediction was CORRECT
        #   dl = "OVER"  → WIN if actual_pts > line,  LOSS if actual_pts <= line
        #   dl = "UNDER" → WIN if actual_pts <= line, LOSS if actual_pts > line
        #
        # The Props Excel has BOTH Over Odds and Under Odds — the model picks
        # which side to bet. Grading measures prediction accuracy, not "did the
        # player go over". Training on direction-correctness teaches the model to
        # pick the right side, not just find always-over patterns.
        #
        # STABILITY: dl is derived from prb12 (V12 direction classifier). Once a play
        # is graded and the direction is stored on the play, subsequent generate runs
        # preserve the original direction — see the "prev_directions" lookup below.
        result = ""; actual_pts = None; delta = None
        actual_row = played[
            (played["PLAYER_NAME"] == player) &
            (played["GAME_DATE"] == pd.Timestamp(date_str))
        ]
        if not actual_row.empty:
            actual_pts = float(actual_row["PTS"].iloc[0])
            if abs(actual_pts - line) < 0.05:
                result = "PUSH"
            elif dl == "OVER":
                result = "WIN" if actual_pts > line else "LOSS"
            else:  # dl == "UNDER"
                result = "WIN" if actual_pts <= line else "LOSS"
            delta = round(actual_pts - line, 1)
        elif pd.Timestamp(date_str) < pd.Timestamp("today"):
            # Past game — player not in played df (MIN_NUM > 0)
            # Check if ANY games happened on that date (combined_df has all rows incl. DNP)
            date_games = combined_df[combined_df["GAME_DATE"] == pd.Timestamp(date_str)]
            if not date_games.empty:
                # Games DID happen that day — player sat out (injury/rest/scratch)
                result = "DNP"
            # else: no games recorded that day yet (data gap) — leave result="" as pending

        # Recent 20 games
        pts_vals   = prior["PTS"].values[-20:]
        dates_vals = prior["GAME_DATE"].values[-20:]
        home_vals  = (prior["IS_HOME"].values[-20:]
                      if "IS_HOME" in prior.columns
                      else [True] * min(20, len(prior)))

        play = {
            "player":       player,   "date":         date_str,
            "match":        game or f"{away_team} @ {home_team}",
            "fullMatch":    game or f"{away_team} @ {home_team}",
            "game":         game,     "home":         home_team,
            "away":         away_team,"opponent":     opp,
            "position":     pos_raw,  "isHome":       is_home,
            "gameTime":     gtime,    "game_time":    gtime,
            "line":         line,
            "overOdds":     prop.get("over_odds",  -110),
            "underOdds":    prop.get("under_odds", -110),
            "books":        prop.get("books", 0),
            "min_line":     prop.get("min_line"),
            "max_line":     prop.get("max_line"),
            "lineHistory":  [],
            "elite_tier":   et,   "elite_prob":  round(ep, 4),
            "elite_stake":  est,  "elite_rank":  0,
            "v12_clf_prob": round(prb12, 4), "v12_clf_conv": round(cv12, 3),
            "v14_clf_prob": round(prb14, 4),
            "all_clf_agree":     bool(ev.get("all_clf_agree", 0)),
            "v92_v12_agree":     bool(ev.get("v92_v12clf_agree", 0)),
            "v12_v14_agree":     bool(ev.get("v12_v14_agree", 0)),
            "reg_consensus":     bool(ev.get("reg_consensus", 0)),
            "v12_extreme":       bool(ev.get("_vex", 0)),
            "trust_v12":         round(t12, 3),
            "trust_v14":         round(t14v, 3),
            "trust_mean":        round(tm, 3),
            "q25_v12":           round(q25, 2),
            "q75_v12":           round(q75, 2),
            "q_confidence":      round(ev.get("q_confidence", 0), 3),
            "real_gap_v92":      round(g92, 2),
            "real_gap_v12":      round(g12, 2),
            "real_gap_mean":     round(ev.get("gap_mean_real", 0), 2),
            "is_under":          dl.endswith("UNDER"),
            "dir":               dl,   "direction":   dl,
            "predPts":           round(pmean, 1),
            "predGap":           round(gmean, 2),
            "conf":              round(ep, 4),
            "calProb":           round(prb12, 4),
            "tierLabel":         ELITE_TIER_LABEL.get(et, "T3"),
            "tier":              ELITE_TIER_NUM.get(et, 3),
            "units":             est,
            "enginesAgree":      bool(ev.get("all_clf_agree", 0)),
            "flags":             nf,   "flagsStr":    f"{nf}/10",
            "l3":   round(f.get("L3",  0), 1), "l5":  round(f.get("L5",  0), 1),
            "l10":  round(f.get("L10", 0), 1), "l20": round(f.get("L20", 0), 1),
            "l30":  round(f.get("L30", 0), 1), "std10": round(f.get("std10", 0), 1),
            "hr10": round(f.get("hr10", 0), 3),"hr30": round(f.get("hr30", 0), 3),
            "volume":  round(f.get("volume",  0), 1),
            "trend":   round(f.get("trend",   0), 1),
            "momentum":round(f.get("momentum",0), 1),
            "min_l10": round(f.get("min_l10", 28), 1),
            "minL10":  round(f.get("min_l10", 28), 1),
            "min_l30": round(f.get("min_l30", 28), 1),
            "pts_per_min":  round(f.get("pts_per_min",  0.5),  3),
            "usage_l10":    round(f.get("usage_l10",    0.18), 3),
            "fta_l10":      round(f.get("fta_l10",      0),    1),
            "ft_rate":      round(f.get("ft_rate",      0),    3),
            "fg3a_l10":     round(f.get("fg3a_l10",     0),    1),
            "fga_l10":      round(f.get("fga_l10",      0),    1),
            "l10_fg_pct":   round(f.get("fg_pct_l10",  0.45),  3),
            "line_vs_l30":  round(f.get("line_vs_l30",  0),    2),
            "homeAvgPts":   round(f.get("home_l10", f.get("L10", 0)), 1),
            "awayAvgPts":   round(f.get("away_l10", f.get("L10", 0)), 1),
            "home_away_split": round(f.get("home_away_split", 0), 1),
            "ptm":          ptm,
            "team":         ptm,
            "extreme_hot":  bool(f.get("extreme_hot",  False)),
            "extreme_cold": bool(f.get("extreme_cold", False)),
            "level_ewm":    round(f.get("level_ewm", f.get("L10", 0)), 1),
            "usage_l30":    round(f.get("usage_l30", f.get("usage_l10", 0.18)), 3),
            "min_cv":       round(f.get("min_cv", 0.2), 3),
            "seasonProgress": round(f.get("season_progress", 0.5), 3),
            "season_progress": round(f.get("season_progress", 0.5), 3),
            "is_b2b":           bool(f.get("is_b2b", 0)),
            "rest_days":        int(rd),
            "defP":             int(f.get("defP_dynamic", 15)),
            "pace_rank":        int(f.get("pace_rank", 15)),
            "defP_dynamic":     int(f.get("defP_dynamic", 15)),
            "meanReversionRisk":round(f.get("mean_reversion_risk", 0), 2),
            "mean_reversion_risk": round(f.get("mean_reversion_risk", 0), 2),
            "earlySeasonW":     round(f.get("early_season_weight", 1), 3),
            "early_season_weight": round(f.get("early_season_weight", 1), 3),
            "is_long_rest":     bool(f.get("is_long_rest", 0)),
            "h2hG":             int(hG),
            "h2h":              round(havg, 1) if havg else None,
            "h2hAvg":           round(havg, 1) if havg else None,
            "h2h_avg":          round(havg, 1) if havg else None,
            "h2h_games":        int(hG),
            "h2hTsDev":         round(f.get("h2h_ts_dev",  0), 2),
            "h2h_ts_dev":       round(f.get("h2h_ts_dev",  0), 2),
            "h2hFgaDev":        round(f.get("h2h_fga_dev", 0), 2),
            "h2hConfidence":    round(f.get("h2h_conf",    0), 3),
            "result":           result,
            "actualPts":        actual_pts,
            "delta":            delta,
            "lossType":         None,
            "postMatchReason":  None,
            "preMatchReason":   "",
            "flagDetails":      flag_details(ev),
            "recent20":         [float(v) for v in pts_vals],
            "recent20dates":    [str(pd.Timestamp(d).date()) for d in dates_vals],
            "recent20homes":    [bool(v) for v in home_vals],
            "source":           source,
            "season":           season,
        }

        play["preMatchReason"] = generate_pre_match_reason(play)

        if result in ("WIN", "LOSS"):
            box_data: dict = {"actual_pts": actual_pts}
            if not actual_row.empty:
                r0 = actual_row.iloc[0]
                box_data["actual_min"] = float(r0.get("MIN_NUM", r0.get("MIN", 0)) or 0)
                box_data["actual_fga"] = float(r0.get("FGA", 0) or 0)
                box_data["actual_fgm"] = float(r0.get("FGM", 0) or 0)
            # Store box stats on play for ML dataset
            play_box_fga = box_data.get("actual_fga") or 0
            play_box_fgm = box_data.get("actual_fgm") or 0
            post_narrative, loss_type = generate_post_match_reason(play, box_data)
            play["postMatchReason"] = post_narrative
            play["lossType"] = loss_type if result == "LOSS" else None
            play["actual_min"]    = box_data.get("actual_min")
            play["actual_fga"]    = box_data.get("actual_fga")
            play["actual_fgm"]    = box_data.get("actual_fgm")
            play["actual_fg_pct"] = round(play_box_fgm / play_box_fga * 100, 1) if play_box_fga > 0 else None
        elif result == "PUSH":
            play["postMatchReason"] = f"PUSH — scored exactly {actual_pts:.0f}."

        scored.append(play)

    print(f"  Scored: {len(scored):,} | Skipped: {skipped:,}")
    return scored


# ── Main ───────────────────────────────────────────────────────────────────────
def _print_season_stats(label: str, plays: list[dict]) -> None:
    """Print a detailed breakdown of scored/graded plays for one season."""
    total   = len(plays)
    graded  = [p for p in plays if p.get("result") in ("WIN", "LOSS")]
    wins    = [p for p in graded if p.get("result") == "WIN"]
    dnp     = [p for p in plays if p.get("result") == "DNP"]
    push    = [p for p in plays if p.get("result") == "PUSH"]
    pending = [p for p in plays if p.get("result") not in ("WIN","LOSS","DNP","PUSH")]
    hr      = f"{len(wins)/len(graded)*100:.1f}%" if graded else "—"

    # Flag if graded rate is suspiciously low
    grade_pct = len(graded) / total * 100 if total else 0
    flag = "  ⚠ Low graded rate — check game log coverage" if grade_pct < 50 else ""

    print(f"  ✓ season_{label.replace('-','_').lower()}.json" if label != "COMBINED"
          else "  ✓ Combined", end="")
    print(f" → {total:,} plays | "
          f"{len(graded):,} graded (HR:{hr}) | "
          f"{len(dnp):,} DNP | "
          f"{len(push):,} PUSH | "
          f"{len(pending):,} pending{flag}")

    # Warn about players with 0 graded plays (name mismatch indicator)
    if label != "COMBINED":
        from collections import Counter
        player_results = Counter()
        for p in plays:
            player_results[p.get("player","")] += 1 if p.get("result") in ("WIN","LOSS") else 0
        zero_graded = [pl for pl, cnt in player_results.items() if cnt == 0 and pl]
        if zero_graded:
            print(f"    ⚠ {len(zero_graded)} players with 0 graded plays "
                  f"(possible name mismatch): {zero_graded[:5]}")


def main():
    print(f"\n  PropEdge {VERSION} — generate_season_json.py")

    # ── Lock file — prevent concurrent generate runs ─────────────────────
    lock = FILE_SEASON_2526.parent / ".generate.lock"
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        if age < 3600:  # lock younger than 1 hour = another process running
            print(f"  ✗ generate already running (lock file {age:.0f}s old). "
                  f"Delete {lock} if this is stale.")
            return
    lock.write_text(str(time.time()))
    try:
        _run_generate_locked()
    finally:
        lock.unlink(missing_ok=True)


def _run_generate_locked() -> None:
    """Inner generate — called by main() after acquiring lock."""
    import shutil

    # ── Backup existing season JSONs before overwriting ────────────────────
    for f in (FILE_SEASON_2425, FILE_SEASON_2526):
        if f.exists():
            bak = f.with_suffix(".bak")
            shutil.copy2(f, bak)
            print(f"  ✓ Backed up {f.name} → {bak.name}")

    # [1] Load game logs
    print("\n[1/5] Loading game logs...")
    dfs = []
    for fp in (FILE_GL_2425, FILE_GL_2526):
        if fp.exists():
            try:
                dfs.append(pd.read_csv(fp, parse_dates=["GAME_DATE"], low_memory=False))
            except Exception as e:
                print(f"  ⚠ {fp.name}: {e}")
        else:
            print(f"  ⚠ Missing: {fp.name}")
    if not dfs:
        print("  ✗ No game log CSVs found. Add to source-files/ and retry.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    played   = filter_played(combined)
    pidx     = build_player_index(played)
    print(f"  {len(played):,} played rows | {len(pidx):,} players")

    # [2] Caches
    print("\n[2/5] Building caches...")
    dvp       = build_dynamic_dvp(played)
    pace      = build_pace_rank(played)
    otr, ovr  = build_opp_def_caches(played)
    rmap      = build_rest_days_map(played)

    # [3] H2H
    print("\n[3/5] Loading H2H...")
    h2h = {}
    if FILE_H2H.exists():
        try:
            df2 = pd.read_csv(FILE_H2H, low_memory=False)
            h2h = {
                (_norm(str(r.get("PLAYER_NAME", ""))),
                 str(r.get("OPPONENT", "")).strip().upper()): r.to_dict()
                for _, r in df2.iterrows()
            }
            print(f"  H2H pairs: {len(h2h):,}")
        except Exception as e:
            print(f"  ⚠ H2H load failed: {e}")
    else:
        print("  ⚠ h2h_database.csv not found — run: python3 run.py 0 (or h2h_builder.py)")

    kwargs = dict(
        pidx=pidx, played=played, combined_df=combined,
        h2h_lkp=h2h, dvp=dvp, pace=pace, otr=otr, ovr=ovr, rmap=rmap,
    )

    # [4] 2024-25 season
    print("\n[4/5] 2024-25 season JSON...")
    props_2425 = load_props_for_season(
        FILE_PROPS_2425, "2024-10-01", "2025-09-30", "2024-25"
    )
    if not props_2425:
        print("  ✗ Cannot build 2024-25 JSON — no props loaded. "
              "Ensure PropEdge_2024_25_Match_and_Player_Prop_lines.xlsx is in source-files/")
    else:
        # Load existing season JSON to lock direction for already-graded plays
        # This prevents direction from flipping when game log updates change prb12
        prev_2425: dict[tuple, str] = {}
        if FILE_SEASON_2425.exists():
            try:
                prev_plays = json.loads(FILE_SEASON_2425.read_text())
                prev_2425 = {
                    (p.get("player",""), p.get("date","")): p.get("direction", p.get("dir","OVER"))
                    for p in prev_plays
                    if p.get("result") in ("WIN","LOSS","DNP","PUSH")
                }
                print(f"  ✓ Locked directions from existing JSON: {len(prev_2425):,} plays")
            except Exception as e:
                print(f"  ⚠ Could not load existing 2024-25 JSON for direction lock: {e}")
        plays_2425 = score_and_grade(props_2425, season="2024-25",
                                     prev_directions=prev_2425, **kwargs)
        plays_2425.sort(key=lambda p: (p.get("date", ""), p.get("player", "")))
        FILE_SEASON_2425.parent.mkdir(parents=True, exist_ok=True)
        with open(FILE_SEASON_2425, "w") as f:
            json.dump(clean_json(plays_2425), f, indent=2)
        _print_season_stats("2024-25", plays_2425)
        # Monthly split — replaces single large file for GitHub Pages
        counts_2425 = write_monthly_split(plays_2425, "2024_25")
        ok, msg = verify_monthly_integrity("2024_25", plays_2425)
        if ok:
            print(f"  ✓ Monthly split 2024-25: {len(counts_2425)} files | {msg}")
        else:
            print(f"  ⚠ Monthly split verification: {msg}")

    # [5] 2025-26 season
    print("\n[5/5] 2025-26 season JSON...")
    props_2526 = load_props_for_season(
        FILE_PROPS, "2025-10-01", "2026-09-30", "2025-26"
    )
    if not props_2526:
        print("  ✗ Cannot build 2025-26 JSON — no props loaded. "
              "Ensure PropEdge_-_Match_and_Player_Prop_lines_.xlsx is in source-files/")
    else:
        prev_2526: dict[tuple, str] = {}
        if FILE_SEASON_2526.exists():
            try:
                prev_plays = json.loads(FILE_SEASON_2526.read_text())
                prev_2526 = {
                    (p.get("player",""), p.get("date","")): p.get("direction", p.get("dir","OVER"))
                    for p in prev_plays
                    if p.get("result") in ("WIN","LOSS","DNP","PUSH")
                }
                print(f"  ✓ Locked directions from existing JSON: {len(prev_2526):,} plays")
            except Exception as e:
                print(f"  ⚠ Could not load existing 2025-26 JSON for direction lock: {e}")
        plays_2526 = score_and_grade(props_2526, season="2025-26",
                                     prev_directions=prev_2526, **kwargs)
        plays_2526.sort(key=lambda p: (p.get("date", ""), p.get("player", "")))
        FILE_SEASON_2526.parent.mkdir(parents=True, exist_ok=True)
        with open(FILE_SEASON_2526, "w") as f:
            json.dump(clean_json(plays_2526), f, indent=2)
        _print_season_stats("2025-26", plays_2526)
        # Monthly split
        counts_2526 = write_monthly_split(plays_2526, "2025_26")
        ok, msg = verify_monthly_integrity("2025_26", plays_2526)
        if ok:
            print(f"  ✓ Monthly split 2025-26: {len(counts_2526)} files | {msg}")
        else:
            print(f"  ⚠ Monthly split verification: {msg}")

    # ── Combined summary ──────────────────────────────────────────────────
    all_plays = (plays_2425 or []) + (plays_2526 or [])
    if all_plays:
        print("\n  ── Combined (both seasons) ──")
        _print_season_stats("COMBINED", all_plays)
        try:
            write_ml_dataset(all_plays)
        except Exception as e:
            print(f"  ⚠ ML Dataset write failed: {e}")

    # Push both season JSONs + today + dvp to GitHub Pages
    try:
        from git_push import push
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        push(f"generate {ts}", generate=True)
        print(f"  ✓ Data files pushed to GitHub Pages")
    except Exception as e:
        print(f"  ⚠ GitHub push failed: {e}")
        print(f"     Run: python3 run.py sync then commit via GitHub Desktop")

    print(f"\n  ✓ Done. Next: python3 run.py retrain")


if __name__ == "__main__":
    main()
