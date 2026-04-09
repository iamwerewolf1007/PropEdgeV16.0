"""
PropEdge V16.0 — batch0_grade.py
B0 job: runs at 07:00 UK.

Actions:
  1. Fetch yesterday's box scores from NBA API (ScoreboardV3 + BoxScoreTraditionalV3)
  2. Grade all open plays in season_2025_26.json and today.json
  3. Append new game rows to nba_gamelogs_2025_26.csv (keeps rolling stats current)
  4. Write post-match analysis (lossType, postMatchReason, delta)
  5. Rebuild H2H database
  6. Monthly trigger: retrain Elite V2 GBT on all 2025-26 graded plays
  7. Git push

Grading notes:
  - WIN  : direction correct AND actual_pts > line  (OVER)
             or direction correct AND actual_pts <= line (UNDER)
  - LOSS : direction wrong
  - DNP  : player not in box score or played 0 minutes
  - PUSH : actual_pts == line exactly
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    VERSION, FILE_SEASON_2526, FILE_SEASON_2425, FILE_TODAY,
    FILE_GL_2526, FILE_GL_2425, FILE_H2H, FILE_ELITE_MODEL,
    ELITE_FEATURES, clean_json, uk_now,
)
from player_name_aliases import _norm, resolve_grade_name
from reasoning_engine import generate_post_match_reason
from ml_dataset import append_ml_dataset
from monthly_split import update_month


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
# _norm and resolve_grade_name imported from player_name_aliases
# classify_loss and post_reason replaced by reasoning_engine.generate_post_match_reason

def _si(v):
    try: return int(v) if pd.notna(v) else 0
    except: return 0

def _parse_min(v) -> float:
    s = str(v).strip()
    if s in ('', 'None', 'nan', '0', 'PT00M00.00S'): return 0.0
    if s.startswith('PT') and 'M' in s:
        m = re.match(r'PT(\d+)M([\d.]+)S', s)
        return float(m.group(1)) + float(m.group(2)) / 60 if m else 0.0
    if ':' in s:
        p = s.split(':'); return float(p[0]) + float(p[1]) / 60
    try: return float(s)
    except: return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FETCH BOX SCORES  (V3 endpoints — correct for 2025-26 season)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_box_scores(date_str: str) -> tuple[list[dict], dict[str, float]]:
    """
    Fetch NBA box scores using V3 endpoints (correct for 2025-26).

    Returns:
      played_rows   — list of full stat row dicts (one per player who played)
      results_map   — {normalised_player_name: pts} for grading lookups
      players_in_box— set of all player names seen (incl. DNP-in-game)

    Uses:
      ScoreboardV3          → game IDs + context for the date
      BoxScoreTraditionalV3 → player-level box scores per game

    Root cause of V2 failure: BoxScoreTraditionalV2 data is no longer
    published for the 2025-26 season (deprecated). V3 is the current endpoint.
    """
    print(f"\n  Fetching box scores: {date_str}...")

    played_rows:    list[dict] = []
    players_in_box: set[str]  = set()
    results_map:    dict[str, float] = {}

    # ── Load bio cache from existing game log ──────────────────────────────
    bio: dict[int, dict] = {}
    if FILE_GL_2526.exists():
        try:
            df26 = pd.read_csv(FILE_GL_2526, low_memory=False)
            bc = ['PLAYER_ID','PLAYER_NAME','PLAYER_POSITION','PLAYER_POSITION_FULL',
                  'PLAYER_CURRENT_TEAM','GAME_TEAM_ABBREVIATION','GAME_TEAM_NAME',
                  'PLAYER_HEIGHT','PLAYER_WEIGHT','PLAYER_EXPERIENCE','PLAYER_COUNTRY',
                  'PLAYER_DRAFT_YEAR','PLAYER_DRAFT_ROUND','PLAYER_DRAFT_NUMBER']
            bc_avail = [c for c in bc if c in df26.columns]
            for _, r in df26.drop_duplicates('PLAYER_ID', keep='last')[bc_avail].iterrows():
                try: bio[int(r['PLAYER_ID'])] = r.to_dict()
                except: pass
        except Exception as e:
            print(f"  ⚠ Bio cache load: {e}")

    # ── Step 1: ScoreboardV3 → game IDs + context (3 retries) ────────────────
    game_ids: list[str] = []
    ctx:      dict[str, dict] = {}
    try:
        from nba_api.stats.endpoints import ScoreboardV3
        for attempt in range(3):
            try:
                time.sleep(1 + attempt * 2)  # 1s, 3s, 5s
                sb = ScoreboardV3(game_date=date_str, league_id='00')
                gh = sb.game_header.get_data_frame()
                ls = sb.line_score.get_data_frame()
                if gh.empty:
                    print("  No games found for date.")
                    return played_rows, results_map, players_in_box
                game_ids = gh['gameId'].tolist()
                print(f"  ScoreboardV3: {len(game_ids)} games")
                for g in game_ids:
                    r = ls[ls['gameId'] == g]
                    if len(r) >= 2:
                        ctx[str(g)] = {
                            'htid': r.iloc[0]['teamId'],
                            'ht':   r.iloc[0]['teamTricode'],
                            'at':   r.iloc[1]['teamTricode'],
                            'hs':   _si(r.iloc[0].get('score', 0)),
                            'as_':  _si(r.iloc[1].get('score', 0)),
                        }
                break  # success
            except Exception as e:
                print(f"  ⚠ ScoreboardV3 attempt {attempt+1}/3: {e}")
                if attempt == 2:
                    raise
    except Exception as e:
        print(f"  ⚠ ScoreboardV3 failed after 3 attempts: {e}")

    if not game_ids:
        print("  ⚠ No game IDs found — cannot fetch box scores.")
        return played_rows, results_map, players_in_box

    # ── Step 2: BoxScoreTraditionalV3 per game (3 retries each) ───────────────
    try:
        from nba_api.stats.endpoints import BoxScoreTraditionalV3
    except ImportError:
        print("  ⚠ BoxScoreTraditionalV3 not available in this nba_api version.")
        return played_rows, results_map, players_in_box

    for g in game_ids:
        ps = None
        for attempt in range(3):
            try:
                time.sleep(0.8 + attempt * 2)  # 0.8s, 2.8s, 4.8s
                box = BoxScoreTraditionalV3(game_id=g)
                ps  = box.player_stats.get_data_frame()
                break  # success
            except Exception as e:
                print(f"  ⚠ BoxScore game {g} attempt {attempt+1}/3: {e}")
                if attempt == 2:
                    print(f"  ✗ Skipping game {g} after 3 failed attempts")
        if ps is None or ps.empty:
            continue

        try:
            # Rename V3 column names → our standard names
            col_map = {
                'personId':'PLAYER_ID', 'teamId':'TEAM_ID',
                'teamTricode':'TEAM_ABBREVIATION',
                'firstName':'FN', 'familyName':'LN', 'minutes':'MR',
                'fieldGoalsMade':'FGM', 'fieldGoalsAttempted':'FGA',
                'threePointersMade':'FG3M', 'threePointersAttempted':'FG3A',
                'freeThrowsMade':'FTM', 'freeThrowsAttempted':'FTA',
                'reboundsOffensive':'OREB', 'reboundsDefensive':'DREB',
                'reboundsTotal':'REB', 'assists':'AST', 'steals':'STL',
                'blocks':'BLK', 'turnovers':'TOV', 'foulsPersonal':'PF',
                'points':'PTS', 'plusMinusPoints':'PLUS_MINUS',
            }
            ps = ps.rename(columns={k: v for k, v in col_map.items() if k in ps.columns})
            if 'PLAYER_NAME' not in ps.columns and 'FN' in ps.columns:
                ps['PLAYER_NAME'] = (
                    ps['FN'].fillna('').str.strip() + ' ' +
                    ps['LN'].fillna('').str.strip()
                ).str.strip()

            c = ctx.get(str(g), {})

            for _, p in ps.iterrows():
                pname = str(p.get('PLAYER_NAME', '')).strip()
                if pname:
                    players_in_box.add(pname)

                mn = _parse_min(p.get('MR', 0))
                if mn <= 0:
                    continue  # DNP in game — not a played row

                pid = _si(p.get('PLAYER_ID', 0))
                tid = _si(p.get('TEAM_ID', 0))
                ta  = str(p.get('TEAM_ABBREVIATION', ''))
                ih  = 1 if tid == c.get('htid') else 0
                opp_rows = ps[ps['TEAM_ID'] != tid]['TEAM_ABBREVIATION']
                opp  = opp_rows.iloc[0] if len(opp_rows) > 0 else 'UNK'
                mu   = f"{ta} vs. {opp}" if ih else f"{ta} @ {opp}"
                wl   = ('W' if c.get('hs', 0) > c.get('as_', 0) else 'L') if ih \
                       else ('W' if c.get('as_', 0) > c.get('hs', 0) else 'L')

                pts  = _si(p.get('PTS', 0));  fgm = _si(p.get('FGM', 0)); fga = _si(p.get('FGA', 0))
                fg3m = _si(p.get('FG3M', 0)); fg3a= _si(p.get('FG3A', 0))
                ftm  = _si(p.get('FTM', 0));  fta = _si(p.get('FTA', 0))
                oreb = _si(p.get('OREB', 0)); dreb= _si(p.get('DREB', 0)); reb = _si(p.get('REB', 0))
                ast  = _si(p.get('AST', 0));  stl = _si(p.get('STL', 0));  blk = _si(p.get('BLK', 0))
                tov  = _si(p.get('TOV', 0));  pf  = _si(p.get('PF', 0));   pm  = _si(p.get('PLUS_MINUS', 0))

                fgp = fgm / fga  if fga  > 0 else 0.0
                f3p = fg3m / fg3a if fg3a > 0 else 0.0
                ftp = ftm / fta  if fta  > 0 else 0.0
                tsa = 2 * (fga + 0.44 * fta)
                ts  = pts / tsa  if tsa  > 0 else 0.0
                usg = (fga + 0.44 * fta + tov) / (mn / 5) if mn > 0 else 0.0
                pra = pts + reb + ast
                ddc = sum(1 for x in [pts, reb, ast, stl, blk] if x >= 10)
                dd  = 1 if ddc >= 2 else 0; td = 1 if ddc >= 3 else 0
                fp  = pts + 1.25*reb + 1.5*ast + 2*stl + 2*blk - 0.5*tov + 0.5*fg3m + 1.5*dd + 3*td
                b   = bio.get(pid, {})

                row = {
                    'PLAYER_ID':   pid,
                    'PLAYER_NAME': pname or b.get('PLAYER_NAME', ''),
                    'SEASON': '2025-26', 'SEASON_TYPE': 'Regular Season',
                    'PLAYER_POSITION':      b.get('PLAYER_POSITION', ''),
                    'PLAYER_POSITION_FULL': b.get('PLAYER_POSITION_FULL', ''),
                    'PLAYER_CURRENT_TEAM':  b.get('PLAYER_CURRENT_TEAM', ta),
                    'GAME_TEAM_ABBREVIATION': ta,
                    'GAME_TEAM_NAME': b.get('GAME_TEAM_NAME', ''),
                    'PLAYER_HEIGHT':       b.get('PLAYER_HEIGHT', ''),
                    'PLAYER_WEIGHT':       b.get('PLAYER_WEIGHT', 0),
                    'PLAYER_EXPERIENCE':   b.get('PLAYER_EXPERIENCE', 0),
                    'PLAYER_COUNTRY':      b.get('PLAYER_COUNTRY', ''),
                    'PLAYER_DRAFT_YEAR':   b.get('PLAYER_DRAFT_YEAR', 0),
                    'PLAYER_DRAFT_ROUND':  b.get('PLAYER_DRAFT_ROUND', 0),
                    'PLAYER_DRAFT_NUMBER': b.get('PLAYER_DRAFT_NUMBER', 0),
                    'GAME_ID': int(g), 'GAME_DATE': date_str,
                    'MATCHUP': mu, 'OPPONENT': opp,
                    'IS_HOME': ih, 'WL': wl,
                    'WL_WIN': 1 if wl == 'W' else 0,
                    'WL_LOSS': 1 if wl == 'L' else 0,
                    'MIN': int(round(mn)), 'MIN_NUM': round(mn, 1),
                    'FGM': fgm, 'FGA': fga, 'FG_PCT': round(fgp, 4),
                    'FG3M': fg3m, 'FG3A': fg3a, 'FG3_PCT': round(f3p, 4),
                    'FTM': ftm, 'FTA': fta, 'FT_PCT': round(ftp, 4),
                    'OREB': oreb, 'DREB': dreb, 'REB': reb, 'AST': ast,
                    'STL': stl, 'BLK': blk, 'TOV': tov, 'PF': pf,
                    'PTS': pts, 'PLUS_MINUS': pm,
                    'TRUE_SHOOTING_PCT': round(ts, 4),
                    'USAGE_APPROX': round(usg, 2),
                    'PTS_REB_AST': pra, 'DOUBLE_DOUBLE': dd, 'TRIPLE_DOUBLE': td,
                    'FANTASY_PTS': round(fp, 2),
                    'SEASON_ID': 22025, 'DNP': 0,
                }
                played_rows.append(row)
                # Build normalised results map for grading
                pnorm = _norm(pname)
                results_map[pnorm] = float(pts)

        except Exception as e:
            print(f"  ⚠ BoxScore game {g}: {e}")

    print(f"  Fetched {len(played_rows)} played rows, {len(players_in_box)} players in box")

    # ── CSV fallback if API returned nothing ───────────────────────────────
    if not played_rows and FILE_GL_2526.exists():
        try:
            df = pd.read_csv(FILE_GL_2526, parse_dates=["GAME_DATE"], low_memory=False)
            day = df[df["GAME_DATE"].dt.date == pd.Timestamp(date_str).date()]
            for _, r in day.iterrows():
                pnorm = _norm(str(r.get("PLAYER_NAME", "")))
                pts   = r.get("PTS")
                if pnorm and pts is not None:
                    try: results_map[pnorm] = float(pts)
                    except: pass
            print(f"  CSV fallback: {len(results_map)} player scores")
        except Exception as e:
            print(f"  ⚠ CSV fallback: {e}")

    return played_rows, results_map, players_in_box


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING STAT RECOMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def _recompute_rolling(df: pd.DataFrame, players: set[str]) -> pd.DataFrame:
    """
    Recompute pre-computed rolling columns for the given players.
    Called after append_gamelogs so that batch_predict has fresh L3/L5/L10/L30
    values for every player who played yesterday.

    Columns recomputed (matching extract_features() expectations):
      L3_PTS, L5_PTS, L10_PTS, L20_PTS, L30_PTS
      L10_MIN_NUM, L30_MIN_NUM, L3_MIN_NUM
      L10_FGA, L10_FG3A, L10_FG3M, L10_FTA, L10_FT_PCT
      L10_USAGE_APPROX, L30_USAGE_APPROX
    """
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    def _roll_mean(arr: pd.Series, n: int) -> pd.Series:
        """Expanding then rolling mean — uses all available data up to n games."""
        return arr.rolling(window=n, min_periods=1).mean()

    for pname in players:
        mask = df['PLAYER_NAME'] == pname
        if mask.sum() == 0:
            continue

        grp = df[mask].sort_values('GAME_DATE').copy()
        idx = grp.index

        # Only use rows where player actually played (MIN_NUM > 0, not DNP)
        played_mask = (grp['MIN_NUM'].fillna(0) > 0) & (grp['DNP'].fillna(0) == 0)

        # PTS rolling — compute on played rows, forward-fill for DNP rows
        pts = grp['PTS'].copy()
        # For rolling, shift(1) so current game doesn't count (prior games only)
        pts_prior = pts.where(played_mask).shift(1)
        grp['L3_PTS']  = _roll_mean(pts_prior, 3)
        grp['L5_PTS']  = _roll_mean(pts_prior, 5)
        grp['L10_PTS'] = _roll_mean(pts_prior, 10)
        grp['L20_PTS'] = _roll_mean(pts_prior, 20)
        grp['L30_PTS'] = _roll_mean(pts_prior, 30)

        # MIN_NUM rolling
        min_prior = grp['MIN_NUM'].where(played_mask).shift(1)
        grp['L3_MIN_NUM']  = _roll_mean(min_prior, 3)
        grp['L10_MIN_NUM'] = _roll_mean(min_prior, 10)
        grp['L30_MIN_NUM'] = _roll_mean(min_prior, 30)

        # Shooting rolling (FGA, FG3A, FG3M, FTA, FT_PCT, USAGE_APPROX)
        for src_col, dst_col, n in [
            ('FGA',           'L10_FGA',           10),
            ('FG3A',          'L10_FG3A',          10),
            ('FG3M',          'L10_FG3M',          10),
            ('FTA',           'L10_FTA',           10),
            ('FT_PCT',        'L10_FT_PCT',        10),
            ('USAGE_APPROX',  'L10_USAGE_APPROX',  10),
            ('USAGE_APPROX',  'L30_USAGE_APPROX',  30),
        ]:
            if src_col in grp.columns:
                col_prior = grp[src_col].where(played_mask).shift(1)
                grp[dst_col] = _roll_mean(col_prior, n)

        # Write back ONLY the rolling columns — never overwrite raw stat columns
        # (df.loc[idx, grp.columns] = grp.values would write ALL cols with numpy
        #  dtypes, risking silent corruption of int/bool columns.)
        _ROLL_COLS = [
            'L3_PTS', 'L5_PTS', 'L10_PTS', 'L20_PTS', 'L30_PTS',
            'L3_MIN_NUM', 'L10_MIN_NUM', 'L30_MIN_NUM',
            'L10_FGA', 'L10_FG3A', 'L10_FG3M', 'L10_FTA',
            'L10_FT_PCT', 'L10_USAGE_APPROX', 'L30_USAGE_APPROX',
        ]
        cols = [c for c in _ROLL_COLS if c in df.columns and c in grp.columns]
        if cols:
            df.loc[idx, cols] = grp[cols].values

    return df


# ─────────────────────────────────────────────────────────────────────────────
# APPEND GAME LOGS  (keeps nba_gamelogs_2025_26.csv current for rolling stats)
# ─────────────────────────────────────────────────────────────────────────────
def append_gamelogs(played_rows: list[dict], dnp_player_names: list[str], date_str: str) -> None:
    """
    Append played rows + DNP stubs to nba_gamelogs_2025_26.csv.
    Critical: without this, rolling stats (L10/L30) go stale and
    batch_predict uses outdated feature windows.
    """
    if not FILE_GL_2526.exists():
        print("  ⚠ Game log CSV missing — cannot append"); return

    try:
        df26 = pd.read_csv(FILE_GL_2526, parse_dates=['GAME_DATE'], low_memory=False)
        rows_before = len(df26)
        if 'DNP' not in df26.columns: df26['DNP'] = 0

        # Build DNP stubs
        dnp_stubs = []
        for pname in dnp_player_names:
            stub = {c: np.nan for c in df26.columns}
            stub.update({
                'PLAYER_NAME': pname, 'GAME_DATE': date_str,
                'DNP': 1, 'MIN_NUM': 0, 'PTS': np.nan,
                'SEASON': '2025-26', 'SEASON_TYPE': 'Regular Season',
                'SEASON_ID': 22025,
            })
            dnp_stubs.append(stub)

        all_new = played_rows + dnp_stubs
        if not all_new:
            print("  No new rows to append"); return

        ndf = pd.DataFrame(all_new)
        ndf['GAME_DATE'] = pd.to_datetime(ndf['GAME_DATE'])
        if 'DNP' not in ndf.columns: ndf['DNP'] = 0

        # Align columns to existing schema — single reindex call (not a loop)
        # The loop pattern triggers a PerformanceWarning for each of the 100+
        # NBA API columns as pandas internally copies the fragmented DataFrame.
        ndf = ndf.reindex(columns=df26.columns)

        updated = pd.concat([df26, ndf], ignore_index=True)
        updated['GAME_DATE'] = pd.to_datetime(updated['GAME_DATE'])
        updated = updated.sort_values(['PLAYER_NAME', 'GAME_DATE']).reset_index(drop=True)
        updated = updated.drop_duplicates(subset=['PLAYER_NAME', 'GAME_DATE'], keep='last')

        # ── Recompute rolling for played AND DNP players ───────────────────────
        # Critical bug if omitted: DNP stub rows are appended with NaN in all
        # rolling columns.  extract_features() reads from the LAST prior row —
        # if that is a DNP stub, L10_PTS falls back to `line` and all features
        # become meaningless for any player who was inactive yesterday.
        new_players = {
            row.get('PLAYER_NAME', '') for row in played_rows if row.get('PLAYER_NAME')
        }
        dnp_players = set(filter(None, dnp_player_names))
        all_new_players = new_players | dnp_players
        if all_new_players:
            updated = _recompute_rolling(updated, all_new_players)

        updated.to_csv(FILE_GL_2526, index=False)
        print(f"  ✓ Game log: {rows_before} → {len(updated)} rows "
              f"(+{len(played_rows)} played, +{len(dnp_stubs)} DNP stubs)"
              f" | Rolling recomputed: {len(new_players)} played + {len(dnp_players)} DNP")

    except Exception as e:
        print(f"  ⚠ append_gamelogs failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# GRADE PLAYS
# ─────────────────────────────────────────────────────────────────────────────
def grade_plays(
    plays: list[dict],
    results_map: dict[str, float],
    players_in_box: set[str],
    date_str: str,
    played_rows: list[dict] | None = None,
) -> tuple[list[dict], int, int, int]:
    """
    Grade plays for date_str. Returns (updated_plays, wins, losses, dnps).

    results_map     — {normalised_player_name: pts}  built from box scores
    players_in_box  — set of all player names seen in box (incl. 0-min DNPs)
    played_rows     — full box stat rows (optional) used by reasoning engine
                      for rich post-match narrative (MIN, FGA, FGM etc.)
    """
    # Build player → full box stats lookup for reasoning engine
    box_stats: dict[str, dict] = {}
    if played_rows:
        for row in played_rows:
            pname = str(row.get("PLAYER_NAME", ""))
            if pname:
                box_stats[_norm(pname)] = row

    for p in plays:
        if p.get("date") != date_str:
            continue
        if p.get("result") in ("WIN", "LOSS", "DNP", "PUSH"):
            continue  # immutable once graded

        pname = p.get("player", "")
        line  = float(p.get("line", 0))
        # Normalise direction — strip legacy LEAN prefix
        dir_  = str(p.get("dir", "")).upper().replace("LEAN ", "")

        # ── Look up actual score ───────────────────────────────────────────
        pnorm = _norm(pname)
        actual_pts: float | None = results_map.get(pnorm)
        if actual_pts is None:
            actual_pts = resolve_grade_name(pname, results_map)

        # Player not in box at all → DNP
        if actual_pts is None:
            # Diagnostic: show what names were in results_map that are close
            close = [k for k in results_map if k[:4] == _norm(pname)[:4]]
            if close:
                print(f"  ⚠ DNP: '{pname}' not found — close matches in box: {close[:3]}")
            p["result"] = "DNP"
            p["actualPts"] = None
            p["lossType"] = None
            p["postMatchReason"] = "🚫 No score found — assumed DNP or game not played."
            continue

        # ── Grade ─────────────────────────────────────────────────────────
        # WIN = model's direction call was CORRECT
        #   OVER predicted → WIN if actual > line
        #   UNDER predicted → WIN if actual <= line
        # dir_ is the model's stored direction from when the play was predicted.
        if abs(actual_pts - line) < 0.05:
            result = "PUSH"
        elif "OVER" in dir_:
            result = "WIN" if actual_pts > line else "LOSS"
        elif "UNDER" in dir_:
            result = "WIN" if actual_pts <= line else "LOSS"
        else:
            result = "WIN" if actual_pts > line else "LOSS"  # fallback to OVER

        p["result"]    = result
        p["actualPts"] = actual_pts
        p["delta"]     = round(actual_pts - line, 1)

        # ── Rich post-match narrative via reasoning_engine ─────────────────
        # Build box_data dict with actual game stats for this player
        brow = box_stats.get(pnorm) or box_stats.get(_norm(pname), {})
        box_data = {
            "actual_pts": actual_pts,
            "actual_min": float(brow.get("MIN_NUM", brow.get("MIN", 0)) or 0),
            "actual_fga": float(brow.get("FGA", 0) or 0),
            "actual_fgm": float(brow.get("FGM", 0) or 0),
        }
        post_narrative, loss_type = generate_post_match_reason(p, box_data)
        p["lossType"]        = loss_type if result == "LOSS" else None
        p["postMatchReason"] = post_narrative
        # Store box stats directly on play for ML dataset
        p["actual_min"] = box_data.get("actual_min")
        p["actual_fga"] = box_data.get("actual_fga")
        p["actual_fgm"] = box_data.get("actual_fgm")
        fga = box_data.get("actual_fga") or 0
        fgm = box_data.get("actual_fgm") or 0
        p["actual_fg_pct"] = round(fgm / fga * 100, 1) if fga > 0 else None

    # Single clean recount
    wins = losses = dnps = 0
    for p in plays:
        if p.get("date") != date_str: continue
        r = p.get("result")
        if r == "WIN":    wins   += 1
        elif r == "LOSS": losses += 1
        elif r == "DNP":  dnps   += 1

    return plays, wins, losses, dnps


# ─────────────────────────────────────────────────────────────────────────────
# SEASON JSON UPDATE
# ─────────────────────────────────────────────────────────────────────────────
def update_season_json(graded_plays: list[dict], date_str: str) -> None:
    existing: list[dict] = []
    if FILE_SEASON_2526.exists():
        try:
            with open(FILE_SEASON_2526) as f: existing = json.load(f)
        except: pass

    # Remove ALL ungraded plays for this date (will be replaced by freshly graded)
    kept = [p for p in existing if not (
        p.get("date") == date_str and
        p.get("result") not in ("WIN", "LOSS", "DNP", "PUSH")
    )]

    # Build dedup index keyed on (player, date) only — NOT line.
    # This handles line changes: if a player has two graded entries for the same
    # date (old line 25.5 graded WIN, new line 26.0 graded LOSS), keep the one
    # with the highest elite_prob (most recent confident prediction).
    # In practice this only occurs when predict runs twice with a moved line.
    graded_by_player_date: dict[tuple, dict] = {}
    for p in kept:
        if p.get("result") in ("WIN", "LOSS", "DNP", "PUSH"):
            k = (p.get("player"), p.get("date"))
            existing_p = graded_by_player_date.get(k)
            if existing_p is None or \
               float(p.get("elite_prob", 0)) >= float(existing_p.get("elite_prob", 0)):
                graded_by_player_date[k] = p

    # Merge newly graded plays — new graded result always wins over old
    for p in graded_plays:
        if p.get("date") != date_str: continue
        k = (p.get("player"), p.get("date"))
        graded_by_player_date[k] = p  # freshly graded always takes priority

    # Rebuild kept: non-date-str plays + deduplicated graded plays for this date
    other_plays = [p for p in kept if p.get("date") != date_str]
    merged = other_plays + list(graded_by_player_date.values())
    merged.sort(key=lambda p: (p.get("date", ""), p.get("player", "")))

    FILE_SEASON_2526.parent.mkdir(parents=True, exist_ok=True)
    with open(FILE_SEASON_2526, "w") as f:
        json.dump(clean_json(merged), f, indent=2)
    print(f"  ✓ season_2025_26.json → {len(merged):,} plays total")


# ─────────────────────────────────────────────────────────────────────────────
# MONTHLY RETRAIN
# ─────────────────────────────────────────────────────────────────────────────
def should_retrain(date_str: str) -> bool:
    return date_str[8:10] == "01"

def retrain_elite_model() -> None:
    print("\n  [RETRAIN] Triggering model_trainer.py...")
    try:
        import subprocess
        r = subprocess.run(
            [sys.executable, str(ROOT / "model_trainer.py")],
            cwd=ROOT, timeout=1800,
        )
        print("  ✓ Retrain complete" if r.returncode == 0 else f"  ✗ Retrain failed (rc={r.returncode})")
    except Exception as e:
        print(f"  ✗ Retrain error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# GIT PUSH
# ─────────────────────────────────────────────────────────────────────────────
def git_push(date_str: str) -> None:
    from git_push import push
    push(f"B0 grade {date_str} — {datetime.now(timezone.utc).strftime('%H:%M')} UTC",
         grade=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run_grade(date_str: str | None = None) -> None:
    import time as _time
    if date_str is None:
        date_str = (uk_now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # ── Lock file — prevent two B0 runs simultaneously ──────────────────────
    lock = FILE_SEASON_2526.parent / ".b0.lock"
    if lock.exists():
        age = _time.time() - lock.stat().st_mtime
        if age < 1800:  # < 30 min = another B0 still running
            print(f"  ⚠ B0 lock found ({age:.0f}s old) — skipping {date_str}. "
                  f"Delete {lock} manually if stale.")
            return
    lock.write_text(str(_time.time()))
    try:
        _run_grade_locked(date_str)
    finally:
        lock.unlink(missing_ok=True)


def _run_grade_locked(date_str: str) -> None:
    print(f"\n  PropEdge {VERSION} — B0 Grade  ({date_str})")

    # 1. Fetch box scores (ScoreboardV3 + BoxScoreTraditionalV3)
    print("[1/4] Fetching box scores...")
    played_rows, results_map, players_in_box = fetch_box_scores(date_str)
    if not results_map:
        print(f"  No scores found for {date_str}. Skipping grade.")
        return

    # 2. Load plays to grade
    print("[2/4] Loading plays to grade...")
    plays: list[dict] = []
    if FILE_TODAY.exists():
        try:
            with open(FILE_TODAY) as f: plays = json.load(f)
        except: pass

    # Also check season JSON for any plays not yet in today.json
    if FILE_SEASON_2526.exists():
        try:
            with open(FILE_SEASON_2526) as f: season_plays = json.load(f)
            existing_keys = {(p.get("player"), p.get("date"), str(p.get("line"))) for p in plays}
            for p in season_plays:
                if p.get("date") == date_str:
                    k = (p.get("player"), p.get("date"), str(p.get("line")))
                    if k not in existing_keys:
                        plays.append(p)
        except: pass

    to_grade = [p for p in plays if p.get("date") == date_str
                and p.get("result") not in ("WIN", "LOSS", "DNP", "PUSH")]
    print(f"  Plays to grade: {len(to_grade):,} | Total loaded: {len(plays):,}")

    wins = losses = dnps = 0
    graded_today: list[dict] = []

    # [3] Grade (only if there are ungraded plays — but we always continue to step 4)
    if to_grade:
        print("[3/4] Grading...")
        plays, wins, losses, dnps = grade_plays(
            plays, results_map, players_in_box, date_str, played_rows
        )
        total_graded = wins + losses + dnps
        hr = f"{wins/total_graded*100:.1f}%" if total_graded > 0 else "—"
        print(f"  Graded: {total_graded} | WIN: {wins} | LOSS: {losses} | DNP: {dnps} | HR: {hr}")

        # Update today.json with graded results
        FILE_TODAY.parent.mkdir(parents=True, exist_ok=True)
        with open(FILE_TODAY, "w") as f:
            json.dump(clean_json(plays), f, indent=2)

        # Update season JSON
        graded_today = [
            p for p in plays
            if p.get("date") == date_str and p.get("result") in ("WIN", "LOSS", "DNP", "PUSH")
        ]
        update_season_json(graded_today, date_str)

        # ── Monthly file update — keeps GitHub Pages monthly files current ─
        try:
            month_str = date_str[:7]  # '2026-04'
            update_month(graded_today, "2025_26", month_str)
        except Exception as e:
            print(f"  ⚠ Monthly file update failed: {e}")

        # ── ML Dataset — append today's graded plays ──────────────────────
        try:
            append_ml_dataset(graded_today, date_str)
        except Exception as e:
            print(f"  ⚠ ML Dataset append failed: {e}")
    else:
        print("  Nothing to grade — box scores will still be appended to game log.")

    # [4] Append game logs — ALWAYS runs so rolling stats stay current for
    #     every player who played, not just those with open props.
    print("[4/4] Appending game logs + rebuilding caches...")
    dnp_player_names = [
        p.get("player", "") for p in graded_today if p.get("result") == "DNP"
    ]
    if played_rows:
        append_gamelogs(played_rows, dnp_player_names, date_str)
    else:
        print("  No played rows fetched — game log unchanged.")

    # Rebuild DVP rankings from updated game log
    try:
        from dvp_updater import compute_and_save_dvp
        from config import FILE_DVP
        compute_and_save_dvp(FILE_GL_2526, FILE_DVP)
        print("  ✓ DVP rebuilt")
    except Exception as e:
        print(f"  ⚠ DVP refresh: {e}")

    # Rebuild H2H database from updated game log
    try:
        from h2h_builder import build_h2h
        build_h2h()
        print("  ✓ H2H rebuilt")
    except Exception as e:
        print(f"  ⚠ H2H rebuild: {e}")

    # Monthly retrain (1st of each month)
    if should_retrain(date_str):
        print("  First of month — triggering retrain...")
        retrain_elite_model()

    # Daily trust score update — cheap to run, high value when current
    # Recomputes per-player direction accuracy from all graded plays.
    # This is the mechanism V12 had that kept tier assignments honest:
    # a player the model consistently gets wrong gets capped at STRONG
    # regardless of Elite prob score.
    try:
        from model_trainer import update_trust_scores
        update_trust_scores()
    except Exception as e:
        print(f"  ⚠ Trust score update failed: {e}")

    git_push(date_str)
    if to_grade:
        print(f"\n  B0 complete. {date_str}  {wins}W / {losses}L / {dnps} DNP")
    else:
        print(f"\n  B0 complete. {date_str}  (game log + caches updated, no new plays graded)")


if __name__ == "__main__":
    override_date = sys.argv[1] if len(sys.argv) > 1 else None
    run_grade(override_date)
