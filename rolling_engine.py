"""
PropEdge V16.0 — rolling_engine.py

The game log CSVs already contain pre-computed rolling stats (L3_PTS, L5_PTS,
L10_PTS, L30_PTS, L10_FGA, L10_FTA, etc.) so we read those directly instead
of recomputing from raw game-by-game values.

Actual CSV column names (confirmed from your files):
  PLAYER_POSITION          — position (not 'POSITION')
  GAME_TEAM_ABBREVIATION   — team abbreviation
  MIN_NUM                  — minutes as float
  OPPONENT, IS_HOME        — already present
  USAGE_APPROX             — already present
  L3_PTS, L5_PTS, L10_PTS, L20_PTS, L30_PTS  — pre-computed rolling PTS
  L10_FGA, L10_FG3A, L10_FTA, L10_FT_PCT     — pre-computed shooting stats
  L10_MIN_NUM, L30_MIN_NUM                    — pre-computed minutes
  L10_USAGE_APPROX                            — pre-computed usage
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from config import V14_ML_FEATURES, season_progress, get_pos_group, MIN_PRIOR_GAMES


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _f(row, col, default=0.0):
    """Safe float read from a row dict or Series."""
    v = row.get(col) if isinstance(row, dict) else (row[col] if col in row.index else default)
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default

def _sm(arr, n):
    sl = arr[-n:] if len(arr) >= n else arr
    return float(np.mean(sl)) if len(sl) > 0 else 0.0

def _std(arr, n):
    sl = arr[-n:] if len(arr) >= n else arr
    return float(np.std(sl)) if len(sl) > 1 else 1.0

def _hr(arr, line, n):
    sl = arr[-n:] if len(arr) >= n else arr
    return float((sl > line).mean()) if len(sl) > 0 else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# FILTER PLAYED
# ─────────────────────────────────────────────────────────────────────────────
def filter_played(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows where player actually played. Normalise GAME_DATE."""
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    mask = (df["MIN_NUM"].fillna(0) > 0) & df["PTS"].notna()
    return df[mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER INDEX  (built once, used for fast lookups)
# ─────────────────────────────────────────────────────────────────────────────
def build_player_index(played: pd.DataFrame) -> dict:
    idx = {}
    for name, grp in played.groupby("PLAYER_NAME"):
        idx[name] = grp.sort_values("GAME_DATE").reset_index(drop=True)
    return idx


def get_prior_games(player_idx: dict, player: str, game_date: str | pd.Timestamp) -> pd.DataFrame:
    """Return all played rows for player strictly before game_date."""
    if player not in player_idx:
        return pd.DataFrame()
    grp = player_idx[player]
    gd  = pd.Timestamp(game_date)
    return grp[grp["GAME_DATE"] < gd].copy()


# ─────────────────────────────────────────────────────────────────────────────
# REST DAYS MAP
# ─────────────────────────────────────────────────────────────────────────────
def build_rest_days_map(played: pd.DataFrame) -> dict:
    """(player, date_str) → rest_days. Vectorised."""
    result = {}
    for name, grp in played.groupby("PLAYER_NAME"):
        grp = grp.sort_values("GAME_DATE").drop_duplicates("GAME_DATE")
        dates_s = grp["GAME_DATE"]
        diffs   = dates_s.diff().dt.days.fillna(7).clip(upper=14).astype(int)
        for gd, rd in zip(dates_s, diffs):
            result[(name, str(pd.Timestamp(gd).date()))] = int(rd)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DVP + PACE CACHES
# ─────────────────────────────────────────────────────────────────────────────
def build_dynamic_dvp(played: pd.DataFrame) -> dict:
    if "OPPONENT" not in played.columns or "PTS" not in played.columns:
        return {}
    pos_col = "PLAYER_POSITION" if "PLAYER_POSITION" in played.columns else None
    recent  = played[played["GAME_DATE"] >= played["GAME_DATE"].max() - pd.Timedelta(days=30)]
    if len(recent) < 100:
        recent = played
    recent  = recent.copy()
    recent["pos_grp"] = recent[pos_col].fillna("G").apply(get_pos_group) if pos_col else "Guard"
    avg_pts = recent.groupby(["OPPONENT","pos_grp"])["PTS"].mean()
    result  = {}
    for pos in ["Guard","Forward","Center"]:
        if pos not in avg_pts.index.get_level_values("pos_grp"):
            continue
        sub  = avg_pts.xs(pos, level="pos_grp")
        rank = sub.rank(ascending=True, method="min").astype(int)
        for team, r in rank.items():
            result[f"{team}|{pos}"] = int(r)
    return result


def build_pace_rank(played: pd.DataFrame) -> dict:
    if "GAME_TEAM_ABBREVIATION" not in played.columns:
        return {}
    recent = played[played["GAME_DATE"] >= played["GAME_DATE"].max() - pd.Timedelta(days=30)]
    if recent.empty:
        recent = played
    totals = recent.groupby("GAME_TEAM_ABBREVIATION")["MIN_NUM"].count()
    return totals.rank(ascending=False, method="min").astype(int).to_dict()


def build_opp_def_caches(played: pd.DataFrame) -> tuple[dict, dict]:
    if "OPPONENT" not in played.columns:
        return {}, {}
    pos_col = "PLAYER_POSITION" if "PLAYER_POSITION" in played.columns else None
    played  = played.copy()
    played["pos_grp"] = played[pos_col].fillna("G").apply(get_pos_group) if pos_col else "Guard"
    trend_d: dict = {}; var_d: dict = {}
    for (opp, pos), grp in played.groupby(["OPPONENT","pos_grp"]):
        pts_arr = grp.sort_values("GAME_DATE")["PTS"].values[-20:].astype(float)
        if len(pts_arr) >= 5:
            trend_d[f"{opp}|{pos}"] = float(pts_arr[-5:].mean() - pts_arr.mean())
            var_d[f"{opp}|{pos}"]   = float(np.std(pts_arr))
    return trend_d, var_d


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION  — reads pre-computed rolling cols from the last prior row
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(
    prior: pd.DataFrame,
    line: float,
    opponent: str,
    rest_days: int,
    pos_raw: str,
    game_date: pd.Timestamp,
    min_line: float | None,
    max_line: float | None,
    dyn_dvp: dict,
    pace_rank: dict,
    opp_trend: dict,
    opp_var: dict,
    is_home: bool | None = None,
    h2h_row: dict | None = None,
) -> dict | None:
    if len(prior) < MIN_PRIOR_GAMES:
        return None

    # ── Use pre-computed rolling stats from the most recent prior row ─────────
    last = prior.iloc[-1]

    def col(c, d=0.0): return _f(last, c, d)

    # ── Rolling PTS — computed directly from raw game scores ─────────────────
    # The NBA API pre-computed columns (L3_PTS, L5_PTS etc.) are built with
    # shift(1) — "form entering each game" — so prior.iloc[-1]["L3_PTS"]
    # excludes the most recent game's score. Cross-check tests confirmed
    # errors up to 4pts (LeBron: L3=23.0 vs correct 19.0).
    # Computing directly from pts_arr gives the correct last-N-games average.
    pts_arr = prior["PTS"].values.astype(float)
    n       = len(pts_arr)
    std10   = max(float(np.std(pts_arr[-10:])) if len(pts_arr) >= 2 else 1.0, 0.5)
    hr10    = _hr(pts_arr, line, 10)
    hr30    = _hr(pts_arr, line, 30)

    L3  = _sm(pts_arr, 3)
    L5  = _sm(pts_arr, 5)
    L10 = _sm(pts_arr, 10)
    L20 = _sm(pts_arr, 20)
    L30 = _sm(pts_arr, 30)

    # Rolling minutes
    min_l10 = col("L10_MIN_NUM", 28.0)
    min_l30 = col("L30_MIN_NUM", 28.0)
    min_l3  = col("L3_MIN_NUM",  28.0)
    mins_arr= prior["MIN_NUM"].fillna(0).values.astype(float)
    min_cv  = float(np.std(mins_arr[-10:]) / max(min_l10, 1.0)) if len(mins_arr) >= 3 else 0.2
    recent_min_trend = float(_sm(mins_arr, 5) - min_l10)

    # Shooting (pre-computed)
    fga_l10  = col("L10_FGA",    0.0)
    fg_pct_l10 = col("L10_FG_PCT", 0.45)   # FG% as ratio 0–1 (pre-computed in CSV)
    fg3a_l10 = col("L10_FG3A",   0.0)
    fg3m_l10 = col("L10_FG3M",   0.0)
    fta_l10  = col("L10_FTA",    0.0)
    ft_rate  = col("L10_FT_PCT", 0.0)

    # Usage
    usage_l10 = col("L10_USAGE_APPROX", 0.18)
    usage_l30 = col("L30_USAGE_APPROX", 0.18)

    # Derived
    pts_per_min    = float(L10 / max(min_l10, 1.0))
    fga_per_min    = float(fga_l10 / max(min_l10, 1.0))
    ppfga_l10      = float(L10 / max(fga_l10, 1.0))
    role_intensity = float(usage_l10 * pts_per_min)

    # V12 orthogonal decomposition
    level        = L30
    reversion    = L10 - L30
    momentum     = L5  - L30
    acceleration = L3  - L5
    level_ewm    = float(pd.Series(pts_arr[-10:]).ewm(span=5,adjust=False).mean().iloc[-1]) if n>=2 else L10

    # V14 z-scores + regime
    z_momentum    = momentum    / (std10 + 1e-6)
    z_reversion   = reversion   / (std10 + 1e-6)
    z_accel       = acceleration/ (std10 + 1e-6)
    mean_rev_risk = abs(L10 - L30) / (std10 + 1e-6)
    extreme_hot   = float(L5 > L30 + std10)
    extreme_cold  = float(L5 < L30 - std10)

    # Season context
    sp             = season_progress(str(game_date.date()))
    games_depth    = min(float(n), 82.0)
    early_season_w = float(max(0.70, min(1.0, 0.70 + 0.30*(games_depth/30))))
    consistency    = float(1.0 / (std10 + 1.0))
    volume         = float(L30 - line)
    trend          = float(L5  - L30)

    # Home/away split
    is_home_arr  = prior["IS_HOME"].fillna(True).astype(float).values if "IS_HOME" in prior.columns else np.ones(n)
    home_pts     = pts_arr[is_home_arr.astype(bool)]
    away_pts     = pts_arr[~is_home_arr.astype(bool)]
    home_l10     = _sm(home_pts[-10:], 10) if len(home_pts) > 0 else L10
    away_l10     = _sm(away_pts[-10:], 10) if len(away_pts) > 0 else L10
    home_away_split = home_l10 - away_l10

    # Rest
    is_b2b     = float(rest_days <= 1)
    rest_cat   = 0 if rest_days<=1 else (1 if rest_days<=2 else (2 if rest_days<=4 else (3 if rest_days<=6 else 4)))
    is_long_rest = float(rest_days >= 6)

    # Matchup
    pos_grp      = get_pos_group(pos_raw)
    defP_dynamic = float(dyn_dvp.get(f"{opponent}|{pos_grp}", 15))
    pace_r       = float(pace_rank.get(opponent, 15))
    opp_def_trend= float(opp_trend.get(f"{opponent}|{pos_grp}", 0.0))
    opp_def_var  = float(opp_var.get(f"{opponent}|{pos_grp}", 5.0))

    # Line context
    line_vs_l30   = float(line - L30)
    line_bucket   = float(min(int(line // 5), 5))
    line_spread   = float((max_line or line) - (min_line or line))
    line_sharpness= float(1.0 / (line_spread + 1.0))
    vol_risk      = float(line_spread * std10 / max(line, 1.0))

    # H2H
    h2h_ts_dev  = 0.0; h2h_fga_dev= 0.0; h2h_min_dev= 0.0
    h2h_conf    = 0.0; h2h_games  = 0.0; h2h_trend  = 0.0
    h2h_avg     = None
    if h2h_row:
        def hf(k): return float(h2h_row.get(k,0) or 0)
        h2h_ts_dev  = hf("H2H_TS_VS_OVERALL")
        h2h_fga_dev = hf("H2H_FGA_VS_OVERALL")
        h2h_min_dev = hf("H2H_MIN_VS_OVERALL")
        h2h_conf    = hf("H2H_CONFIDENCE")
        h2h_games   = hf("H2H_GAMES")
        h2h_trend   = hf("H2H_PTS_TREND")
        _a = h2h_row.get("H2H_AVG_PTS")
        if _a is not None:
            try: h2h_avg = float(_a)
            except: pass

    return {
        "L30":L30,"L10":L10,"L5":L5,"L3":L3,"L20":L20,
        "l30":L30,"l10":L10,"l5":L5,"l3":L3,"l20":L20,
        "std10":std10,"hr10":hr10,"hr30":hr30,"n_games":float(n),
        "level":level,"reversion":reversion,"momentum":momentum,
        "acceleration":acceleration,"level_ewm":level_ewm,
        "z_momentum":z_momentum,"z_reversion":z_reversion,"z_accel":z_accel,
        "mean_reversion_risk":mean_rev_risk,
        "extreme_hot":extreme_hot,"extreme_cold":extreme_cold,
        "season_progress":sp,"early_season_weight":early_season_w,
        "games_depth":games_depth,"consistency":consistency,
        "volume":volume,"trend":trend,
        "min_l10":min_l10,"min_l30":min_l30,"min_l3":min_l3,
        "minL10":min_l10,"minL30":min_l30,
        "min_cv":min_cv,"recent_min_trend":recent_min_trend,
        "pts_per_min":pts_per_min,"fga_per_min":fga_per_min,
        "ppfga_l10":ppfga_l10,"role_intensity":role_intensity,
        "fga_l10":fga_l10,"fg_pct_l10":fg_pct_l10,
        "fg3a_l10":fg3a_l10,"fg3m_l10":fg3m_l10,
        "fta_l10":fta_l10,"ft_rate":ft_rate,"ft_rate_l10":ft_rate,
        "usage_l10":usage_l10,"usage_l30":usage_l30,
        "home_l10":home_l10,"away_l10":away_l10,"home_away_split":home_away_split,
        "rest_days":float(rest_days),"is_b2b":is_b2b,
        "rest_cat":float(rest_cat),"is_long_rest":is_long_rest,
        "defP_dynamic":defP_dynamic,"defP":defP_dynamic,"pace_rank":pace_r,
        "opp_def_trend":opp_def_trend,"opp_def_var":opp_def_var,
        "line":float(line),"line_vs_l30":line_vs_l30,
        "line_bucket":line_bucket,"line_spread":line_spread,
        "line_sharpness":line_sharpness,"vol_risk":vol_risk,
        "h2h_ts_dev":h2h_ts_dev,"h2h_fga_dev":h2h_fga_dev,
        "h2h_min_dev":h2h_min_dev,"h2h_conf":h2h_conf,
        "h2h_games":h2h_games,"h2h_trend":h2h_trend,"h2h_avg":h2h_avg,
        "is_home":float(1 if is_home else 0),
        "pos_grp_str":pos_grp,"books":0,"line_spread":line_spread,
        "volatility":std10,
        # V10 aliases
        "l10_ewm":level_ewm,
        "l5_ewm":float(pd.Series(pts_arr[-5:]).ewm(span=3,adjust=False).mean().iloc[-1]) if n>=2 else L5,
        "season_phase":sp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUB-VERSION FEATURE MATRICES
# ─────────────────────────────────────────────────────────────────────────────
def build_v92_X(f: dict) -> "pd.DataFrame":
    FEAT = ["l30","l10","l5","l3","volume","trend","std10","defP","pace_rank",
            "h2h_ts_dev","h2h_fga_dev","h2h_min_dev","h2h_conf","min_cv",
            "pts_per_min","recent_min_trend","fga_per_min","is_b2b","rest_days",
            "consistency","line"]
    return pd.DataFrame([{k: f.get(k,0.0) for k in FEAT}])[FEAT].fillna(0)

def build_v10_X(f: dict) -> "pd.DataFrame":
    FEAT = ["l30","l10","l5","l3","l10_ewm","l5_ewm","volume","trend","std10",
            "consistency","fg3a_l10","fg3m_l10","fta_l10","ft_rate_l10","usage_l10",
            "usage_l30","min_cv","pts_per_min","recent_min_trend","home_l10","away_l10",
            "home_away_split","fga_per_min","is_b2b","rest_days",
            "defP","defP_dynamic","pace_rank","h2h_ts_dev","h2h_fga_dev","h2h_min_dev",
            "h2h_conf","line","line_bucket"]
    d = {k: f.get(k,0.0) for k in FEAT}
    d["ft_rate_l10"] = f.get("ft_rate",0.0)
    return pd.DataFrame([d])[FEAT].fillna(0)

def build_v11_X(f: dict) -> "pd.DataFrame":
    FEAT = ["l30","l10","l5","l3","volume","trend","std10","defP","pace_rank",
            "h2h_ts_dev","h2h_fga_dev","h2h_min_dev","h2h_conf","min_cv",
            "pts_per_min","recent_min_trend","fga_per_min","is_b2b","rest_days",
            "consistency","line","fg3a_l10","season_phase"]
    return pd.DataFrame([{k: f.get(k,0.0) for k in FEAT}])[FEAT].fillna(0)

def build_v12_X(f: dict) -> "pd.DataFrame":
    # Exact 40-feature list the V12 LightGBM clf was trained on
    FEAT = ["level","reversion","momentum","acceleration","level_ewm","volatility",
            "fg3a_l10","fg3m_l10","fta_l10","ft_rate_l10","ppfga_l10","usage_l10",
            "usage_l30","role_intensity","min_l10","min_l3","min_cv","recent_min_trend",
            "home_l10","away_l10","home_away_split","is_b2b","b2b_pts_delta","rest_cat",
            "is_long_rest","defP","defP_dynamic","opp_def_trend","opp_def_var","pace_rank",
            "h2h_ts_dev","h2h_fga_dev","h2h_min_dev","h2h_conf",
            "line","line_bucket","line_vs_l30","line_bias_l10","usage_segment","season_game_num"]
    d = {k: f.get(k,0.0) for k in FEAT}
    d["ft_rate_l10"]   = f.get("ft_rate", 0.0)
    d["b2b_pts_delta"] = 0.0   # no B2B data available — zero is the neutral value
    d["line_bias_l10"] = 0.0   # no line bias cache — zero is neutral
    d["usage_segment"] = float(2 if f.get("usage_l10",0.18)>0.25 else (1 if f.get("usage_l10",0.18)>0.15 else 0))
    d["season_game_num"]= f.get("n_games", 30.0)
    d["opp_def_trend"] = f.get("opp_def_trend", 0.0)
    d["opp_def_var"]   = f.get("opp_def_var", 5.0)
    return pd.DataFrame([d])[FEAT].fillna(0)

def build_v14_X(f: dict) -> "pd.DataFrame":
    d = {k: f.get(k, 0.0) for k in V14_ML_FEATURES}
    # Ensure specific field mappings
    d["ft_rate"]        = f.get("ft_rate", 0.0)
    d["h2h_games"]      = f.get("h2h_games", 0.0)
    d["h2h_trend"]      = f.get("h2h_trend", 0.0)
    d["line_spread"]    = f.get("line_spread", 0.0)
    d["line_sharpness"] = f.get("line_sharpness", 0.5)
    d["vol_risk"]       = f.get("vol_risk", 0.0)
    d["rest_days"]      = f.get("rest_days", 2.0)
    return pd.DataFrame([d])[V14_ML_FEATURES].fillna(0)
