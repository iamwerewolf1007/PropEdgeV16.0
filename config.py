"""
PropEdge V16.0 — config.py
All constants, paths, Elite tier thresholds, clean_json helper.
"""
from __future__ import annotations
import json, math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION   = "V16.0"
ROOT      = Path(__file__).parent.resolve()
DATA_DIR  = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR   = ROOT / "logs"
SRC_DIR   = ROOT / "source-files"

for _d in (DATA_DIR, MODEL_DIR, LOG_DIR, SRC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Source files ──────────────────────────────────────────────────────────────
FILE_GL_2425 = SRC_DIR / "nba_gamelogs_2024_25.csv"
FILE_GL_2526 = SRC_DIR / "nba_gamelogs_2025_26.csv"
FILE_H2H     = SRC_DIR / "h2h_database.csv"
FILE_PROPS      = SRC_DIR / "PropEdge_-_Match_and_Player_Prop_lines_.xlsx"
FILE_PROPS_2425 = SRC_DIR / "PropEdge_2024_25_Match_and_Player_Prop_lines.xlsx"

# ── Data files ────────────────────────────────────────────────────────────────
FILE_TODAY       = DATA_DIR / "today.json"
FILE_SEASON_2526 = DATA_DIR / "season_2025_26.json"
FILE_SEASON_2425 = DATA_DIR / "season_2024_25.json"
FILE_AUDIT       = DATA_DIR / "audit_log.csv"
FILE_DVP         = DATA_DIR / "dvp_rankings.json"

# ── Model files ───────────────────────────────────────────────────────────────
FILE_V92_REG   = MODEL_DIR / "V9.2" / "projection_model.pkl"
FILE_V10_REG   = MODEL_DIR / "V10"  / "projection_model.pkl"
FILE_V11_REG   = MODEL_DIR / "V11"  / "projection_model.pkl"
FILE_V12_REG   = MODEL_DIR / "V12"  / "projection_model.pkl"
FILE_V12_CLF   = MODEL_DIR / "V12"  / "direction_classifier.pkl"
FILE_V12_CAL   = MODEL_DIR / "V12"  / "calibrator.pkl"
FILE_V12_SEG   = MODEL_DIR / "V12"  / "segment_model.pkl"
FILE_V12_Q     = MODEL_DIR / "V12"  / "quantile_models.pkl"
FILE_V12_TRUST = MODEL_DIR / "V12"  / "player_trust.json"
FILE_V14_REG   = MODEL_DIR / "V14"  / "projection_model.pkl"
FILE_V14_CLF   = MODEL_DIR / "V14"  / "direction_classifier.pkl"
FILE_V14_CAL   = MODEL_DIR / "V14"  / "calibrator.pkl"
FILE_V14_TRUST = MODEL_DIR / "V14"  / "player_trust.json"
FILE_ELITE_MODEL = MODEL_DIR / "elite" / "propedge_elite_v2.pkl"

for _d in (MODEL_DIR/"V9.2", MODEL_DIR/"V10", MODEL_DIR/"V11",
           MODEL_DIR/"V12",  MODEL_DIR/"V14", MODEL_DIR/"elite"):
    _d.mkdir(parents=True, exist_ok=True)

# ── API / Git ─────────────────────────────────────────────────────────────────
ODDS_API_KEY  = "a77b14b513399a472139e58390aac514"
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
GIT_REMOTE    = "git@github.com:iamwerewolf1007/PropEdgeV16.0.git"
LOCAL_DIR     = Path.home() / "Documents" / "GitHub" / "PropEdgeV16.0-Local"
REPO_DIR      = Path.home() / "Documents" / "GitHub" / "PropEdgeV16.0"

# ── Season ────────────────────────────────────────────────────────────────────
SEASON_START = datetime(2025, 10, 1,  tzinfo=timezone.utc)
SEASON_END   = datetime(2026, 4,  20, tzinfo=timezone.utc)
SEASON_DAYS  = (SEASON_END - SEASON_START).days

def season_progress(date_str: str) -> float:
    try:
        y, mo, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
        dt = datetime(y, mo, d, tzinfo=timezone.utc)
        return float(max(0.0, min(1.0, (dt - SEASON_START).days / SEASON_DAYS)))
    except Exception:
        return 0.5

# ── Elite tier system ─────────────────────────────────────────────────────────
ELITE_THRESHOLDS = {"APEX":0.81,"ULTRA":0.78,"ELITE":0.75,"STRONG":0.72,"PLAY+":0.68}
ELITE_STAKES     = {"APEX":2.5,"ULTRA":2.0,"ELITE":2.0,"STRONG":1.5,"PLAY+":1.2,"SKIP":0.0}
ELITE_TIER_NUM   = {"APEX":1,"ULTRA":1,"ELITE":1,"STRONG":2,"PLAY+":2,"SKIP":3}
ELITE_TIER_LABEL = {"APEX":"APEX","ULTRA":"ULTRA","ELITE":"ELITE",
                    "STRONG":"STRONG","PLAY+":"PLAY+","SKIP":"SKIP"}

def assign_elite_tier(prob: float) -> str:
    if prob >= 0.81: return "APEX"
    if prob >= 0.78: return "ULTRA"
    if prob >= 0.75: return "ELITE"
    if prob >= 0.72: return "STRONG"
    if prob >= 0.68: return "PLAY+"
    return "SKIP"

# ── Hard filters ──────────────────────────────────────────────────────────────
TRUST_THRESHOLD  = 0.42
MIN_PRIOR_GAMES  = 5

# ── Position mapping ──────────────────────────────────────────────────────────
POS_MAP = {
    "PG":"Guard","SG":"Guard","G":"Guard","GF":"Guard","FG":"Guard",
    "SF":"Forward","PF":"Forward","F":"Forward","FC":"Forward","CF":"Forward",
    "C":"Center",
}

def get_pos_group(pos: str) -> str:
    p = str(pos).strip().upper()
    return POS_MAP.get(p, POS_MAP.get(p.split("-")[0], "Guard"))

# ── V14 ML feature list ───────────────────────────────────────────────────────
# Exact 56-feature list the V14 GBR/GBC was trained on (from V14 model_trainer.py)
V14_ML_FEATURES = [
    "level","reversion","momentum","acceleration","level_ewm",
    "z_momentum","z_reversion","z_accel",
    "mean_reversion_risk","extreme_hot","extreme_cold",
    "season_progress","early_season_weight","games_depth",
    "volume","trend","std10","consistency","hr10","hr30",
    "min_l10","min_l30","min_cv","recent_min_trend","pts_per_min",
    "fga_l10","fg3a_l10","fg3m_l10","fta_l10","ft_rate","fga_per_min","ppfga_l10",
    "usage_l10","usage_l30","role_intensity",
    "home_l10","away_l10","home_away_split",
    "is_b2b","rest_days","rest_cat","is_long_rest",
    "defP_dynamic","pace_rank",
    "h2h_ts_dev","h2h_fga_dev","h2h_min_dev","h2h_conf","h2h_games","h2h_trend",
    "line","line_vs_l30","line_bucket",
    "line_spread","line_sharpness","vol_risk",
]

# ── Elite V2 meta-model features (82) ─────────────────────────────────────────
ELITE_FEATURES = [
    "v92_v12clf_agree","v12_clf_conv","prob_v12","prob_v14",
    "v12_extreme","v12_strong_under","v12_strong_over",
    "v92_v14clf_agree","v12_v14_agree","all_clf_agree","v12clf_allreg",
    "reg_consensus","reg_all_over","reg_all_under",
    "dir_v92","dir_v10","dir_v11","dir_v14",
    "gap_v92","gap_v10","gap_v11","gap_v12","gap_v14","gap_mean_real","gap_max_real",
    "V9.2_predGap","V12_predGap","V14_predGap",
    "V9.2_conf","V12_conf","V14_conf","conf_mean","v14_clf_conv",
    "h2h_avg_gap","h2hG","h2h_ts_dev","h2h_fga_dev","h2h_v12_align",
    "q25_v12","q75_v12","q_range","q_confidence","line_in_q","line_vs_q25","line_vs_q75",
    "L30","L10","L5","L3","std10","hr30","hr10","minL10","n_games",
    "vol_l30","vol_l10","trend_l5l30","trend_l3l5","low_var","high_var",
    "line","line_sharp2","books_sig","tight_market",
    "rest_days","rest_b2b","rest_sweet","rest_rust","is_b2b",
    "is_guard","is_center","pace_rank","defP_dynamic","is_home",
    "trust_v12","trust_v14","trust_mean","low_trust",
    "tier_sum","V9.2_tn","V12_tn",
]

# ── Timezone helpers ──────────────────────────────────────────────────────────
_UK = ZoneInfo("Europe/London")
_ET = ZoneInfo("America/New_York")

def uk_now() -> datetime:
    return datetime.now(_UK)

def et_window(date_str: str):
    y, mo, d = int(date_str[:4]), int(date_str[5:7]), int(date_str[8:10])
    local_midnight = datetime(y, mo, d, 0, 0, tzinfo=_ET)
    fr = local_midnight.astimezone(timezone.utc)
    to = (local_midnight + timedelta(hours=23, minutes=59)).astimezone(timezone.utc)
    return fr, to

# ── DVP cache invalidation ───────────────────────────────────────────────────
_dvp_cache: dict = {}

def load_dvp_cache() -> dict:
    global _dvp_cache
    if not _dvp_cache and FILE_DVP.exists():
        try:
            with open(FILE_DVP) as f:
                _dvp_cache = json.load(f)
        except Exception:
            _dvp_cache = {}
    return _dvp_cache

def invalidate_dvp_cache() -> None:
    global _dvp_cache
    _dvp_cache = {}

def get_uk() -> ZoneInfo:
    return _UK

# ── JSON serialiser ───────────────────────────────────────────────────────────────
def clean_json(obj):
    try:
        import numpy as np
        if isinstance(obj, dict):   return {k: clean_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [clean_json(v) for v in obj]
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        return obj
    except ImportError:
        if isinstance(obj, dict):   return {k: clean_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)): return [clean_json(v) for v in obj]
        if isinstance(obj, float):
            return None if (math.isnan(obj) or math.isinf(obj)) else obj
        return obj
