"""
PropEdge V16.0 — model_trainer.py
Trains the PropEdge Elite V2 GBT meta-model.

Training data:
  - season_2024_25.json  (real bookmaker lines — full 2024-25 season)
  - season_2025_26.json  (real bookmaker lines — 2025-26 season to date)
  - Only WIN/LOSS graded plays used — DNP and PUSH excluded
  - source="synthetic" plays excluded (legacy guard, no longer produced)
  - Minimum 200 graded plays required to attempt training
  - Plays sorted chronologically across both seasons before walk-forward

Features (82): ELITE_FEATURES from config.py
Target: 1 if result=='WIN', 0 if result=='LOSS'

Walk-forward validation covers the full Oct 2024 → present window,
stepping month-by-month so no future data leaks into any validation fold.

The sub-version pkl models (V9.2, V10, V11, V12, V14) are NOT retrained here.
They are fixed at their shipped versions.  Only the meta-model updates monthly.

Output: models/elite/propedge_elite_v2.pkl
         → {model, scaler, features, trained_at, n_plays, accuracy}
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    VERSION, FILE_SEASON_2526, FILE_SEASON_2425, FILE_ELITE_MODEL,
    FILE_V92_REG, FILE_V10_REG, FILE_V11_REG,
    FILE_V12_REG, FILE_V12_CLF, FILE_V12_CAL, FILE_V12_SEG, FILE_V12_Q,
    FILE_V14_REG, FILE_V14_CLF, FILE_V14_CAL,
    FILE_V12_TRUST, FILE_V14_TRUST,
    FILE_GL_2526, FILE_GL_2425, FILE_H2H,
    ELITE_FEATURES, V14_ML_FEATURES,
    MIN_PRIOR_GAMES, TRUST_THRESHOLD, get_pos_group,
)
from rolling_engine import (
    filter_played, get_prior_games, extract_features,
    build_v92_X, build_v10_X, build_v11_X, build_v12_X, build_v14_X,
    build_player_index, build_pace_rank, build_dynamic_dvp,
    build_opp_def_caches,
)


MIN_PLAYS_TO_TRAIN = 200   # abort if fewer graded plays available


# ─────────────────────────────────────────────────────────────────────────────
# LOAD GRADED PLAYS — both seasons, sorted chronologically
# ─────────────────────────────────────────────────────────────────────────────
def load_training_plays() -> list[dict]:
    """
    Load WIN/LOSS graded plays from both season JSONs.
    Sorted chronologically so walk-forward validation is temporally honest.
    Synthetic plays (source='synthetic') are excluded as a legacy guard —
    the pipeline no longer produces them since both seasons now use real lines.
    """
    all_plays: list[dict] = []

    for season_file, label in [
        (FILE_SEASON_2425, "2024-25"),
        (FILE_SEASON_2526, "2025-26"),
    ]:
        if not season_file.exists():
            print(f"  ⚠ {season_file.name} not found — skipping {label}")
            continue
        with open(season_file) as f:
            plays = json.load(f)
        graded = [
            p for p in plays
            if p.get("result") in ("WIN", "LOSS")
            and p.get("source", "excel") != "synthetic"
        ]
        print(f"  {label}: {len(graded):,} graded plays")
        all_plays.extend(graded)

    # Sort chronologically — critical for walk-forward integrity
    all_plays.sort(key=lambda p: p.get("date", ""))

    total = len(all_plays)
    wins  = sum(1 for p in all_plays if p.get("result") == "WIN")
    print(f"  Combined: {total:,} plays | {wins:,}W / {total-wins:,}L "
          f"| base HR: {wins/total:.1%}")
    return all_plays


# ─────────────────────────────────────────────────────────────────────────────
# BUILD TRAINING MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def build_training_matrix(plays: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    For each play, reconstruct the 82-feature Elite vector from stored
    JSON fields.  This avoids re-running the full feature extraction pipeline
    and uses the field values as they were at prediction time.
    """
    rows: list[dict] = []
    labels: list[int] = []

    for p in plays:
        try:
            line = float(p.get("line", 0))
            prb12 = float(p.get("v12_clf_prob", p.get("calProb", 0.5)))
            prb14 = float(p.get("v14_clf_prob", 0.5))
            g92   = float(p.get("real_gap_v92", p.get("predGap", 1.0)))
            g12   = float(p.get("real_gap_v12", g92))
            g10   = float(p.get("predGap", g92))   # proxy
            g11   = g10
            g14   = float(p.get("V14_predGap", p.get("predGap", g92)))

            L30 = float(p.get("l30", p.get("L30", line)))
            L10 = float(p.get("l10", p.get("L10", line)))
            L5  = float(p.get("l5",  p.get("L5",  line)))
            L3  = float(p.get("l3",  p.get("L3",  line)))
            std10 = float(p.get("std10", 5.0))

            # Direction flags
            dv92 = 1 if (float(p.get("real_gap_v92", 1.0)) > 0 and "OVER" in str(p.get("dir","")).upper()) else -1
            dv10 = dv92; dv11 = dv92
            dv14 = 1 if prb14 >= 0.5 else -1
            dv12 = 1 if prb12 >= 0.5 else -1

            ag12 = int(dv92 == dv12)
            ag14 = int(dv92 == dv14)
            ag12_14 = int(dv12 == dv14)
            all_ag = int(ag12 and ag14)
            v12ar = int(dv12 == dv92 and dv12 == dv10)
            ro = int(all(d == 1  for d in [dv92, dv10, dv11, dv14]))
            ru = int(all(d == -1 for d in [dv92, dv10, dv11, dv14]))
            rc = int(ro or ru)

            cv12 = abs(prb12 - 0.5) * 2
            cv14 = abs(prb14 - 0.5) * 2
            vex  = int(prb12 >= 0.80 or prb12 <= 0.20)
            vso  = int(prb12 >= 0.75); vsu = int(prb12 <= 0.25)

            gmr = (g92 + g10 + g12) / 3.0
            gmx = max(g92, g10, g12)

            vc92 = float(np.clip(0.5 + g92 * 0.04, 0.45, 0.90))
            vc12 = prb12; vc14 = prb14
            cm = (vc92 + vc12 + vc14) / 3.0

            q25 = float(p.get("q25_v12", line - 3.0))
            q75 = float(p.get("q75_v12", line + 3.0))
            qr  = max(q75 - q25, 1.0)
            lq25 = line - q25; lq75 = line - q75
            qc = float(1.0 - (max(lq25, 0) + abs(min(lq75, 0))) / qr)
            liq = int(q25 <= line <= q75)

            h2hG = float(p.get("h2hG", p.get("h2h_games", 0)) or 0)
            h2h_avg = p.get("h2hAvg") or p.get("h2h_avg")
            hgap = float((h2h_avg - line) if h2h_avg is not None else 0.0)
            hts  = float(p.get("h2h_ts_dev", 0) or 0)
            hfga = float(p.get("h2h_fga_dev", 0) or 0)
            hal  = int(h2hG >= 3 and ((hgap > 0) == (prb12 >= 0.5)))

            vol10 = L10 - line; vol30 = L30 - line
            tr530 = L5 - L30;   tr35  = L3 - L5
            lv = int(std10 <= 4); hv = int(std10 >= 8)

            ls = float((p.get("max_line") or line) - (p.get("min_line") or line))
            ls2 = 1.0 / (ls + 1.0)
            bk  = int((p.get("books", 0) or 0) >= 6)
            tm  = int(ls <= 0.5)

            rd  = float(p.get("rest_days", 2) or 2)
            rb  = int(rd <= 1); rsw = int(rd == 3); rru = int(rd >= 4)

            pos = str(p.get("position", "G"))
            pg  = get_pos_group(pos)
            ig  = int(pg == "Guard"); ic = int(pg == "Center")

            t12 = float(p.get("trust_v12", 0.68) or 0.68)
            t14 = float(p.get("trust_v14", 0.67) or 0.67)
            tmn = (t12 + t14) / 2.0; lt = int(tmn < 0.50)

            def tn(c): return 4.0 if c>=0.75 else (3.0 if c>=0.72 else (2.0 if c>=0.68 else (1.0 if c>=0.60 else 0.0)))
            vtn = tn(vc92); v12tn = tn(vc12); ts = vtn + v12tn + tn(vc14)

            is_home = p.get("isHome")

            row = {
                "v92_v12clf_agree": float(ag12), "v12_clf_conv": cv12,
                "prob_v12": prb12, "prob_v14": prb14,
                "v12_extreme": float(vex), "v12_strong_under": float(vsu),
                "v12_strong_over": float(vso), "v92_v14clf_agree": float(ag14),
                "v12_v14_agree": float(ag12_14), "all_clf_agree": float(all_ag),
                "v12clf_allreg": float(v12ar), "reg_consensus": float(rc),
                "reg_all_over": float(ro), "reg_all_under": float(ru),
                "dir_v92": float(dv92), "dir_v10": float(dv10),
                "dir_v11": float(dv11), "dir_v14": float(dv14),
                "gap_v92": g92, "gap_v10": g10, "gap_v11": g11,
                "gap_v12": g12, "gap_v14": g14,
                "gap_mean_real": gmr, "gap_max_real": gmx,
                "V9.2_predGap": g92, "V12_predGap": g12, "V14_predGap": g14,
                "V9.2_conf": vc92, "V12_conf": vc12, "V14_conf": vc14,
                "conf_mean": cm, "v14_clf_conv": cv14,
                "h2h_avg_gap": hgap, "h2hG": float(h2hG),
                "h2h_ts_dev": hts, "h2h_fga_dev": hfga,
                "h2h_v12_align": float(hal),
                "q25_v12": q25, "q75_v12": q75, "q_range": qr,
                "q_confidence": qc, "line_in_q": float(liq),
                "line_vs_q25": lq25, "line_vs_q75": lq75,
                "L30": L30, "L10": L10, "L5": L5, "L3": L3,
                "std10": std10,
                "hr30": float(p.get("hr30", 0.5) or 0.5),
                "hr10": float(p.get("hr10", 0.5) or 0.5),
                "minL10": float(p.get("minL10", p.get("min_l10", 28)) or 28),
                "n_games": float(p.get("h2hG", 30) or 30),
                "vol_l30": vol30, "vol_l10": vol10,
                "trend_l5l30": tr530, "trend_l3l5": tr35,
                "low_var": float(lv), "high_var": float(hv),
                "line": float(line), "line_sharp2": ls2,
                "books_sig": float(bk), "tight_market": float(tm),
                "rest_days": rd, "rest_b2b": float(rb),
                "rest_sweet": float(rsw), "rest_rust": float(rru),
                "is_b2b": float(p.get("is_b2b", False) or 0),
                "is_guard": float(ig), "is_center": float(ic),
                "pace_rank": float(p.get("pace_rank", p.get("pace", 15)) or 15),
                "defP_dynamic": float(p.get("defP_dynamic", p.get("defP", 15)) or 15),
                "is_home": float(1 if is_home else 0),
                "trust_v12": t12, "trust_v14": t14, "trust_mean": tmn,
                "low_trust": float(lt), "tier_sum": ts,
                "V9.2_tn": vtn, "V12_tn": v12tn,
            }
            rows.append(row)
            labels.append(1 if p.get("result") == "WIN" else 0)
        except Exception:
            continue

    X = pd.DataFrame(rows)[ELITE_FEATURES].fillna(0).values
    y = np.array(labels)
    print(f"  Training matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_validate(
    plays: list[dict], X: np.ndarray, y: np.ndarray
) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    months = sorted(set(p.get("date","")[:7] for p in plays))
    oof_prob = np.zeros(len(plays))
    results  = []

    for i in range(2, len(months)):
        tr_months = set(months[:i]); te_month = months[i]
        tr = np.array([p.get("date","")[:7] in tr_months for p in plays])
        te = np.array([p.get("date","")[:7] == te_month  for p in plays])
        if tr.sum() < 100 or te.sum() < 5:
            continue
        sc  = StandardScaler().fit(X[tr])
        gbt = GradientBoostingClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.025,
            subsample=0.68, min_samples_leaf=15, max_features=0.72,
            n_iter_no_change=40, validation_fraction=0.12, random_state=42,
        )
        gbt.fit(sc.transform(X[tr]), y[tr])
        prob = gbt.predict_proba(sc.transform(X[te]))[:, 1]
        oof_prob[te] = prob
        for thresh, tname in [
            (0.68, "PLAY+"), (0.72, "STRONG"), (0.75, "ELITE"),
            (0.78, "ULTRA"), (0.81, "APEX"),
        ]:
            mk = prob >= thresh
            if mk.sum() < 5:
                continue
            results.append({
                "month": te_month, "tier": tname,
                "n": int(mk.sum()), "acc": float(y[te][mk].mean()),
            })

    # Summary
    summary: dict = {}
    for tier in ["PLAY+", "STRONG", "ELITE", "ULTRA", "APEX"]:
        sub = [r for r in results if r["tier"] == tier]
        if not sub:
            continue
        total_n   = sum(r["n"] for r in sub)
        total_w   = sum(r["n"] * r["acc"] for r in sub)
        wacc      = total_w / total_n if total_n > 0 else 0.0
        m80       = sum(1 for r in sub if r["acc"] >= 0.80)
        summary[tier] = {"acc": round(wacc, 4), "n": total_n,
                          "months_gte80": m80, "months_total": len(sub)}
        print(f"    {tier:8s}: {wacc:.1%} | n={total_n:,} | M≥80%: {m80}/{len(sub)}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN FINAL MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_final_model(X: np.ndarray, y: np.ndarray) -> tuple:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    sc  = StandardScaler().fit(X)
    gbt = GradientBoostingClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.025,
        subsample=0.68, min_samples_leaf=15, max_features=0.72,
        n_iter_no_change=40, validation_fraction=0.12, random_state=42,
    )
    print("  Training final model on all available data...")
    gbt.fit(sc.transform(X), y)
    train_acc = float(gbt.score(sc.transform(X), y))
    print(f"  Training accuracy (in-sample): {train_acc:.1%}")
    return gbt, sc


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────
def save_model(model, scaler, n_plays: int, oof_summary: dict,
               seasons: list[str] | None = None) -> None:
    FILE_ELITE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    pkg = {
        "model":        model,
        "scaler":       scaler,
        "features":     ELITE_FEATURES,
        "version":      VERSION,
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "n_plays":      n_plays,
        "oof_summary":  oof_summary,
        "seasons_used": seasons or ["2024-25", "2025-26"],
    }
    with open(FILE_ELITE_MODEL, "wb") as f:
        pickle.dump(pkg, f)
    print(f"  ✓ Saved: {FILE_ELITE_MODEL}")
    print(f"    n_plays={n_plays} | seasons={','.join(pkg['seasons_used'])} "
          f"| trained_at={pkg['trained_at'][:19]}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n  PropEdge {VERSION} — model_trainer.py")
    print("  Training data: season_2024_25.json + season_2025_26.json (real bookmaker lines)")

    plays = load_training_plays()
    if len(plays) < MIN_PLAYS_TO_TRAIN:
        print(f"  ✗ Only {len(plays)} graded plays — need ≥{MIN_PLAYS_TO_TRAIN}. Aborting.")
        return

    print(f"\n  Building training matrix from {len(plays):,} plays...")
    X, y = build_training_matrix(plays)
    if len(X) < MIN_PLAYS_TO_TRAIN:
        print(f"  ✗ Feature extraction yielded only {len(X)} rows. Aborting.")
        return

    months = sorted(set(p.get("date", "")[:7] for p in plays))
    print(f"\n  Walk-forward validation ({len(months)} months: {months[0]} → {months[-1]})...")
    summary = walk_forward_validate(plays, X, y)

    print("\n  Training final model (full dataset)...")
    model, scaler = train_final_model(X, y)

    seasons = sorted(set(
        ("2024-25" if p.get("date","") < "2025-07" else "2025-26")
        for p in plays
    ))
    save_model(model, scaler, len(X), summary, seasons)
    print("\n  ✓ model_trainer.py complete")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# DAILY TRUST SCORE UPDATE
# ─────────────────────────────────────────────────────────────────────────────

def update_trust_scores(min_plays: int = 10) -> None:
    """
    Recompute per-player direction accuracy from all graded plays across
    both season JSONs and write fresh player_trust.json files.

    Called daily after batch0_grade — cheap (no model training), high value.
    Trust scores degrade quickly when stale: a player who was unreliable in
    November may be reliable in March. This ensures tiers reflect current
    accuracy rather than static historical values.

    Trust = fraction of plays where model direction (OVER/UNDER) was correct.
    Minimum min_plays required for a player to get a score; others default to 0.68.

    Both V12 and V14 trust files are updated to the same values since both
    sub-models have similar per-player accuracy patterns and using the same
    trust file keeps the system consistent.
    """
    from config import FILE_SEASON_2425, FILE_SEASON_2526, FILE_V12_TRUST, FILE_V14_TRUST

    all_graded: list[dict] = []
    for season_file in (FILE_SEASON_2425, FILE_SEASON_2526):
        if not season_file.exists():
            continue
        try:
            with open(season_file) as f:
                plays = json.load(f)
            graded = [
                p for p in plays
                if p.get("result") in ("WIN", "LOSS")
                and p.get("source", "excel") != "synthetic"
            ]
            all_graded.extend(graded)
        except Exception as e:
            print(f"  ⚠ Trust update: could not load {season_file.name}: {e}")

    if len(all_graded) < min_plays:
        print(f"  ⚠ Trust update: only {len(all_graded)} graded plays — skipping")
        return

    # Build per-player accuracy: direction correct = WIN means model was right
    from collections import defaultdict
    player_stats: dict[str, dict] = defaultdict(lambda: {"plays": 0, "correct": 0})

    for p in all_graded:
        player = p.get("player", "")
        if not player:
            continue
        player_stats[player]["plays"] += 1
        if p.get("result") == "WIN":
            player_stats[player]["correct"] += 1

    # Only compute trust for players with enough data
    trust: dict[str, float] = {}
    for player, stats in player_stats.items():
        if stats["plays"] >= min_plays:
            trust[player] = round(stats["correct"] / stats["plays"], 3)

    # Stats for the log
    values = list(trust.values())
    avg    = round(sum(values) / len(values), 3) if values else 0
    below_threshold = sum(1 for v in values if v < 0.42)

    # Write both trust files (same values — both sub-models track the same players)
    for trust_file in (FILE_V12_TRUST, FILE_V14_TRUST):
        trust_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(trust_file, "w") as f:
                json.dump(trust, f, indent=2, sort_keys=True)
        except Exception as e:
            print(f"  ⚠ Trust update: could not write {trust_file.name}: {e}")
            return

    print(f"  ✓ Trust scores updated: {len(trust):,} players | "
          f"avg={avg:.3f} | {below_threshold} below threshold (0.42)")
    print(f"    Based on {len(all_graded):,} graded plays across both seasons")

    # Invalidate the cached trust tables in batch_predict memory
    try:
        from batch_predict import M
        M._tv12 = None
        M._tv14 = None
    except Exception:
        pass  # cache will reload naturally on next predict run
