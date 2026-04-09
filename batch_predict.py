"""
PropEdge V16.0 — batch_predict.py  (B1-B5)
python3 batch_predict.py 1   # 08:30
python3 batch_predict.py 2   # 11:00
python3 batch_predict.py 3   # 16:00
python3 batch_predict.py 4   # 18:30
python3 batch_predict.py 5   # 21:00
"""
from __future__ import annotations
import json, pickle, re, sys, unicodedata, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    VERSION, FILE_TODAY, FILE_GL_2425, FILE_GL_2526, FILE_PROPS,
    FILE_H2H, FILE_V92_REG, FILE_V10_REG, FILE_V11_REG,
    FILE_V12_REG, FILE_V12_CLF, FILE_V12_CAL, FILE_V12_SEG, FILE_V12_Q,
    FILE_V12_TRUST, FILE_V14_REG, FILE_V14_CLF, FILE_V14_CAL, FILE_V14_TRUST,
    FILE_ELITE_MODEL, ODDS_API_KEY, ODDS_BASE_URL,
    ELITE_THRESHOLDS, ELITE_STAKES, ELITE_TIER_NUM, ELITE_TIER_LABEL,
    ELITE_FEATURES, TRUST_THRESHOLD, MIN_PRIOR_GAMES,
    assign_elite_tier, get_pos_group, clean_json, uk_now, et_window,
)
from rolling_engine import (
    filter_played, build_player_index, get_prior_games, build_rest_days_map,
    build_dynamic_dvp, build_pace_rank, build_opp_def_caches,
    extract_features,
    build_v92_X, build_v10_X, build_v11_X, build_v12_X, build_v14_X,
)
from reasoning_engine import generate_pre_match_reason

BATCH = int(sys.argv[1]) if len(sys.argv)>1 and sys.argv[1].isdigit() else 2


# ── Model store ───────────────────────────────────────────────────────────────
class MS:
    def __init__(self): self._c={}
    def _p(self, path):
        k=str(path)
        if k not in self._c:
            if not path.exists(): print(f"  ⚠ Missing: {path.name}"); self._c[k]=None
            else:
                try:
                    with open(path,"rb") as f: self._c[k]=pickle.load(f)
                    print(f"  ✓ {path.name}")
                except Exception as e: print(f"  ✗ {path.name}: {e}"); self._c[k]=None
        return self._c[k]
    def _j(self,path,k):
        if k not in self._c:
            try:
                with open(path) as f: self._c[k]=json.load(f)
            except: self._c[k]={}
        return self._c[k]
    @property
    def v92(self):  return self._p(FILE_V92_REG)
    @property
    def v10(self):  return self._p(FILE_V10_REG)
    @property
    def v11(self):  return self._p(FILE_V11_REG)
    @property
    def v12r(self): return self._p(FILE_V12_REG)
    @property
    def v12c(self): return self._p(FILE_V12_CLF)
    @property
    def v12a(self): return self._p(FILE_V12_CAL)
    @property
    def v12s(self): return self._p(FILE_V12_SEG)
    @property
    def v12q(self): return self._p(FILE_V12_Q)
    @property
    def v14r(self): return self._p(FILE_V14_REG)
    @property
    def v14c(self): return self._p(FILE_V14_CLF)
    @property
    def v14a(self): return self._p(FILE_V14_CAL)
    @property
    def elite(self):return self._p(FILE_ELITE_MODEL)
    @property
    def tv12(self): return self._j(FILE_V12_TRUST,"_tv12")
    @property
    def tv14(self): return self._j(FILE_V14_TRUST,"_tv14")

M = MS()


# ── Name normalisation ────────────────────────────────────────────────────────
from player_name_aliases import resolve_name as resolve, _norm


# ── Props loader ──────────────────────────────────────────────────────────────
def load_props(date_str):
    props = []
    today_str = uk_now().strftime("%Y-%m-%d")
    is_today  = (date_str == today_str)

    # ── Source 1: Excel (always for historical dates; fallback for today) ─────
    # For today's date we first try the Odds API to get the FRESHEST lines.
    # If the API fails or has no credits, we fall back to Excel.
    # This ensures line moves between B1→B2→B3 etc. are reflected.
    if FILE_PROPS.exists() and not is_today:
        try:
            xl  = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
            xl["Date"] = pd.to_datetime(xl["Date"])
            td  = pd.Timestamp(date_str).date()
            for _, r in xl[xl["Date"].dt.date == td].iterrows():
                try:
                    props.append({
                        "player":       str(r["Player"]).strip(),
                        "date":         date_str,
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
            print(f"  Props from Excel: {len(props):,}")
        except Exception as e:
            print(f"  ⚠ Excel: {e}")

    if props:
        return props  # historical date: Excel is source of truth

    # ── Source 2: Odds API ────────────────────────────────────────────────────
    # Uses broad window (date-6h to date+30h) so games are found regardless
    # of UTC offset — matches V12 behaviour.
    # Fetches player_points + spreads + totals per event.
    # Writes fetched props back to Excel so they persist for re-runs and
    # season JSON rebuilds.
    import requests
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    _ET = ZoneInfo("America/New_York")
    d   = pd.Timestamp(date_str)
    fr_s = (d - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    to_s = (d + timedelta(hours=30)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        ev_r = requests.get(
            f"{ODDS_BASE_URL}/sports/basketball_nba/events",
            params={
                "apiKey":           ODDS_API_KEY,
                "commenceTimeFrom": fr_s,
                "commenceTimeTo":   to_s,
                "dateFormat":       "iso",
            },
            timeout=30,
        )
        ev_r.raise_for_status()
        rem = ev_r.headers.get("x-requests-remaining", "?")

        # Keep only events that actually tip off on date_str ET
        events = []
        for ev in ev_r.json():
            try:
                ct_et = pd.Timestamp(ev.get("commence_time","")).tz_convert(_ET)
                if ct_et.strftime("%Y-%m-%d") == date_str:
                    events.append(ev)
            except Exception:
                events.append(ev)

        print(f"  Odds API: {len(events)} NBA games for {date_str} | Credits: {rem}")
        if not events:
            print(f"  No NBA games scheduled for {date_str}.")
            return props

        seen       = set()
        excel_rows = []

        for ev in events:
            eid  = ev.get("id", "")
            home = ev.get("home_team", "")
            away = ev.get("away_team", "")
            gn   = f"{away} @ {home}"
            ct   = ev.get("commence_time", "")
            try:
                gt = pd.Timestamp(ct).tz_convert(_ET).strftime("%-I:%M %p ET")
            except Exception:
                gt = ""

            try:
                od_r = requests.get(
                    f"{ODDS_BASE_URL}/sports/basketball_nba/events/{eid}/odds",
                    params={
                        "apiKey":     ODDS_API_KEY,
                        "markets":    "player_points,spreads,totals",
                        "regions":    "us",
                        "oddsFormat": "american",
                        "dateFormat": "iso",
                    },
                    timeout=30,
                )
                rem = od_r.headers.get("x-requests-remaining", rem)
                od_r.raise_for_status()
                od  = od_r.json()

                game_props: dict = {}
                for bk in od.get("bookmakers", []):
                    for mk in bk.get("markets", []):
                        if mk.get("key") != "player_points":
                            continue
                        for out in mk.get("outcomes", []):
                            pl  = (out.get("description") or "").strip()
                            nm  = out.get("name", "").upper()
                            pt  = out.get("point")
                            pr  = out.get("price", -110)
                            if not pl or pt is None:
                                continue
                            if pl not in game_props:
                                game_props[pl] = {"line": pt, "over": -110, "under": -110,
                                                  "books": 0, "min_line": pt, "max_line": pt}
                            if nm == "OVER":
                                game_props[pl]["over"]     = pr
                                game_props[pl]["books"]   += 1
                                game_props[pl]["min_line"] = min(game_props[pl]["min_line"], pt)
                                game_props[pl]["max_line"] = max(game_props[pl]["max_line"], pt)
                            elif nm == "UNDER":
                                game_props[pl]["under"] = pr

                for pl, pd_ in game_props.items():
                    key = f"{pl}_{pd_['line']}_{eid}"
                    if key in seen:
                        continue
                    seen.add(key)
                    props.append({
                        "player":       pl,
                        "date":         date_str,
                        "line":         pd_["line"],
                        "over_odds":    pd_["over"],
                        "under_odds":   pd_["under"],
                        "books":        pd_["books"],
                        "min_line":     pd_["min_line"],
                        "max_line":     pd_["max_line"],
                        "game":         gn,
                        "home":         home,
                        "away":         away,
                        "game_time_et": gt,
                        "source":       "api",
                    })
                    excel_rows.append({
                        "Date":         pd.Timestamp(date_str),
                        "Game_Time_ET": gt,
                        "Player":       pl,
                        "Position":     "",
                        "Game":         gn,
                        "Home":         home,
                        "Away":         away,
                        "Line":         pd_["line"],
                        "Over Odds":    pd_["over"],
                        "Under Odds":   pd_["under"],
                        "Books":        pd_["books"],
                        "Min Line":     pd_["min_line"],
                        "Max Line":     pd_["max_line"],
                        "Commence":     ct,
                        "Event ID":     eid,
                    })

                print(f"    ✓ {gn}: {len(game_props)} props")

            except Exception as e:
                print(f"  ⚠ Event {eid} ({gn}): {e}")

        print(f"  Props from API: {len(props):,} | Credits remaining: {rem}")

        # Write fetched props back to Excel so they persist for re-runs / season JSON rebuilds
        if excel_rows:
            try:
                import openpyxl

                new_p = pd.DataFrame(excel_rows)

                if FILE_PROPS.exists():
                    ep = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
                    try:
                        es = pd.read_excel(FILE_PROPS, sheet_name="Team_Spreads_Totals")
                    except Exception:
                        es = pd.DataFrame()
                else:
                    ep = pd.DataFrame()
                    es = pd.DataFrame()

                if "Date" in ep.columns:
                    ep["Date"] = pd.to_datetime(ep["Date"], errors="coerce")

                # Dedup: remove existing rows for same date+player before appending
                if not ep.empty and "Player" in ep.columns and "Date" in ep.columns:
                    keep = ~(
                        ep["Date"].dt.strftime("%Y-%m-%d").eq(date_str) &
                        ep["Player"].isin(new_p["Player"])
                    )
                    ep = ep[keep]

                ep = pd.concat([ep, new_p], ignore_index=True)
                ep = ep.sort_values(["Date", "Player"]).reset_index(drop=True)

                # Guard: never write an empty DataFrame (would wipe the Excel file)
                if ep.empty:
                    print("  ⚠ Excel: nothing to write — skipping update")
                else:
                    FILE_PROPS.parent.mkdir(parents=True, exist_ok=True)
                    # Use openpyxl directly so we never lose Team_Spreads_Totals
                    # if it fails to load above.  mode="a" + if_sheet_exists="replace"
                    # requires openpyxl ≥ 3.1; fall back to full rewrite if not supported.
                    try:
                        with pd.ExcelWriter(
                            FILE_PROPS, engine="openpyxl",
                            mode="a", if_sheet_exists="replace",
                        ) as w:
                            ep.to_excel(w, sheet_name="Player_Points_Props", index=False)
                            if not es.empty:
                                es.to_excel(w, sheet_name="Team_Spreads_Totals", index=False)
                    except TypeError:
                        # openpyxl < 3.1 — fall back to full rewrite
                        with pd.ExcelWriter(FILE_PROPS, engine="openpyxl") as w:
                            ep.to_excel(w, sheet_name="Player_Points_Props", index=False)
                            if not es.empty:
                                es.to_excel(w, sheet_name="Team_Spreads_Totals", index=False)

                    print(f"  ✓ Excel updated: {len(ep):,} rows in Player_Points_Props")
            except Exception as e:
                print(f"  ⚠ Excel write failed: {e}")

    except Exception as e:
        print(f"  ⚠ Odds API: {e}")

    # ── Fallback: load today from Excel if API failed ─────────────────────
    if is_today and not props and FILE_PROPS.exists():
        try:
            xl  = pd.read_excel(FILE_PROPS, sheet_name="Player_Points_Props")
            xl["Date"] = pd.to_datetime(xl["Date"])
            td  = pd.Timestamp(date_str).date()
            for _, r in xl[xl["Date"].dt.date == td].iterrows():
                try:
                    props.append({
                        "player":       str(r["Player"]).strip(),
                        "date":         date_str,
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
                        "source":       "excel_fallback",
                    })
                except Exception:
                    continue
            if props:
                print(f"  Props from Excel (API fallback): {len(props):,}")
        except Exception as e2:
            print(f"  ⚠ Excel fallback: {e2}")

    return props


# ── H2H loader ────────────────────────────────────────────────────────────────
def load_h2h():
    if not FILE_H2H.exists(): return {}
    try:
        df=pd.read_csv(FILE_H2H,low_memory=False)
        return {(_norm(str(r.get("PLAYER_NAME",""))),str(r.get("OPPONENT","")).strip().upper()):r.to_dict()
                for _,r in df.iterrows()}
    except Exception as e: print(f"  ⚠ H2H: {e}"); return {}


# ── Sub-version scoring ───────────────────────────────────────────────────────
def _surr(f,line):
    return float(0.45*f.get("L10",line)+0.30*f.get("L30",line)+0.15*f.get("L5",line)+0.10*line)

def sv92(f,line):
    m=M.v92
    if m is None: return _surr(f,line),0.0
    try: pp=float(m.predict(build_v92_X(f))[0]); return pp,round(pp-line,2)
    except: return _surr(f,line),0.0

def sv10(f,line):
    m=M.v10
    if m is None: return _surr(f,line),0.0
    try: pp=float(m.predict(build_v10_X(f))[0]); return pp,round(pp-line,2)
    except: return _surr(f,line),0.0

def sv11(f,line):
    m=M.v11
    if m is None: return _surr(f,line),0.0
    try: pp=float(m.predict(build_v11_X(f))[0]); return pp,round(pp-line,2)
    except: return _surr(f,line),0.0

def sv12(f,line):
    X=build_v12_X(f); seg=M.v12s; reg=M.v12r; clf=M.v12c; cal=M.v12a; qm=M.v12q
    pp=_surr(f,line)
    if seg is not None:
        try: pp=float(seg["fallback"].predict(X)[0])
        except:
            if reg is not None:
                try: pp=float(reg.predict(X)[0])
                except: pass
    elif reg is not None:
        try: pp=float(reg.predict(X)[0])
        except: pass
    q25,q75=line-3.0,line+3.0
    if qm is not None:
        try: q25=float(qm["q25"].predict(X)[0]); q75=float(qm["q75"].predict(X)[0])
        except: pass
    prob=0.5
    if clf is not None:
        try:
            raw=float(clf.predict_proba(X)[0,1])
            prob=float(cal.transform([raw])[0]) if cal else raw
        except: pass
    if prob==0.5: prob=float(np.clip(0.5+(pp-line)/max(f.get("std10",5),1)*0.15,0.30,0.70))
    return pp,round(pp-line,2),prob,q25,q75

def sv14(f,line):
    X=build_v14_X(f).values; reg=M.v14r; clf=M.v14c; cal=M.v14a
    pp=_surr(f,line)
    if reg is not None:
        try: pp=float(reg.predict(X)[0])
        except: pass
    prob=0.5
    if clf is not None:
        try:
            raw=float(clf.predict_proba(X)[0,1])
            prob=float(cal.transform([raw])[0]) if cal else raw
        except: pass
    if prob==0.5: prob=float(np.clip(0.5+(pp-line)/max(f.get("std10",5),1)*0.15,0.30,0.70))
    return pp,round(pp-line,2),prob


# ── Elite vector builder ──────────────────────────────────────────────────────
def build_ev(f,line,p92,g92,p10,g10,p11,g11,p12,g12,prb12,q25,q75,p14,g14,prb14,hG,havg,t12,t14,is_home):
    L30=f.get("L30",line); L10=f.get("L10",line); L5=f.get("L5",line); L3=f.get("L3",line)
    std10=f.get("std10",5.0)
    dv92=1 if p92>line else -1; dv10=1 if p10>line else -1
    dv11=1 if p11>line else -1; dv14=1 if prb14>=0.5 else -1; dv12=1 if prb12>=0.5 else -1
    ag12=int(dv92==dv12); ag14=int(dv92==dv14); ag1214=int(dv12==dv14)
    all_ag=int(ag12 and ag14); v12ar=int(dv12==dv92 and dv12==dv10)
    ro=int(all(d==1  for d in [dv92,dv10,dv11,dv14]))
    ru=int(all(d==-1 for d in [dv92,dv10,dv11,dv14])); rc=int(ro or ru)
    cv12=abs(prb12-0.5)*2; cv14=abs(prb14-0.5)*2
    vex=int(prb12>=0.80 or prb12<=0.20); vso=int(prb12>=0.75); vsu=int(prb12<=0.25)
    gmr=(g92+g10+g12)/3  # signed mean gap — positive=over, negative=under
    gmx=max([g92,g10,g12], key=abs)  # signed gap with largest magnitude
    vc92=float(np.clip(0.5+g92*0.04,0.45,0.90)); vc12=prb12; vc14=prb14; cm=(vc92+vc12+vc14)/3
    qr=max(q75-q25,1.0); lq25=line-q25; lq75=line-q75
    qc=float(1.0-(max(lq25,0)+abs(min(lq75,0)))/qr); liq=int(q25<=line<=q75)
    hgap=float((havg-line) if havg else 0.0)
    hal=int(hG>=3 and ((hgap>0)==(prb12>=0.5)))
    vol10=L10-line; vol30=L30-line; tr530=L5-L30; tr35=L3-L5
    lv=int(std10<=4); hv=int(std10>=8)
    ls=f.get("line_spread",1.0); ls2=1.0/(ls+1.0)
    bk=int((f.get("books",0) or 0)>=6); tm=int(ls<=0.5)
    rd=float(f.get("rest_days",2)); rb=int(rd<=1); rsw=int(rd==3); rru=int(rd>=4)
    pg=f.get("pos_grp_str","Guard"); ig=int(pg=="Guard"); ic=int(pg=="Center")
    tmn=(t12+t14)/2; lt=int(tmn<0.50)
    def tn(c): return 4.0 if c>=0.75 else (3.0 if c>=0.72 else (2.0 if c>=0.68 else (1.0 if c>=0.60 else 0.0)))
    vtn=tn(vc92); v12tn=tn(vc12); ts=vtn+v12tn+tn(vc14)
    return {
        "v92_v12clf_agree":float(ag12),"v12_clf_conv":cv12,"prob_v12":prb12,
        "prob_v14":prb14,"v12_extreme":float(vex),"v12_strong_under":float(vsu),
        "v12_strong_over":float(vso),"v92_v14clf_agree":float(ag14),
        "v12_v14_agree":float(ag1214),"all_clf_agree":float(all_ag),
        "v12clf_allreg":float(v12ar),"reg_consensus":float(rc),
        "reg_all_over":float(ro),"reg_all_under":float(ru),
        "dir_v92":float(dv92),"dir_v10":float(dv10),"dir_v11":float(dv11),"dir_v14":float(dv14),
        "gap_v92":g92,"gap_v10":g10,"gap_v11":g11,"gap_v12":g12,"gap_v14":g14,
        "gap_mean_real":gmr,"gap_max_real":gmx,
        "V9.2_predGap":g92,"V12_predGap":g12,"V14_predGap":g14,
        "V9.2_conf":vc92,"V12_conf":vc12,"V14_conf":vc14,"conf_mean":cm,"v14_clf_conv":cv14,
        "h2h_avg_gap":hgap,"h2hG":float(hG),"h2h_ts_dev":f.get("h2h_ts_dev",0.0),
        "h2h_fga_dev":f.get("h2h_fga_dev",0.0),"h2h_v12_align":float(hal),
        "q25_v12":q25,"q75_v12":q75,"q_range":qr,"q_confidence":qc,
        "line_in_q":float(liq),"line_vs_q25":lq25,"line_vs_q75":lq75,
        "L30":L30,"L10":L10,"L5":L5,"L3":L3,"std10":std10,
        "hr30":f.get("hr30",0.5),"hr10":f.get("hr10",0.5),
        "minL10":f.get("min_l10",28.0),"n_games":f.get("n_games",30.0),
        "vol_l30":vol30,"vol_l10":vol10,"trend_l5l30":tr530,"trend_l3l5":tr35,
        "low_var":float(lv),"high_var":float(hv),"line":float(line),
        "line_sharp2":ls2,"books_sig":float(bk),"tight_market":float(tm),
        "rest_days":rd,"rest_b2b":float(rb),"rest_sweet":float(rsw),"rest_rust":float(rru),
        "is_b2b":float(f.get("is_b2b",0)),"is_guard":float(ig),"is_center":float(ic),
        "pace_rank":f.get("pace_rank",15.0),"defP_dynamic":f.get("defP_dynamic",15.0),
        "is_home":float(1 if is_home else 0),
        "trust_v12":t12,"trust_v14":t14,"trust_mean":tmn,"low_trust":float(lt),
        "tier_sum":ts,"V9.2_tn":vtn,"V12_tn":v12tn,
        "_vex":vex,"_cv12":cv12,"_ag":all_ag,"_rc":rc,"_rru":rru,
        "_rsw":rsw,"_tm":tm,"_hgap":hgap,"_lv":lv,"_hal":hal,
    }

def score_elite(ev):
    pkg=M.elite
    if pkg is None:
        # Calibrated fallback: V12 clf conviction as primary signal
        # Produces SKIP<0.68, PLAY+≈0.68-0.72, STRONG≈0.72-0.75, ELITE≈0.75-0.78, ULTRA≈0.78-0.81, APEX≈0.81+
        p12   = ev.get("prob_v12", 0.5)
        p14   = ev.get("prob_v14", 0.5)
        cv12  = abs(p12 - 0.5) * 2          # conviction 0→1
        base  = 0.58 + cv12 * 0.18          # 0.58 (no conv) → 0.76 (full conv)
        bonus = 0.0
        if ev.get("all_clf_agree",0):   bonus += 0.04
        elif ev.get("v92_v12clf_agree",0): bonus += 0.02
        if ev.get("reg_consensus",0):   bonus += 0.03
        bonus += min(ev.get("gap_mean_real",0) / 25.0, 0.03)
        p14cv = abs(p14 - 0.5) * 2
        if (p12 > 0.5) == (p14 > 0.5):  bonus += p14cv * 0.04
        return float(np.clip(base + bonus, 0.40, 0.81))
    try:
        X=pd.DataFrame([{k:ev.get(k,0.0) for k in pkg["features"]}])[pkg["features"]].fillna(0).values
        return float(np.clip(pkg["model"].predict_proba(pkg["scaler"].transform(X))[0,1],0.0,1.0))
    except Exception as e: print(f"  ⚠ Elite: {e}"); return 0.5


# ── Pre-match reasoning ───────────────────────────────────────────────────────
# Delegated to reasoning_engine.generate_pre_match_reason(play)
# Called after the play dict is fully built below.

def flag_details(ev):
    tm=ev.get("trust_mean",0.68)
    return [
        {"name":"V12 clf direction",  "agrees":bool(ev.get("v92_v12clf_agree",0)), "detail":f"prob={ev.get('prob_v12',0.5):.0%}"},
        {"name":"V14 clf direction",  "agrees":bool(ev.get("v92_v14clf_agree",0)), "detail":f"prob={ev.get('prob_v14',0.5):.0%}"},
        {"name":"All clfs agree",     "agrees":bool(ev.get("all_clf_agree",0)),     "detail":""},
        {"name":"Reg consensus",      "agrees":bool(ev.get("reg_consensus",0)),     "detail":"All 4 regs"},
        {"name":"V12 extreme conv",   "agrees":bool(ev.get("_vex",0)),              "detail":f"{ev.get('_cv12',0):.0%}"},
        {"name":"H2H aligned",        "agrees":bool(ev.get("_hal",0)),              "detail":f"gap {ev.get('_hgap',0):+.1f}"},
        {"name":"No rust",            "agrees":not bool(ev.get("_rru",0)),          "detail":f"{int(ev.get('rest_days',2))}d rest"},
        {"name":"Low variance",       "agrees":bool(ev.get("_lv",0)),               "detail":f"σ={ev.get('std10',5):.1f}"},
        {"name":"Tight market",       "agrees":bool(ev.get("_tm",0)),               "detail":"spread ≤0.5"},
        {"name":"High trust",         "agrees":float(tm)>=0.65,                     "detail":f"trust={tm:.2f}"},
    ]


# ── today.json helpers ────────────────────────────────────────────────────────
def pkey(p): return (p.get("player",""),p.get("date",""),str(p.get("line","")))

def load_today():
    if FILE_TODAY.exists():
        try:
            with open(FILE_TODAY) as f: return json.load(f)
        except: pass
    return []

def save_today(plays,date_str):
    ex=load_today()
    graded={pkey(p):p for p in ex if p.get("result") in ("WIN","LOSS") and p.get("date")==date_str}
    merged=list(graded.values()); gk=set(graded)
    for p in plays:
        k=pkey(p)
        if k in gk: continue
        old=next((e for e in ex if pkey(e)==k and e.get("date")==date_str),None)
        if old: p["lineHistory"]=old.get("lineHistory",[])
        lh=p.setdefault("lineHistory",[])
        if not any(h.get("batch")==f"B{BATCH}" for h in lh):
            lh.append({"line":p["line"],"batch":f"B{BATCH}",
                        "ts":datetime.now(timezone.utc).strftime("%H:%M")})
        merged.append(p)
    FILE_TODAY.parent.mkdir(parents=True, exist_ok=True)
    with open(FILE_TODAY,"w") as f: json.dump(clean_json(merged),f,indent=2)
    print(f"  ✓ today.json → {len(merged)} plays")

def git_push():
    from git_push import push
    push(f"B{BATCH} {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
         grade=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def run_batch(date_str):
    print(f"\n  PropEdge {VERSION} — B{BATCH} ({date_str})")
    print("[1/5] Loading game logs...")
    dfs=[]
    for fp in (FILE_GL_2425,FILE_GL_2526):
        if fp.exists():
            try: dfs.append(pd.read_csv(fp,parse_dates=["GAME_DATE"],low_memory=False))
            except Exception as e: print(f"  ⚠ {fp.name}: {e}")
    if not dfs: print("  No CSVs. Exiting."); return
    combined=pd.concat(dfs,ignore_index=True)
    played=filter_played(combined)
    pidx=build_player_index(played)
    nmap={_norm(k):k for k in pidx}
    print(f"  {len(played):,} rows | {len(pidx)} players")

    print("[2/5] Caches...")
    # DVP: run dvp_updater first (live data + hardcoded fallback blend → dvp_rankings.json)
    # then load the result. Falls back to in-memory build_dynamic_dvp if updater fails.
    try:
        from dvp_updater import compute_and_save_dvp
        from config import FILE_DVP, load_dvp_cache, invalidate_dvp_cache
        compute_and_save_dvp(FILE_GL_2526, FILE_DVP)  # FILE_GL_2526 from top-level import
        invalidate_dvp_cache()
        dvp = load_dvp_cache()
        print(f"  DVP: {len(dvp)} team|pos rankings (live + fallback blend)")
    except Exception as _e:
        print(f"  ⚠ DVP updater failed ({_e}), using in-memory fallback")
        dvp = build_dynamic_dvp(played)
    pace=build_pace_rank(played)
    otr,ovr=build_opp_def_caches(played); rmap=build_rest_days_map(played)

    print("[3/5] Props + H2H + models...")
    h2h=load_h2h(); props=load_props(date_str)
    if not props: print(f"  No props for {date_str}."); return
    for a in ("v92","v10","v11","v12r","v12c","v12a","v12s","v12q","v14r","v14c","v14a","elite"):
        getattr(M,a)
    tv12=M.tv12; tv14=M.tv14

    print(f"[4/5] Scoring {len(props):,} props...")
    scored=[]; skipped=0; TORD=["APEX","ULTRA","ELITE","STRONG","PLAY+","SKIP"]

    for prop in props:
        line=float(prop["line"]); player_raw=prop["player"]
        home=str(prop.get("home","")).upper(); away=str(prop.get("away","")).upper()
        game=str(prop.get("game","")); gtime=str(prop.get("game_time_et",""))

        player=resolve(player_raw,nmap)
        if player is None: skipped+=1; continue
        prior=get_prior_games(pidx,player,prop["date"])
        if len(prior)<MIN_PRIOR_GAMES: skipped+=1; continue

        ptm=str(prior["GAME_TEAM_ABBREVIATION"].iloc[-1]).upper() if "GAME_TEAM_ABBREVIATION" in prior.columns else ""
        opp=home if ptm==away else away
        if not opp and game: tms=re.findall(r"\b([A-Z]{2,3})\b",game); opp=tms[-1] if tms else ""
        is_home=(ptm==home) if ptm and home else None
        pos_raw=str(prior["PLAYER_POSITION"].iloc[-1]) if "PLAYER_POSITION" in prior.columns else "G"
        rd=rmap.get((player,date_str),2)
        h2h_row=h2h.get((_norm(player),opp.upper()))

        f=extract_features(prior=prior,line=line,opponent=opp,rest_days=rd,
            pos_raw=pos_raw,game_date=pd.Timestamp(date_str),
            min_line=prop.get("min_line"),max_line=prop.get("max_line"),
            dyn_dvp=dvp,pace_rank=pace,opp_trend=otr,opp_var=ovr,
            is_home=is_home,h2h_row=h2h_row)
        if f is None: skipped+=1; continue

        f["pos_grp_str"]=get_pos_group(pos_raw)
        f["books"]=prop.get("books",0)
        f["line_spread"]=float((prop.get("max_line") or line)-(prop.get("min_line") or line))

        p92,g92=sv92(f,line); p10,g10=sv10(f,line); p11,g11=sv11(f,line)
        p12,g12,prb12,q25,q75=sv12(f,line); p14,g14,prb14=sv14(f,line)
        t12=float(tv12.get(player,0.68)); t14v=float(tv14.get(player,0.67))
        hG=0.0; havg=None
        if h2h_row:
            hG=float(h2h_row.get("H2H_GAMES") or 0)
            try: v=h2h_row.get("H2H_AVG_PTS"); havg=float(v) if v is not None else None
            except: pass

        ev=build_ev(f,line,p92,g92,p10,g10,p11,g11,p12,g12,prb12,q25,q75,p14,g14,prb14,hG,havg,t12,t14v,is_home)
        ep=score_elite(ev); et=assign_elite_tier(ep)
        if rd>=4 and et in ("APEX","ULTRA","ELITE","STRONG"):
            et="PLAY+" if ep>=ELITE_THRESHOLDS["PLAY+"] else "SKIP"
        tm=(t12+t14v)/2
        if tm<TRUST_THRESHOLD and et in ("APEX","ULTRA","ELITE"): et="STRONG"
        est=ELITE_STAKES.get(et,0.0)
        pmean=float(0.30*p92+0.25*p10+0.10*p11+0.25*p12+0.10*p14); gmean=round(pmean-line,2)
        cv12=abs(prb12-0.5)*2
        dl="OVER" if prb12>=0.5 else "UNDER"
        # No LEAN direction — prediction model outputs OVER or UNDER only
        nf=sum([int(ev.get("v92_v12clf_agree",0)),int(ev.get("v92_v14clf_agree",0)),
                int(ev.get("all_clf_agree",0)),int(ev.get("reg_consensus",0)),
                int(ev.get("_vex",0)),int(ev.get("_hal",0)),int(not ev.get("_rru",0)),
                int(ev.get("_lv",0)),int(ev.get("_tm",0)),int(tm>=0.65)])

        # Build recent20 from raw PTS values
        pts_vals = prior["PTS"].values[-20:]
        dates_vals= prior["GAME_DATE"].values[-20:]
        home_vals = (prior["IS_HOME"].values[-20:] if "IS_HOME" in prior.columns
                     else [True]*min(20,len(prior)))

        play={
            "player":player,"date":date_str,
            "match":game or f"{away} @ {home}","fullMatch":game or f"{away} @ {home}",
            "game":game,"home":home,"away":away,"opponent":opp,"position":pos_raw,
            "isHome":is_home,"gameTime":gtime,"game_time":gtime,
            "batchTs":datetime.now(timezone.utc).strftime("%H:%M"),
            "ptm":ptm,"team":ptm,
            "line":line,"overOdds":prop.get("over_odds",-110),
            "underOdds":prop.get("under_odds",-110),"books":prop.get("books",0),
            "min_line":prop.get("min_line"),"max_line":prop.get("max_line"),"lineHistory":[],
            "elite_tier":et,"elite_prob":round(ep,4),"elite_stake":est,"elite_rank":0,
            "v12_clf_prob":round(prb12,4),"v12_clf_conv":round(cv12,3),
            "v14_clf_prob":round(prb14,4),"all_clf_agree":bool(ev.get("all_clf_agree",0)),
            "v92_v12_agree":bool(ev.get("v92_v12clf_agree",0)),
            "v12_v14_agree":bool(ev.get("v12_v14_agree",0)),
            "reg_consensus":bool(ev.get("reg_consensus",0)),
            "v12_extreme":bool(ev.get("_vex",0)),"trust_v12":round(t12,3),
            "trust_v14":round(t14v,3),"trust_mean":round(tm,3),
            "q25_v12":round(q25,2),"q75_v12":round(q75,2),
            "q_confidence":round(ev.get("q_confidence",0),3),
            "real_gap_v92":round(g92,2),"real_gap_v12":round(g12,2),
            "real_gap_mean":round(ev.get("gap_mean_real",0),2),
            "is_under":dl.endswith("UNDER"),
            "dir":dl,"direction":dl,"predPts":round(pmean,1),"predGap":round(gmean,2),
            "conf":round(ep,4),"calProb":round(prb12,4),
            "tierLabel":ELITE_TIER_LABEL.get(et,"T3"),"tier":ELITE_TIER_NUM.get(et,3),
            "units":est,"enginesAgree":bool(ev.get("all_clf_agree",0)),
            "flags":nf,"flagsStr":f"{nf}/10",
            "l3":round(f.get("L3",0),1),"l5":round(f.get("L5",0),1),
            "l10":round(f.get("L10",0),1),"l20":round(f.get("L20",0),1),
            "l30":round(f.get("L30",0),1),"std10":round(f.get("std10",0),1),
            "hr10":round(f.get("hr10",0),3),"hr30":round(f.get("hr30",0),3),
            "volume":round(f.get("volume",0),1),"trend":round(f.get("trend",0),1),
            "momentum":round(f.get("momentum",0),1),
            "min_l10":round(f.get("min_l10",28),1),"minL10":round(f.get("min_l10",28),1),
            "min_l30":round(f.get("min_l30",28),1),
            "pts_per_min":round(f.get("pts_per_min",0.5),3),
            "usage_l10":round(f.get("usage_l10",0.18),3),
            "usage_l30":round(f.get("usage_l30",0.18),3),
            "min_cv":round(f.get("min_cv",0.2),3),
            "fta_l10":round(f.get("fta_l10",0),1),
            "ft_rate":round(f.get("ft_rate",0),3),
            "fg3a_l10":round(f.get("fg3a_l10",0),1),
            "fga_l10":round(f.get("fga_l10",0),1),
            "l10_fg_pct":round(f.get("fg_pct_l10",0.45),3),
            "line_vs_l30":round(f.get("line_vs_l30",0),2),
            "level_ewm":round(f.get("level_ewm",f.get("L10",0)),1),
            "homeAvgPts":round(f.get("home_l10",f.get("L10",0)),1),
            "awayAvgPts":round(f.get("away_l10",f.get("L10",0)),1),
            "home_away_split":round(f.get("home_away_split",0),1),
            "is_long_rest":bool(f.get("is_long_rest",0)),"is_b2b":bool(f.get("is_b2b",0)),
            "rest_days":int(rd),"extreme_hot":bool(f.get("extreme_hot",0)),
            "extreme_cold":bool(f.get("extreme_cold",0)),
            "defP":int(f.get("defP_dynamic",15)),"pace_rank":int(f.get("pace_rank",15)),
            "defP_dynamic":int(f.get("defP_dynamic",15)),
            "meanReversionRisk":round(f.get("mean_reversion_risk",0),2),
            "mean_reversion_risk":round(f.get("mean_reversion_risk",0),2),
            "earlySeasonW":round(f.get("early_season_weight",1),3),
            "early_season_weight":round(f.get("early_season_weight",1),3),
            "seasonProgress":round(f.get("season_progress",0.5),3),
            "h2hG":int(hG),"h2h":round(havg,1) if havg else None,
            "h2hAvg":round(havg,1) if havg else None,
            "h2h_avg":round(havg,1) if havg else None,
            "h2h_games":int(hG),
            "h2hTsDev":round(f.get("h2h_ts_dev",0),2),
            "h2h_ts_dev":round(f.get("h2h_ts_dev",0),2),
            "h2hFgaDev":round(f.get("h2h_fga_dev",0),2),
            "h2hConfidence":round(f.get("h2h_conf",0),3),
            "result":"","actualPts":None,"delta":None,"lossType":None,"postMatchReason":None,
            "preMatchReason":"","flagDetails":flag_details(ev),
            "recent20":[float(v) for v in pts_vals],
            "recent20dates":[str(pd.Timestamp(d).date()) for d in dates_vals],
            "recent20homes":[bool(v) for v in home_vals],
            "source":prop.get("source","excel"),
        }
        # Generate rich 5-part pre-match narrative from reasoning_engine
        play["preMatchReason"] = generate_pre_match_reason(play)
        scored.append(play)

    scored.sort(key=lambda p:(TORD.index(p.get("elite_tier","SKIP")),-p.get("elite_prob",0)))
    cnt={}
    for p in scored:
        et2=p.get("elite_tier","SKIP"); cnt[et2]=cnt.get(et2,0)+1; p["elite_rank"]=cnt[et2]

    print(f"  Scored: {len(scored):,} | Skipped: {skipped}")
    for t in TORD:
        n=sum(1 for p in scored if p.get("elite_tier")==t)
        if n: print(f"    {t}: {n}")

    print("[5/5] Saving...")
    save_today(scored,date_str); git_push()

if __name__=="__main__":
    run_batch(uk_now().strftime("%Y-%m-%d"))
