"""
analyze_game1_finals.py — Combined Analysis: Both Models on Game 1 of 2017 NBA Finals

Model 1 (Morale): Computes GSW momentum index at each scenario.
  - If momentum is NEGATIVE → GSW is losing momentum → CALL TIMEOUT
  - If momentum is POSITIVE → GSW has momentum → DON'T CALL TIMEOUT

Model 2 (Statistics): XGBoost prediction on whether timeout is beneficial.

Decision Logic:
  - If both models agree → that's the verdict
  - If they disagree → weighted average of confidence scores breaks the tie
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import joblib

MORALE_DIR = os.path.join(os.path.dirname(__file__), 'nba_morale_model')
STATS_DIR = os.path.join(os.path.dirname(__file__), 'nba_statistics_model')
TEAM_ID = 1610612744  # GSW

FEATURES = ['period', 'clock_seconds', 'score_diff', 'opp_run_before', 'own_run_before',
            'opp_fg_pct_before', 'own_fg_pct_before', 'own_turnovers_before', 'opp_turnovers_before']


def clock_to_sec(cl):
    if not isinstance(cl, str) or ":" not in cl:
        return None
    try:
        mm, ss = cl.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return None


# ============================================================
# MODEL 1: MORALE MODEL — Compute momentum at each scenario
# ============================================================

def build_momentum_timeline():
    """Build the full per-play momentum timeline for Game 1."""
    pbp_path = os.path.join(MORALE_DIR, 'data', 'nba_data', 'datanba_po_2016.csv')
    pbp = pd.read_csv(pbp_path)
    
    game_ids = pbp['GAME_ID'].astype(str).unique()
    finals_g1 = [gid for gid in game_ids if '41600401' in str(gid)]
    if not finals_g1:
        CLE_ID = 1610612739
        for gid in game_ids:
            df_game = pbp[pbp['GAME_ID'].astype(str) == str(gid)]
            teams = df_game['tid'].dropna().unique()
            if TEAM_ID in teams and CLE_ID in teams:
                finals_g1.append(gid)
                break
    
    game_id = str(finals_g1[0])
    df = pbp[pbp['GAME_ID'].astype(str) == game_id].copy().sort_values('evt').reset_index(drop=True)
    
    # Load weights
    weights_path = os.path.join(MORALE_DIR, 'data', 'momentum_weights.json')
    weights = None
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            weights = json.load(f)
    
    # Infer GSW home
    gsw_is_home = True
    last_h, last_a = 0, 0
    for _, row in df.iterrows():
        h = pd.to_numeric(row.get("hs"), errors="coerce")
        a = pd.to_numeric(row.get("vs"), errors="coerce")
        if pd.isna(h) or pd.isna(a): continue
        h, a = int(h), int(a)
        if h == last_h and a == last_a: continue
        if row.get("tid") == TEAM_ID:
            gsw_is_home = (h > last_h and a == last_a)
            break
        last_h, last_a = h, a
    
    df["hs"] = pd.to_numeric(df["hs"], errors="coerce").ffill().fillna(0)
    df["vs"] = pd.to_numeric(df["vs"], errors="coerce").ffill().fillna(0)
    
    last_home_score, last_away_score = 0, 0
    gsw_run, opp_run = 0, 0
    gsw_run_start, opp_run_start = None, None
    margin_history = []
    gsw_3_times, opp_3_times = [], []
    
    timeline = []
    
    for _, row in df.iterrows():
        desc = str(row.get("de", ""))
        tid = row.get("tid")
        cl = row.get("cl")
        home_score, away_score = int(row.get("hs", 0)), int(row.get("vs", 0))
        home_delta = home_score - last_home_score
        away_delta = away_score - last_away_score
        is_scoring = (home_delta > 0 or away_delta > 0)
        
        gsw_score = home_score if gsw_is_home else away_score
        opp_score = away_score if gsw_is_home else home_score
        
        if is_scoring:
            if home_delta > 0 and away_delta == 0:
                scoring_gsw = gsw_is_home
            elif away_delta > 0 and home_delta == 0:
                scoring_gsw = not gsw_is_home
            else:
                scoring_gsw = None
            
            if scoring_gsw is True:
                gsw_run += max(home_delta if gsw_is_home else away_delta, 0)
                opp_run = 0
                if gsw_run_start is None: gsw_run_start = clock_to_sec(cl)
                opp_run_start = None
            elif scoring_gsw is False:
                opp_run += max(away_delta if gsw_is_home else home_delta, 0)
                gsw_run = 0
                if opp_run_start is None: opp_run_start = clock_to_sec(cl)
                gsw_run_start = None
            
            if "3-pt" in desc.lower() or "3pt" in desc.lower():
                if scoring_gsw: gsw_3_times.append(cl)
                else: opp_3_times.append(cl)
            
            last_home_score, last_away_score = home_score, away_score
        
        margin = gsw_score - opp_score
        margin_history.append(margin)
        if len(margin_history) > 5: margin_history = margin_history[-5:]
        margin_swing = (margin - margin_history[0]) if len(margin_history) >= 5 else 0
        
        cur_sec = clock_to_sec(cl)
        gsw_ri = gsw_run / max((gsw_run_start - cur_sec), 1) if gsw_run_start and cur_sec else 0
        opp_ri = opp_run / max((opp_run_start - cur_sec), 1) if opp_run_start and cur_sec else 0
        
        if cur_sec is not None:
            gsw_3_times = [t for t in gsw_3_times if clock_to_sec(t) and (clock_to_sec(t) - cur_sec) <= 180]
            opp_3_times = [t for t in opp_3_times if clock_to_sec(t) and (clock_to_sec(t) - cur_sec) <= 180]
        gsw_b2b3 = 1 if len(gsw_3_times) >= 2 else 0
        opp_b2b3 = 1 if len(opp_3_times) >= 2 else 0
        
        if weights:
            gsw_mom = (weights.get("intercept", 0) + weights.get("gsw_run_points", 0) * gsw_run
                      + weights.get("gsw_run_intensity", 0) * gsw_ri
                      + weights.get("margin_swing", 0) * margin_swing
                      + weights.get("gsw_back_to_back_3s_3min", 0) * gsw_b2b3)
            opp_mom = (weights.get("intercept", 0) + weights.get("gsw_run_points", 0) * opp_run
                      + weights.get("gsw_run_intensity", 0) * opp_ri
                      + weights.get("margin_swing", 0) * (-margin_swing)
                      + weights.get("gsw_back_to_back_3s_3min", 0) * opp_b2b3)
        else:
            gsw_mom = 0.40 * gsw_run + 0.25 * gsw_ri + 0.15 * margin_swing + 0.20 * gsw_b2b3
            opp_mom = 0.40 * opp_run + 0.25 * opp_ri - 0.15 * margin_swing + 0.20 * opp_b2b3
        
        momentum_index = gsw_mom - opp_mom
        
        timeline.append({
            'period': row.get('PERIOD'),
            'clock': cl,
            'clock_sec': cur_sec,
            'gsw_score': gsw_score, 'opp_score': opp_score,
            'momentum_index': momentum_index,
            'gsw_run': gsw_run, 'opp_run': opp_run,
            'desc': desc,
        })
    
    return pd.DataFrame(timeline)


def morale_model_verdict(timeline, period, clock_seconds):
    """
    Get Model 1 verdict for a scenario.
    Finds the closest play in the timeline and returns the avg momentum
    in a window around that point.
    
    If avg momentum is NEGATIVE → GSW losing momentum → CALL TIMEOUT (True)
    If avg momentum is POSITIVE → GSW has momentum → DON'T CALL TIMEOUT (False)
    """
    # Find plays in the matching period
    period_plays = timeline[timeline['period'] == period].copy()
    if period_plays.empty:
        return False, 0.0, 0.0
    
    # Find closest play by clock
    period_plays = period_plays.dropna(subset=['clock_sec'])
    if period_plays.empty:
        return False, 0.0, 0.0
    
    period_plays['clock_diff'] = (period_plays['clock_sec'] - clock_seconds).abs()
    closest_idx = period_plays['clock_diff'].idxmin()
    
    # Get a window of ~10 plays around this point for the average
    window_start = max(closest_idx - 5, timeline.index.min())
    window_end = min(closest_idx + 5, timeline.index.max())
    window = timeline.loc[window_start:window_end]
    
    avg_momentum = window['momentum_index'].mean()
    point_momentum = timeline.loc[closest_idx, 'momentum_index']
    
    # Negative momentum = GSW losing momentum = CALL TIMEOUT
    should_timeout = avg_momentum < 0
    
    # Confidence: how far from zero (stronger signal = more confident)
    # Normalize to 0-1 scale (cap at |2.0| for practical purposes)
    confidence = min(abs(avg_momentum) / 2.0, 1.0)
    # Minimum 50% confidence
    confidence = 0.5 + confidence * 0.5
    
    return should_timeout, avg_momentum, confidence


# ============================================================
# MODEL 2: STATISTICS MODEL — XGBoost verdict
# ============================================================

def stats_model_verdict(model, situation):
    """Get Model 2 verdict for a scenario."""
    X = pd.DataFrame([situation])[FEATURES]
    prob = model.predict_proba(X)[0]
    prediction = int(model.predict(X)[0])
    
    return bool(prediction), float(prob[1]), float(max(prob))


# ============================================================
# COMBINED VERDICT
# ============================================================

def combined_verdict(m1_timeout, m1_confidence, m2_timeout, m2_confidence):
    """
    If both agree → return that verdict.
    If they disagree → weighted average of confidence scores.
    
    Weight Model 2 (Stats) slightly higher (0.55) since it was trained
    specifically on timeout outcomes. Model 1 (Morale) gets 0.45.
    """
    W1 = 0.30  # Morale model weight — momentum signal, not trained on timeout outcomes
    W2 = 0.70  # Statistics model weight — trained on actual timeout outcomes with known labels
    
    if m1_timeout == m2_timeout:
        # Both agree
        combined_conf = m1_confidence * W1 + m2_confidence * W2
        return m1_timeout, combined_conf, "AGREE"
    else:
        # Disagree — weighted average
        # Convert to a "timeout score": 1.0 = definitely timeout, 0.0 = definitely not
        m1_score = m1_confidence if m1_timeout else (1 - m1_confidence)
        m2_score = m2_confidence if m2_timeout else (1 - m2_confidence)
        
        weighted_score = m1_score * W1 + m2_score * W2
        
        should_timeout = weighted_score > 0.5
        return should_timeout, weighted_score, "DISAGREE"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("GAME 1 — 2017 NBA FINALS: GSW vs CLE (GSW 113, CLE 91)")
    print("Dual-Model Timeout Analysis")
    print("=" * 70)
    
    # Build morale model timeline
    print("\n📈 Building momentum timeline (Model 1)...")
    timeline = build_momentum_timeline()
    print(f"   {len(timeline)} plays tracked")
    
    # Load stats model
    print("📊 Loading XGBoost timeout model (Model 2)...")
    model_path = os.path.join(STATS_DIR, 'timeout_model.pkl')
    stats_model = joblib.load(model_path)
    print("   Model loaded\n")
    
    # Define scenarios
    scenarios = [
        {
            'name': 'Q1 — CLE opens with 8-0 run',
            'period': 1, 'clock_seconds': 540, 'score_diff': -8,
            'opp_run_before': 8, 'own_run_before': 0,
            'opp_fg_pct_before': 0.75, 'own_fg_pct_before': 0.0,
            'own_turnovers_before': 1, 'opp_turnovers_before': 0,
        },
        {
            'name': 'Q1 — GSW responds, ties game',
            'period': 1, 'clock_seconds': 360, 'score_diff': 0,
            'opp_run_before': 0, 'own_run_before': 8,
            'opp_fg_pct_before': 0.333, 'own_fg_pct_before': 0.667,
            'own_turnovers_before': 0, 'opp_turnovers_before': 1,
        },
        {
            'name': 'Q2 — GSW up 7, CLE 5-pt mini-run',
            'period': 2, 'clock_seconds': 420, 'score_diff': 4,
            'opp_run_before': 5, 'own_run_before': 0,
            'opp_fg_pct_before': 0.60, 'own_fg_pct_before': 0.40,
            'own_turnovers_before': 1, 'opp_turnovers_before': 0,
        },
        {
            'name': 'Q3 — GSW on massive 12-0 run, up 18',
            'period': 3, 'clock_seconds': 400, 'score_diff': 18,
            'opp_run_before': 0, 'own_run_before': 12,
            'opp_fg_pct_before': 0.25, 'own_fg_pct_before': 0.80,
            'own_turnovers_before': 0, 'opp_turnovers_before': 2,
        },
        {
            'name': 'Q3 — CLE fights back, 5-0 run (GSW up 14)',
            'period': 3, 'clock_seconds': 200, 'score_diff': 14,
            'opp_run_before': 5, 'own_run_before': 0,
            'opp_fg_pct_before': 0.50, 'own_fg_pct_before': 0.333,
            'own_turnovers_before': 1, 'opp_turnovers_before': 0,
        },
        {
            'name': 'Q4 — CLE 7-0 run, GSW still up 16',
            'period': 4, 'clock_seconds': 480, 'score_diff': 16,
            'opp_run_before': 7, 'own_run_before': 0,
            'opp_fg_pct_before': 0.667, 'own_fg_pct_before': 0.25,
            'own_turnovers_before': 2, 'opp_turnovers_before': 0,
        },
        {
            'name': 'Q4 — Garbage time, GSW up 22',
            'period': 4, 'clock_seconds': 120, 'score_diff': 22,
            'opp_run_before': 0, 'own_run_before': 4,
            'opp_fg_pct_before': 0.333, 'own_fg_pct_before': 0.50,
            'own_turnovers_before': 0, 'opp_turnovers_before': 1,
        },
    ]
    
    print("=" * 70)
    print(f"{'SCENARIO':<45} {'MODEL 1':^14} {'MODEL 2':^14} {'FINAL':^14}")
    print(f"{'':45} {'(Morale)':^14} {'(Stats)':^14} {'VERDICT':^14}")
    print("-" * 70)
    
    results = []
    
    for s in scenarios:
        name = s.pop('name')
        
        # Model 1: Morale
        m1_timeout, m1_avg_mom, m1_conf = morale_model_verdict(
            timeline, s['period'], s['clock_seconds']
        )
        
        # Model 2: Stats
        m2_timeout, m2_prob_beneficial, m2_conf = stats_model_verdict(stats_model, s)
        
        # Combined
        final_timeout, final_conf, agreement = combined_verdict(
            m1_timeout, m1_conf, m2_timeout, m2_conf
        )
        
        m1_str = "🟢 TIMEOUT" if m1_timeout else "🔴 NO"
        m2_str = "🟢 TIMEOUT" if m2_timeout else "🔴 NO"
        final_str = "✅ TIMEOUT" if final_timeout else "❌ NO"
        
        print(f"{name:<45} {m1_str:^14} {m2_str:^14} {final_str:^14}")
        
        results.append({
            'name': name,
            'm1_timeout': m1_timeout, 'm1_avg_momentum': m1_avg_mom, 'm1_confidence': m1_conf,
            'm2_timeout': m2_timeout, 'm2_prob_beneficial': m2_prob_beneficial, 'm2_confidence': m2_conf,
            'final_timeout': final_timeout, 'final_confidence': final_conf, 'agreement': agreement,
            **s
        })
        
        s['name'] = name  # restore
    
    # Detailed breakdown
    print("\n" + "=" * 70)
    print("DETAILED BREAKDOWN")
    print("=" * 70)
    
    for r in results:
        agree_icon = "🤝" if r['agreement'] == "AGREE" else "⚖️"
        print(f"\n{agree_icon} {r['name']}")
        print(f"   Model 1 (Morale):  {'CALL TIMEOUT' if r['m1_timeout'] else 'NO TIMEOUT':<15} "
              f"| Avg Momentum: {r['m1_avg_momentum']:+.3f} "
              f"({'negative → timeout' if r['m1_avg_momentum'] < 0 else 'positive → no timeout'}) "
              f"| Conf: {r['m1_confidence']:.0%}")
        print(f"   Model 2 (Stats):   {'CALL TIMEOUT' if r['m2_timeout'] else 'NO TIMEOUT':<15} "
              f"| P(beneficial): {r['m2_prob_beneficial']:.1%}  "
              f"| Conf: {r['m2_confidence']:.0%}")
        
        if r['agreement'] == "AGREE":
            print(f"   ➜ BOTH AGREE → {'✅ CALL TIMEOUT' if r['final_timeout'] else '❌ NO TIMEOUT'}")
        else:
            print(f"   ➜ MODELS DISAGREE → Weighted avg "
                  f"(Morale 30% + Stats 70%) = {r['final_confidence']:.2f} "
                  f"→ {'✅ CALL TIMEOUT' if r['final_timeout'] else '❌ NO TIMEOUT'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    timeout_count = sum(1 for r in results if r['final_timeout'])
    agree_count = sum(1 for r in results if r['agreement'] == 'AGREE')
    print(f"\n  Total scenarios: {len(results)}")
    print(f"  Timeouts recommended: {timeout_count}")
    print(f"  Models agreed: {agree_count}/{len(results)}")
    print(f"  Models disagreed: {len(results) - agree_count}/{len(results)}")
    
    print(f"\n  Weights: Model 1 (Morale) = 30% | Model 2 (Stats) = 70%")
    print(f"  Model 1 rule: avg momentum < 0 → call timeout (GSW losing momentum)")
    print(f"  Model 2 rule: XGBoost P(beneficial) > 50% → call timeout")


if __name__ == '__main__':
    main()
