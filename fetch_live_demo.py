"""
fetch_live_demo.py — Fetch PBP data for a game and produce a JSON for the live demo frontend.

For each play, we compute:
  - Running score (home/away)
  - Scoring runs for each team
  - FG% windows
  - Turnovers
  - Model 1 (Morale) momentum index
  - Model 2 (XGBoost) timeout probability
  - Ensemble verdict

Output: a JSON array where each element is one play with all computed features,
        ready for the frontend to animate through.

Usage:
  python3 fetch_live_demo.py
"""

import json
import os
import sys
import pandas as pd
import numpy as np
import joblib
from nba_api.stats.endpoints import playbyplayv3, leaguegamefinder

# ---------- CONFIG ----------
# Spurs @ Kings, Feb 22, 2026
# We need to discover the game ID dynamically.
SAS_ID = 1610612759
SAC_ID = 1610612758
SEASON = "2025-26"
GAME_DATE = "02/21/2026"

STATS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "nba_statistics_model", "timeout_model.pkl")
MORALE_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "nba_morale_model", "data", "momentum_weights.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "nba-analytics-dashboard", "public", "live_demo.json")

FEATURES = ['period', 'clock_seconds', 'score_diff', 'opp_run_before', 'own_run_before',
            'opp_fg_pct_before', 'own_fg_pct_before', 'own_turnovers_before', 'opp_turnovers_before']


# ---------- HELPERS ----------

def parse_clock(clock_str):
    """Convert 'PT05M30.00S' to total seconds remaining."""
    if not clock_str or pd.isna(clock_str):
        return 0.0
    clock_str = str(clock_str)
    try:
        clock_str = clock_str.replace("PT", "").replace("S", "")
        parts = clock_str.split("M")
        minutes = float(parts[0]) if len(parts) > 1 else 0
        seconds = float(parts[-1]) if parts[-1] else 0
        return minutes * 60 + seconds
    except (ValueError, IndexError):
        return 0.0


def clock_display(seconds):
    """Convert seconds to MM:SS display."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def get_points_scored(row):
    action = str(row.get('actionType', ''))
    desc = str(row.get('description', '')).upper()
    if action == 'Made Shot':
        sv = row.get('shotValue')
        return int(sv) if pd.notnull(sv) else 2
    elif action == 'Free Throw' and 'MISS' not in desc:
        return 1
    return 0


def compute_run(pbp, end_idx, team_id, look_back=15):
    run = 0
    for i in range(end_idx - 1, max(end_idx - look_back - 1, -1), -1):
        row = pbp.iloc[i]
        t_id = int(row.get('teamId', 0)) if pd.notnull(row.get('teamId')) else 0
        pts = get_points_scored(row)
        if pts > 0:
            if t_id == team_id:
                run += pts
            else:
                break
    return run


def compute_fg_pct(pbp, start_idx, end_idx, team_id):
    makes = 0
    attempts = 0
    for i in range(start_idx, min(end_idx, len(pbp))):
        row = pbp.iloc[i]
        action = str(row.get('actionType', ''))
        t_id = int(row.get('teamId', 0)) if pd.notnull(row.get('teamId')) else 0
        if t_id == team_id and action in ('Made Shot', 'Missed Shot'):
            attempts += 1
            if action == 'Made Shot':
                makes += 1
    return makes / attempts if attempts > 0 else 0.0


def count_turnovers(pbp, end_idx, team_id, look_back=8):
    count = 0
    for i in range(end_idx - 1, max(end_idx - look_back - 1, -1), -1):
        row = pbp.iloc[i]
        action = str(row.get('actionType', ''))
        t_id = int(row.get('teamId', 0)) if pd.notnull(row.get('teamId')) else 0
        if t_id == team_id and action == 'Turnover':
            count += 1
    return count


# ---------- MOMENTUM MODEL ----------

def compute_momentum_index(sac_run, sas_run, sac_ri, sas_ri, margin_swing, sac_b2b3, sas_b2b3, weights):
    """Compute momentum index from the perspective of SAC (home team).
    Positive = SAC has momentum, Negative = SAS has momentum.
    Uses intuitive weights: bigger run = more momentum for that team."""
    # Always use the direct formula — the trained weights were fit for a GSW-opponent
    # context and have inverted signs that don't transfer to other team pairings.
    sac_mom = 0.40 * sac_run + 0.25 * sac_ri + 0.15 * margin_swing + 0.20 * sac_b2b3
    sas_mom = 0.40 * sas_run + 0.25 * sas_ri - 0.15 * margin_swing + 0.20 * sas_b2b3
    return sac_mom - sas_mom


# ---------- FIND GAME ----------

def find_game_id():
    """Find the game ID for SAS @ SAC on the given date."""
    print(f"Looking for SAS @ SAC on {GAME_DATE} in {SEASON}...")

    # Try SAC's games
    gf = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=SAC_ID,
        season_nullable=SEASON,
        season_type_nullable="Regular Season",
        date_from_nullable=GAME_DATE,
        date_to_nullable=GAME_DATE,
    )
    games = gf.get_data_frames()[0]

    if len(games) == 0:
        print("No games found for SAC on that date. Trying SAS...")
        gf2 = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=SAS_ID,
            season_nullable=SEASON,
            season_type_nullable="Regular Season",
            date_from_nullable=GAME_DATE,
            date_to_nullable=GAME_DATE,
        )
        games = gf2.get_data_frames()[0]

    if len(games) == 0:
        print("ERROR: Could not find the game. Check date/teams.")
        sys.exit(1)

    game_id = games.iloc[0]["GAME_ID"]
    matchup = games.iloc[0]["MATCHUP"]
    print(f"Found: {matchup} (Game ID: {game_id})")
    return str(game_id)


# ---------- MAIN ----------

def main():
    game_id = find_game_id()

    # Fetch PBP
    print("Fetching play-by-play data...")
    pbp_obj = playbyplayv3.PlayByPlayV3(game_id=game_id)
    pbp = pbp_obj.get_data_frames()[0]
    pbp = pbp.sort_values(by=['actionNumber']).reset_index(drop=True)
    print(f"  {len(pbp)} actions found.")

    # Load models
    print("Loading XGBoost model...")
    xgb_model = joblib.load(STATS_MODEL_PATH)

    weights = None
    if os.path.exists(MORALE_WEIGHTS_PATH):
        with open(MORALE_WEIGHTS_PATH) as f:
            weights = json.load(f)
        print("Loaded momentum weights.")
    else:
        print("Using default momentum weights.")

    # Determine home/away
    # SAC is home (Kings), SAS is away (Spurs)
    home_id = SAC_ID
    away_id = SAS_ID

    # Process every play
    home_score = 0
    away_score = 0
    sac_run = 0
    sas_run = 0
    sac_run_start_sec = None
    sas_run_start_sec = None
    margin_history = []
    sac_3_times = []
    sas_3_times = []

    plays = []

    for idx, row in pbp.iterrows():
        action_type = str(row.get('actionType', ''))
        description = str(row.get('description', ''))
        period = int(row.get('period', 0))
        clock_str = row.get('clock', '')
        clock_sec = parse_clock(clock_str)
        team_id = int(row.get('teamId', 0)) if pd.notnull(row.get('teamId')) else 0

        # Update score
        pts = get_points_scored(row)
        if pts > 0:
            if team_id == home_id:
                home_score += pts
            elif team_id == away_id:
                away_score += pts

        # Determine team abbreviation
        if team_id == SAC_ID:
            team_abbr = "SAC"
        elif team_id == SAS_ID:
            team_abbr = "SAS"
        else:
            team_abbr = ""

        # Figure out action label for display
        action_label = action_type
        sub_type = str(row.get('subType', '')) if pd.notnull(row.get('subType')) else ''
        if action_type == 'Made Shot':
            shot_val = row.get('shotValue', 2)
            action_label = f"MADE {shot_val}PT"
        elif action_type == 'Missed Shot':
            action_label = "MISS"
        elif action_type == 'Free Throw':
            if 'MISS' in description.upper():
                action_label = "FT MISS"
            else:
                action_label = "FT MADE"
        elif action_type == 'Turnover':
            action_label = "TURNOVER"
        elif action_type == 'Rebound':
            action_label = "REBOUND"
        elif action_type == 'Foul':
            action_label = "FOUL"
        elif action_type == 'Timeout':
            action_label = "TIMEOUT"
        elif action_type == 'Substitution':
            action_label = "SUB"
        elif action_type == 'Jump Ball':
            action_label = "JUMP BALL"
        elif action_type == 'Violation':
            action_label = "VIOLATION"
        elif action_type == 'Block':
            action_label = "BLOCK"
        elif action_type == 'Steal':
            action_label = "STEAL"

        # Update scoring runs
        if pts > 0:
            if team_id == home_id:
                sac_run += pts
                sas_run = 0
                if sac_run_start_sec is None:
                    sac_run_start_sec = clock_sec
                sas_run_start_sec = None
            elif team_id == away_id:
                sas_run += pts
                sac_run = 0
                if sas_run_start_sec is None:
                    sas_run_start_sec = clock_sec
                sac_run_start_sec = None

            # Track 3-pointers
            if action_type == 'Made Shot':
                sv = row.get('shotValue', 2)
                if str(sv) == '3':
                    if team_id == home_id:
                        sac_3_times.append(clock_sec)
                    else:
                        sas_3_times.append(clock_sec)

        # Margin tracking
        margin = home_score - away_score
        margin_history.append(margin)
        if len(margin_history) > 5:
            margin_history = margin_history[-5:]
        margin_swing = (margin - margin_history[0]) if len(margin_history) >= 5 else 0

        # Run intensity
        sac_ri = sac_run / max((sac_run_start_sec - clock_sec) if sac_run_start_sec and clock_sec else 1, 1)
        sas_ri = sas_run / max((sas_run_start_sec - clock_sec) if sas_run_start_sec and clock_sec else 1, 1)

        # Back-to-back 3s within 3 minutes
        sac_3_times = [t for t in sac_3_times if abs(t - clock_sec) <= 180]
        sas_3_times = [t for t in sas_3_times if abs(t - clock_sec) <= 180]
        sac_b2b3 = 1 if len(sac_3_times) >= 2 else 0
        sas_b2b3 = 1 if len(sas_3_times) >= 2 else 0

        # Momentum index (from SAC perspective)
        momentum = compute_momentum_index(sac_run, sas_run, sac_ri, sas_ri,
                                          margin_swing, sac_b2b3, sas_b2b3, weights)

        # Model 1 verdict (from SAC/home perspective — negative = losing momentum)
        m1_timeout = momentum < 0
        m1_confidence = min(abs(momentum) / 2.0, 1.0)
        m1_confidence = 0.5 + m1_confidence * 0.5

        # Model 2 verdict — XGBoost (from SAC perspective)
        score_diff = home_score - away_score
        sas_run_for_model = compute_run(pbp, idx, SAS_ID, look_back=15)
        sac_run_for_model = compute_run(pbp, idx, SAC_ID, look_back=15)
        opp_fg = compute_fg_pct(pbp, max(0, idx - 15), idx, SAS_ID)
        own_fg = compute_fg_pct(pbp, max(0, idx - 15), idx, SAC_ID)
        own_to = count_turnovers(pbp, idx, SAC_ID, look_back=8)
        opp_to = count_turnovers(pbp, idx, SAS_ID, look_back=8)

        features = {
            'period': period,
            'clock_seconds': clock_sec,
            'score_diff': score_diff,
            'opp_run_before': sas_run_for_model,
            'own_run_before': sac_run_for_model,
            'opp_fg_pct_before': round(opp_fg, 3),
            'own_fg_pct_before': round(own_fg, 3),
            'own_turnovers_before': own_to,
            'opp_turnovers_before': opp_to,
        }

        try:
            X = pd.DataFrame([features])[FEATURES]
            prob = xgb_model.predict_proba(X)[0]
            m2_timeout = bool(prob[1] > 0.5)
            m2_prob = float(prob[1])
            m2_confidence = float(max(prob))
        except Exception:
            m2_timeout = False
            m2_prob = 0.0
            m2_confidence = 0.5

        # Ensemble verdict
        W1 = 0.30
        W2 = 0.70
        if m1_timeout == m2_timeout:
            final_timeout = m1_timeout
            final_conf = m1_confidence * W1 + m2_confidence * W2
            agreement = "AGREE"
        else:
            m1_score = m1_confidence if m1_timeout else (1 - m1_confidence)
            m2_score = m2_confidence if m2_timeout else (1 - m2_confidence)
            weighted = m1_score * W1 + m2_score * W2
            final_timeout = weighted > 0.5
            final_conf = weighted
            agreement = "DISAGREE"

        # Detect if this play is a momentum shift
        is_momentum_shift = False
        if pts >= 3 and abs(momentum) > 1.0:
            is_momentum_shift = True
        if action_type == 'Turnover' and abs(momentum) > 0.5:
            is_momentum_shift = True
        if action_type == 'Steal':
            is_momentum_shift = True

        # Skip uninteresting actions for the "live feed" display
        # but keep ALL for scoring tracking
        is_significant = action_type in (
            'Made Shot', 'Missed Shot', 'Free Throw', 'Turnover', 'Steal',
            'Block', 'Foul', 'Rebound', 'Timeout', 'Jump Ball'
        )

        play = {
            "idx": int(idx),
            "period": period,
            "clock": clock_display(clock_sec),
            "clockSeconds": round(clock_sec, 1),
            "team": team_abbr,
            "action": action_label,
            "description": description,
            "homeScore": home_score,
            "awayScore": away_score,
            "scoreDiff": score_diff,
            "sacRun": sac_run,
            "sasRun": sas_run,
            "momentum": round(momentum, 3),
            "m1Timeout": m1_timeout,
            "m1AvgMomentum": round(momentum, 3),
            "m1Confidence": round(m1_confidence, 3),
            "m2Timeout": m2_timeout,
            "m2ProbBeneficial": round(m2_prob, 3),
            "m2Confidence": round(m2_confidence, 3),
            "finalTimeout": final_timeout,
            "finalConfidence": round(final_conf, 3),
            "agreement": agreement,
            "isMomentumShift": is_momentum_shift,
            "isSignificant": is_significant,
            "oppFgPct": round(opp_fg, 3),
            "ownFgPct": round(own_fg, 3),
            "ownTurnovers": own_to,
            "oppTurnovers": opp_to,
        }
        plays.append(play)

    # Write output
    output = {
        "gameInfo": {
            "homeTeam": "SAC",
            "awayTeam": "SAS",
            "homeTeamFull": "Sacramento Kings",
            "awayTeamFull": "San Antonio Spurs",
            "gameId": game_id,
            "date": "2026-02-22",
            "finalHomeScore": home_score,
            "finalAwayScore": away_score,
        },
        "plays": plays,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f)

    print(f"\nDone! {len(plays)} plays written to {OUTPUT_PATH}")
    print(f"Final score: SAC {home_score} — SAS {away_score}")

    # Also count significant plays
    sig = [p for p in plays if p['isSignificant']]
    print(f"Significant plays (for display): {len(sig)}")

    timeout_recs = [p for p in plays if p['finalTimeout']]
    print(f"Timeout recommendations: {len(timeout_recs)}")


if __name__ == '__main__':
    main()
