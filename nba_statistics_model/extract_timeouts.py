"""
extract_timeouts.py — NBA Timeout Feature Extractor (v2)

Pulls play-by-play data from the NBA API and builds a dataset of every timeout
called during a season. For each timeout, we capture:

BEFORE the timeout (the "trigger"):
  - Opponent scoring run (unanswered points)
  - Own scoring run
  - Score differential at the moment
  - Period + time remaining
  - Opponent FG% over recent plays
  - Turnovers committed in recent plays

AFTER the timeout (the "result"):
  - Score differential change over next N plays
  - Did the team that called it score first after?
  - Did the opponent's run stop?

LABEL:
  - "beneficial" = positive score swing after timeout or opponent run stopped
"""

import pandas as pd
import numpy as np
import time
import os
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def parse_clock(clock_str):
    """Convert clock string like 'PT05M30.00S' to total seconds remaining."""
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


def get_score_at_action(pbp, idx):
    """
    Walk backward from idx to find the most recent *real* score.
    In PlayByPlayV3, non-scoring rows have scoreHome='0' and scoreAway='0',
    so we only trust scores on rows that are actual scoring plays.
    """
    for i in range(idx, -1, -1):
        row = pbp.iloc[i]
        action = str(row.get('actionType', ''))

        # Only trust score values on scoring plays
        if action not in ('Made Shot', 'Free Throw'):
            continue

        sh = row.get('scoreHome')
        sa = row.get('scoreAway')
        if pd.notnull(sh) and pd.notnull(sa):
            try:
                h = int(float(str(sh)))
                a = int(float(str(sa)))
                # Extra safety: scores should be non-negative
                if h >= 0 and a >= 0:
                    return h, a
            except (ValueError, TypeError):
                continue
    return 0, 0


def get_points_scored(row):
    """Return how many points were scored on this action (0 if none)."""
    action = str(row.get('actionType', ''))
    desc = str(row.get('description', '')).upper()

    if action == 'Made Shot':
        sv = row.get('shotValue')
        return int(sv) if pd.notnull(sv) else 2
    elif action == 'Free Throw' and 'MISS' not in desc:
        return 1
    return 0


def compute_run(pbp, end_idx, team_id, look_back=15):
    """
    Compute the unanswered scoring run for `team_id` looking backward from end_idx.
    """
    run = 0
    for i in range(end_idx - 1, max(end_idx - look_back - 1, -1), -1):
        row = pbp.iloc[i]
        t_id = int(row.get('teamId', 0)) if pd.notnull(row.get('teamId')) else 0
        pts = get_points_scored(row)

        if pts > 0:
            if t_id == team_id:
                run += pts
            else:
                break  # Other team scored — run is over
    return run


def compute_fg_pct(pbp, start_idx, end_idx, team_id):
    """Compute FG% for a team in a range of plays."""
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
    """Count turnovers by team_id in the last `look_back` plays before end_idx."""
    count = 0
    for i in range(end_idx - 1, max(end_idx - look_back - 1, -1), -1):
        row = pbp.iloc[i]
        action = str(row.get('actionType', ''))
        t_id = int(row.get('teamId', 0)) if pd.notnull(row.get('teamId')) else 0
        if t_id == team_id and action == 'Turnover':
            count += 1
    return count


def who_scores_next(pbp, after_idx):
    """Return the team_id of whoever scores first after after_idx, or 0."""
    for i in range(after_idx, min(after_idx + 25, len(pbp))):
        row = pbp.iloc[i]
        t_id = int(row.get('teamId', 0)) if pd.notnull(row.get('teamId')) else 0
        pts = get_points_scored(row)
        if pts > 0:
            return t_id
    return 0


def score_diff_change(pbp, timeout_idx, team_id, home_team_id, look_ahead=15):
    """
    Compute the change in score differential for `team_id` over the next
    `look_ahead` plays after the timeout.
    """
    h_before, a_before = get_score_at_action(pbp, timeout_idx)
    is_home = (team_id == home_team_id)
    diff_before = (h_before - a_before) if is_home else (a_before - h_before)

    end = min(timeout_idx + look_ahead, len(pbp) - 1)
    h_after, a_after = get_score_at_action(pbp, end)
    diff_after = (h_after - a_after) if is_home else (a_after - h_after)

    return diff_after - diff_before


# Comprehensive mapping of team names/nicknames that appear in PBP descriptions
# to their NBA team IDs
TEAM_NAME_TO_ID = {
    'WARRIORS': 1610612744, 'GOLDEN STATE': 1610612744, 'GSW': 1610612744,
    'CAVALIERS': 1610612739, 'CLEVELAND': 1610612739, 'CLE': 1610612739, 'CAVS': 1610612739,
    'CELTICS': 1610612738, 'BOSTON': 1610612738, 'BOS': 1610612738,
    'LAKERS': 1610612747, 'LOS ANGELES LAKERS': 1610612747, 'LAL': 1610612747,
    'HEAT': 1610612748, 'MIAMI': 1610612748, 'MIA': 1610612748,
    'SPURS': 1610612759, 'SAN ANTONIO': 1610612759, 'SAS': 1610612759,
    'BULLS': 1610612741, 'CHICAGO': 1610612741, 'CHI': 1610612741,
    'ROCKETS': 1610612745, 'HOUSTON': 1610612745, 'HOU': 1610612745,
    '76ERS': 1610612755, 'SIXERS': 1610612755, 'PHILADELPHIA': 1610612755, 'PHI': 1610612755,
    'BUCKS': 1610612749, 'MILWAUKEE': 1610612749, 'MIL': 1610612749,
    'NUGGETS': 1610612743, 'DENVER': 1610612743, 'DEN': 1610612743,
    'MAVERICKS': 1610612742, 'DALLAS': 1610612742, 'DAL': 1610612742, 'MAVS': 1610612742,
    'SUNS': 1610612756, 'PHOENIX': 1610612756, 'PHX': 1610612756,
    'TIMBERWOLVES': 1610612750, 'MINNESOTA': 1610612750, 'MIN': 1610612750, 'WOLVES': 1610612750,
    'THUNDER': 1610612760, 'OKLAHOMA CITY': 1610612760, 'OKC': 1610612760,
    'RAPTORS': 1610612761, 'TORONTO': 1610612761, 'TOR': 1610612761,
    'HAWKS': 1610612737, 'ATLANTA': 1610612737, 'ATL': 1610612737,
    'NETS': 1610612751, 'BROOKLYN': 1610612751, 'BKN': 1610612751,
    'HORNETS': 1610612766, 'CHARLOTTE': 1610612766, 'CHA': 1610612766,
    'PISTONS': 1610612765, 'DETROIT': 1610612765, 'DET': 1610612765,
    'PACERS': 1610612754, 'INDIANA': 1610612754, 'IND': 1610612754,
    'GRIZZLIES': 1610612763, 'MEMPHIS': 1610612763, 'MEM': 1610612763,
    'PELICANS': 1610612740, 'NEW ORLEANS': 1610612740, 'NOP': 1610612740,
    'KNICKS': 1610612752, 'NEW YORK': 1610612752, 'NYK': 1610612752,
    'MAGIC': 1610612753, 'ORLANDO': 1610612753, 'ORL': 1610612753,
    'TRAIL BLAZERS': 1610612757, 'BLAZERS': 1610612757, 'PORTLAND': 1610612757, 'POR': 1610612757,
    'KINGS': 1610612758, 'SACRAMENTO': 1610612758, 'SAC': 1610612758,
    'JAZZ': 1610612762, 'UTAH': 1610612762, 'UTA': 1610612762,
    'WIZARDS': 1610612764, 'WASHINGTON': 1610612764, 'WAS': 1610612764,
    'CLIPPERS': 1610612746, 'LOS ANGELES CLIPPERS': 1610612746, 'LAC': 1610612746,
}


def detect_team_from_description(description, game_team_id, game_opp_team_id):
    """
    In PlayByPlayV3, timeout rows have teamId=0.
    The team is identified from the description, e.g. 'WARRIORS Timeout: Regular...'
    """
    desc_upper = str(description).upper()

    # Extract team name from description (format: "TEAM_NAME Timeout: ...")
    if 'TIMEOUT' not in desc_upper:
        return 0

    team_part = desc_upper.split('TIMEOUT')[0].strip()
    if not team_part:
        return 0

    # Look up in our comprehensive name mapping
    detected_id = TEAM_NAME_TO_ID.get(team_part, 0)

    if detected_id == 0:
        # Try partial matching — some names might be slightly different
        for name, tid in TEAM_NAME_TO_ID.items():
            if name in team_part or team_part in name:
                detected_id = tid
                break

    # Verify it's one of the two teams in this game
    if detected_id == game_team_id or detected_id == game_opp_team_id:
        return detected_id

    return 0


# ---------------------------------------------------------------------------
# MAIN EXTRACTION
# ---------------------------------------------------------------------------

def extract_timeouts_from_game(game_id, team_id, home_team_id, opp_team_id):
    """
    Extract all timeouts (by either team) in a game, with pre/post features.
    Returns data for timeouts called by BOTH teams (we label which team called it).
    """
    print(f"  Fetching PBP for game {game_id}...")
    # Retry with exponential backoff for rate limiting
    for attempt in range(3):
        try:
            pbp_obj = playbyplayv3.PlayByPlayV3(game_id=game_id)
            pbp = pbp_obj.get_data_frames()[0]
            break
        except Exception as e:
            wait_time = 10 * (2 ** attempt)  # 10s, 20s, 40s
            print(f"    Retry {attempt+1}/3 after {wait_time}s: {e}")
            import time as _time
            _time.sleep(wait_time)
            if attempt == 2:
                raise
    pbp = pbp.sort_values(by=['actionNumber']).reset_index(drop=True)

    timeouts = []

    for idx, row in pbp.iterrows():
        action_type = str(row.get('actionType', ''))
        sub_type = str(row.get('subType', '')).lower() if pd.notnull(row.get('subType')) else ''
        description = str(row.get('description', ''))

        # Skip non-timeout actions, and skip official/TV timeouts
        if action_type != 'Timeout':
            continue
        if sub_type == 'official':
            continue

        # Detect which team called the timeout
        calling_team = detect_team_from_description(description, team_id, opp_team_id)
        if calling_team == 0:
            continue  # Can't determine who called it

        is_our_team = (calling_team == team_id)
        # For feature engineering, "own" is whoever called the timeout
        own_team = calling_team
        opp_team = opp_team_id if is_our_team else team_id

        # ---- PRE-TIMEOUT FEATURES ----
        period = int(row.get('period', 0))
        clock_seconds = parse_clock(row.get('clock'))
        h_score, a_score = get_score_at_action(pbp, idx)
        is_home = (own_team == home_team_id)
        score_diff = (h_score - a_score) if is_home else (a_score - h_score)

        opp_run = compute_run(pbp, idx, opp_team, look_back=15)
        own_run = compute_run(pbp, idx, own_team, look_back=15)

        opp_fg_before = compute_fg_pct(pbp, max(0, idx - 15), idx, opp_team)
        own_fg_before = compute_fg_pct(pbp, max(0, idx - 15), idx, own_team)

        own_turnovers_before = count_turnovers(pbp, idx, own_team, look_back=8)
        opp_turnovers_before = count_turnovers(pbp, idx, opp_team, look_back=8)

        # ---- POST-TIMEOUT FEATURES ----
        diff_change = score_diff_change(pbp, idx, own_team, home_team_id, look_ahead=15)

        first_scorer = who_scores_next(pbp, idx + 1)
        team_scores_first = 1 if first_scorer == own_team else 0

        opp_run_after = compute_run(pbp, min(idx + 16, len(pbp)), opp_team, look_back=15)

        # ---- LABEL: WAS IT BENEFICIAL? ----
        # Beneficial if: positive score swing, or stopped a big opponent run
        beneficial = 1 if (diff_change > 0) or (opp_run >= 7 and opp_run_after <= 2) else 0

        timeouts.append({
            'game_id': game_id,
            'period': period,
            'clock_seconds': round(clock_seconds, 1),
            'score_diff': score_diff,
            'opp_run_before': opp_run,
            'own_run_before': own_run,
            'opp_fg_pct_before': round(opp_fg_before, 3),
            'own_fg_pct_before': round(own_fg_before, 3),
            'own_turnovers_before': own_turnovers_before,
            'opp_turnovers_before': opp_turnovers_before,
            'diff_change_after': diff_change,
            'team_scores_first_after': team_scores_first,
            'opp_run_stopped': 1 if (opp_run >= 5 and opp_run_after <= 2) else 0,
            'beneficial': beneficial,
            'calling_team': 'own' if is_our_team else 'opp',
            'description': description,
        })

    return timeouts


def extract_season_timeouts(team_id, season='2016-17', max_games=None):
    """
    Extract timeout data for games involving a team across a season.
    """
    print(f"Finding games for team {team_id} in {season}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable='Regular Season'
    )
    games = gamefinder.get_data_frames()[0]

    if max_games:
        games = games.head(max_games)

    print(f"Found {len(games)} games. Extracting timeouts...\n")

    # Get ALL teams' games to find opponent IDs efficiently (one API call)
    print("Loading league game index for opponent lookup...")
    all_games = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable='Regular Season'
    ).get_data_frames()[0]
    time.sleep(1)

    all_timeouts = []

    for _, game in games.iterrows():
        game_id = game['GAME_ID']
        matchup = game['MATCHUP']

        # Find opponent from the all_games dataframe
        game_rows = all_games[all_games['GAME_ID'] == game_id]
        opp_row = game_rows[game_rows['TEAM_ID'] != team_id]

        if len(opp_row) == 0:
            print(f"  Skipping {game_id} — can't find opponent")
            continue

        opp_team_id = int(opp_row.iloc[0]['TEAM_ID'])

        # Determine home/away
        is_home = 'vs.' in matchup
        home_team_id = team_id if is_home else opp_team_id

        try:
            game_timeouts = extract_timeouts_from_game(
                game_id, team_id, home_team_id, opp_team_id
            )
            for t in game_timeouts:
                t['matchup'] = matchup
                t['outcome'] = 1 if game['WL'] == 'W' else 0

            all_timeouts.extend(game_timeouts)
            print(f"  {matchup}: found {len(game_timeouts)} timeouts ✓")
        except Exception as e:
            print(f"  Error processing {game_id}: {e}")

        time.sleep(1.5)  # Rate limit — be gentle with NBA API

    return pd.DataFrame(all_timeouts)


# ---------------------------------------------------------------------------
# TEAM IDS
# ---------------------------------------------------------------------------

TEAM_IDS = {
    'GSW': 1610612744, 'CLE': 1610612739, 'BOS': 1610612738,
    'LAL': 1610612747, 'MIA': 1610612748, 'SAS': 1610612759,
    'CHI': 1610612741, 'HOU': 1610612745, 'PHI': 1610612755,
    'MIL': 1610612749, 'DEN': 1610612743, 'DAL': 1610612742,
    'PHX': 1610612756, 'MIN': 1610612750, 'OKC': 1610612760,
    'TOR': 1610612761, 'ATL': 1610612737, 'BKN': 1610612751,
    'CHA': 1610612766, 'DET': 1610612765, 'IND': 1610612754,
    'MEM': 1610612763, 'NOP': 1610612740, 'NYK': 1610612752,
    'ORL': 1610612753, 'POR': 1610612757, 'SAC': 1610612758,
    'UTA': 1610612762, 'WAS': 1610612764, 'LAC': 1610612746,
}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract NBA timeout data')
    parser.add_argument('--team', default='GSW', help='Team abbreviation (e.g. GSW, CLE, BOS)')
    parser.add_argument('--season', default='2016-17', help='Season (e.g. 2016-17)')
    parser.add_argument('--max-games', type=int, default=10, help='Max games to process')
    parser.add_argument('--output', default='timeout_data.csv', help='Output CSV file')
    args = parser.parse_args()

    team_id = TEAM_IDS.get(args.team.upper())
    if not team_id:
        print(f"Unknown team '{args.team}'. Available: {list(TEAM_IDS.keys())}")
        exit(1)

    df = extract_season_timeouts(team_id, args.season, args.max_games)

    if len(df) > 0:
        print(f"\n{'='*60}")
        print(f"Extracted {len(df)} timeouts from {args.max_games or 'all'} games")
        print(f"Beneficial: {df['beneficial'].sum()} / {len(df)} ({df['beneficial'].mean()*100:.1f}%)")
        print(f"{'='*60}")

        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")
        print(df.head(10).to_string())
    else:
        print("\nNo timeouts extracted. Check the data.")
