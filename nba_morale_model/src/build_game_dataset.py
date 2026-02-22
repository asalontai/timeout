import pandas as pd
from pathlib import Path
from collections import deque

from config import FEATURES_CSV, GAMES_CSV, DATA_DIR

TEAM_ABBR_TO_NAME = {
    "ATL": "Hawks",
    "BOS": "Celtics",
    "BKN": "Nets",
    "BRK": "Nets",
    "CHA": "Hornets",
    "CHH": "Hornets",
    "CHI": "Bulls",
    "CLE": "Cavaliers",
    "DAL": "Mavericks",
    "DEN": "Nuggets",
    "DET": "Pistons",
    "GSW": "Warriors",
    "HOU": "Rockets",
    "IND": "Pacers",
    "LAC": "Clippers",
    "LAL": "Lakers",
    "MEM": "Grizzlies",
    "MIA": "Heat",
    "MIL": "Bucks",
    "MIN": "Timberwolves",
    "NOP": "Pelicans",
    "NOH": "Hornets",
    "NYK": "Knicks",
    "OKC": "Thunder",
    "ORL": "Magic",
    "PHI": "76ers",
    "PHX": "Suns",
    "POR": "Blazers",
    "SAC": "Kings",
    "SAS": "Spurs",
    "TOR": "Raptors",
    "UTA": "Jazz",
    "WAS": "Wizards",
}


def opponent_abbr(matchup: str) -> str:
    if not isinstance(matchup, str):
        return ""
    parts = matchup.split(" ")
    return parts[-1].strip()


def build_recent_injury_counts(inj: pd.DataFrame, team_name: str, dates, window_days=7):
    events = inj[inj["Team"] == team_name].copy()
    if events.empty:
        return {d: 0 for d in dates}

    events["Date"] = pd.to_datetime(events["Date"]).dt.date
    events = events.sort_values("Date")

    active = {}
    counts = {}
    idx = 0
    event_rows = events.to_dict("records")

    for d in dates:
        # Add events up to date
        while idx < len(event_rows) and event_rows[idx]["Date"] <= d:
            row = event_rows[idx]
            acquired = row.get("Acquired")
            relinquised = row.get("Relinquised")

            if isinstance(acquired, str) and acquired.strip():
                active[acquired.strip()] = row["Date"]
            if isinstance(relinquised, str) and relinquised.strip():
                active.pop(relinquised.strip(), None)

            idx += 1

        # Drop injuries older than window
        cutoff = pd.to_datetime(d) - pd.Timedelta(days=window_days)
        cutoff = cutoff.date()
        active = {p: dt for p, dt in active.items() if dt >= cutoff}

        counts[d] = len(active)

    return counts


def compute_gsw_win_pct(games: pd.DataFrame) -> pd.Series:
    wins = 0
    total = 0
    win_pct = []
    last_season = None
    for _, row in games.iterrows():
        season = row.get("SEASON")
        if season != last_season:
            wins = 0
            total = 0
            last_season = season
        if row.get("WL") in ("W", "L"):
            total += 1
            if row.get("WL") == "W":
                wins += 1
        win_pct.append(wins / total if total > 0 else 0)
    return pd.Series(win_pct, index=games.index)


def main():
    games = pd.read_csv(GAMES_CSV)
    features = pd.read_csv(FEATURES_CSV)
    injuries_path = DATA_DIR / "injuries.csv"
    if not injuries_path.exists():
        raise RuntimeError(f"Missing injuries file: {injuries_path}")

    injuries = pd.read_csv(injuries_path)

    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.date
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    games["opponent_abbr"] = games["MATCHUP"].apply(opponent_abbr)
    games["opponent_name"] = games["opponent_abbr"].map(TEAM_ABBR_TO_NAME)
    games["team_name"] = "Warriors"
    games["is_home"] = games["MATCHUP"].str.contains("vs.").astype(int)

    # GSW strength control: running win pct
    games["gsw_win_pct_to_date"] = compute_gsw_win_pct(games)

    # Injury counts per team per game date (7-day rolling window)
    unique_dates = sorted(games["GAME_DATE"].unique())

    gsw_counts = build_recent_injury_counts(injuries, "Warriors", unique_dates, window_days=7)
    opp_counts_cache = {}

    opp_counts = []
    for _, row in games.iterrows():
        opp = row["opponent_name"]
        if opp not in opp_counts_cache:
            opp_counts_cache[opp] = build_recent_injury_counts(injuries, opp, unique_dates, window_days=7)
        opp_counts.append(opp_counts_cache[opp].get(row["GAME_DATE"], 0))

    games["gsw_injuries_7d"] = games["GAME_DATE"].map(gsw_counts)
    games["opp_injuries_7d"] = opp_counts

    # Aggregate 3s run features by game
    agg = features.groupby("game_id").agg(
        gsw_3s_run_any=("gsw_3s_run_flag", "max"),
        gsw_3s_run_count=("gsw_3s_run_flag", "sum"),
        gsw_max_consec_3s=("gsw_consecutive_3s", "max"),
        avg_msi=("msi", "mean"),
        max_msi=("msi", "max"),
    ).reset_index()

    out = games.merge(agg, left_on="GAME_ID", right_on="game_id", how="left")

    out_path = DATA_DIR / "game_features.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
