import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

from config import SEASONS, TEAM_ID, GAMES_CSV


def main():
    all_games = []
    for season in SEASONS:
        finder = leaguegamefinder.LeagueGameFinder(team_id_nullable=TEAM_ID, season_nullable=season)
        games = finder.get_data_frames()[0]

        # Normalize columns
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.date.astype(str)
        games["SEASON"] = season

        # Keep only regular season (exclude playoffs)
        if "SEASON_TYPE" in games.columns:
            games = games[games["SEASON_TYPE"] == "Regular Season"].copy()
        else:
            # SEASON_ID format: "2YYYY" regular season
            games = games[games["SEASON_ID"].astype(str).str.startswith("2")].copy()

        all_games.append(games)
        time.sleep(0.6)

    games = pd.concat(all_games, ignore_index=True)

    # Sort chronologically
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    GAMES_CSV.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(GAMES_CSV, index=False)

    print(f"Saved {len(games)} games to {GAMES_CSV}")


if __name__ == "__main__":
    main()
