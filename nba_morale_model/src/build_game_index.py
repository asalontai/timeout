import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder
from tqdm import tqdm

from config import SEASONS, DATA_DIR


def main():
    all_games = []
    for season in SEASONS:
        finder = leaguegamefinder.LeagueGameFinder(season_nullable=season, season_type_nullable="Regular Season")
        games = finder.get_data_frames()[0]

        # Normalize date
        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.date.astype(str)
        games["SEASON"] = season

        # We need one row per game with home/away team IDs
        # The endpoint returns two rows per game (one for each team)
        subset = games[["GAME_ID", "TEAM_ID", "GAME_DATE", "MATCHUP", "SEASON"]].copy()
        subset["IS_HOME"] = subset["MATCHUP"].str.contains("vs.").astype(int)

        # Pivot to home/away IDs
        home = subset[subset["IS_HOME"] == 1][["GAME_ID", "TEAM_ID", "GAME_DATE", "SEASON"]]
        away = subset[subset["IS_HOME"] == 0][["GAME_ID", "TEAM_ID"]]
        home = home.rename(columns={"TEAM_ID": "HOME_TEAM_ID"})
        away = away.rename(columns={"TEAM_ID": "AWAY_TEAM_ID"})

        merged = home.merge(away, on="GAME_ID", how="inner")
        all_games.append(merged)

        time.sleep(0.6)

    out = pd.concat(all_games, ignore_index=True)
    out_path = DATA_DIR / "game_index.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} games to {out_path}")


if __name__ == "__main__":
    main()
