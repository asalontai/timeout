import time
import pandas as pd
from nba_api.stats.static import teams

from config import DATA_DIR


def main():
    team_list = teams.get_teams()
    df = pd.DataFrame(team_list)

    # Keep only NBA teams
    df = df[df["is_nba"] == True].copy()

    out = df[["id", "full_name", "abbreviation", "nickname", "city"]].copy()
    out = out.rename(columns={"id": "TEAM_ID", "full_name": "TEAM_NAME"})

    out_path = DATA_DIR / "team_id_map.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} teams to {out_path}")


if __name__ == "__main__":
    main()
