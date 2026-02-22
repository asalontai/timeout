import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import PBP_DIR, DATA_DIR

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


def load_pbp_all():
    files = sorted(PBP_DIR.glob("datanba_*.csv"))
    if not files:
        raise RuntimeError("No PBP CSVs found in data/nba_data")
    frames = []
    for p in files:
        df = pd.read_csv(p)
        stem = p.stem
        season = int(stem.split("_")[-1])
        season_type = "playoffs" if "po" in stem else "regular"
        df["season"] = season
        df["season_type"] = season_type
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clock_to_sec(cl):
    if not isinstance(cl, str) or ":" not in cl:
        return None
    try:
        mm, ss = cl.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return None


def infer_team_is_home(df: pd.DataFrame, team_id: int) -> bool:
    last_home = 0
    last_away = 0
    for _, row in df.iterrows():
        home = pd.to_numeric(row.get("hs"), errors="coerce")
        away = pd.to_numeric(row.get("vs"), errors="coerce")
        if pd.isna(home) or pd.isna(away):
            continue
        home = int(home)
        away = int(away)
        if home == last_home and away == last_away:
            continue
        tid = row.get("tid")
        if tid == team_id:
            if home > last_home and away == last_away:
                return True
            if away > last_away and home == last_home:
                return False
        last_home = home
        last_away = away
    return True


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
        while idx < len(event_rows) and event_rows[idx]["Date"] <= d:
            row = event_rows[idx]
            acquired = row.get("Acquired")
            relinquised = row.get("Relinquised")

            if isinstance(acquired, str) and acquired.strip():
                active[acquired.strip()] = row["Date"]
            if isinstance(relinquised, str) and relinquised.strip():
                active.pop(relinquised.strip(), None)

            idx += 1

        cutoff = pd.to_datetime(d) - pd.Timedelta(days=window_days)
        cutoff = cutoff.date()
        active = {p: dt for p, dt in active.items() if dt >= cutoff}

        counts[d] = len(active)

    return counts


def main():
    pbp = load_pbp_all()

    # Load injuries
    injuries_path = DATA_DIR / "injuries.csv"
    injuries = pd.read_csv(injuries_path)

    # Load game index with dates and home/away
    game_index_path = DATA_DIR / "game_index.csv"
    if not game_index_path.exists():
        raise RuntimeError("Missing game_index.csv. Run build_game_index.py on a machine with NBA API access.")
    game_index = pd.read_csv(game_index_path)
    game_index["GAME_DATE"] = pd.to_datetime(game_index["GAME_DATE"]).dt.date

    # Load tuned weights if available
    weights_path = DATA_DIR / "momentum_weights.json"
    weights = None
    if weights_path.exists():
        with open(weights_path, "r") as f:
            weights = json.load(f)

    out_rows = []

    game_ids = pbp["GAME_ID"].astype(str).unique().tolist()

    # Precompute injury counts by date for all teams
    unique_dates = sorted(game_index["GAME_DATE"].unique())
    team_names = sorted(injuries["Team"].dropna().unique())
    injury_counts = {team: build_recent_injury_counts(injuries, team, unique_dates, window_days=7) for team in team_names}

    game_index_map = game_index.set_index("GAME_ID").to_dict("index")

    for game_id in tqdm(game_ids, total=len(game_ids)):
        if str(game_id) not in game_index_map:
            continue
        meta = game_index_map[str(game_id)]
        game_date = meta["GAME_DATE"]
        home_id = meta["HOME_TEAM_ID"]
        away_id = meta["AWAY_TEAM_ID"]

        df = pbp[pbp["GAME_ID"].astype(str) == str(game_id)].copy()
        if df.empty:
            continue

        df = df.sort_values("evt").reset_index(drop=True)

        # forward-fill scores
        df["hs"] = pd.to_numeric(df["hs"], errors="coerce")
        df["vs"] = pd.to_numeric(df["vs"], errors="coerce")
        df["hs"] = df["hs"].ffill().fillna(0)
        df["vs"] = df["vs"].ffill().fillna(0)

        team_ids = [home_id, away_id]

        for team_id in team_ids:
            team_is_home = (team_id == home_id)

            # Map team id to name for injuries
            team_name = None
            opp_name = None
            # team names aren't in PBP; use game_index team ids mapping from injuries list if available
            if team_id == home_id:
                opp_id = away_id
            else:
                opp_id = home_id

            # We can't map ID->name without a lookup; skip injury adjustment if not found
            # Expect user to provide a mapping if needed; for now, set to None

            last_home_score = 0
            last_away_score = 0
            team_run_points = 0
            opp_run_points = 0
            team_run_start = None
            opp_run_start = None
            margin_history = []

            team_3_times = []
            opp_3_times = []

            team_momentum = 0.0
            opp_momentum = 0.0
            prev_index = 0.0

            for _, row in df.iterrows():
                desc = str(row.get("de", ""))
                tid = row.get("tid")
                cl = row.get("cl")

                home_score = int(row.get("hs", 0))
                away_score = int(row.get("vs", 0))

                home_delta = home_score - last_home_score
                away_delta = away_score - last_away_score
                is_scoring = (home_delta > 0 or away_delta > 0)

                if team_is_home:
                    team_score = home_score
                    opp_score = away_score
                else:
                    team_score = away_score
                    opp_score = home_score

                is_team = (tid == team_id)

                if is_scoring:
                    if home_delta > 0 and away_delta == 0:
                        scoring_team_is_team = team_is_home
                    elif away_delta > 0 and home_delta == 0:
                        scoring_team_is_team = not team_is_home
                    else:
                        scoring_team_is_team = None

                    if scoring_team_is_team is True:
                        team_run_points += max(home_delta if team_is_home else away_delta, 0)
                        opp_run_points = 0
                        if team_run_start is None:
                            team_run_start = clock_to_sec(cl)
                        opp_run_start = None
                    elif scoring_team_is_team is False:
                        opp_run_points += max(away_delta if team_is_home else home_delta, 0)
                        team_run_points = 0
                        if opp_run_start is None:
                            opp_run_start = clock_to_sec(cl)
                        team_run_start = None

                    if "3-pt" in desc.lower() or "3pt" in desc.lower():
                        if scoring_team_is_team:
                            team_3_times.append(cl)
                        else:
                            opp_3_times.append(cl)

                    last_home_score = home_score
                    last_away_score = away_score

                # margin swing (team perspective)
                margin = team_score - opp_score
                margin_history.append(margin)
                if len(margin_history) > 5:
                    margin_history = margin_history[-5:]
                if len(margin_history) >= 5:
                    margin_swing = margin - margin_history[0]
                else:
                    margin_swing = 0

                # run intensity
                cur_sec = clock_to_sec(cl)
                team_run_intensity = 0
                opp_run_intensity = 0
                if team_run_start is not None and cur_sec is not None:
                    team_run_intensity = team_run_points / max((team_run_start - cur_sec), 1)
                if opp_run_start is not None and cur_sec is not None:
                    opp_run_intensity = opp_run_points / max((opp_run_start - cur_sec), 1)

                # back-to-back 3s in 3-minute window
                if cur_sec is not None:
                    team_3_times = [t for t in team_3_times if clock_to_sec(t) is not None and (clock_to_sec(t) - cur_sec) <= 180]
                    opp_3_times = [t for t in opp_3_times if clock_to_sec(t) is not None and (clock_to_sec(t) - cur_sec) <= 180]
                team_b2b3 = 1 if len(team_3_times) >= 2 else 0
                opp_b2b3 = 1 if len(opp_3_times) >= 2 else 0

                # injury adjustment (only if mapping exists)
                team_injuries = 0
                opp_injuries = 0
                injury_factor = 1.0

                if team_name and opp_name:
                    team_injuries = injury_counts.get(team_name, {}).get(game_date, 0)
                    opp_injuries = injury_counts.get(opp_name, {}).get(game_date, 0)
                    injury_factor = 1 + 0.02 * (opp_injuries - team_injuries)

                # momentum using tuned weights if available
                if weights:
                    team_momentum = (
                        weights.get("intercept", 0.0)
                        + weights.get("gsw_run_points", 0.0) * team_run_points
                        + weights.get("gsw_run_intensity", 0.0) * team_run_intensity
                        + weights.get("margin_swing", 0.0) * margin_swing
                        + weights.get("gsw_back_to_back_3s_3min", 0.0) * team_b2b3
                    )
                    opp_momentum = (
                        weights.get("intercept", 0.0)
                        + weights.get("gsw_run_points", 0.0) * opp_run_points
                        + weights.get("gsw_run_intensity", 0.0) * opp_run_intensity
                        + weights.get("margin_swing", 0.0) * (-margin_swing)
                        + weights.get("gsw_back_to_back_3s_3min", 0.0) * opp_b2b3
                    )
                else:
                    team_momentum = 0.40 * team_run_points + 0.25 * team_run_intensity + 0.15 * margin_swing + 0.20 * team_b2b3
                    opp_momentum = 0.40 * opp_run_points + 0.25 * opp_run_intensity - 0.15 * margin_swing + 0.20 * opp_b2b3

                team_momentum *= injury_factor
                opp_momentum *= (2 - injury_factor)

                momentum_index = team_momentum - opp_momentum
                momentum_shift = momentum_index - prev_index
                prev_index = momentum_index

                out_rows.append(
                    {
                        "game_id": game_id,
                        "team_id": int(team_id),
                        "event_num": row.get("evt"),
                        "period": row.get("PERIOD"),
                        "clock": cl,
                        "description": desc,
                        "tid": tid,
                        "is_team": 1 if is_team else 0,
                        "team_score": team_score,
                        "opp_score": opp_score,
                        "team_momentum": team_momentum,
                        "opp_momentum": opp_momentum,
                        "momentum_index": momentum_index,
                        "momentum_shift": momentum_shift,
                        "team_b2b3_3min": team_b2b3,
                        "opp_b2b3_3min": opp_b2b3,
                        "season": row.get("season"),
                        "season_type": row.get("season_type"),
                    }
                )

    out = pd.DataFrame(out_rows)
    out_path = Path("/Users/randytran/nba_morale_model/data/momentum_per_play_allteams.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
