import json
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import GAMES_CSV, PBP_DIR, TEAM_ID, DATA_DIR


def infer_gsw_home_from_pbp(df: pd.DataFrame) -> bool:
    # Infer by first scoring event by GSW and which side score changed
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
        if tid == TEAM_ID:
            if home > last_home and away == last_away:
                return True
            if away > last_away and home == last_home:
                return False
        last_home = home
        last_away = away
    # Fallback
    return True


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


def main():
    games = pd.read_csv(GAMES_CSV)
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    pbp = load_pbp_all()

    # Load tuned weights if available
    weights_path = DATA_DIR / "momentum_weights.json"
    weights = None
    if weights_path.exists():
        with open(weights_path, "r") as f:
            weights = json.load(f)

    out_rows = []

    game_ids = pbp["GAME_ID"].astype(str).unique().tolist()
    if len(sys.argv) > 1:
        game_ids = [str(sys.argv[1]).strip()]

    # Use all PBP games (regular + playoffs) unless GAME_ID is provided
    for game_id in tqdm(game_ids, total=len(game_ids)):
        df = pbp[pbp["GAME_ID"].astype(str) == str(game_id)].copy()
        if df.empty:
            continue

        df = df.sort_values("evt").reset_index(drop=True)
        gsw_is_home = infer_gsw_home_from_pbp(df)

        # forward-fill scores
        df["hs"] = pd.to_numeric(df["hs"], errors="coerce")
        df["vs"] = pd.to_numeric(df["vs"], errors="coerce")
        df["hs"] = df["hs"].ffill().fillna(0)
        df["vs"] = df["vs"].ffill().fillna(0)

        last_home_score = 0
        last_away_score = 0
        gsw_run_points = 0
        opp_run_points = 0
        gsw_run_start = None
        opp_run_start = None
        margin_history = []

        gsw_3_times = []
        opp_3_times = []

        gsw_momentum = 0.0
        opp_momentum = 0.0
        prev_index = 0.0

        for _, row in df.iterrows():
            desc = str(row.get("de", ""))
            etype = row.get("etype")
            tid = row.get("tid")
            cl = row.get("cl")

            home_score = int(row.get("hs", 0))
            away_score = int(row.get("vs", 0))

            home_delta = home_score - last_home_score
            away_delta = away_score - last_away_score
            is_scoring = (home_delta > 0 or away_delta > 0)

            if gsw_is_home:
                gsw_score = home_score
                opp_score = away_score
            else:
                gsw_score = away_score
                opp_score = home_score

            is_gsw = (tid == TEAM_ID)

            if is_scoring:
                if home_delta > 0 and away_delta == 0:
                    scoring_team_is_gsw = gsw_is_home
                elif away_delta > 0 and home_delta == 0:
                    scoring_team_is_gsw = not gsw_is_home
                else:
                    scoring_team_is_gsw = None

                if scoring_team_is_gsw is True:
                    gsw_run_points += max(home_delta if gsw_is_home else away_delta, 0)
                    opp_run_points = 0
                    if gsw_run_start is None:
                        gsw_run_start = clock_to_sec(cl)
                    opp_run_start = None
                elif scoring_team_is_gsw is False:
                    opp_run_points += max(away_delta if gsw_is_home else home_delta, 0)
                    gsw_run_points = 0
                    if opp_run_start is None:
                        opp_run_start = clock_to_sec(cl)
                    gsw_run_start = None

                if "3-pt" in desc.lower() or "3pt" in desc.lower():
                    if scoring_team_is_gsw:
                        gsw_3_times.append(cl)
                    else:
                        opp_3_times.append(cl)

                last_home_score = home_score
                last_away_score = away_score

            # compute margin swing
            margin = gsw_score - opp_score
            margin_history.append(margin)
            if len(margin_history) > 5:
                margin_history = margin_history[-5:]
            if len(margin_history) >= 5:
                margin_swing = margin - margin_history[0]
            else:
                margin_swing = 0

            # run intensity
            cur_sec = clock_to_sec(cl)
            gsw_run_intensity = 0
            opp_run_intensity = 0
            if gsw_run_start is not None and cur_sec is not None:
                gsw_run_intensity = gsw_run_points / max((gsw_run_start - cur_sec), 1)
            if opp_run_start is not None and cur_sec is not None:
                opp_run_intensity = opp_run_points / max((opp_run_start - cur_sec), 1)

            # back-to-back 3s in 3-minute window
            if cur_sec is not None:
                gsw_3_times = [t for t in gsw_3_times if clock_to_sec(t) is not None and (clock_to_sec(t) - cur_sec) <= 180]
                opp_3_times = [t for t in opp_3_times if clock_to_sec(t) is not None and (clock_to_sec(t) - cur_sec) <= 180]
            gsw_b2b3 = 1 if len(gsw_3_times) >= 2 else 0
            opp_b2b3 = 1 if len(opp_3_times) >= 2 else 0

            # momentum using tuned weights if available
            if weights:
                gsw_momentum = (
                    weights.get("intercept", 0.0)
                    + weights.get("gsw_run_points", 0.0) * gsw_run_points
                    + weights.get("gsw_run_intensity", 0.0) * gsw_run_intensity
                    + weights.get("margin_swing", 0.0) * margin_swing
                    + weights.get("gsw_back_to_back_3s_3min", 0.0) * gsw_b2b3
                )
                opp_momentum = (
                    weights.get("intercept", 0.0)
                    + weights.get("gsw_run_points", 0.0) * opp_run_points
                    + weights.get("gsw_run_intensity", 0.0) * opp_run_intensity
                    + weights.get("margin_swing", 0.0) * (-margin_swing)
                    + weights.get("gsw_back_to_back_3s_3min", 0.0) * opp_b2b3
                )
            else:
                gsw_momentum = 0.40 * gsw_run_points + 0.25 * gsw_run_intensity + 0.15 * margin_swing + 0.20 * gsw_b2b3
                opp_momentum = 0.40 * opp_run_points + 0.25 * opp_run_intensity - 0.15 * margin_swing + 0.20 * opp_b2b3

            momentum_index = gsw_momentum - opp_momentum
            momentum_shift = momentum_index - prev_index
            prev_index = momentum_index

            out_rows.append(
                {
                    "game_id": game_id,
                    "event_num": row.get("evt"),
                    "period": row.get("PERIOD"),
                    "clock": cl,
                    "description": desc,
                    "tid": tid,
                    "is_gsw": 1 if is_gsw else 0,
                    "gsw_score": gsw_score,
                    "opp_score": opp_score,
                    "gsw_momentum": gsw_momentum,
                    "opp_momentum": opp_momentum,
                    "momentum_index": momentum_index,
                    "momentum_shift": momentum_shift,
                    "gsw_back_to_back_3s_3min": gsw_b2b3,
                    "opp_back_to_back_3s_3min": opp_b2b3,
                    "season": row.get("season"),
                    "season_type": row.get("season_type"),
                }
            )

    out = pd.DataFrame(out_rows)
    out_path = Path("/Users/randytran/nba_morale_model/data/momentum_per_play.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
