import time
import re
import random
from typing import Tuple, Optional
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    GAMES_CSV,
    FEATURES_CSV,
    PBP_DIR,
    TEAM_ID,
    STAR_PLAYER_IDS,
    MARGIN_SWING_WINDOW,
)

THREE_PT_RE = re.compile(r"\b3-?pt\b", re.IGNORECASE)


def parse_score(score: str) -> Optional[Tuple[int, int]]:
    if not isinstance(score, str) or "-" not in score:
        return None
    try:
        home, away = score.split("-")
        return int(home), int(away)
    except ValueError:
        return None


def infer_gsw_home(matchup: str) -> bool:
    if isinstance(matchup, str) and "vs." in matchup:
        return True
    if isinstance(matchup, str) and "@" in matchup:
        return False
    return True


def game_seconds_remaining(period: int, pctimestring: str) -> Optional[int]:
    if not isinstance(pctimestring, str) or ":" not in pctimestring:
        return None
    try:
        minutes, seconds = pctimestring.split(":")
        remaining_in_period = int(minutes) * 60 + int(seconds)
    except ValueError:
        return None

    if period <= 4:
        future_periods = 4 - period
        future_seconds = future_periods * 12 * 60
    else:
        future_seconds = 0

    return remaining_in_period + future_seconds


def is_three_pointer(desc: str) -> bool:
    if not isinstance(desc, str):
        return False
    return bool(THREE_PT_RE.search(desc))


def build_features_for_game(game_row: pd.Series, pbp_game: pd.DataFrame) -> pd.DataFrame:
    game_id = game_row["GAME_ID"]
    gsw_win = 1 if game_row.get("WL") == "W" else 0
    gsw_is_home = infer_gsw_home(game_row.get("MATCHUP", ""))

    df = pbp_game.sort_values("evt").reset_index(drop=True)

    last_home_score = 0
    last_away_score = 0
    gsw_score = 0
    opp_score = 0

    gsw_run_points = 0
    opp_run_points = 0
    gsw_run_start_time = None
    opp_run_start_time = None

    gsw_consecutive_3s = 0
    opp_consecutive_3s = 0
    gsw_3_pts_times = []
    opp_3_pts_times = []

    margin_history = []
    opp_timeout_since_last_score = 0

    rows = []

    for _, row in df.iterrows():
        period = row.get("PERIOD")
        time_str = row.get("cl")
        seconds_remaining = game_seconds_remaining(period, time_str)
        if seconds_remaining is None:
            seconds_remaining = 0

        # Timeout tracking
        desc = row.get("de", "")
        mtype = row.get("mtype", "")
        etype = row.get("etype", "")
        if (isinstance(desc, str) and "timeout" in desc.lower()) or mtype == 9 or etype == 9:
            timeout_team_id = row.get("tid")
            if timeout_team_id and timeout_team_id != TEAM_ID:
                opp_timeout_since_last_score = 1

        # Scores are in hs (home), vs (visitor)
        try:
            home_score = int(row.get("hs"))
            away_score = int(row.get("vs"))
        except (TypeError, ValueError):
            continue

        home_delta = home_score - last_home_score
        away_delta = away_score - last_away_score
        if home_delta <= 0 and away_delta <= 0:
            continue
        if home_delta > 0 and away_delta > 0:
            last_home_score, last_away_score = home_score, away_score
            continue

        scoring_team_id = row.get("tid")
        if not scoring_team_id:
            if home_delta > 0 and away_delta == 0:
                scoring_team_id = TEAM_ID if gsw_is_home else None
            elif away_delta > 0 and home_delta == 0:
                scoring_team_id = TEAM_ID if not gsw_is_home else None

        if not scoring_team_id:
            last_home_score, last_away_score = home_score, away_score
            continue

        last_home_score, last_away_score = home_score, away_score

        if gsw_is_home:
            gsw_score = home_score
            opp_score = away_score
        else:
            gsw_score = away_score
            opp_score = home_score

        three_pt = is_three_pointer(desc)

        if rows:
            prev_gsw_score = rows[-1]["gsw_score"]
            prev_opp_score = rows[-1]["opp_score"]
        else:
            prev_gsw_score = 0
            prev_opp_score = 0

        gsw_points_delta = gsw_score - prev_gsw_score
        opp_points_delta = opp_score - prev_opp_score

        if scoring_team_id == TEAM_ID:
            gsw_run_points += gsw_points_delta
            opp_run_points = 0
            if gsw_run_start_time is None:
                gsw_run_start_time = seconds_remaining
            opp_run_start_time = None
        else:
            opp_run_points += opp_points_delta
            gsw_run_points = 0
            if opp_run_start_time is None:
                opp_run_start_time = seconds_remaining
            gsw_run_start_time = None

        if scoring_team_id == TEAM_ID:
            gsw_consecutive_3s = gsw_consecutive_3s + 1 if three_pt else 0
            opp_consecutive_3s = 0
            if three_pt:
                gsw_3_pts_times.append(seconds_remaining)
        else:
            opp_consecutive_3s = opp_consecutive_3s + 1 if three_pt else 0
            gsw_consecutive_3s = 0
            if three_pt:
                opp_3_pts_times.append(seconds_remaining)

        if scoring_team_id == TEAM_ID and gsw_run_start_time is not None:
            gsw_run_duration = max(gsw_run_start_time - seconds_remaining, 1)
        else:
            gsw_run_duration = 0

        if scoring_team_id != TEAM_ID and opp_run_start_time is not None:
            opp_run_duration = max(opp_run_start_time - seconds_remaining, 1)
        else:
            opp_run_duration = 0

        gsw_run_intensity = gsw_run_points / max(gsw_run_duration, 1) if gsw_run_points else 0
        opp_run_intensity = opp_run_points / max(opp_run_duration, 1) if opp_run_points else 0

        margin = gsw_score - opp_score
        margin_history.append(margin)
        if len(margin_history) > MARGIN_SWING_WINDOW:
            margin_history = margin_history[-MARGIN_SWING_WINDOW:]

        if len(margin_history) >= MARGIN_SWING_WINDOW:
            margin_swing = margin - margin_history[0]
        else:
            margin_swing = 0

        pid = row.get("pid")
        star_impact = 1 if (scoring_team_id == TEAM_ID and pid in STAR_PLAYER_IDS) else 0

        msi = (
            0.35 * gsw_run_points
            + 0.25 * gsw_run_intensity
            + 0.20 * gsw_consecutive_3s
            + 0.10 * star_impact
            + 0.10 * margin_swing
            - 0.15 * opp_timeout_since_last_score
        )

        # Momentum score (3-minute window for back-to-back 3s)
        window_seconds = 180
        gsw_3_recent = [t for t in gsw_3_pts_times if (seconds_remaining - t) <= window_seconds]
        opp_3_recent = [t for t in opp_3_pts_times if (seconds_remaining - t) <= window_seconds]
        gsw_back_to_back_3s = 1 if len(gsw_3_recent) >= 2 else 0
        opp_back_to_back_3s = 1 if len(opp_3_recent) >= 2 else 0

        momentum_score = (
            0.40 * gsw_run_points
            + 0.25 * gsw_run_intensity
            + 0.15 * margin_swing
            + 0.20 * gsw_back_to_back_3s
        )

        rows.append(
            {
                "game_id": game_id,
                "event_num": row.get("evt"),
                "period": period,
                "seconds_remaining": seconds_remaining,
                "gsw_score": gsw_score,
                "opp_score": opp_score,
                "margin": margin,
                "is_gsw_score": 1 if scoring_team_id == TEAM_ID else 0,
                "gsw_run_points": gsw_run_points,
                "opp_run_points": opp_run_points,
                "gsw_run_intensity": gsw_run_intensity,
                "opp_run_intensity": opp_run_intensity,
                "gsw_consecutive_3s": gsw_consecutive_3s,
                "opp_consecutive_3s": opp_consecutive_3s,
                "margin_swing": margin_swing,
                "star_impact": star_impact,
                "opp_timeout_since_last_score": opp_timeout_since_last_score,
                "msi": msi,
                "momentum_score": momentum_score,
                "gsw_back_to_back_3s_3min": gsw_back_to_back_3s,
                "opp_back_to_back_3s_3min": opp_back_to_back_3s,
                "gsw_points_delta": gsw_points_delta,
                "opp_points_delta": opp_points_delta,
                "gsw_3s_run_flag": 1 if (scoring_team_id == TEAM_ID and gsw_consecutive_3s >= 3) else 0,
                "gsw_win": gsw_win,
            }
        )

        opp_timeout_since_last_score = 0

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        opp_points = df_out["opp_points_delta"].fillna(0).tolist()
        flags = df_out["gsw_3s_run_flag"].fillna(0).tolist()
        morale_label = []
        opp_points_next3 = []
        for i in range(len(df_out)):
            if flags[i] == 1:
                future = opp_points[i + 1:i + 4]
                morale_label.append(1 if sum(future) <= 2 else 0)
            else:
                morale_label.append(0)
            future_any = opp_points[i + 1:i + 4]
            opp_points_next3.append(sum(future_any))
        df_out["opp_le2_next3_scoring_events"] = morale_label
        df_out["opp_points_next3_scoring_events"] = opp_points_next3
        df_out["momentum_shift"] = df_out["momentum_score"].diff().fillna(0)
    else:
        df_out["opp_le2_next3_scoring_events"] = []
        df_out["opp_points_next3_scoring_events"] = []
        df_out["momentum_shift"] = []

    return df_out


def _load_pbp_regular_season() -> pd.DataFrame:
    if not PBP_DIR.exists():
        raise RuntimeError(f"PBP directory not found: {PBP_DIR}")

    # Use only regular season files: datanba_YYYY.csv
    reg_files = sorted(PBP_DIR.glob("datanba_*.csv"))
    reg_files = [p for p in reg_files if "po" not in p.stem]
    if not reg_files:
        raise RuntimeError("No regular season PBP CSVs found. Expected datanba_YYYY.csv files.")

    frames = []
    for path in reg_files:
        season = path.stem.split("_")[-1]
        df = pd.read_csv(path)
        df["season"] = int(season)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main():
    games = pd.read_csv(GAMES_CSV)
    pbp_all = _load_pbp_regular_season()

    if "GAME_ID" not in pbp_all.columns:
        raise RuntimeError("Could not find GAME_ID column in play-by-play CSVs.")

    all_rows = []

    # Ensure regular season only
    if "SEASON_TYPE" in games.columns:
        games = games[games["SEASON_TYPE"] == "Regular Season"].copy()
    elif "SEASON_ID" in games.columns:
        games = games[games["SEASON_ID"].astype(str).str.startswith("2")].copy()

    for _, game_row in tqdm(games.iterrows(), total=len(games)):
        game_id = game_row["GAME_ID"]
        success = False
        for attempt in range(4):
            try:
                pbp_game = pbp_all[pbp_all["GAME_ID"].astype(str) == str(game_id)]
                if pbp_game.empty:
                    print(f"Skipping game {game_id}: no PBP rows (likely playoffs or missing).")
                    success = True
                    break
                df = build_features_for_game(game_row, pbp_game)
                if not df.empty:
                    # add season from PBP
                    season_val = int(pbp_game["season"].iloc[0])
                    df["season"] = season_val
                    all_rows.append(df)
                success = True
                break
            except Exception as exc:
                print(f"Failed game {game_id} (attempt {attempt+1}/4): {exc}")
                time.sleep(2 + attempt * 2 + random.random())
        if not success:
            continue

    if not all_rows:
        raise RuntimeError("No features generated.")

    features = pd.concat(all_rows, ignore_index=True)
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {len(features)} rows to {FEATURES_CSV}")


if __name__ == "__main__":
    main()
