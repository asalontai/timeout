import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from config import DATA_DIR


def main():
    df = pd.read_csv(DATA_DIR / "momentum_per_play_allteams.csv")

    # Build game outcome labels from final scores
    final_scores = df.sort_values("event_num").groupby(["game_id", "team_id"]).tail(1)
    final_scores = final_scores[["game_id", "team_id", "team_score", "opp_score"]]
    final_scores["team_win"] = (final_scores["team_score"] > final_scores["opp_score"]).astype(int)

    df = df.merge(final_scores, on=["game_id", "team_id"], suffixes=("", "_final"))

    # Build time remaining in game
    def clock_to_sec(cl):
        if not isinstance(cl, str) or ":" not in cl:
            return None
        try:
            mm, ss = cl.split(":")
            return int(mm) * 60 + int(ss)
        except Exception:
            return None

    df["clock_sec"] = df["clock"].apply(clock_to_sec).fillna(0)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1)
    df["time_remaining"] = df["clock_sec"] + (4 - df["period"]).clip(lower=0) * 12 * 60

    # Features per play
    feature_cols = [
        "momentum_index",
        "momentum_shift",
        "team_score",
        "opp_score",
        "time_remaining",
        "team_b2b3_3min",
        "opp_b2b3_3min",
    ]

    X = df[feature_cols].fillna(0)
    y = df["team_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    df["win_prob"] = model.predict_proba(X)[:, 1]

    out_path = DATA_DIR / "per_play_winprob.csv"
    df[["game_id", "team_id", "event_num", "period", "clock", "team_score", "opp_score", "momentum_index", "momentum_shift", "win_prob"]].to_csv(out_path, index=False)

    print(f"Saved per-play win probabilities to {out_path}")


if __name__ == "__main__":
    main()
