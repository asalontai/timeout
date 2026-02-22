import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import DATA_DIR


def main():
    df = pd.read_csv(DATA_DIR / "game_features.csv")
    df = df.dropna(subset=["WL", "GAME_DATE"]).copy()
    df["gsw_win"] = (df["WL"] == "W").astype(int)

    feature_cols = [
        "gsw_injuries_7d",
        "opp_injuries_7d",
        "gsw_win_pct_to_date",
        "is_home",
        "gsw_3s_run_any",
        "gsw_3s_run_count",
        "gsw_max_consec_3s",
        "avg_msi",
        "max_msi",
    ]

    # Train on all available data
    X = df[feature_cols].fillna(0)
    y = df["gsw_win"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    df["win_prob"] = model.predict_proba(X)[:, 1]

    # Output a compact prediction table
    out = df[["GAME_DATE", "MATCHUP", "WL", "win_prob"]].copy()
    out = out.sort_values(["GAME_DATE", "MATCHUP"]).reset_index(drop=True)

    out_path = DATA_DIR / "game_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    print(out.head(10))


if __name__ == "__main__":
    main()
