import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from config import FEATURES_CSV, GAMES_CSV


def main():
    df = pd.read_csv(FEATURES_CSV)
    games = pd.read_csv(GAMES_CSV)

    # Build chronological order by game date
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    game_order = {gid: idx for idx, gid in enumerate(games["GAME_ID"].astype(str).tolist())}

    df["game_order"] = df["game_id"].astype(str).map(game_order)
    df = df.dropna(subset=["game_order"]).sort_values("game_order")

    target = "opp_points_next3_scoring_events"
    if target not in df.columns:
        raise RuntimeError(f"Missing target column: {target}. Rebuild features first.")

    feature_cols = [
        "seconds_remaining",
        "margin",
        "is_gsw_score",
        "gsw_run_points",
        "opp_run_points",
        "gsw_run_intensity",
        "opp_run_intensity",
        "gsw_consecutive_3s",
        "opp_consecutive_3s",
        "margin_swing",
        "star_impact",
        "opp_timeout_since_last_score",
        "msi",
        "gsw_3s_run_flag",
    ]

    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)

    # Time-based split: first 80% games train, last 20% test
    split_idx = int(df["game_order"].max() * 0.8)
    train_mask = df["game_order"] <= split_idx
    test_mask = df["game_order"] > split_idx

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=5,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Time-split MAE: {mae:.3f}")
    print(f"Time-split R2: {r2:.3f}")

    # Effect check
    avg_on = df[df["gsw_3s_run_flag"] == 1][target].mean()
    avg_off = df[df["gsw_3s_run_flag"] == 0][target].mean()
    print(f"Avg opp points next3 when 3s-run=1: {avg_on:.3f}")
    print(f"Avg opp points next3 when 3s-run=0: {avg_off:.3f}")


if __name__ == "__main__":
    main()
