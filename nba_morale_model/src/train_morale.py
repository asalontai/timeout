import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

from config import FEATURES_CSV


def main():
    df = pd.read_csv(FEATURES_CSV)

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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

    print(f"Morale regression MAE: {mae:.3f}")
    print(f"Morale regression R2: {r2:.3f}")

    # Simple effect check: average target when 3-in-a-row flag is on/off
    avg_on = df[df["gsw_3s_run_flag"] == 1][target].mean()
    avg_off = df[df["gsw_3s_run_flag"] == 0][target].mean()
    print(f"Avg opp points next3 when 3s-run=1: {avg_on:.3f}")
    print(f"Avg opp points next3 when 3s-run=0: {avg_off:.3f}")


if __name__ == "__main__":
    main()
