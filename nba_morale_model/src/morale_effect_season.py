import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from config import FEATURES_CSV


def main():
    df = pd.read_csv(FEATURES_CSV)

    target = "opp_points_next3_scoring_events"
    if target not in df.columns:
        raise RuntimeError(f"Missing target column: {target}. Rebuild features first.")
    if "season" not in df.columns:
        raise RuntimeError("Missing season column. Rebuild features with multi-season PBP.")

    # Per-season simple effect (unmatched)
    print("Per-season effect (mean diff):")
    for season, g in df.groupby("season"):
        avg_on = g[g["gsw_3s_run_flag"] == 1][target].mean()
        avg_off = g[g["gsw_3s_run_flag"] == 0][target].mean()
        print(f"  {season}: diff={avg_on - avg_off:.3f} (on={avg_on:.3f}, off={avg_off:.3f})")

    # Pooled regression with season fixed effects
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
    season_dummies = pd.get_dummies(df["season"], prefix="season", drop_first=True)
    X = pd.concat([X, season_dummies], axis=1)
    y = df[target].fillna(0)

    model = LinearRegression()
    model.fit(X, y)

    coef = pd.Series(model.coef_, index=X.columns)
    effect = coef.get("gsw_3s_run_flag", 0.0)

    print(f"\nPooled regression effect (gsw_3s_run_flag coef): {effect:.3f}")


if __name__ == "__main__":
    main()
