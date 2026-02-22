import json
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import FEATURES_CSV, DATA_DIR


def main():
    df = pd.read_csv(FEATURES_CSV)

    # Build target: net points over next 3 scoring events
    gsw_pts = df["gsw_points_delta"].fillna(0).tolist()
    opp_pts = df["opp_points_delta"].fillna(0).tolist()

    net_next3 = []
    for i in range(len(df)):
        future_gsw = sum(gsw_pts[i + 1:i + 4])
        future_opp = sum(opp_pts[i + 1:i + 4])
        net_next3.append(future_gsw - future_opp)

    df["net_points_next3"] = net_next3

    feature_cols = [
        "gsw_run_points",
        "gsw_run_intensity",
        "margin_swing",
        "gsw_back_to_back_3s_3min",
    ]

    X = df[feature_cols].fillna(0)
    y = df["net_points_next3"].fillna(0)

    model = LinearRegression()
    model.fit(X, y)

    weights = {col: float(w) for col, w in zip(feature_cols, model.coef_)}
    weights["intercept"] = float(model.intercept_)

    out_path = DATA_DIR / "momentum_weights.json"
    with open(out_path, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"Saved weights to {out_path}")
    print(weights)


if __name__ == "__main__":
    main()
