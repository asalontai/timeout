import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from config import FEATURES_CSV


def main():
    df = pd.read_csv(FEATURES_CSV)

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
    ]

    X = df[feature_cols].fillna(0)
    y = df["gsw_win"]

    # Simple random split baseline; replace with time-based split if desired
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, (preds >= 0.5).astype(int))

    print(f"AUC: {auc:.3f}")
    print(f"Accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
