import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

from config import DATA_DIR


def main():
    df = pd.read_csv(DATA_DIR / "game_features.csv")

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

    df = df.dropna(subset=["WL", "GAME_DATE"]).copy()
    df["gsw_win"] = (df["WL"] == "W").astype(int)

    # Time-based split: first 80% games train, last 20% test
    df = df.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["gsw_win"]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df["gsw_win"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    bal_acc = balanced_accuracy_score(y_test, preds)

    print(f"Time-split AUC: {auc:.3f}")
    print(f"Time-split PR-AUC: {pr_auc:.3f}")
    print(f"Time-split Balanced Accuracy: {bal_acc:.3f}")

    coefs = pd.Series(model.coef_[0], index=feature_cols).sort_values()
    print("\nFeature coefficients (log-odds):")
    for k, v in coefs.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
