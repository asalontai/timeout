import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR


def main():
    per_play = pd.read_csv(DATA_DIR / "momentum_per_play.csv")

    # Normalize momentum within each game (z-score)
    per_play["momentum_index_z"] = per_play.groupby("game_id")["momentum_index"].transform(
        lambda x: (x - x.mean()) / (x.std() if x.std() != 0 else 1)
    )

    # Aggregate momentum per game with progress
    records = []
    grouped = per_play.groupby("game_id")
    for game_id, g in tqdm(grouped, total=grouped.ngroups):
        records.append(
            {
                "game_id": game_id,
                "momentum_max": g["momentum_index_z"].max(),
                "momentum_min": g["momentum_index_z"].min(),
                "momentum_mean": g["momentum_index_z"].mean(),
                "momentum_std": g["momentum_index_z"].std(),
                "positive_shifts": (g["momentum_shift"] > 0).sum(),
                "negative_shifts": (g["momentum_shift"] < 0).sum(),
                "b2b3_gsw": g["gsw_back_to_back_3s_3min"].max(),
                "b2b3_opp": g["opp_back_to_back_3s_3min"].max(),
                "final_gsw_score": g["gsw_score"].iloc[-1],
                "final_opp_score": g["opp_score"].iloc[-1],
            }
        )

    agg = pd.DataFrame(records)

    agg["gsw_win"] = (agg["final_gsw_score"] > agg["final_opp_score"]).astype(int)

    feature_cols = [
        "momentum_max",
        "momentum_min",
        "momentum_mean",
        "momentum_std",
        "positive_shifts",
        "negative_shifts",
        "b2b3_gsw",
        "b2b3_opp",
    ]

    X = agg[feature_cols].fillna(0)
    y = agg["gsw_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.5)),
        ]
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    bal_acc = balanced_accuracy_score(y_test, preds)

    print(f"Momentum-only AUC: {auc:.3f}")
    print(f"Momentum-only Balanced Accuracy: {bal_acc:.3f}")

    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=feature_cols).sort_values()
    print("\nFeature coefficients (log-odds):")
    for k, v in coefs.items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
