"""
timeout_model.py — Timeout Benefit Prediction Model

Trains an XGBoost model to predict whether calling a timeout in a given
game situation will be beneficial.

Features used (all computed from pre-timeout game state):
  - score_diff: current score margin for the team
  - opp_run_before: opponent's unanswered scoring run
  - own_run_before: team's own scoring run
  - opp_fg_pct_before: opponent FG% in recent plays
  - own_fg_pct_before: team's FG% in recent plays
  - own_turnovers_before: turnovers committed recently
  - opp_turnovers_before: opponent turnovers recently
  - period: which quarter
  - clock_seconds: time remaining in period

Target:
  - beneficial: 1 if timeout improved the team's position, 0 otherwise
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score
)


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, 'timeout_model.pkl')
FEEDBACK_PATH = os.path.join(MODEL_DIR, 'feedback_data.json')

# The features the model uses for prediction (pre-timeout only)
FEATURES = [
    'period',
    'clock_seconds',
    'score_diff',
    'opp_run_before',
    'own_run_before',
    'opp_fg_pct_before',
    'own_fg_pct_before',
    'own_turnovers_before',
    'opp_turnovers_before',
]

TARGET = 'beneficial'


def load_data(filepath='timeout_data.csv'):
    """Load timeout dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} timeout records from {filepath}")
    return df


def load_feedback():
    """Load human/AI feedback corrections."""
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} feedback corrections")
        return data
    return []


def augment_with_feedback(df, feedback_data):
    """
    Merge feedback corrections into the training data.
    Feedback can either override labels on existing data or add synthetic scenarios.
    """
    if not feedback_data:
        return df

    feedback_df = pd.DataFrame(feedback_data)

    # Only keep feedback entries that have all required features
    required = FEATURES + [TARGET]
    feedback_df = feedback_df.dropna(subset=[c for c in required if c in feedback_df.columns])

    if len(feedback_df) == 0:
        return df

    # Give feedback data higher weight by duplicating it
    # (feedback from the coaching loop is more valuable signal)
    feedback_df = pd.concat([feedback_df] * 3, ignore_index=True)

    combined = pd.concat([df, feedback_df[required]], ignore_index=True)
    print(f"Training data augmented: {len(df)} original + {len(feedback_df)} feedback = {len(combined)} total")
    return combined


def train_model(df, feedback=None):
    """
    Train the XGBoost timeout prediction model.
    Returns the trained model and evaluation metrics.
    """
    if feedback:
        df = augment_with_feedback(df, feedback)

    X = df[FEATURES]
    y = df[TARGET]

    print(f"\nTraining on {len(X)} samples...")
    print(f"Class balance: {y.value_counts().to_dict()}")

    # XGBoost with moderate parameters
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    # Cross-validation if enough data
    if len(df) >= 10:
        cv = StratifiedKFold(n_splits=min(5, len(df) // 2), shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        print(f"\nCross-validated accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # Train on full data
    model.fit(X, y)

    # Feature importances
    importances = dict(zip(FEATURES, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Feature Importances ---")
    for feat, imp in sorted_imp:
        bar = '█' * int(imp * 40)
        print(f"  {feat:25s} {imp:.4f}  {bar}")

    # Training accuracy (not a true evaluation, but useful for feedback loop)
    y_pred = model.predict(X)
    train_acc = accuracy_score(y, y_pred)
    print(f"\nTraining accuracy: {train_acc:.3f}")

    metrics = {
        'train_accuracy': train_acc,
        'feature_importances': importances,
        'n_samples': len(X),
        'class_balance': y.value_counts().to_dict()
    }

    return model, metrics


def save_model(model):
    """Save trained model to disk."""
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model():
    """Load trained model from disk."""
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
        return model
    print("No saved model found.")
    return None


def predict_timeout(model, situation):
    """
    Predict whether calling a timeout in this situation would be beneficial.

    Args:
        model: trained XGBClassifier
        situation: dict with feature values

    Returns:
        dict with prediction, confidence, and feature contributions
    """
    X = pd.DataFrame([situation])[FEATURES]
    prob = model.predict_proba(X)[0]
    prediction = int(model.predict(X)[0])

    result = {
        'should_call_timeout': bool(prediction),
        'confidence': float(max(prob)),
        'prob_beneficial': float(prob[1]),
        'prob_not_beneficial': float(prob[0]),
        'situation': situation
    }

    return result


def explain_prediction(result, importances):
    """
    Generate a human-readable explanation of why the model made this prediction.
    """
    sit = result['situation']
    lines = []

    lines.append(f"\n{'='*60}")
    if result['should_call_timeout']:
        lines.append(f"🟢 MODEL SAYS: CALL TIMEOUT (confidence: {result['confidence']:.1%})")
    else:
        lines.append(f"🔴 MODEL SAYS: DON'T CALL TIMEOUT (confidence: {result['confidence']:.1%})")
    lines.append(f"{'='*60}")

    lines.append(f"\n📊 Game Situation:")
    lines.append(f"  Period: Q{sit.get('period', '?')} | Clock: {sit.get('clock_seconds', 0):.0f}s remaining")
    lines.append(f"  Score margin: {sit.get('score_diff', 0):+d}")
    lines.append(f"  Opponent run: {sit.get('opp_run_before', 0)} unanswered pts")
    lines.append(f"  Own run: {sit.get('own_run_before', 0)} unanswered pts")
    lines.append(f"  Opp FG%: {sit.get('opp_fg_pct_before', 0):.1%} | Own FG%: {sit.get('own_fg_pct_before', 0):.1%}")
    lines.append(f"  Own turnovers: {sit.get('own_turnovers_before', 0)} | Opp turnovers: {sit.get('opp_turnovers_before', 0)}")

    # Show which factors weighed most
    lines.append(f"\n🔍 Key factors driving this prediction:")
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    for feat, imp in sorted_imp:
        val = sit.get(feat, 'N/A')
        lines.append(f"  • {feat}: {val} (importance: {imp:.3f})")

    return '\n'.join(lines)


def save_feedback(feedback_entry):
    """Append a feedback entry to the feedback file."""
    data = load_feedback()
    data.append(feedback_entry)
    with open(FEEDBACK_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Feedback saved ({len(data)} total entries)")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train timeout prediction model')
    parser.add_argument('--data', default='timeout_data.csv', help='Training data CSV')
    parser.add_argument('--use-feedback', action='store_true', help='Include feedback data')
    args = parser.parse_args()

    df = load_data(args.data)
    feedback = load_feedback() if args.use_feedback else None

    model, metrics = train_model(df, feedback)
    save_model(model)

    print(f"\n{'='*60}")
    print("Model trained and saved successfully!")
    print(f"{'='*60}")
