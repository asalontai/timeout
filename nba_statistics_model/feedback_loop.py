"""
feedback_loop.py — Interactive AI-Human-Model Feedback Loop

This is the core of the system. It creates an interactive loop where:

1. A game situation is presented (real from data or hypothetical)
2. The ML model makes a prediction: "call timeout" or "don't call timeout"
3. The AI (me) analyzes the prediction and provides coaching context
4. You (the human) can agree, disagree, or adjust
5. Your feedback gets stored and used to retrain the model

Run modes:
  --analyze    : Analyze the trained model's patterns (no interaction needed)
  --simulate   : Walk through real timeout situations from the data
  --scenario   : Input a custom game scenario and get a prediction
  --retrain    : Retrain the model incorporating all feedback
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from timeout_model import (
    load_model, load_data, predict_timeout, explain_prediction,
    save_feedback, train_model, save_model, load_feedback,
    FEATURES, TARGET
)


# ---------------------------------------------------------------------------
# AI ANALYSIS — This is where I (the AI) provide coaching insight
# ---------------------------------------------------------------------------

def ai_analyze_situation(result, historical_data=None):
    """
    AI-generated analysis of a timeout situation.
    This is the 'AI coaching assistant' part of the feedback loop.
    """
    sit = result['situation']
    analysis = []

    opp_run = sit.get('opp_run_before', 0)
    own_run = sit.get('own_run_before', 0)
    score_diff = sit.get('score_diff', 0)
    period = sit.get('period', 1)
    clock = sit.get('clock_seconds', 720)
    own_to = sit.get('own_turnovers_before', 0)
    opp_fg = sit.get('opp_fg_pct_before', 0)

    analysis.append("\n🤖 AI COACHING ANALYSIS:")
    analysis.append("-" * 40)

    # Momentum analysis
    if opp_run >= 10:
        analysis.append("⚠️  CRITICAL: Opponent is on a {}-0 run. This is a textbook timeout situation.".format(opp_run))
        analysis.append("   Historical data shows runs of 10+ rarely stop without intervention.")
    elif opp_run >= 7:
        analysis.append("⚠️  WARNING: Opponent has {} unanswered points. Momentum is shifting.".format(opp_run))
        analysis.append("   Consider calling timeout to disrupt their rhythm.")
    elif opp_run >= 4:
        analysis.append("📊 Opponent has a modest {}-pt run. May not warrant a timeout yet.".format(opp_run))
    elif own_run >= 5:
        analysis.append("🔥 Your team is on a {}-pt run! Calling timeout here would kill YOUR momentum.".format(own_run))
        analysis.append("   Only call if players need rest or there's a critical strategic adjustment.")

    # Score context
    if score_diff < -15:
        analysis.append("📉 Down by {}. In Q{}, this is a desperation situation.".format(abs(score_diff), period))
        if period >= 4:
            analysis.append("   Late in the game and down big — timeouts should be saved strategically.")
    elif score_diff < -8:
        analysis.append("📉 Down by {}. Need to regroup, timeout could help refocus.".format(abs(score_diff)))
    elif abs(score_diff) <= 5:
        analysis.append("🏀 Close game (margin: {}). Every possession matters.".format(score_diff))
        if opp_run >= 6:
            analysis.append("   In close games, letting a run go unchecked is the #1 way to lose.")
    elif score_diff > 10:
        analysis.append("✅ Leading by {}. Save your timeouts unless opponent is making a surge.".format(score_diff))

    # Turnover analysis
    if own_to >= 3:
        analysis.append("🔄 {} turnovers in recent plays! Timeout can settle the team down.".format(own_to))
    
    # Shooting analysis
    if opp_fg >= 0.6:
        analysis.append("🎯 Opponent shooting {}% recently — they're in rhythm.".format(int(opp_fg * 100)))
        analysis.append("   A timeout can break their shooting rhythm and allow defensive adjustment.")

    # Period/clock analysis
    if period == 4 and clock <= 120:
        analysis.append("⏰ Under 2 minutes in Q4 — timeout usage is critical for endgame strategy.")
    elif period >= 4 and clock <= 300:
        analysis.append("⏰ Crunch time. Each timeout is precious — use wisely.")

    # Historical comparison
    if historical_data is not None and len(historical_data) > 0:
        similar = find_similar_situations(sit, historical_data)
        if len(similar) > 0:
            pct_beneficial = similar[TARGET].mean() * 100
            analysis.append(f"\n📚 Historical comparison: Found {len(similar)} similar situations.")
            analysis.append(f"   Timeouts were beneficial {pct_beneficial:.0f}% of the time in comparable spots.")

    # Agreement/disagreement with model
    if result['should_call_timeout']:
        if opp_run < 3 and own_run >= 5 and score_diff > 5:
            analysis.append("\n⚡ I DISAGREE with the model here. Team has momentum and a lead.")
            analysis.append("   The model may be overweighting a minor factor. Consider overriding.")
        else:
            analysis.append("\n✅ I AGREE with the model's recommendation to call timeout.")
    else:
        if opp_run >= 10 and score_diff < 0:
            analysis.append("\n⚡ I DISAGREE with the model here. A 10+ run while trailing demands a timeout.")
            analysis.append("   The model may need correction on this pattern.")
        else:
            analysis.append("\n✅ I AGREE — no timeout needed in this situation.")

    return '\n'.join(analysis)


def find_similar_situations(situation, df, tolerance=0.3):
    """Find historical timeout situations similar to the given one."""
    mask = pd.Series([True] * len(df))

    for feat in ['opp_run_before', 'own_run_before']:
        val = situation.get(feat, 0)
        mask &= (df[feat] >= val - 3) & (df[feat] <= val + 3)

    score_diff = situation.get('score_diff', 0)
    mask &= (df['score_diff'] >= score_diff - 5) & (df['score_diff'] <= score_diff + 5)
    mask &= (df['period'] == situation.get('period', 1))

    return df[mask]


# ---------------------------------------------------------------------------
# INTERACTIVE MODES
# ---------------------------------------------------------------------------

def mode_analyze(data_path):
    """Analyze the trained model's learned patterns."""
    model = load_model()
    df = load_data(data_path)

    if model is None or df is None:
        print("Need a trained model and data first!")
        return

    print("\n" + "=" * 60)
    print("📊 MODEL PATTERN ANALYSIS")
    print("=" * 60)

    # Overall stats
    print(f"\nTotal timeouts analyzed: {len(df)}")
    print(f"Beneficial: {df['beneficial'].sum()} ({df['beneficial'].mean()*100:.1f}%)")
    print(f"Not beneficial: {(1-df['beneficial']).sum()} ({(1-df['beneficial']).mean()*100:.1f}%)")

    # Patterns
    print("\n--- When are timeouts most effective? ---")
    
    # By opponent run size
    for threshold in [0, 5, 8, 10, 12]:
        subset = df[df['opp_run_before'] >= threshold]
        if len(subset) > 0:
            pct = subset['beneficial'].mean() * 100
            print(f"  Opp run ≥ {threshold:2d} pts: {pct:5.1f}% beneficial  (n={len(subset)})")

    # By score differential
    print()
    for lo, hi, label in [(-99, -10, "Down 10+"), (-10, -1, "Down 1-9"),
                            (0, 0, "Tied"), (1, 10, "Up 1-10"), (10, 99, "Up 10+")]:
        subset = df[(df['score_diff'] >= lo) & (df['score_diff'] <= hi)]
        if len(subset) > 0:
            pct = subset['beneficial'].mean() * 100
            print(f"  {label:12s}: {pct:5.1f}% beneficial  (n={len(subset)})")

    # By period
    print()
    for q in range(1, 5):
        subset = df[df['period'] == q]
        if len(subset) > 0:
            pct = subset['beneficial'].mean() * 100
            print(f"  Q{q}: {pct:5.1f}% beneficial  (n={len(subset)})")

    # Key insight
    print("\n--- Key Correlations Found ---")
    corrs = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    for feat, corr in corrs.items():
        direction = "↑" if corr > 0 else "↓"
        print(f"  {feat:25s}  r = {corr:+.3f}  {direction}")


def mode_simulate(data_path):
    """Walk through real timeout situations from the dataset interactively."""
    model = load_model()
    df = load_data(data_path)
    importances = dict(zip(FEATURES, model.feature_importances_))

    if model is None or df is None:
        print("Need a trained model and data first!")
        return

    print("\n" + "=" * 60)
    print("🏀 INTERACTIVE TIMEOUT SIMULATION")
    print("Walk through real timeout situations. The model predicts,")
    print("the AI analyzes, and you provide coaching feedback.")
    print("=" * 60)

    # Shuffle for variety
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    for i, row in df_shuffled.iterrows():
        situation = {feat: row[feat] for feat in FEATURES}
        actual_beneficial = row['beneficial']

        print(f"\n\n{'#'*60}")
        print(f"  TIMEOUT #{i+1}  |  {row.get('matchup', 'Unknown matchup')}")
        print(f"{'#'*60}")

        # Model prediction
        result = predict_timeout(model, situation)
        print(explain_prediction(result, importances))

        # AI analysis
        print(ai_analyze_situation(result, df))

        # Reveal actual outcome
        print(f"\n📋 ACTUAL OUTCOME: {'✅ BENEFICIAL' if actual_beneficial else '❌ NOT BENEFICIAL'}")
        print(f"   Score swing after timeout: {row.get('diff_change_after', '?'):+d}")
        print(f"   Team scored first after: {'Yes' if row.get('team_scores_first_after', 0) else 'No'}")
        print(f"   Opp run stopped: {'Yes' if row.get('opp_run_stopped', 0) else 'No'}")

        # Model correctness
        model_correct = (result['should_call_timeout'] == bool(actual_beneficial))
        if model_correct:
            print(f"\n   ✅ Model was CORRECT")
        else:
            print(f"\n   ❌ Model was WRONG")

        # Get human feedback
        print(f"\n💬 YOUR COACHING FEEDBACK:")
        print(f"   [a] Agree with actual outcome label")
        print(f"   [d] Disagree — I think {'it was' if not actual_beneficial else 'it was NOT'} beneficial")
        print(f"   [s] Skip / no feedback")
        print(f"   [q] Quit simulation")

        choice = input("   Your choice: ").strip().lower()

        if choice == 'q':
            break
        elif choice == 'a':
            feedback_entry = {**situation, TARGET: int(actual_beneficial),
                              'source': 'human_agree'}
            save_feedback(feedback_entry)
            print("   ✓ Feedback recorded: agree with outcome")
        elif choice == 'd':
            corrected = 0 if actual_beneficial else 1
            feedback_entry = {**situation, TARGET: corrected,
                              'source': 'human_disagree'}
            save_feedback(feedback_entry)
            print(f"   ✓ Feedback recorded: corrected to {'beneficial' if corrected else 'not beneficial'}")
        else:
            print("   ↩ Skipped")

    print(f"\n{'='*60}")
    print("Simulation complete! Run with --retrain to incorporate feedback.")
    print(f"{'='*60}")


def mode_scenario():
    """Input a custom game scenario and get a prediction."""
    model = load_model()
    if model is None:
        print("No trained model found! Train one first.")
        return

    importances = dict(zip(FEATURES, model.feature_importances_))

    print("\n" + "=" * 60)
    print("🎯 CUSTOM SCENARIO PREDICTOR")
    print("Enter a game situation to get a timeout recommendation.")
    print("=" * 60)

    try:
        df = load_data()
    except:
        df = None

    while True:
        print("\nEnter the game situation (or 'q' to quit):\n")

        try:
            period = int(input("  Period (1-4, 5 for OT): "))
            clock = float(input("  Seconds remaining in period: "))
            score_diff = int(input("  Your score margin (+/- pts): "))
            opp_run = int(input("  Opponent's unanswered scoring run: "))
            own_run = int(input("  Your team's unanswered scoring run: "))
            opp_fg = float(input("  Opponent FG% in recent plays (0.0-1.0): "))
            own_fg = float(input("  Your FG% in recent plays (0.0-1.0): "))
            own_to = int(input("  Your turnovers in recent plays: "))
            opp_to = int(input("  Opponent turnovers in recent plays: "))
        except (ValueError, EOFError):
            print("Invalid input, try again.")
            continue

        situation = {
            'period': period,
            'clock_seconds': clock,
            'score_diff': score_diff,
            'opp_run_before': opp_run,
            'own_run_before': own_run,
            'opp_fg_pct_before': opp_fg,
            'own_fg_pct_before': own_fg,
            'own_turnovers_before': own_to,
            'opp_turnovers_before': opp_to,
        }

        result = predict_timeout(model, situation)
        print(explain_prediction(result, importances))
        print(ai_analyze_situation(result, df))

        # Feedback
        print(f"\n💬 Do you agree with this prediction?")
        print(f"   [y] Yes, good call   [n] No, I'd do the opposite   [s] Skip")
        choice = input("   Your choice: ").strip().lower()

        if choice == 'y':
            label = 1 if result['should_call_timeout'] else 0
            save_feedback({**situation, TARGET: label, 'source': 'human_scenario_agree'})
        elif choice == 'n':
            label = 0 if result['should_call_timeout'] else 1
            save_feedback({**situation, TARGET: label, 'source': 'human_scenario_disagree'})

        cont = input("\n  Another scenario? [y/n]: ").strip().lower()
        if cont != 'y':
            break


def mode_retrain(data_path):
    """Retrain the model incorporating all feedback data."""
    df = load_data(data_path)
    feedback = load_feedback()

    if len(feedback) == 0:
        print("No feedback data to incorporate. Run --simulate or --scenario first!")
        return

    print(f"\n{'='*60}")
    print(f"🔄 RETRAINING WITH {len(feedback)} FEEDBACK ENTRIES")
    print(f"{'='*60}")

    model, metrics = train_model(df, feedback)
    save_model(model)

    print(f"\n{'='*60}")
    print("✅ Model retrained successfully with feedback!")
    print(f"New training accuracy: {metrics['train_accuracy']:.3f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Timeout Feedback Loop')
    parser.add_argument('--analyze', action='store_true', help='Analyze model patterns')
    parser.add_argument('--simulate', action='store_true', help='Walk through real situations')
    parser.add_argument('--scenario', action='store_true', help='Input custom scenario')
    parser.add_argument('--retrain', action='store_true', help='Retrain with feedback')
    parser.add_argument('--data', default='timeout_data.csv', help='Data file path')
    args = parser.parse_args()

    if args.analyze:
        mode_analyze(args.data)
    elif args.simulate:
        mode_simulate(args.data)
    elif args.scenario:
        mode_scenario()
    elif args.retrain:
        mode_retrain(args.data)
    else:
        print("Usage: python feedback_loop.py --analyze|--simulate|--scenario|--retrain")
        print("\nWorkflow:")
        print("  1. python extract_timeouts.py         — Get timeout data from NBA API")
        print("  2. python timeout_model.py             — Train initial model")
        print("  3. python feedback_loop.py --analyze   — See what the model learned")
        print("  4. python feedback_loop.py --simulate  — Walk through situations & give feedback")
        print("  5. python feedback_loop.py --retrain   — Improve model with your feedback")
        print("  6. Repeat steps 3-5!")
