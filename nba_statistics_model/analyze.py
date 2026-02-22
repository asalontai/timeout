"""
analyze.py — Visual Correlation Analysis for Timeout Optimization

This script produces clear charts and statistical tests that show you
exactly what the data says about when timeouts work and when they don't.

No ML model needed — this is pure data analysis to find genuine patterns.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(path='timeout_data.csv'):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} timeouts")
    return df


def create_full_analysis(df, save_path=None):
    """
    Produce a comprehensive visual analysis of timeout effectiveness.
    Creates one big figure with multiple subplots.
    """
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, 'timeout_analysis.png')

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('NBA Timeout Effectiveness Analysis\n'
                 f'({len(df)} timeouts analyzed)',
                 fontsize=22, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.96, top=0.92, bottom=0.06)

    colors = {'beneficial': '#2ecc71', 'not_beneficial': '#e74c3c',
              'neutral': '#3498db', 'bg': '#f8f9fa'}

    # ================================================================
    # PLOT 1: Opponent Run vs Timeout Success Rate
    # This is THE key question: how big does the run need to be?
    # ================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    bins = [0, 2, 4, 6, 8, 10, 100]
    labels = ['0-1', '2-3', '4-5', '6-7', '8-9', '10+']
    df['opp_run_bin'] = pd.cut(df['opp_run_before'], bins=bins, labels=labels, right=False)

    run_stats = df.groupby('opp_run_bin', observed=True).agg(
        success_rate=('beneficial', 'mean'),
        count=('beneficial', 'count')
    ).reset_index()

    bar_colors = [colors['not_beneficial'] if r < 0.5 else colors['beneficial']
                  for r in run_stats['success_rate']]
    bars = ax1.bar(run_stats['opp_run_bin'], run_stats['success_rate'] * 100,
                   color=bar_colors, edgecolor='white', linewidth=1.5)

    # Add count labels on bars
    for bar, count in zip(bars, run_stats['count']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (coin flip)')
    ax1.set_xlabel('Opponent Scoring Run (points)', fontsize=11)
    ax1.set_ylabel('Timeout Success Rate (%)', fontsize=11)
    ax1.set_title('🔑 KEY FINDING:\nDoes Run Size Predict Timeout Success?', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=9)

    # ================================================================
    # PLOT 2: Score Differential vs Timeout Success
    # ================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    bins_sd = [-100, -15, -10, -5, 0, 5, 10, 15, 100]
    labels_sd = ['-15+', '-10 to -15', '-5 to -10', '-5 to 0', '0 to +5', '+5 to +10', '+10 to +15', '+15+']
    df['score_diff_bin'] = pd.cut(df['score_diff'], bins=bins_sd, labels=labels_sd, right=False)

    sd_stats = df.groupby('score_diff_bin', observed=True).agg(
        success_rate=('beneficial', 'mean'),
        count=('beneficial', 'count')
    ).reset_index()

    bar_colors2 = [colors['not_beneficial'] if r < 0.5 else colors['beneficial']
                   for r in sd_stats['success_rate']]
    bars2 = ax2.bar(sd_stats['score_diff_bin'], sd_stats['success_rate'] * 100,
                    color=bar_colors2, edgecolor='white', linewidth=1.5)

    for bar, count in zip(bars2, sd_stats['count']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Score Margin When Timeout Called', fontsize=11)
    ax2.set_ylabel('Timeout Success Rate (%)', fontsize=11)
    ax2.set_title('Does Being Ahead/Behind Affect\nTimeout Effectiveness?', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=45)

    # ================================================================
    # PLOT 3: Quarter-by-Quarter Analysis
    # ================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    q_stats = df.groupby('period').agg(
        success_rate=('beneficial', 'mean'),
        count=('beneficial', 'count'),
        avg_opp_run=('opp_run_before', 'mean')
    ).reset_index()

    x = np.arange(len(q_stats))
    width = 0.4
    bars3a = ax3.bar(x - width/2, q_stats['success_rate'] * 100,
                     width, label='Success Rate %', color=colors['beneficial'], alpha=0.8)
    ax3b = ax3.twinx()
    bars3b = ax3b.bar(x + width/2, q_stats['avg_opp_run'],
                      width, label='Avg Opp Run', color=colors['neutral'], alpha=0.8)

    ax3.set_xlabel('Quarter', fontsize=11)
    ax3.set_ylabel('Timeout Success Rate (%)', fontsize=11, color=colors['beneficial'])
    ax3b.set_ylabel('Avg Opponent Run (pts)', fontsize=11, color=colors['neutral'])
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Q{int(p)}' for p in q_stats['period']])
    ax3.set_title('When in the Game Are\nTimeouts Most Effective?', fontsize=13, fontweight='bold')
    ax3.set_ylim(0, 80)

    for bar, count in zip(bars3a, q_stats['count']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    # ================================================================
    # PLOT 4: Scatter — Opp Run vs Score Swing After Timeout
    # This shows the ACTUAL IMPACT of each timeout
    # ================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    beneficial_mask = df['beneficial'] == 1
    ax4.scatter(df[beneficial_mask]['opp_run_before'],
               df[beneficial_mask]['diff_change_after'],
               color=colors['beneficial'], alpha=0.6, s=50,
               label=f'Beneficial (n={beneficial_mask.sum()})', edgecolors='white', linewidth=0.5)
    ax4.scatter(df[~beneficial_mask]['opp_run_before'],
               df[~beneficial_mask]['diff_change_after'],
               color=colors['not_beneficial'], alpha=0.6, s=50,
               label=f'Not Beneficial (n={(~beneficial_mask).sum()})', edgecolors='white', linewidth=0.5)

    # Trend line
    if len(df) > 5:
        z = np.polyfit(df['opp_run_before'], df['diff_change_after'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(df['opp_run_before'].min(), df['opp_run_before'].max(), 100)
        ax4.plot(x_trend, p(x_trend), '--', color='black', alpha=0.7,
                label=f'Trend (slope={z[0]:.2f})')

    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Opponent Run Before Timeout (pts)', fontsize=11)
    ax4.set_ylabel('Score Swing After Timeout (pts)', fontsize=11)
    ax4.set_title('Each Dot = One Timeout\nDid Bigger Runs Lead to Better Timeout Results?', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)

    # ================================================================
    # PLOT 5: Opponent FG% Before vs Timeout Success
    # ================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    fg_bins = [0, 0.25, 0.40, 0.50, 0.60, 0.75, 1.01]
    fg_labels = ['0-25%', '25-40%', '40-50%', '50-60%', '60-75%', '75%+']
    df['opp_fg_bin'] = pd.cut(df['opp_fg_pct_before'], bins=fg_bins, labels=fg_labels, right=False)

    fg_stats = df.groupby('opp_fg_bin', observed=True).agg(
        success_rate=('beneficial', 'mean'),
        count=('beneficial', 'count')
    ).reset_index()

    bar_colors5 = [colors['not_beneficial'] if r < 0.5 else colors['beneficial']
                   for r in fg_stats['success_rate']]
    bars5 = ax5.bar(fg_stats['opp_fg_bin'], fg_stats['success_rate'] * 100,
                    color=bar_colors5, edgecolor='white', linewidth=1.5)

    for bar, count in zip(bars5, fg_stats['count']):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Opponent FG% Before Timeout', fontsize=11)
    ax5.set_ylabel('Timeout Success Rate (%)', fontsize=11)
    ax5.set_title('Does Opponent Shooting Quality\nPredict Timeout Impact?', fontsize=13, fontweight='bold')
    ax5.set_ylim(0, 100)

    # ================================================================
    # PLOT 6: Combined "Danger Score" — multivariable
    # ================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    # Create a composite "danger" metric combining multiple factors
    df['danger_score'] = (
        df['opp_run_before'] * 2 +
        df['opp_fg_pct_before'] * 10 -
        df['own_run_before'] * 1.5 -
        df['score_diff'] * 0.3 +
        df['own_turnovers_before'] * 3
    )

    danger_bins = pd.qcut(df['danger_score'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    df['danger_bin'] = danger_bins

    danger_stats = df.groupby('danger_bin', observed=True).agg(
        success_rate=('beneficial', 'mean'),
        count=('beneficial', 'count')
    ).reset_index()

    bar_colors6 = [colors['not_beneficial'] if r < 0.5 else colors['beneficial']
                   for r in danger_stats['success_rate']]
    bars6 = ax6.bar(danger_stats['danger_bin'], danger_stats['success_rate'] * 100,
                    color=bar_colors6, edgecolor='white', linewidth=1.5)

    for bar, count in zip(bars6, danger_stats['count']):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax6.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Danger Level When Timeout Called', fontsize=11)
    ax6.set_ylabel('Timeout Success Rate (%)', fontsize=11)
    ax6.set_title('Combined "Danger Score"\n(Run + FG% + Turnovers + Deficit)', fontsize=13, fontweight='bold')
    ax6.set_ylim(0, 100)

    # ================================================================
    # PLOT 7: Statistical Significance Tests
    # ================================================================
    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.axis('off')

    # Run actual statistical tests
    text_lines = []
    text_lines.append("📊 STATISTICAL SIGNIFICANCE TESTS")
    text_lines.append("=" * 60)

    # Test 1: Are timeouts after big runs (5+) more beneficial?
    big_run = df[df['opp_run_before'] >= 5]['beneficial']
    small_run = df[df['opp_run_before'] < 5]['beneficial']
    if len(big_run) > 1 and len(small_run) > 1:
        stat, p_val = stats.mannwhitneyu(big_run, small_run, alternative='greater')
        sig = "✅ YES" if p_val < 0.05 else "❌ NO"
        text_lines.append(f"\n1. Timeouts after opponent 5+ pt runs are more beneficial?")
        text_lines.append(f"   Big run success: {big_run.mean()*100:.1f}% vs Small run: {small_run.mean()*100:.1f}%")
        text_lines.append(f"   p-value = {p_val:.4f}  |  Significant: {sig}")

    # Test 2: Are timeouts when behind more beneficial?
    behind = df[df['score_diff'] < 0]['beneficial']
    ahead = df[df['score_diff'] > 0]['beneficial']
    if len(behind) > 1 and len(ahead) > 1:
        stat, p_val = stats.mannwhitneyu(behind, ahead, alternative='greater')
        sig = "✅ YES" if p_val < 0.05 else "❌ NO"
        text_lines.append(f"\n2. Timeouts when trailing are more beneficial than when leading?")
        text_lines.append(f"   Trailing success: {behind.mean()*100:.1f}% vs Leading: {ahead.mean()*100:.1f}%")
        text_lines.append(f"   p-value = {p_val:.4f}  |  Significant: {sig}")

    # Test 3: Does opponent FG% matter?
    hot_opp = df[df['opp_fg_pct_before'] >= 0.5]['beneficial']
    cold_opp = df[df['opp_fg_pct_before'] < 0.5]['beneficial']
    if len(hot_opp) > 1 and len(cold_opp) > 1:
        stat, p_val = stats.mannwhitneyu(hot_opp, cold_opp, alternative='greater')
        sig = "✅ YES" if p_val < 0.05 else "❌ NO"
        text_lines.append(f"\n3. Timeouts when opponent is shooting hot (50%+) are more beneficial?")
        text_lines.append(f"   Hot opp success: {hot_opp.mean()*100:.1f}% vs Cold opp: {cold_opp.mean()*100:.1f}%")
        text_lines.append(f"   p-value = {p_val:.4f}  |  Significant: {sig}")

    # Test 4: Turnovers
    high_to = df[df['own_turnovers_before'] >= 2]['beneficial']
    low_to = df[df['own_turnovers_before'] < 2]['beneficial']
    if len(high_to) > 1 and len(low_to) > 1:
        stat, p_val = stats.mannwhitneyu(high_to, low_to, alternative='two-sided')
        sig = "✅ YES" if p_val < 0.05 else "❌ NO"
        text_lines.append(f"\n4. Do turnovers before timeout affect its success?")
        text_lines.append(f"   High TO success: {high_to.mean()*100:.1f}% vs Low TO: {low_to.mean()*100:.1f}%")
        text_lines.append(f"   p-value = {p_val:.4f}  |  Significant: {sig}")

    # Overall correlation summary
    text_lines.append(f"\n{'='*60}")
    text_lines.append(f"CORRELATION STRENGTHS (Pearson r):")
    corrs = df[['opp_run_before', 'score_diff', 'opp_fg_pct_before',
                'own_run_before', 'own_turnovers_before', 'clock_seconds',
                'period', 'beneficial']].corr()['beneficial'].drop('beneficial')
    corrs = corrs.sort_values(key=abs, ascending=False)
    for feat, r in corrs.items():
        strength = "Strong" if abs(r) > 0.3 else "Moderate" if abs(r) > 0.15 else "Weak"
        text_lines.append(f"   {feat:25s}  r = {r:+.3f}  ({strength})")

    ax7.text(0.02, 0.98, '\n'.join(text_lines), transform=ax7.transAxes,
             fontsize=11, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor=colors['bg'], alpha=0.8))

    # ================================================================
    # PLOT 8: Own team vs Opponent timeouts
    # ================================================================
    ax8 = fig.add_subplot(gs[2, 2])
    if 'calling_team' in df.columns:
        team_stats = df.groupby('calling_team').agg(
            success_rate=('beneficial', 'mean'),
            count=('beneficial', 'count'),
            avg_opp_run=('opp_run_before', 'mean'),
            avg_diff=('diff_change_after', 'mean')
        ).reset_index()

        x = np.arange(len(team_stats))
        bars8 = ax8.bar(x, team_stats['success_rate'] * 100,
                       color=[colors['beneficial'], colors['neutral']],
                       edgecolor='white', linewidth=1.5)

        for bar, row in zip(bars8, team_stats.itertuples()):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'n={row.count}\navg run: {row.avg_opp_run:.1f}\navg swing: {row.avg_diff:+.1f}',
                    ha='center', va='bottom', fontsize=9)

        ax8.set_xticks(x)
        ax8.set_xticklabels(['Opponent Called', 'Our Team Called'])
        ax8.set_ylabel('Timeout Success Rate (%)', fontsize=11)
        ax8.set_title('Who Called the Timeout:\nDoes It Matter?', fontsize=13, fontweight='bold')
        ax8.set_ylim(0, 100)
        ax8.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Analysis saved to: {save_path}")
    print(f"   Open this image to see all the patterns!\n")

    # Also print a plain-English summary
    print_summary(df)


def print_summary(df):
    """Print a coaching-friendly plain-English summary of findings."""
    print("\n" + "=" * 60)
    print("🏀 PLAIN-ENGLISH SUMMARY FOR COACHES")
    print("=" * 60)

    # Finding 1: Run threshold
    for threshold in [4, 5, 6, 7, 8]:
        subset = df[df['opp_run_before'] >= threshold]
        if len(subset) >= 10:
            rate = subset['beneficial'].mean() * 100
            if rate > 55:
                print(f"\n✅ FINDING 1: Call timeout when the opponent goes on a {threshold}+ point run.")
                print(f"   Success rate: {rate:.0f}% (based on {len(subset)} timeouts)")
                print(f"   This is the strongest signal in the data.")
                break

    # Finding 2: Score margin
    behind = df[df['score_diff'] < 0]
    ahead = df[df['score_diff'] > 0]
    if len(behind) > 10 and len(ahead) > 10:
        behind_rate = behind['beneficial'].mean() * 100
        ahead_rate = ahead['beneficial'].mean() * 100
        if behind_rate > ahead_rate:
            print(f"\n✅ FINDING 2: Timeouts are MORE effective when trailing ({behind_rate:.0f}%) vs leading ({ahead_rate:.0f}%).")
            print(f"   Don't waste timeouts when you're comfortably ahead.")
        else:
            print(f"\n📊 FINDING 2: Score margin doesn't have a clear effect.")
            print(f"   Trailing: {behind_rate:.0f}% | Leading: {ahead_rate:.0f}%")

    # Finding 3: Quarter
    q_rates = df.groupby('period')['beneficial'].agg(['mean', 'count'])
    best_q = q_rates[q_rates['count'] >= 10]['mean'].idxmax()
    worst_q = q_rates[q_rates['count'] >= 10]['mean'].idxmin()
    print(f"\n📊 FINDING 3: Timeouts work best in Q{best_q} ({q_rates.loc[best_q, 'mean']*100:.0f}%)")
    print(f"   and worst in Q{worst_q} ({q_rates.loc[worst_q, 'mean']*100:.0f}%).")

    # Finding 4: Own momentum
    own_run_timeouts = df[df['own_run_before'] >= 5]
    if len(own_run_timeouts) >= 5:
        rate = own_run_timeouts['beneficial'].mean() * 100
        print(f"\n{'✅' if rate > 50 else '⚠️'} FINDING 4: Calling timeout during YOUR OWN {5}+ point run:")
        print(f"   Success rate: {rate:.0f}% (n={len(own_run_timeouts)})")
        if rate < 50:
            print(f"   ⚠️  This kills your momentum — avoid unless absolutely necessary!")

    # Finding 5: Opponent shooting
    hot = df[df['opp_fg_pct_before'] >= 0.6]
    if len(hot) >= 5:
        rate = hot['beneficial'].mean() * 100
        print(f"\n📊 FINDING 5: Timeout when opponent shooting 60%+: {rate:.0f}% success (n={len(hot)})")

    # Overall
    overall_rate = df['beneficial'].mean() * 100
    print(f"\n{'='*60}")
    print(f"OVERALL: {overall_rate:.0f}% of all timeouts are beneficial.")
    if overall_rate < 50:
        print("Most timeouts don't actually help — timing is everything!")
    print(f"{'='*60}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='timeout_data.csv')
    args = parser.parse_args()

    df = load_data(args.data)
    create_full_analysis(df)
