import pandas as pd
import numpy as np

from config import DATA_DIR


def clock_to_sec(cl):
    if not isinstance(cl, str) or ":" not in cl:
        return None
    try:
        mm, ss = cl.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return None


def main():
    df = pd.read_csv(DATA_DIR / "momentum_per_play_allteams.csv")
    ctx = pd.read_csv(DATA_DIR / "timeout_optimal_contexts.csv")

    if df.empty or ctx.empty:
        raise RuntimeError("Missing data. Run timeout_optimal_analysis.py first.")

    # Rebuild context bins to match timeout_optimal_analysis.py (close games only)
    df = df.sort_values(["game_id", "team_id", "event_num"]).reset_index(drop=True)
    df["is_timeout"] = df["description"].astype(str).str.contains("timeout", case=False, na=False).astype(int)

    df["clock_sec"] = df["clock"].apply(clock_to_sec).fillna(0)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1)
    df["time_remaining"] = df["clock_sec"] + (4 - df["period"]).clip(lower=0) * 12 * 60
    df["margin"] = df["team_score"] - df["opp_score"]
    df = df[df["margin"].between(-10, 10)].copy()

    df["momentum_trend"] = (
        df.groupby(["game_id", "team_id"])["momentum_shift"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    margin_bins = [-10, -5, -3, 3, 5, 10]
    time_bins = [0, 180, 360, 720, 1440, 2880]
    trend_bins = [-10, -0.2, 0.2, 10]

    df["margin_bin"] = pd.cut(df["margin"], bins=margin_bins).astype(str)
    df["time_bin"] = pd.cut(df["time_remaining"], bins=time_bins).astype(str)
    df["trend_bin"] = pd.cut(df["momentum_trend"], bins=trend_bins).astype(str)

    ctx["margin_bin"] = ctx["margin_bin"].astype(str)
    ctx["time_bin"] = ctx["time_bin"].astype(str)
    ctx["trend_bin"] = ctx["trend_bin"].astype(str)

    # Merge with context table
    merged = df.merge(
        ctx,
        on=["margin_bin", "time_bin", "trend_bin"],
        how="inner",
    )

    # Rank contexts by timeout advantage
    ctx_rank = ctx.copy()
    ctx_rank["rank"] = ctx_rank["timeout_advantage"].rank(pct=True)
    ctx_rank = ctx_rank[["margin_bin", "time_bin", "trend_bin", "rank"]]

    merged = merged.merge(ctx_rank, on=["margin_bin", "time_bin", "trend_bin"], how="left")

    # Compare actual timeouts vs all plays
    timeouts = merged[merged["is_timeout"] == 1]
    non_timeouts = merged[merged["is_timeout"] == 0]

    if timeouts.empty:
        raise RuntimeError("No timeouts in merged data.")

    print("Timeout optimality comparison (close games):")
    print(f"  Avg context rank for actual timeouts: {timeouts['rank'].mean():.3f}")
    print(f"  Avg context rank for non-timeouts: {non_timeouts['rank'].mean():.3f}")

    # % of timeouts in top 20% contexts
    top20 = timeouts[timeouts["rank"] >= 0.8]
    print(f"  % of timeouts in top 20% contexts: {len(top20)/len(timeouts):.3f}")

    # Compare distribution by time bin
    dist = timeouts.groupby("time_bin")["rank"].mean().reset_index()
    print("\nAvg context rank by time bin (timeouts only):")
    print(dist)


if __name__ == "__main__":
    main()
