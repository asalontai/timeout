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
    if df.empty:
        raise RuntimeError("momentum_per_play_allteams.csv is empty. Run momentum_per_play_allteams.py first.")

    df = df.sort_values(["game_id", "team_id", "event_num"]).reset_index(drop=True)

    # Timeout flag
    df["is_timeout"] = df["description"].astype(str).str.contains("timeout", case=False, na=False).astype(int)

    # Time remaining
    df["clock_sec"] = df["clock"].apply(clock_to_sec).fillna(0)
    df["period"] = pd.to_numeric(df["period"], errors="coerce").fillna(1)
    df["time_remaining"] = df["clock_sec"] + (4 - df["period"]).clip(lower=0) * 12 * 60

    # Margin (team perspective)
    df["margin"] = df["team_score"] - df["opp_score"]

    # Momentum trend: rolling mean of last 3 shifts
    df["momentum_trend"] = (
        df.groupby(["game_id", "team_id"])["momentum_shift"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Next 3 plays momentum shift sum
    grp = df.groupby(["game_id", "team_id"])
    df["next3_shift_sum"] = (
        grp["momentum_shift"].shift(-1).fillna(0)
        + grp["momentum_shift"].shift(-2).fillna(0)
        + grp["momentum_shift"].shift(-3).fillna(0)
    )

    # Context bins (close games only)
    df = df[df["margin"].between(-10, 10)].copy()
    margin_bins = [-10, -5, -3, 3, 5, 10]
    time_bins = [0, 180, 360, 720, 1440, 2880]  # last 3, 6, 12, 24, 48 minutes
    trend_bins = [-10, -0.2, 0.2, 10]

    df["margin_bin"] = pd.cut(df["margin"], bins=margin_bins)
    df["time_bin"] = pd.cut(df["time_remaining"], bins=time_bins)
    df["trend_bin"] = pd.cut(df["momentum_trend"], bins=trend_bins)

    # Aggregate
    grouped = df.groupby(["margin_bin", "time_bin", "trend_bin"], observed=True)
    rows = []
    for key, g in grouped:
        timeout = g[g["is_timeout"] == 1]
        notime = g[g["is_timeout"] == 0]
        if len(timeout) < 20 or len(notime) < 100:
            continue
        rows.append({
            "margin_bin": str(key[0]),
            "time_bin": str(key[1]),
            "trend_bin": str(key[2]),
            "timeout_n": len(timeout),
            "no_timeout_n": len(notime),
            "avg_next3_timeout": timeout["next3_shift_sum"].mean(),
            "avg_next3_no_timeout": notime["next3_shift_sum"].mean(),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No contexts with enough samples. Relax thresholds.")

    out["timeout_advantage"] = out["avg_next3_timeout"] - out["avg_next3_no_timeout"]
    out = out.sort_values("timeout_advantage", ascending=False)

    out_path = DATA_DIR / "timeout_optimal_contexts.csv"
    out.to_csv(out_path, index=False)

    print(f"Saved timeout context table to {out_path}")
    print("\nTop 10 contexts where timeouts help most:")
    print(out.head(10))
    print("\nBottom 10 contexts where timeouts hurt most:")
    print(out.tail(10))


if __name__ == "__main__":
    main()
