import pandas as pd
import numpy as np

from config import FEATURES_CSV


def main():
    df = pd.read_csv(FEATURES_CSV)

    target = "opp_points_next3_scoring_events"
    if target not in df.columns:
        raise RuntimeError(f"Missing target column: {target}. Rebuild features first.")

    # Bin by context to reduce confounding
    # Margin bins: [-inf,-10],[-10,0],[0,10],[10,inf]
    margin_bins = [-1e9, -10, 0, 10, 1e9]
    df["margin_bin"] = pd.cut(df["margin"], bins=margin_bins)

    # Time remaining bins: coarser buckets across game
    df["time_bin"] = pd.cut(df["seconds_remaining"], bins=4)

    # Only keep bins with both treatment and control
    grouped = df.groupby(["margin_bin", "time_bin"], observed=True)

    diffs = []
    weights = []

    for _, g in grouped:
        treated = g[g["gsw_3s_run_flag"] == 1][target]
        control = g[g["gsw_3s_run_flag"] == 0][target]
        if len(treated) < 2 or len(control) < 2:
            continue
        diff = treated.mean() - control.mean()
        diffs.append(diff)
        weights.append(len(treated) + len(control))

    if not diffs:
        raise RuntimeError("No bins with enough treated/control samples. Try relaxing thresholds.")

    # Weighted average treatment effect
    diffs = np.array(diffs)
    weights = np.array(weights)
    ate = np.average(diffs, weights=weights)

    # Bootstrap for a simple CI
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(500):
        idx = rng.integers(0, len(diffs), len(diffs))
        boot.append(np.average(diffs[idx], weights=weights[idx]))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    print(f"ATE (3s-run vs not) on opp points next3: {ate:.3f}")
    print(f"95% bootstrap CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"Bins used: {len(diffs)}")


if __name__ == "__main__":
    main()
