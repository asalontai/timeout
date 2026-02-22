import pandas as pd

from config import DATA_DIR


def is_timeout(desc: str) -> bool:
    return isinstance(desc, str) and "timeout" in desc.lower()


def main():
    df = pd.read_csv(DATA_DIR / "momentum_per_play_allteams.csv")

    if df.empty:
        raise RuntimeError("momentum_per_play_allteams.csv is empty. Run momentum_per_play_allteams.py first.")

    df = df.sort_values(["game_id", "team_id", "event_num"]).reset_index(drop=True)

    # Identify timeout events
    df["is_timeout"] = df["description"].apply(is_timeout).astype(int)

    # Momentum shift at timeout and next events
    df["next_momentum_shift"] = df.groupby(["game_id", "team_id"])["momentum_shift"].shift(-1)
    df["next3_momentum_shift_sum"] = (
        df.groupby(["game_id", "team_id"])["momentum_shift"].shift(-1).fillna(0)
        + df.groupby(["game_id", "team_id"])["momentum_shift"].shift(-2).fillna(0)
        + df.groupby(["game_id", "team_id"])["momentum_shift"].shift(-3).fillna(0)
    )

    timeouts = df[df["is_timeout"] == 1].copy()

    if timeouts.empty:
        raise RuntimeError("No timeout events found in data.")

    # Compare momentum shift after timeouts vs non-timeouts
    avg_after_timeout = timeouts["next_momentum_shift"].mean()
    avg_after_non = df[df["is_timeout"] == 0]["next_momentum_shift"].mean()
    avg_after_timeout_3 = timeouts["next3_momentum_shift_sum"].mean()
    avg_after_non_3 = df[df["is_timeout"] == 0]["next3_momentum_shift_sum"].mean()

    # Split by which side called timeout
    timeouts_team = timeouts[timeouts["is_team"] == 1]
    timeouts_opp = timeouts[timeouts["is_team"] == 0]

    avg_after_team = timeouts_team["next_momentum_shift"].mean()
    avg_after_opp = timeouts_opp["next_momentum_shift"].mean()
    avg_after_team_3 = timeouts_team["next3_momentum_shift_sum"].mean()
    avg_after_opp_3 = timeouts_opp["next3_momentum_shift_sum"].mean()

    print("Timeout momentum impact:")
    print(f"  Avg momentum shift after any timeout: {avg_after_timeout:.4f}")
    print(f"  Avg momentum shift after non-timeouts: {avg_after_non:.4f}")
    print(f"  Avg momentum shift after TEAM timeout: {avg_after_team:.4f}")
    print(f"  Avg momentum shift after OPP timeout: {avg_after_opp:.4f}")
    print(f"  Avg momentum shift over next 3 plays after timeout: {avg_after_timeout_3:.4f}")
    print(f"  Avg momentum shift over next 3 plays after non-timeout: {avg_after_non_3:.4f}")
    print(f"  Avg momentum shift over next 3 plays after TEAM timeout: {avg_after_team_3:.4f}")
    print(f"  Avg momentum shift over next 3 plays after OPP timeout: {avg_after_opp_3:.4f}")

    # Contextual: only large momentum swings before timeout
    df["prev_momentum_shift"] = df.groupby(["game_id", "team_id"])["momentum_shift"].shift(1)
    timeouts["prev_momentum_shift"] = df.loc[timeouts.index, "prev_momentum_shift"]
    big_shift_timeouts = timeouts[timeouts["prev_momentum_shift"].abs() >= 0.5]
    if not big_shift_timeouts.empty:
        avg_after_big = big_shift_timeouts["next_momentum_shift"].mean()
        print(f"  Avg momentum shift after timeout following big shift (>=0.5): {avg_after_big:.4f}")


if __name__ == "__main__":
    main()
