import pandas as pd

from config import FEATURES_CSV


def main():
    df = pd.read_csv(FEATURES_CSV)

    required = [
        "momentum_score",
        "momentum_shift",
        "gsw_back_to_back_3s_3min",
        "opp_back_to_back_3s_3min",
        "is_gsw_score",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Rebuild features first.")

    # Effect of GSW back-to-back 3s in 3-minute window
    gsw_events = df[df["gsw_back_to_back_3s_3min"] == 1]
    base_events = df[df["gsw_back_to_back_3s_3min"] == 0]

    if gsw_events.empty:
        raise RuntimeError("No GSW back-to-back 3s events found.")

    avg_shift_on = gsw_events["momentum_shift"].mean()
    avg_shift_off = base_events["momentum_shift"].mean()

    # Momentum flip: momentum_shift > 0
    flip_rate_on = (gsw_events["momentum_shift"] > 0).mean()
    flip_rate_off = (base_events["momentum_shift"] > 0).mean()

    print("Momentum analysis (3-minute window, back-to-back 3s):")
    print(f"  Avg momentum shift when GSW back-to-back 3s: {avg_shift_on:.3f}")
    print(f"  Avg momentum shift otherwise: {avg_shift_off:.3f}")
    print(f"  Momentum-up rate when GSW back-to-back 3s: {flip_rate_on:.3f}")
    print(f"  Momentum-up rate otherwise: {flip_rate_off:.3f}")

    # Opponent back-to-back 3s
    opp_events = df[df["opp_back_to_back_3s_3min"] == 1]
    if not opp_events.empty:
        avg_shift_opp = opp_events["momentum_shift"].mean()
        print(f"  Avg momentum shift when opponent back-to-back 3s: {avg_shift_opp:.3f}")


if __name__ == "__main__":
    main()
