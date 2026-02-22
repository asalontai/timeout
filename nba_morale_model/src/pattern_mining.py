import pandas as pd
from collections import defaultdict

from config import DATA_DIR


def classify_event(desc: str, is_gsw: int) -> str:
    d = (desc or "").lower()
    team = "GSW" if is_gsw == 1 else "OPP"

    if "turnover" in d:
        return f"{team}_TO"
    if "3-pt" in d or "3pt" in d:
        if "miss" in d:
            return f"{team}_3MISS"
        return f"{team}_3"
    if "free throw" in d:
        if "miss" in d:
            return f"{team}_FTMISS"
        return f"{team}_FT"
    if any(k in d for k in ["layup", "dunk", "jumper", "shot", "tip-in", "hook"]):
        if "miss" in d:
            return f"{team}_2MISS"
        return f"{team}_2"
    if "foul" in d:
        return f"{team}_FOUL"
    if "rebound" in d:
        return f"{team}_REB"
    if "steal" in d:
        return f"{team}_STL"
    if "block" in d:
        return f"{team}_BLK"

    return f"{team}_OTHER"


def mine_patterns(df, window=2, min_count=50):
    pattern_stats = defaultdict(lambda: {"count": 0, "sum_shift": 0.0})

    for game_id, g in df.groupby("game_id"):
        g = g.sort_values("event_num")
        tokens = [classify_event(d, i) for d, i in zip(g["description"], g["is_gsw"])]
        shifts = g["momentum_shift"].tolist()

        for i in range(len(tokens) - window):
            pattern = tuple(tokens[i:i + window])
            # effect measured at the event immediately after the pattern
            effect = shifts[i + window]
            pattern_stats[pattern]["count"] += 1
            pattern_stats[pattern]["sum_shift"] += effect

    rows = []
    for pattern, stats in pattern_stats.items():
        if stats["count"] < min_count:
            continue
        avg_shift = stats["sum_shift"] / stats["count"]
        rows.append({
            "pattern": " -> ".join(pattern),
            "count": stats["count"],
            "avg_momentum_shift": avg_shift,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("avg_momentum_shift", ascending=False)
    return out


def main():
    df = pd.read_csv(DATA_DIR / "momentum_per_play.csv")

    if df.empty:
        raise RuntimeError("momentum_per_play.csv is empty. Run momentum_per_play.py first.")

    # Mine 2-step and 3-step patterns
    top2 = mine_patterns(df, window=2, min_count=50)
    top3 = mine_patterns(df, window=3, min_count=50)

    out2 = DATA_DIR / "momentum_patterns_2step.csv"
    out3 = DATA_DIR / "momentum_patterns_3step.csv"
    top2.to_csv(out2, index=False)
    top3.to_csv(out3, index=False)

    print(f"Saved 2-step patterns to {out2}")
    print(f"Saved 3-step patterns to {out3}")

    print("\nTop 10 positive 2-step patterns:")
    print(top2.head(10))

    print("\nTop 10 negative 2-step patterns:")
    print(top2.tail(10))


if __name__ == "__main__":
    main()
