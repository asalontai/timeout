import pandas as pd
from collections import defaultdict
import math

from config import DATA_DIR


def classify_event(desc: str, is_team: int) -> str:
    d = (desc or "").lower()
    team = "TEAM" if is_team == 1 else "OPP"

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


def update_stats(stats, value):
    stats["count"] += 1
    delta = value - stats["mean"]
    stats["mean"] += delta / stats["count"]
    delta2 = value - stats["mean"]
    stats["m2"] += delta * delta2


def mine_patterns_for_team(df, window=2, min_count=50):
    pattern_stats = defaultdict(lambda: {"count": 0, "mean": 0.0, "m2": 0.0})

    for game_id, g in df.groupby("game_id"):
        g = g.sort_values("event_num")
        tokens = [classify_event(d, i) for d, i in zip(g["description"], g["is_team"])]
        shifts = g["momentum_shift"].tolist()

        for i in range(len(tokens) - window):
            pattern = tuple(tokens[i:i + window])
            effect = shifts[i + window]
            update_stats(pattern_stats[pattern], effect)

    rows = []
    for pattern, stats in pattern_stats.items():
        if stats["count"] < min_count:
            continue
        count = stats["count"]
        mean = stats["mean"]
        if count > 1:
            var = stats["m2"] / (count - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        se = std / math.sqrt(count) if count > 0 else 0.0
        z = mean / se if se > 0 else 0.0
        rows.append({
            "pattern": " -> ".join(pattern),
            "count": count,
            "avg_momentum_shift": mean,
            "std": std,
            "z_score": z,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values("z_score", ascending=False)
    return out


def main():
    df = pd.read_csv(DATA_DIR / "momentum_per_play_allteams.csv")
    if df.empty:
        raise RuntimeError("momentum_per_play_allteams.csv is empty. Run momentum_per_play_allteams.py first.")

    results = []
    for team_id, g in df.groupby("team_id"):
        top2 = mine_patterns_for_team(g, window=2, min_count=50)
        if not top2.empty:
            top2["team_id"] = team_id
            results.append(top2)

    if not results:
        raise RuntimeError("No patterns found. Try lowering min_count.")

    out = pd.concat(results, ignore_index=True)
    out_path = DATA_DIR / "momentum_patterns_allteams_2step.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved patterns to {out_path}")


if __name__ == "__main__":
    main()
