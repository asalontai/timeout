import sys
import pandas as pd
from collections import deque

from config import DATA_DIR


def clock_to_sec(cl):
    if not isinstance(cl, str) or ":" not in cl:
        return None
    try:
        mm, ss = cl.split(":")
        return int(mm) * 60 + int(ss)
    except Exception:
        return None


def estimate_shift(desc: str, is_team: int) -> float:
    d = (desc or "").lower()
    delta = 0.0

    # scoring
    if ("3-pt" in d or "3pt" in d) and "miss" not in d:
        delta += 3
    elif "free throw" in d and "miss" not in d:
        delta += 1
    elif any(k in d for k in ["layup", "dunk", "jumper", "shot", "tip-in", "hook"]) and "miss" not in d:
        delta += 2

    # turnovers
    if "turnover" in d:
        delta -= 1.5

    # rebounds
    if "rebound" in d:
        delta += 0.4

    # fouls
    if "foul" in d:
        delta -= 0.3

    # blocks / steals
    if "block" in d:
        delta += 0.6
    if "steal" in d:
        delta += 0.8

    # flip sign for opponent
    if is_team == 0:
        delta = -delta

    return delta


def main():
    # Load context ranks
    ctx = pd.read_csv(DATA_DIR / "timeout_optimal_contexts.csv")
    ctx["margin_bin"] = ctx["margin_bin"].astype(str)
    ctx["time_bin"] = ctx["time_bin"].astype(str)
    ctx["trend_bin"] = ctx["trend_bin"].astype(str)

    ctx["rank"] = ctx["timeout_advantage"].rank(pct=True)

    # Use full margin range
    margin_bins = [-100, -15, -8, -3, 3, 8, 15, 100]
    time_bins = [0, 180, 360, 720, 1440, 2880]
    trend_bins = [-10, -0.2, 0.2, 10]

    threshold = 0.8
    if len(sys.argv) > 1:
        try:
            threshold = float(sys.argv[1])
        except Exception:
            pass

    trend_buf = deque(maxlen=3)

    print("Enter plays as: period<TAB>clock<TAB>team_score<TAB>opp_score<TAB>team_tos<TAB>opp_tos<TAB>is_team<TAB>description")
    print("Example: 4\t02:15\t98\t95\t2\t1\t1\tCurry makes 3-pt jumper")
    print(f"Timeout recommendation threshold: top {int(threshold*100)}% contexts")
    print("Type 'quit' to exit.")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            break

        try:
            period_s, clock_s, team_s, opp_s, team_tos_s, opp_tos_s, is_team_s, desc = line.split("\t", 7)
            period = int(period_s)
            team_score = int(team_s)
            opp_score = int(opp_s)
            team_tos = int(team_tos_s)
            opp_tos = int(opp_tos_s)
            is_team = int(is_team_s)
        except Exception:
            print("Invalid format. Use: period<TAB>clock<TAB>team_score<TAB>opp_score<TAB>team_tos<TAB>opp_tos<TAB>is_team<TAB>description")
            continue

        # No recommendation if team has no timeouts
        if team_tos <= 0:
            print("No timeout available.")
            continue

        shift = estimate_shift(desc, is_team)
        trend_buf.append(shift)
        momentum_trend = sum(trend_buf) / len(trend_buf)

        clock_sec = clock_to_sec(clock_s) or 0
        time_remaining = clock_sec + max(0, 4 - period) * 12 * 60
        margin = team_score - opp_score

        margin_bin = pd.cut([margin], bins=margin_bins)[0]
        time_bin = pd.cut([time_remaining], bins=time_bins)[0]
        trend_bin = pd.cut([momentum_trend], bins=trend_bins)[0]

        key = {
            "margin_bin": str(margin_bin),
            "time_bin": str(time_bin),
            "trend_bin": str(trend_bin),
        }

        row = ctx[
            (ctx["margin_bin"] == key["margin_bin"]) &
            (ctx["time_bin"] == key["time_bin"]) &
            (ctx["trend_bin"] == key["trend_bin"]) 
        ]

        if row.empty:
            print("No recommendation (context not in table).")
            continue

        rank = row["rank"].iloc[0]
        adv = row["timeout_advantage"].iloc[0]

        # Adjust threshold if opponent has more timeouts remaining (conservative)
        adj_threshold = threshold + 0.05 if opp_tos > team_tos else threshold

        if rank >= adj_threshold:
            print(f"TIMEOUT RECOMMENDED | rank={rank:.3f} advantage={adv:.3f}")
        else:
            print(f"No timeout | rank={rank:.3f} advantage={adv:.3f}")


if __name__ == "__main__":
    main()
