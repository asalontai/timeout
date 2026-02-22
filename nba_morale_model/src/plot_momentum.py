import sys
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import DATA_DIR


def main():
    path = DATA_DIR / "momentum_per_play.csv"
    print(f"Loading {path} ...")
    df = pd.read_csv(path)

    game_id = sys.argv[1].strip() if len(sys.argv) > 1 else input("Enter GAME_ID to plot: ").strip()
    g = df[df["game_id"].astype(str) == game_id]
    if g.empty:
        raise RuntimeError(f"No rows for GAME_ID {game_id}")

    g = g.sort_values("event_num")

    plt.figure(figsize=(12, 5))
    plt.plot(g["event_num"], g["momentum_index"], label="Momentum Index")
    plt.axhline(0, color="gray", linewidth=1)
    plt.title(f"Momentum Index per Play - Game {game_id}")
    plt.xlabel("Event Number")
    plt.ylabel("Momentum Index")
    plt.legend()
    plt.tight_layout()

    out_path = DATA_DIR / f"momentum_{game_id}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
