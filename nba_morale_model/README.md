# NBA Morale Model (2016-17 Warriors)

This project builds a play-by-play feature dataset for the 2016-17 Warriors season (regular season + playoffs) and excludes the 2017 Finals. It computes a Momentum Shock Index (MSI) from scoring runs, 3-point streaks, star impact, margin swings, and opponent timeouts.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Steps

1. Download the Warriors game list (regular season + playoffs, excluding Finals):

```bash
python3 src/download_games.py
```

2. Build the feature dataset from play-by-play:

```bash
python3 src/build_dataset.py
```

3. Train a simple baseline model:

```bash
python3 src/train_baseline.py
```

## Outputs

- `data/games.csv`: game list
- `data/features.csv`: feature dataset for modeling

## Notes

- The Finals exclusion uses these dates (GSW vs CLE): 2017-06-01, 2017-06-04, 2017-06-07, 2017-06-09, 2017-06-12.
- The MSI weights are heuristic and intended to be tuned via validation.
- If you want a strict “no Finals context” evaluation, keep Finals as a hold-out set and never include them in training.

