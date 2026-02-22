from pathlib import Path

# Season config (regular season only)
SEASONS = ["2016-17", "2017-18", "2018-19", "2019-20"]
TEAM_ID = 1610612744  # Golden State Warriors
TEAM_ABBR = "GSW"

# 2017 NBA Finals game dates (GSW vs CLE). Used to exclude Finals from training.
FINALS_DATES = {
    "2017-06-01",
    "2017-06-04",
    "2017-06-07",
    "2017-06-09",
    "2017-06-12",
}

# Star players for 2016-17 Warriors; used for star_impact feature.
STAR_PLAYERS = {
    "Stephen Curry",
    "Klay Thompson",
    "Kevin Durant",
    "Draymond Green",
}

# Player IDs for 2016-17 Warriors (nba_data CSV uses pid)
STAR_PLAYER_IDS = {
    201939,  # Stephen Curry
    202691,  # Klay Thompson
    201142,  # Kevin Durant
    203110,  # Draymond Green
}

# Feature window sizes (scoring events)
MARGIN_SWING_WINDOW = 5

# Data paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
GAMES_CSV = DATA_DIR / "games.csv"
FEATURES_CSV = DATA_DIR / "features.csv"
PBP_DIR = DATA_DIR / "nba_data"
PBP_REG_CSV = PBP_DIR / "datanba_2016.csv"
PBP_PO_CSV = PBP_DIR / "datanba_po_2016.csv"
