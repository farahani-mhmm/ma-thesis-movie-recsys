"""Shared settings for the cold-start lab. Every step imports this file, so nothing is hard-coded twice.

Inputs are READ from the original "CSP Dataset" folder (never modified).
Everything this lab writes goes under cold_start_lab/outputs/.
"""
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LAB_DIR.parent                  # ...\MA Thesis
DATA_DIR = PROJECT_DIR / "CSP Dataset"        # read-only inputs
OUT_DIR = LAB_DIR / "outputs"
CSV_ENCODING = "utf-8-sig"

RANDOM_STATE = 42

# "Liked" rules. Both are reported in every experiment.
#   gt7 : rate > 7   (the rule used in the paper and in the old pipeline)
#   ge7 : rate >= 7
LIKE_RULES = {"gt7": (">", 7), "ge7": (">=", 7)}

# How many of a new user's own ratings are revealed to the model (N = 0 means profile only).
N_GRID = [0, 1, 3, 5, 10, 20, 50]

# User-grouped cross-validation: 5 folds, repeated 3 times with different shuffles.
N_FOLDS = 5
N_REPEATS = 3

# The 12 Twitter-profile columns of user_profiles.csv (id and twitterName are NOT features).
PROFILE_COLS = [
    "account_creation_year",
    "friends_count",
    "preferred_hour",
    "weekend_tweeter",
    "preferred_weekday",
    "early_bird",
    "night_owl",
    "geo_enabled",
    "week_tweeter",
    "favourites_count",
    "followers_count",
    "number_of_tweets",
]

# Every user's ratings are put in a random order (position 0, 1, 2, ...).
#   revealed to the model : positions 0 .. N-1
#   evaluation pool       : positions >= EVAL_POOL_START (the SAME pool for every N, so curves are comparable)
# A user is eligible as a test user only if the pool has at least MIN_POOL_SIZE ratings.
EVAL_POOL_START = max(N_GRID)
MIN_POOL_SIZE = 10
MIN_RATINGS_FOR_EVAL = EVAL_POOL_START + MIN_POOL_SIZE


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
