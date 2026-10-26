from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import sys
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

MOVIES_CSV = BASE_DIR / "movies.csv"
RATINGS_CSV = BASE_DIR / "ratings.csv"
USER_PROFILES_CSV = BASE_DIR / "user_profiles.csv"

OUTPUT_ROOT = BASE_DIR / "recommender_training_outputs"
CSV_ENCODING = "utf-8-sig"
RANDOM_STATE = 42

USER_ID_COL = "user_id"
MOVIE_ID_COL = "movie_id"
RATE_COL = "rate"
TARGET_COL = "liked"

LIKE_THRESHOLD = 7.0
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
MIN_USER_INTERACTIONS_FOR_TEST = 5

TOP_N_RECOMMENDATIONS = 10
RANKING_EVAL_K = 10
RUN_RANKING_EVAL = True
RUN_FINAL_RECOMMENDATIONS = True

# Set e.g. 30000 for a quick smoke test. Keep None for the real run.
DEBUG_MAX_INTERACTIONS: Optional[int] = None

DIRECTOR_FAME_COL = "director_fame_score_llm"
ACTORS_FAME_COL = "actors_fame_score_llm"
DIRECTORS_COL = "directors"
ACTORS_COL = "actors"

# Actor lists can be long. First cast names usually carry the strongest signal.
MAX_ACTORS_PER_MOVIE_FOR_AFFINITY = 8

# Fast, focused search space. The previous broad grid was intentionally reduced.
BALANCING_METHODS = ["none", "sample_weight_balanced"]
TOP_VALIDATION_CONFIGS_FOR_FINAL_TEST = 15

# Keep only ablations that answer the paper question:
# 1) behavior without LLM, 2) raw LLM fame, 3) personalized LLM fame + director/actor affinity,
# 4) same as 3 plus original user profile columns.
RUN_BEHAVIOR_V2_ONLY_EXPERIMENT = True
RUN_BEHAVIOR_V2_PLUS_LLM_RAW_EXPERIMENT = True
RUN_LLM_PERSONALIZED_AFFINITY_EXPERIMENT = True
RUN_USER_PROFILE_PLUS_LLM_PERSONALIZED_AFFINITY_EXPERIMENT = True

# Threshold tuning is still useful, but the grid is now smaller and is not saved for every model.
THRESHOLD_SELECTION_METRIC = "f1"
MIN_THRESHOLD = 0.20
MAX_THRESHOLD = 0.80
THRESHOLD_GRID_SIZE = 61

# Model sizes chosen to stay practical on your server.
HGB_MAX_ITER = 320
HGB_DEEP_MAX_ITER = 360

# Smoothing priors. Larger values shrink sparse user/movie/person histories toward global means.
USER_HISTORY_SMOOTHING = 20.0
MOVIE_HISTORY_SMOOTHING = 30.0
PERSON_HISTORY_SMOOTHING = 5.0
FAME_PREF_SMOOTHING = 10.0

# Leakage fix (v4): features of TRAINING rows are built out-of-fold, so a row's own label never
# enters its own user/movie/director/actor statistics. Eval rows still use all of the training data.
# More folds = training features see more data (closer to what eval rows see) but cost more time.
OOF_FOLDS = 10


# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

@dataclass
class OutputDirs:
    root: Path
    metrics: Path
    figures: Path
    recommendations: Path
    predictions: Path
    model: Path


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_output_dirs() -> OutputDirs:
    run_dir = ensure_dir(OUTPUT_ROOT / f"run_{now_stamp()}")
    return OutputDirs(
        root=run_dir,
        metrics=ensure_dir(run_dir / "03_metrics"),
        figures=ensure_dir(run_dir / "04_figures"),
        recommendations=ensure_dir(run_dir / "06_recommendations"),
        predictions=ensure_dir(run_dir / "07_predictions"),
        model=ensure_dir(run_dir / "08_model"),
    )


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# Stage timings, written to 03_metrics/stage_timings.csv at the end of the run. They show where the
# time goes, so optimization effort can go to the stages that actually dominate.
STAGE_TIMINGS: List[Dict[str, Any]] = []


@contextmanager
def stage(name: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        seconds = time.perf_counter() - start
        STAGE_TIMINGS.append({"stage": name, "seconds": round(seconds, 2)})
        log(f"[timing] {name}: {seconds:.1f}s")


def timed(name: str, fn, *args, **kwargs):
    with stage(name):
        return fn(*args, **kwargs)


def file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment_info() -> Dict[str, Any]:
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "lightgbm", "xgboost", "catboost"]
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count(),
        "random_state": RANDOM_STATE,
        "package_versions": {name: package_version(name) for name in packages},
        "input_sha256": {
            "movies.csv": file_sha256(MOVIES_CSV),
            "ratings.csv": file_sha256(RATINGS_CSV),
            "user_profiles.csv": file_sha256(USER_PROFILES_CSV),
        },
    }


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, encoding=CSV_ENCODING, low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding=CSV_ENCODING)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_std(series: pd.Series) -> float:
    value = series.std()
    return float(value) if pd.notna(value) else 0.0


def split_people(value: Any, max_items: Optional[int] = None) -> List[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    # Dataset uses pipes. Avoid splitting on comma because names can contain commas in some exports.
    parts = [p.strip() for p in text.split("|")]
    parts = [p for p in parts if p]
    if max_items is not None:
        parts = parts[:max_items]
    return parts


# =============================================================================
# CLEANING
# =============================================================================


def clean_movies(raw_movies: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    df = raw_movies.copy()
    report["movies_raw_rows"] = int(len(df))
    report["movies_raw_columns"] = list(df.columns)

    if "id" not in df.columns and MOVIE_ID_COL in df.columns:
        df = df.rename(columns={MOVIE_ID_COL: "id"})
    if "id" not in df.columns:
        raise ValueError("movies.csv must contain 'id' or 'movie_id'.")

    df["id"] = safe_to_numeric(df["id"])
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)

    duplicate_count = int(df.duplicated(subset=["id"]).sum())
    df = df.drop_duplicates(subset=["id"], keep="first").copy()
    report["movies_duplicate_ids_removed"] = duplicate_count

    for col in ["main_title", "original_title", DIRECTORS_COL, ACTORS_COL]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    for col in [DIRECTOR_FAME_COL, ACTORS_FAME_COL]:
        if col not in df.columns:
            raise ValueError(f"movies.csv must contain '{col}'.")
        df[col] = safe_to_numeric(df[col])

    df["directors_tokens"] = df[DIRECTORS_COL].apply(split_people)
    df["actors_tokens"] = df[ACTORS_COL].apply(lambda x: split_people(x, MAX_ACTORS_PER_MOVIE_FOR_AFFINITY))

    keep_cols = [
        "id",
        "main_title",
        "original_title",
        DIRECTORS_COL,
        ACTORS_COL,
        "directors_tokens",
        "actors_tokens",
        DIRECTOR_FAME_COL,
        ACTORS_FAME_COL,
    ]
    df = df[keep_cols].rename(columns={"id": MOVIE_ID_COL})

    report["movies_clean_rows"] = int(len(df))
    report["movies_clean_unique_ids"] = int(df[MOVIE_ID_COL].nunique())
    report["llm_movie_features_used"] = [DIRECTOR_FAME_COL, ACTORS_FAME_COL]
    report["content_columns_used"] = [DIRECTORS_COL, ACTORS_COL]
    return df


def clean_users(raw_users: pd.DataFrame, report: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    df = raw_users.copy()
    report["users_raw_rows"] = int(len(df))
    report["users_raw_columns"] = list(df.columns)

    if "id" not in df.columns:
        raise ValueError("user_profiles.csv must contain an 'id' column.")

    df["id"] = safe_to_numeric(df["id"])
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)

    duplicate_count = int(df.duplicated(subset=["id"]).sum())
    df = df.drop_duplicates(subset=["id"], keep="first").copy()
    report["user_duplicate_ids_removed"] = duplicate_count

    if "twitterName" in df.columns:
        df = df.drop(columns=["twitterName"])

    df = df.rename(columns={"id": USER_ID_COL})

    user_feature_cols: List[str] = []
    for col in df.columns:
        if col == USER_ID_COL:
            continue
        df[col] = safe_to_numeric(df[col])
        if pd.api.types.is_numeric_dtype(df[col]):
            user_feature_cols.append(col)

    df = df[[USER_ID_COL] + user_feature_cols].copy()
    report["users_clean_rows"] = int(len(df))
    report["users_clean_unique_ids"] = int(df[USER_ID_COL].nunique())
    report["user_profile_features_used"] = user_feature_cols
    return df, user_feature_cols


def clean_ratings(raw_ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
    df = raw_ratings.copy()
    report["ratings_raw_rows"] = int(len(df))
    report["ratings_raw_columns"] = list(df.columns)

    required = ["id", MOVIE_ID_COL, RATE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ratings.csv is missing required columns: {missing}")

    df = df.rename(columns={"id": USER_ID_COL})
    df[USER_ID_COL] = safe_to_numeric(df[USER_ID_COL])
    df[MOVIE_ID_COL] = safe_to_numeric(df[MOVIE_ID_COL])
    df[RATE_COL] = safe_to_numeric(df[RATE_COL])
    df = df.dropna(subset=[USER_ID_COL, MOVIE_ID_COL, RATE_COL]).copy()
    df[USER_ID_COL] = df[USER_ID_COL].astype(int)
    df[MOVIE_ID_COL] = df[MOVIE_ID_COL].astype(int)

    before = len(df)
    df = df[(df[RATE_COL] >= 0) & (df[RATE_COL] <= 10)].copy()
    report["ratings_removed_invalid_rate_range"] = int(before - len(df))

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    report["ratings_full_duplicates_removed"] = duplicate_count

    before = len(df)
    df = df.sort_values([USER_ID_COL, MOVIE_ID_COL, "date"], na_position="first")
    df = df.drop_duplicates(subset=[USER_ID_COL, MOVIE_ID_COL], keep="last").copy()
    report["ratings_duplicate_user_movie_removed_keep_latest"] = int(before - len(df))

    movie_ids = set(movies[MOVIE_ID_COL].astype(int))
    user_ids = set(users[USER_ID_COL].astype(int))

    before = len(df)
    df = df[df[MOVIE_ID_COL].isin(movie_ids)].copy()
    report["ratings_removed_no_matching_movie"] = int(before - len(df))

    before = len(df)
    df = df[df[USER_ID_COL].isin(user_ids)].copy()
    report["ratings_removed_no_matching_user_profile"] = int(before - len(df))

    df[TARGET_COL] = (df[RATE_COL] > LIKE_THRESHOLD).astype(int)

    if DEBUG_MAX_INTERACTIONS is not None and len(df) > DEBUG_MAX_INTERACTIONS:
        df = df.sample(n=DEBUG_MAX_INTERACTIONS, random_state=RANDOM_STATE).copy()
        report["debug_max_interactions_applied"] = int(DEBUG_MAX_INTERACTIONS)

    report["ratings_clean_rows"] = int(len(df))
    report["ratings_clean_unique_users"] = int(df[USER_ID_COL].nunique())
    report["ratings_clean_unique_movies"] = int(df[MOVIE_ID_COL].nunique())
    report["positive_like_ratio"] = float(df[TARGET_COL].mean()) if len(df) else None
    return df


# =============================================================================
# SPLITS AND FEATURE BUILDING
# =============================================================================


def per_user_temporal_split(ratings: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: List[pd.DataFrame] = []
    test_parts: List[pd.DataFrame] = []
    sorted_df = ratings.sort_values([USER_ID_COL, "date", MOVIE_ID_COL], na_position="first")

    for _, group in sorted_df.groupby(USER_ID_COL, sort=False):
        n = len(group)
        if n < MIN_USER_INTERACTIONS_FOR_TEST:
            train_parts.append(group)
            continue
        n_test = max(1, int(math.ceil(n * test_size)))
        if n_test >= n:
            n_test = n - 1
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=ratings.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=ratings.columns)
    return train_df, test_df


def class_distribution(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    total = int(len(df))
    positives = int(df[TARGET_COL].sum()) if total else 0
    negatives = total - positives
    return {
        "split": name,
        "rows": total,
        "liked_1": positives,
        "not_liked_0": negatives,
        "liked_ratio": float(positives / total) if total else np.nan,
        "not_liked_ratio": float(negatives / total) if total else np.nan,
    }


def merge_base_features(ratings_df: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    df = ratings_df.merge(users, on=USER_ID_COL, how="left", validate="many_to_one")
    df = df.merge(movies, on=MOVIE_ID_COL, how="left", validate="many_to_one")
    return df


def fit_behavior_aggregates(train_ratings: pd.DataFrame) -> Dict[str, Any]:
    global_mean_rate = float(train_ratings[RATE_COL].mean())
    global_like_ratio = float(train_ratings[TARGET_COL].mean())
    global_rate_std = safe_std(train_ratings[RATE_COL])

    user_stats = train_ratings.groupby(USER_ID_COL).agg(
        user_history_count=(RATE_COL, "size"),
        user_history_sum_rate=(RATE_COL, "sum"),
        user_history_mean_rate_raw=(RATE_COL, "mean"),
        user_history_rating_std=(RATE_COL, "std"),
        user_history_like_count=(TARGET_COL, "sum"),
    ).reset_index()
    user_stats["user_history_dislike_count"] = user_stats["user_history_count"] - user_stats["user_history_like_count"]
    user_stats["user_history_like_ratio_raw"] = user_stats["user_history_like_count"] / user_stats["user_history_count"].clip(lower=1)
    user_stats["user_history_mean_rate"] = (
        user_stats["user_history_sum_rate"] + USER_HISTORY_SMOOTHING * global_mean_rate
    ) / (user_stats["user_history_count"] + USER_HISTORY_SMOOTHING)
    user_stats["user_history_like_ratio"] = (
        user_stats["user_history_like_count"] + USER_HISTORY_SMOOTHING * global_like_ratio
    ) / (user_stats["user_history_count"] + USER_HISTORY_SMOOTHING)
    user_stats["user_history_count_log"] = np.log1p(user_stats["user_history_count"])
    user_stats["user_history_like_count_log"] = np.log1p(user_stats["user_history_like_count"])
    user_stats["user_history_dislike_count_log"] = np.log1p(user_stats["user_history_dislike_count"])
    user_stats["user_history_rating_std"] = user_stats["user_history_rating_std"].fillna(0.0)

    movie_stats = train_ratings.groupby(MOVIE_ID_COL).agg(
        movie_history_count=(RATE_COL, "size"),
        movie_history_sum_rate=(RATE_COL, "sum"),
        movie_history_mean_rate_raw=(RATE_COL, "mean"),
        movie_history_rating_std=(RATE_COL, "std"),
        movie_history_like_count=(TARGET_COL, "sum"),
    ).reset_index()
    movie_stats["movie_history_dislike_count"] = movie_stats["movie_history_count"] - movie_stats["movie_history_like_count"]
    movie_stats["movie_history_like_ratio_raw"] = movie_stats["movie_history_like_count"] / movie_stats["movie_history_count"].clip(lower=1)
    movie_stats["movie_history_mean_rate"] = (
        movie_stats["movie_history_sum_rate"] + MOVIE_HISTORY_SMOOTHING * global_mean_rate
    ) / (movie_stats["movie_history_count"] + MOVIE_HISTORY_SMOOTHING)
    movie_stats["movie_history_like_ratio"] = (
        movie_stats["movie_history_like_count"] + MOVIE_HISTORY_SMOOTHING * global_like_ratio
    ) / (movie_stats["movie_history_count"] + MOVIE_HISTORY_SMOOTHING)
    movie_stats["movie_history_count_log"] = np.log1p(movie_stats["movie_history_count"])
    movie_stats["movie_history_like_count_log"] = np.log1p(movie_stats["movie_history_like_count"])
    movie_stats["movie_history_dislike_count_log"] = np.log1p(movie_stats["movie_history_dislike_count"])
    movie_stats["movie_history_rating_std"] = movie_stats["movie_history_rating_std"].fillna(0.0)

    return {
        "global": {
            "global_mean_rate": global_mean_rate,
            "global_like_ratio": global_like_ratio,
            "global_rate_std": global_rate_std,
        },
        "user_stats": user_stats,
        "movie_stats": movie_stats,
    }


def apply_behavior_features(df: pd.DataFrame, aggregates: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out = out.merge(aggregates["user_stats"], on=USER_ID_COL, how="left")
    out = out.merge(aggregates["movie_stats"], on=MOVIE_ID_COL, how="left")

    g = aggregates["global"]
    global_mean_rate = g["global_mean_rate"]
    global_like_ratio = g["global_like_ratio"]
    global_rate_std = g["global_rate_std"]

    fill_values = {
        "user_history_count": 0.0,
        "user_history_sum_rate": 0.0,
        "user_history_mean_rate_raw": global_mean_rate,
        "user_history_mean_rate": global_mean_rate,
        "user_history_rating_std": global_rate_std,
        "user_history_like_count": 0.0,
        "user_history_dislike_count": 0.0,
        "user_history_like_ratio_raw": global_like_ratio,
        "user_history_like_ratio": global_like_ratio,
        "user_history_count_log": 0.0,
        "user_history_like_count_log": 0.0,
        "user_history_dislike_count_log": 0.0,
        "movie_history_count": 0.0,
        "movie_history_sum_rate": 0.0,
        "movie_history_mean_rate_raw": global_mean_rate,
        "movie_history_mean_rate": global_mean_rate,
        "movie_history_rating_std": global_rate_std,
        "movie_history_like_count": 0.0,
        "movie_history_dislike_count": 0.0,
        "movie_history_like_ratio_raw": global_like_ratio,
        "movie_history_like_ratio": global_like_ratio,
        "movie_history_count_log": 0.0,
        "movie_history_like_count_log": 0.0,
        "movie_history_dislike_count_log": 0.0,
    }
    for col, value in fill_values.items():
        if col in out.columns:
            out[col] = out[col].fillna(value)

    out["user_bias_rate"] = out["user_history_mean_rate"] - global_mean_rate
    out["movie_bias_rate"] = out["movie_history_mean_rate"] - global_mean_rate
    out["user_bias_like"] = out["user_history_like_ratio"] - global_like_ratio
    out["movie_bias_like"] = out["movie_history_like_ratio"] - global_like_ratio
    out["user_movie_rate_affinity"] = out["user_history_mean_rate"] * out["movie_history_mean_rate"]
    out["user_movie_like_affinity"] = out["user_history_like_ratio"] * out["movie_history_like_ratio"]
    out["movie_minus_user_like_tendency"] = out["movie_history_like_ratio"] - out["user_history_like_ratio"]
    out["movie_minus_user_rate_tendency"] = out["movie_history_mean_rate"] - out["user_history_mean_rate"]
    out["combined_like_bias"] = out["user_bias_like"] + out["movie_bias_like"]
    out["combined_rate_bias"] = out["user_bias_rate"] + out["movie_bias_rate"]
    return out


def behavior_v2_features() -> List[str]:
    return [
        "user_history_count_log",
        "user_history_mean_rate",
        "user_history_mean_rate_raw",
        "user_history_rating_std",
        "user_history_like_ratio",
        "user_history_like_ratio_raw",
        "user_history_like_count_log",
        "user_history_dislike_count_log",
        "movie_history_count_log",
        "movie_history_mean_rate",
        "movie_history_mean_rate_raw",
        "movie_history_rating_std",
        "movie_history_like_ratio",
        "movie_history_like_ratio_raw",
        "movie_history_like_count_log",
        "movie_history_dislike_count_log",
        "user_bias_rate",
        "movie_bias_rate",
        "user_bias_like",
        "movie_bias_like",
        "user_movie_rate_affinity",
        "user_movie_like_affinity",
        "movie_minus_user_like_tendency",
        "movie_minus_user_rate_tendency",
        "combined_like_bias",
        "combined_rate_bias",
    ]


def raw_llm_features() -> List[str]:
    return [DIRECTOR_FAME_COL, ACTORS_FAME_COL]


def personalized_llm_fame_features() -> List[str]:
    return [
        "user_director_fame_liked_mean",
        "user_actor_fame_liked_mean",
        "user_director_fame_all_mean",
        "user_actor_fame_all_mean",
        "director_fame_distance_from_user_liked_pref",
        "actor_fame_distance_from_user_liked_pref",
        "director_fame_minus_user_liked_pref",
        "actor_fame_minus_user_liked_pref",
        "director_fame_x_user_liked_pref",
        "actor_fame_x_user_liked_pref",
        "user_high_director_fame_like_ratio",
        "user_high_actor_fame_like_ratio",
    ]


def content_affinity_features() -> List[str]:
    return [
        "same_director_rated_count_log",
        "same_director_liked_count_log",
        "same_director_like_ratio_mean",
        "same_director_like_ratio_max",
        "same_actor_rated_count_log",
        "same_actor_liked_count_log",
        "same_actor_disliked_count_log",
        "same_actor_like_ratio_mean",
        "same_actor_like_ratio_max",
        "same_actor_like_ratio_min",
    ]


def _update_person_stats(stats: Dict[Tuple[int, str], List[float]], uid: int, person: str, rate: float, liked: int) -> None:
    key = (uid, person)
    if key not in stats:
        stats[key] = [0.0, 0.0, 0.0, 0.0]
    stats[key][0] += 1.0
    stats[key][1] += float(liked)
    stats[key][2] += 1.0 - float(liked)
    stats[key][3] += float(rate)


def fit_content_affinity_aggregates(train_ratings: pd.DataFrame, movies: pd.DataFrame) -> Dict[str, Any]:
    train_m = train_ratings.merge(movies, on=MOVIE_ID_COL, how="left", validate="many_to_one")
    global_like_ratio = float(train_m[TARGET_COL].mean())
    global_director_fame = float(train_m[DIRECTOR_FAME_COL].mean())
    global_actor_fame = float(train_m[ACTORS_FAME_COL].mean())
    director_high_threshold = float(train_m[DIRECTOR_FAME_COL].median())
    actor_high_threshold = float(train_m[ACTORS_FAME_COL].median())

    train_m["is_high_director_fame"] = (train_m[DIRECTOR_FAME_COL] >= director_high_threshold).astype(int)
    train_m["is_high_actor_fame"] = (train_m[ACTORS_FAME_COL] >= actor_high_threshold).astype(int)

    user_all = train_m.groupby(USER_ID_COL).agg(
        user_director_fame_all_mean=(DIRECTOR_FAME_COL, "mean"),
        user_actor_fame_all_mean=(ACTORS_FAME_COL, "mean"),
        user_seen_high_director_fame_count=("is_high_director_fame", "sum"),
        user_seen_high_actor_fame_count=("is_high_actor_fame", "sum"),
    ).reset_index()

    liked_rows = train_m[train_m[TARGET_COL] == 1]
    if liked_rows.empty:
        user_liked = pd.DataFrame(columns=[USER_ID_COL, "user_director_fame_liked_mean", "user_actor_fame_liked_mean"])
    else:
        user_liked = liked_rows.groupby(USER_ID_COL).agg(
            user_director_fame_liked_mean=(DIRECTOR_FAME_COL, "mean"),
            user_actor_fame_liked_mean=(ACTORS_FAME_COL, "mean"),
        ).reset_index()

    high_director = train_m[train_m["is_high_director_fame"] == 1].groupby(USER_ID_COL).agg(
        high_director_seen=(TARGET_COL, "size"),
        high_director_liked=(TARGET_COL, "sum"),
    ).reset_index()
    high_actor = train_m[train_m["is_high_actor_fame"] == 1].groupby(USER_ID_COL).agg(
        high_actor_seen=(TARGET_COL, "size"),
        high_actor_liked=(TARGET_COL, "sum"),
    ).reset_index()

    user_pref = user_all.merge(user_liked, on=USER_ID_COL, how="left")
    user_pref = user_pref.merge(high_director, on=USER_ID_COL, how="left")
    user_pref = user_pref.merge(high_actor, on=USER_ID_COL, how="left")

    for col in ["high_director_seen", "high_director_liked", "high_actor_seen", "high_actor_liked"]:
        if col in user_pref.columns:
            user_pref[col] = user_pref[col].fillna(0.0)

    user_pref["user_director_fame_all_mean"] = user_pref["user_director_fame_all_mean"].fillna(global_director_fame)
    user_pref["user_actor_fame_all_mean"] = user_pref["user_actor_fame_all_mean"].fillna(global_actor_fame)
    user_pref["user_director_fame_liked_mean"] = user_pref["user_director_fame_liked_mean"].fillna(user_pref["user_director_fame_all_mean"])
    user_pref["user_actor_fame_liked_mean"] = user_pref["user_actor_fame_liked_mean"].fillna(user_pref["user_actor_fame_all_mean"])

    user_pref["user_high_director_fame_like_ratio"] = (
        user_pref["high_director_liked"] + FAME_PREF_SMOOTHING * global_like_ratio
    ) / (user_pref["high_director_seen"] + FAME_PREF_SMOOTHING)
    user_pref["user_high_actor_fame_like_ratio"] = (
        user_pref["high_actor_liked"] + FAME_PREF_SMOOTHING * global_like_ratio
    ) / (user_pref["high_actor_seen"] + FAME_PREF_SMOOTHING)

    keep_pref_cols = [
        USER_ID_COL,
        "user_director_fame_liked_mean",
        "user_actor_fame_liked_mean",
        "user_director_fame_all_mean",
        "user_actor_fame_all_mean",
        "user_high_director_fame_like_ratio",
        "user_high_actor_fame_like_ratio",
    ]
    user_pref = user_pref[keep_pref_cols].copy()

    director_stats_raw: Dict[Tuple[int, str], List[float]] = {}
    actor_stats_raw: Dict[Tuple[int, str], List[float]] = {}

    for row in train_m[[USER_ID_COL, RATE_COL, TARGET_COL, "directors_tokens", "actors_tokens"]].itertuples(index=False):
        uid = int(row[0])
        rate = float(row[1])
        liked = int(row[2])
        directors = row[3] if isinstance(row[3], list) else []
        actors = row[4] if isinstance(row[4], list) else []
        for director in directors:
            _update_person_stats(director_stats_raw, uid, director, rate, liked)
        for actor in actors:
            _update_person_stats(actor_stats_raw, uid, actor, rate, liked)

    def finalize(raw: Dict[Tuple[int, str], List[float]]) -> Dict[Tuple[int, str], Tuple[float, float, float, float, float]]:
        finalized: Dict[Tuple[int, str], Tuple[float, float, float, float, float]] = {}
        for key, values in raw.items():
            rated_count, liked_count, disliked_count, rate_sum = values
            like_ratio = (liked_count + PERSON_HISTORY_SMOOTHING * global_like_ratio) / (rated_count + PERSON_HISTORY_SMOOTHING)
            mean_rate = rate_sum / rated_count if rated_count > 0 else np.nan
            finalized[key] = (rated_count, liked_count, disliked_count, mean_rate, like_ratio)
        return finalized

    return {
        "global_like_ratio": global_like_ratio,
        "global_director_fame": global_director_fame,
        "global_actor_fame": global_actor_fame,
        "director_high_threshold": director_high_threshold,
        "actor_high_threshold": actor_high_threshold,
        "user_pref": user_pref,
        "director_stats": finalize(director_stats_raw),
        "actor_stats": finalize(actor_stats_raw),
    }


def apply_content_affinity_features(df: pd.DataFrame, aggregates: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out = out.merge(aggregates["user_pref"], on=USER_ID_COL, how="left")

    global_like_ratio = aggregates["global_like_ratio"]
    global_director_fame = aggregates["global_director_fame"]
    global_actor_fame = aggregates["global_actor_fame"]

    pref_fill = {
        "user_director_fame_liked_mean": global_director_fame,
        "user_actor_fame_liked_mean": global_actor_fame,
        "user_director_fame_all_mean": global_director_fame,
        "user_actor_fame_all_mean": global_actor_fame,
        "user_high_director_fame_like_ratio": global_like_ratio,
        "user_high_actor_fame_like_ratio": global_like_ratio,
    }
    for col, value in pref_fill.items():
        out[col] = out[col].fillna(value)

    out[DIRECTOR_FAME_COL] = out[DIRECTOR_FAME_COL].fillna(global_director_fame)
    out[ACTORS_FAME_COL] = out[ACTORS_FAME_COL].fillna(global_actor_fame)

    out["director_fame_distance_from_user_liked_pref"] = (out[DIRECTOR_FAME_COL] - out["user_director_fame_liked_mean"]).abs()
    out["actor_fame_distance_from_user_liked_pref"] = (out[ACTORS_FAME_COL] - out["user_actor_fame_liked_mean"]).abs()
    out["director_fame_minus_user_liked_pref"] = out[DIRECTOR_FAME_COL] - out["user_director_fame_liked_mean"]
    out["actor_fame_minus_user_liked_pref"] = out[ACTORS_FAME_COL] - out["user_actor_fame_liked_mean"]
    out["director_fame_x_user_liked_pref"] = out[DIRECTOR_FAME_COL] * out["user_director_fame_liked_mean"]
    out["actor_fame_x_user_liked_pref"] = out[ACTORS_FAME_COL] * out["user_actor_fame_liked_mean"]

    director_stats: Dict[Tuple[int, str], Tuple[float, float, float, float, float]] = aggregates["director_stats"]
    actor_stats: Dict[Tuple[int, str], Tuple[float, float, float, float, float]] = aggregates["actor_stats"]

    rows = []
    for uid, directors, actors in out[[USER_ID_COL, "directors_tokens", "actors_tokens"]].itertuples(index=False):
        uid_int = int(uid) if pd.notna(uid) else -1
        directors = directors if isinstance(directors, list) else []
        actors = actors if isinstance(actors, list) else []

        director_matches = [director_stats[(uid_int, d)] for d in directors if (uid_int, d) in director_stats]
        actor_matches = [actor_stats[(uid_int, a)] for a in actors if (uid_int, a) in actor_stats]

        if director_matches:
            d_rated = sum(x[0] for x in director_matches)
            d_liked = sum(x[1] for x in director_matches)
            d_ratios = [x[4] for x in director_matches]
            d_ratio_mean = float(np.mean(d_ratios))
            d_ratio_max = float(np.max(d_ratios))
        else:
            d_rated = 0.0
            d_liked = 0.0
            d_ratio_mean = global_like_ratio
            d_ratio_max = global_like_ratio

        if actor_matches:
            a_rated = sum(x[0] for x in actor_matches)
            a_liked = sum(x[1] for x in actor_matches)
            a_disliked = sum(x[2] for x in actor_matches)
            a_ratios = [x[4] for x in actor_matches]
            a_ratio_mean = float(np.mean(a_ratios))
            a_ratio_max = float(np.max(a_ratios))
            a_ratio_min = float(np.min(a_ratios))
        else:
            a_rated = 0.0
            a_liked = 0.0
            a_disliked = 0.0
            a_ratio_mean = global_like_ratio
            a_ratio_max = global_like_ratio
            a_ratio_min = global_like_ratio

        rows.append({
            "same_director_rated_count_log": math.log1p(d_rated),
            "same_director_liked_count_log": math.log1p(d_liked),
            "same_director_like_ratio_mean": d_ratio_mean,
            "same_director_like_ratio_max": d_ratio_max,
            "same_actor_rated_count_log": math.log1p(a_rated),
            "same_actor_liked_count_log": math.log1p(a_liked),
            "same_actor_disliked_count_log": math.log1p(a_disliked),
            "same_actor_like_ratio_mean": a_ratio_mean,
            "same_actor_like_ratio_max": a_ratio_max,
            "same_actor_like_ratio_min": a_ratio_min,
        })

    affinity = pd.DataFrame(rows, index=out.index)
    out = pd.concat([out, affinity], axis=1)
    return out


def affinity_experiments_enabled() -> bool:
    return RUN_LLM_PERSONALIZED_AFFINITY_EXPERIMENT or RUN_USER_PROFILE_PLUS_LLM_PERSONALIZED_AFFINITY_EXPERIMENT


def build_oof_train_frame(
    train_ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    with_affinity: bool,
) -> pd.DataFrame:
    """Build training features out-of-fold.

    Rows are split randomly into OOF_FOLDS folds. The features of the rows in fold k come from
    aggregates fitted on the other folds only, so the row's own label is never part of its own
    user/movie/director/actor statistics (the leakage of the earlier versions).
    """
    oof_start = time.perf_counter()
    ratings = train_ratings.reset_index(drop=True)
    fold_ids = np.random.RandomState(RANDOM_STATE).permutation(len(ratings)) % OOF_FOLDS

    parts: List[pd.DataFrame] = []
    for fold in range(OOF_FOLDS):
        in_fold = fold_ids == fold
        fit_part = ratings[~in_fold]
        apply_part = ratings[in_fold]
        if apply_part.empty:
            continue
        log(f"OOF training features: fold {fold + 1}/{OOF_FOLDS}")
        frame = apply_behavior_features(
            merge_base_features(apply_part, users, movies),
            fit_behavior_aggregates(fit_part),
        )
        if with_affinity:
            frame = apply_content_affinity_features(frame, fit_content_affinity_aggregates(fit_part, movies))
        parts.append(frame)

    seconds = time.perf_counter() - oof_start
    STAGE_TIMINGS.append({"stage": f"oof_training_features(rows={len(ratings)}, affinity={with_affinity})", "seconds": round(seconds, 2)})
    log(f"[timing] OOF training features: {seconds:.1f}s")
    return pd.concat(parts, ignore_index=True)


def build_experiment_frames(
    train_ratings: pd.DataFrame,
    eval_ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    user_feature_cols: List[str],
    behavior_aggregates: Dict[str, Any],
    content_aggregates: Dict[str, Any],
    skip_train_features: bool = False,
) -> Dict[str, Dict[str, Any]]:
    eval_base = merge_base_features(eval_ratings, users, movies)
    eval_b = apply_behavior_features(eval_base, behavior_aggregates)
    eval_a = apply_content_affinity_features(eval_b, content_aggregates)

    if skip_train_features:
        # Only the feature-column bookkeeping is needed; avoid recomputing the OOF training frame.
        train_a = eval_a.head(0).copy()
    else:
        train_a = build_oof_train_frame(train_ratings, users, movies, with_affinity=affinity_experiments_enabled())
    train_b = train_a

    b2 = behavior_v2_features()
    llm = raw_llm_features()
    personalized = personalized_llm_fame_features()
    content = content_affinity_features()

    experiments: Dict[str, Dict[str, Any]] = {}

    if RUN_BEHAVIOR_V2_ONLY_EXPERIMENT:
        experiments["behavior_v2_only"] = {
            "train": train_b.copy(),
            "eval": eval_b.copy(),
            "features": b2,
            "note": "Out-of-fold user/movie behavior features, no LLM fame columns.",
        }

    if RUN_BEHAVIOR_V2_PLUS_LLM_RAW_EXPERIMENT:
        experiments["behavior_v2_plus_llm_raw"] = {
            "train": train_b.copy(),
            "eval": eval_b.copy(),
            "features": b2 + llm,
            "note": "Behavior features plus raw LLM fame columns.",
        }

    if RUN_LLM_PERSONALIZED_AFFINITY_EXPERIMENT:
        experiments["llm_personalized_affinity"] = {
            "train": train_a.copy(),
            "eval": eval_a.copy(),
            "features": b2 + llm + personalized + content,
            "note": "Behavior + raw LLM fame + personalized fame preference + director/actor affinity.",
        }

    if RUN_USER_PROFILE_PLUS_LLM_PERSONALIZED_AFFINITY_EXPERIMENT:
        experiments["user_profile_plus_llm_personalized_affinity"] = {
            "train": train_a.copy(),
            "eval": eval_a.copy(),
            "features": user_feature_cols + b2 + llm + personalized + content,
            "note": "User profile + behavior + raw LLM fame + personalized fame preference + director/actor affinity.",
        }

    for exp_name, exp in experiments.items():
        feature_cols: List[str] = []
        for col in exp["features"]:
            if col in exp["train"].columns:
                exp["train"][col] = safe_to_numeric(exp["train"][col])
                exp["eval"][col] = safe_to_numeric(exp["eval"][col])
                if pd.api.types.is_numeric_dtype(exp["train"][col]):
                    feature_cols.append(col)
        exp["features"] = feature_cols
        if not feature_cols:
            raise RuntimeError(f"Experiment {exp_name} has no valid numeric features.")

    return experiments


# =============================================================================
# BALANCING, MODELS, METRICS
# =============================================================================


def resample_training_frame(train_df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Optional[np.ndarray], Dict[str, Any]]:
    if method not in BALANCING_METHODS:
        raise ValueError(f"Unknown balancing method: {method}")

    y = train_df[TARGET_COL].astype(int)
    n0 = int((y == 0).sum())
    n1 = int((y == 1).sum())
    info: Dict[str, Any] = {
        "balancing_method": method,
        "original_rows": int(len(train_df)),
        "original_0": n0,
        "original_1": n1,
        "original_positive_ratio": float(n1 / len(train_df)) if len(train_df) else np.nan,
        "balanced_rows": int(len(train_df)),
        "balanced_0": n0,
        "balanced_1": n1,
        "balanced_positive_ratio": float(n1 / len(train_df)) if len(train_df) else np.nan,
    }

    if method == "none" or n0 == 0 or n1 == 0:
        info["sample_weight_used"] = False
        return train_df.copy(), None, info

    weights = compute_sample_weight(class_weight="balanced", y=y.values).astype(float)
    info.update({
        "sample_weight_used": True,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
    })
    return train_df.copy(), weights, info

def optional_external_models() -> Dict[str, Optional[BaseEstimator]]:
    external: Dict[str, Optional[BaseEstimator]] = {
        "lightgbm": None,
        "lightgbm_wide": None,
        "xgboost_hist": None,
        "catboost": None,
    }

    try:
        from lightgbm import LGBMClassifier

        external["lightgbm"] = LGBMClassifier(
            n_estimators=700,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=35,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

        external["lightgbm_wide"] = LGBMClassifier(
            n_estimators=900,
            learning_rate=0.025,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=45,
            subsample=0.85,
            colsample_bytree=0.90,
            reg_alpha=0.10,
            reg_lambda=1.20,
            objective="binary",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as exc:
        log(f"LightGBM not available: {exc}")

    try:
        from xgboost import XGBClassifier

        external["xgboost_hist"] = XGBClassifier(
            n_estimators=750,
            learning_rate=0.03,
            max_depth=5,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.20,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except Exception as exc:
        log(f"XGBoost not available: {exc}")

    try:
        from catboost import CatBoostClassifier

        external["catboost"] = CatBoostClassifier(
            iterations=750,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5.0,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
    except Exception as exc:
        log(f"CatBoost not available: {exc}")

    return external

def make_pipeline(model: BaseEstimator, scale: bool = False) -> Pipeline:
    steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)

def make_models() -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        # 1
        "hist_gradient_boosting": make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=HGB_MAX_ITER,
                learning_rate=0.04,
                max_leaf_nodes=31,
                min_samples_leaf=25,
                l2_regularization=0.05,
                random_state=RANDOM_STATE,
            )
        ),

        # 2
        "hist_gradient_boosting_deeper": make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=HGB_DEEP_MAX_ITER,
                learning_rate=0.035,
                max_leaf_nodes=63,
                min_samples_leaf=30,
                l2_regularization=0.08,
                random_state=RANDOM_STATE,
            )
        ),

        # 3
        "random_forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                bootstrap=True,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),

        # 4
        "extra_trees": make_pipeline(
            ExtraTreesClassifier(
                n_estimators=600,
                max_depth=None,
                min_samples_leaf=2,
                max_features="sqrt",
                bootstrap=False,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),

        # 5
        "gradient_boosting": make_pipeline(
            GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.035,
                max_depth=3,
                min_samples_leaf=25,
                subsample=0.85,
                random_state=RANDOM_STATE,
            )
        ),

        # 6
        "adaboost": make_pipeline(
            AdaBoostClassifier(
                n_estimators=300,
                learning_rate=0.035,
                random_state=RANDOM_STATE,
            )
        ),

        # 7 - baseline خطی قوی برای مقایسه
        "logistic_regression": make_pipeline(
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            scale=True,
        ),

        # 8 - شبکه عصبی کوچک برای tabular
        "mlp_small": make_pipeline(
            MLPClassifier(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=512,
                learning_rate_init=0.001,
                max_iter=120,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=10,
                random_state=RANDOM_STATE,
            ),
            scale=True,
        ),
    }

    external = optional_external_models()

    # 9
    if external.get("lightgbm") is not None:
        models["lightgbm"] = make_pipeline(external["lightgbm"])

    # 10
    if external.get("lightgbm_wide") is not None:
        models["lightgbm_wide"] = make_pipeline(external["lightgbm_wide"])

    # 11
    if external.get("xgboost_hist") is not None:
        models["xgboost_hist"] = make_pipeline(external["xgboost_hist"])

    # 12
    if external.get("catboost") is not None:
        models["catboost"] = make_pipeline(external["catboost"])

    log(f"Active models: {', '.join(models.keys())}")
    return models

def model_accepts_sample_weight(pipeline: Pipeline) -> bool:
    estimator = pipeline.named_steps.get("model")
    if estimator is None:
        return False
    try:
        signature = inspect.signature(estimator.fit)
        return "sample_weight" in signature.parameters
    except Exception:
        return False


def fit_pipeline(model_template: Pipeline, X: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray]) -> Pipeline:
    fitted = clone(model_template)
    if sample_weight is not None and model_accepts_sample_weight(fitted):
        try:
            fitted.fit(X, y, model__sample_weight=sample_weight)
            return fitted
        except TypeError:
            pass
    fitted.fit(X, y)
    return fitted


def predict_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 1:
            return np.clip(proba.astype(float), 0.0, 1.0)
        if proba.shape[1] == 1:
            classes = getattr(model.named_steps.get("model"), "classes_", np.array([0]))
            only_class = int(classes[0]) if len(classes) else 0
            return np.ones(len(X), dtype=float) if only_class == 1 else np.zeros(len(X), dtype=float)
        return np.clip(proba[:, 1].astype(float), 0.0, 1.0)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return np.clip(1.0 / (1.0 + np.exp(-scores)), 0.0, 1.0)

    return np.clip(model.predict(X).astype(float), 0.0, 1.0)


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    result: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "rmse_binary": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "brier_score": float(brier_score_loss(y_true, np.clip(y_proba, 0.0, 1.0))),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except Exception:
        result["roc_auc"] = float("nan")
    try:
        result["average_precision"] = float(average_precision_score(y_true, y_proba))
    except Exception:
        result["average_precision"] = float("nan")
    try:
        result["log_loss"] = float(log_loss(y_true, np.clip(y_proba, 1e-6, 1.0 - 1e-6)))
    except Exception:
        result["log_loss"] = float("nan")
    return result


def threshold_scan(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    thresholds = np.linspace(MIN_THRESHOLD, MAX_THRESHOLD, THRESHOLD_GRID_SIZE)
    rows: List[Dict[str, float]] = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        rows.append({"threshold": float(threshold), **evaluate_classifier(y_true, y_pred, y_proba)})
    return pd.DataFrame(rows)


def select_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    scan = threshold_scan(y_true, y_proba)
    metric = THRESHOLD_SELECTION_METRIC
    best_row = scan.sort_values(
        by=[metric, "balanced_accuracy", "accuracy"],
        ascending=[False, False, False],
        na_position="last",
    ).iloc[0]
    best_metrics = {k: float(best_row[k]) for k in scan.columns if k != "threshold"}
    return float(best_row["threshold"]), best_metrics, scan


def select_best_row(metrics_df: pd.DataFrame) -> pd.Series:
    sort_cols = ["roc_auc", "average_precision", "f1", "balanced_accuracy", "accuracy"]
    return metrics_df.sort_values(
        by=sort_cols,
        ascending=[False, False, False, False, False],
        na_position="last",
    ).iloc[0]


# =============================================================================
# TRAIN / VALIDATE / TEST
# =============================================================================


def train_validate_choose_thresholds(
    inner_train_ratings: pd.DataFrame,
    validation_ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    user_feature_cols: List[str],
    dirs: OutputDirs,
) -> pd.DataFrame:
    log("Preparing focused inner train/validation experiments")
    behavior_aggregates = fit_behavior_aggregates(inner_train_ratings)
    content_aggregates = fit_content_affinity_aggregates(inner_train_ratings, movies)
    experiments = timed(
        "validation_build_experiment_frames",
        build_experiment_frames,
        inner_train_ratings,
        validation_ratings,
        users,
        movies,
        user_feature_cols,
        behavior_aggregates,
        content_aggregates,
    )
    models = make_models()

    rows: List[Dict[str, Any]] = []
    threshold_best_rows: List[Dict[str, Any]] = []
    balance_rows: List[Dict[str, Any]] = []

    for exp_name, exp in experiments.items():
        train_df = exp["train"]
        val_df = exp["eval"]
        feature_cols = exp["features"]
        X_val = val_df[feature_cols]
        y_val = val_df[TARGET_COL].astype(int).values

        for balancing_method in BALANCING_METHODS:
            balanced_train_df, sample_weight, balance_info = resample_training_frame(train_df, balancing_method)
            balance_info.update({"phase": "inner_validation", "experiment": exp_name})
            balance_rows.append(balance_info)

            X_train = balanced_train_df[feature_cols]
            y_train = balanced_train_df[TARGET_COL].astype(int).values

            for model_name, model_template in models.items():
                log(f"Validation fit: {exp_name} / {model_name} / {balancing_method}")
                t0 = time.perf_counter()
                fitted = fit_pipeline(model_template, X_train, y_train, sample_weight)
                t1 = time.perf_counter()
                y_proba = predict_probability(fitted, X_val)
                t2 = time.perf_counter()
                selected_threshold, threshold_best_metrics, scan = select_threshold(y_val, y_proba)
                t3 = time.perf_counter()
                y_pred = (y_proba >= selected_threshold).astype(int)
                metrics = evaluate_classifier(y_val, y_pred, y_proba)

                row = {
                    **metrics,
                    "fit_seconds": round(t1 - t0, 2),
                    "predict_seconds": round(t2 - t1, 2),
                    "threshold_scan_seconds": round(t3 - t2, 2),
                    "phase": "validation",
                    "experiment": exp_name,
                    "experiment_note": exp.get("note", ""),
                    "model": model_name,
                    "balancing_method": balancing_method,
                    "selected_threshold": selected_threshold,
                    "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
                    "inner_train_rows_effective": int(len(balanced_train_df)),
                    "validation_rows": int(len(val_df)),
                    "feature_count": int(len(feature_cols)),
                    "features": "|".join(feature_cols),
                }
                for key, value in threshold_best_metrics.items():
                    row[f"threshold_best_{key}"] = value
                rows.append(row)

                threshold_best_rows.append({
                    "phase": "validation",
                    "experiment": exp_name,
                    "model": model_name,
                    "balancing_method": balancing_method,
                    "selected_threshold": selected_threshold,
                    **threshold_best_metrics,
                })

    validation_metrics = pd.DataFrame(rows).sort_values(
        by=["roc_auc", "average_precision", "f1", "balanced_accuracy", "accuracy"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )
    write_csv(validation_metrics, dirs.metrics / "validation_model_metrics.csv")
    write_csv(pd.DataFrame(threshold_best_rows), dirs.metrics / "validation_threshold_best_by_model.csv")
    write_csv(pd.DataFrame(balance_rows), dirs.metrics / "training_balance_report_inner_validation.csv")
    return validation_metrics


def train_and_evaluate_on_test(
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    user_feature_cols: List[str],
    validation_metrics: pd.DataFrame,
    dirs: OutputDirs,
) -> Tuple[str, str, str, float, Pipeline, List[str], pd.DataFrame, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    log("Preparing full train/test experiments")
    behavior_aggregates = fit_behavior_aggregates(train_ratings)
    content_aggregates = fit_content_affinity_aggregates(train_ratings, movies)
    experiments = timed(
        "test_build_experiment_frames",
        build_experiment_frames,
        train_ratings,
        test_ratings,
        users,
        movies,
        user_feature_cols,
        behavior_aggregates,
        content_aggregates,
    )
    models = make_models()

    # Only test top validation configs. This prevents the long final-test loop that got killed before.
    validation_config = validation_metrics.head(TOP_VALIDATION_CONFIGS_FOR_FINAL_TEST)[[
        "experiment",
        "model",
        "balancing_method",
        "selected_threshold",
    ]].drop_duplicates()

    rows: List[Dict[str, Any]] = []
    prediction_frames: List[pd.DataFrame] = []
    balance_rows: List[Dict[str, Any]] = []
    fitted_models: Dict[Tuple[str, str, str], Pipeline] = {}

    for _, config in validation_config.iterrows():
        exp_name = str(config["experiment"])
        model_name = str(config["model"])
        balancing_method = str(config["balancing_method"])
        selected_threshold = float(config["selected_threshold"])

        if exp_name not in experiments or model_name not in models:
            continue

        exp = experiments[exp_name]
        train_df = exp["train"]
        test_df = exp["eval"]
        feature_cols = exp["features"]

        balanced_train_df, sample_weight, balance_info = resample_training_frame(train_df, balancing_method)
        balance_info.update({"phase": "final_test", "experiment": exp_name, "model": model_name})
        balance_rows.append(balance_info)

        X_train = balanced_train_df[feature_cols]
        y_train = balanced_train_df[TARGET_COL].astype(int).values
        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COL].astype(int).values

        log(f"Final test fit: {exp_name} / {model_name} / {balancing_method} / threshold={selected_threshold:.3f}")
        t0 = time.perf_counter()
        fitted = fit_pipeline(models[model_name], X_train, y_train, sample_weight)
        fit_seconds = time.perf_counter() - t0
        y_proba = predict_probability(fitted, X_test)
        y_pred = (y_proba >= selected_threshold).astype(int)
        metrics = evaluate_classifier(y_test, y_pred, y_proba)

        rows.append({
            **metrics,
            "fit_seconds": round(fit_seconds, 2),
            "phase": "test",
            "experiment": exp_name,
            "experiment_note": exp.get("note", ""),
            "model": model_name,
            "balancing_method": balancing_method,
            "selected_threshold": selected_threshold,
            "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
            "train_rows_effective": int(len(balanced_train_df)),
            "test_rows": int(len(test_df)),
            "feature_count": int(len(feature_cols)),
            "features": "|".join(feature_cols),
        })
        fitted_models[(exp_name, model_name, balancing_method)] = fitted

        pred_df = test_df[[USER_ID_COL, MOVIE_ID_COL, RATE_COL, TARGET_COL]].copy()
        pred_df["experiment"] = exp_name
        pred_df["model"] = model_name
        pred_df["balancing_method"] = balancing_method
        pred_df["selected_threshold"] = selected_threshold
        pred_df["predicted_like_probability"] = y_proba
        pred_df["predicted_liked"] = y_pred
        prediction_frames.append(pred_df)

    metrics_df = pd.DataFrame(rows).sort_values(
        by=["roc_auc", "average_precision", "f1", "balanced_accuracy", "accuracy"],
        ascending=[False, False, False, False, False],
        na_position="last",
    )
    write_csv(metrics_df, dirs.metrics / "model_metrics.csv")
    write_csv(pd.DataFrame(balance_rows), dirs.metrics / "training_balance_report_final_test.csv")

    # Baselines on the same test rows; every experiment frame shares the same behavior features.
    evaluate_classification_baselines(
        next(iter(experiments.values()))["eval"],
        behavior_aggregates["global"]["global_like_ratio"],
        dirs,
    )

    all_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    write_csv(all_predictions, dirs.predictions / "test_predictions_top_validation_configs.csv")

    best = select_best_row(metrics_df)
    best_experiment = str(best["experiment"])
    best_model_name = str(best["model"])
    best_balancing_method = str(best["balancing_method"])
    best_threshold = float(best["selected_threshold"])
    best_features = experiments[best_experiment]["features"]
    best_model = fitted_models[(best_experiment, best_model_name, best_balancing_method)]

    best_predictions = all_predictions[
        (all_predictions["experiment"] == best_experiment)
        & (all_predictions["model"] == best_model_name)
        & (all_predictions["balancing_method"] == best_balancing_method)
    ].copy()
    write_csv(best_predictions, dirs.predictions / "test_predictions_best_model.csv")

    summary = pd.DataFrame([{
        "best_experiment": best_experiment,
        "best_model": best_model_name,
        "best_balancing_method": best_balancing_method,
        "selected_threshold": best_threshold,
        "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
        "feature_count": len(best_features),
        "features": "|".join(best_features),
        "roc_auc": best.get("roc_auc"),
        "average_precision": best.get("average_precision"),
        "f1": best.get("f1"),
        "balanced_accuracy": best.get("balanced_accuracy"),
        "accuracy": best.get("accuracy"),
        "precision": best.get("precision"),
        "recall": best.get("recall"),
        "brier_score": best.get("brier_score"),
        "log_loss": best.get("log_loss"),
    }])
    write_csv(summary, dirs.metrics / "best_model_summary.csv")
    log(f"Best test model: {best_experiment} / {best_model_name} / {best_balancing_method} / threshold={best_threshold:.3f}")

    return (
        best_experiment,
        best_model_name,
        best_balancing_method,
        best_threshold,
        best_model,
        best_features,
        metrics_df,
        all_predictions,
        behavior_aggregates,
        content_aggregates,
    )


# =============================================================================
# RECOMMENDER AND RANKING EVALUATION
# =============================================================================


def make_candidate_frame(users_subset: pd.DataFrame, movies: pd.DataFrame, exclude_pairs: pd.DataFrame) -> pd.DataFrame:
    exclude_map = exclude_pairs.groupby(USER_ID_COL)[MOVIE_ID_COL].apply(lambda s: set(s.astype(int))).to_dict()
    movie_ids = movies[MOVIE_ID_COL].astype(int).values
    frames: List[pd.DataFrame] = []

    for uid in users_subset[USER_ID_COL].astype(int).unique():
        excluded = exclude_map.get(uid, set())
        candidates = [int(mid) for mid in movie_ids if int(mid) not in excluded]
        if candidates:
            frames.append(pd.DataFrame({USER_ID_COL: uid, MOVIE_ID_COL: candidates}))

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[USER_ID_COL, MOVIE_ID_COL])


def experiment_uses_affinity(experiment_name: str) -> bool:
    return "affinity" in experiment_name


def prepare_candidate_features(
    candidates: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    behavior_aggregates: Dict[str, Any],
    content_aggregates: Optional[Dict[str, Any]],
    use_affinity: bool,
) -> pd.DataFrame:
    df = candidates.merge(users, on=USER_ID_COL, how="left", validate="many_to_one")
    df = df.merge(movies, on=MOVIE_ID_COL, how="left", validate="many_to_one")
    df = apply_behavior_features(df, behavior_aggregates)
    if use_affinity:
        if content_aggregates is None:
            raise ValueError("Content aggregates are required for affinity experiments.")
        df = apply_content_affinity_features(df, content_aggregates)
    return df


def precision_recall_ndcg_at_k(recommended: List[int], positives: set, k: int) -> Dict[str, float]:
    top_k = recommended[:k]
    hits = [1 if mid in positives else 0 for mid in top_k]
    hit_count = sum(hits)
    precision = hit_count / k if k else 0.0
    recall = hit_count / len(positives) if positives else 0.0
    hit_rate = 1.0 if hit_count > 0 else 0.0

    dcg = 0.0
    for i, hit in enumerate(hits, start=1):
        if hit:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(positives), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {"precision_at_k": precision, "recall_at_k": recall, "hit_rate_at_k": hit_rate, "ndcg_at_k": ndcg}


RANKING_METRIC_COLS = ["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]
RANDOM_BASELINE_DRAWS = 5

# Non-personalized rankers. They need no model; the scores come from train-only movie statistics.
POPULARITY_BASELINES = {
    "popularity_like_ratio": "movie_history_like_ratio",
    "popularity_rated_count": "movie_history_count",
    "movie_mean_rate": "movie_history_mean_rate",
}


def recommendation_diversity_metrics(
    recs: pd.DataFrame,
    popularity: pd.Series,
) -> Dict[str, float]:
    """Diversity / coverage of top-K lists.

    recs: one row per recommended (user, movie). popularity: rating count per movie for the whole
    catalog (index = movie id, includes movies that were never recommended).
    """
    catalog_size = int(len(popularity))
    counts = recs[MOVIE_ID_COL].value_counts()
    n_users = int(recs[USER_ID_COL].nunique())
    n_slots = int(len(recs))
    if n_users == 0 or catalog_size == 0:
        return {}

    # Gini of how often each catalog item is recommended (0 = every item equally often, 1 = a single item).
    x = np.sort(np.concatenate([counts.values.astype(float), np.zeros(max(catalog_size - len(counts), 0))]))
    n = len(x)
    gini = float(2.0 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1.0) / n)

    # Personalization = 1 - mean pairwise Jaccard similarity between users' lists.
    user_codes = pd.factorize(recs[USER_ID_COL])[0]
    item_codes = pd.factorize(recs[MOVIE_ID_COL])[0]
    matrix = np.zeros((n_users, int(item_codes.max()) + 1), dtype=np.float32)
    matrix[user_codes, item_codes] = 1.0
    inter = matrix @ matrix.T
    sizes = matrix.sum(axis=1)
    union = sizes[:, None] + sizes[None, :] - inter
    upper = np.triu_indices(n_users, k=1)
    jaccard = float(np.mean(inter[upper] / np.maximum(union[upper], 1.0))) if n_users > 1 else float("nan")

    recommended_pop = popularity.reindex(recs[MOVIE_ID_COL].values).fillna(0.0).values
    return {
        "users": n_users,
        "unique_items_recommended": int(len(counts)),
        "item_coverage": float(len(counts) / catalog_size),
        "gini_item_exposure": gini,
        "top1_item_user_share": float(counts.iloc[0] / n_users),
        "top10_items_slot_share": float(counts.head(10).sum() / n_slots),
        "mean_popularity_vs_catalog": float(recommended_pop.mean() / max(float(popularity.mean()), 1e-12)),
        "personalization": float(1.0 - jaccard),
    }


def ranking_metrics_for_scores(
    candidates: pd.DataFrame,
    score_col: str,
    positives_by_user: Dict[int, set],
    k: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    top_parts: List[pd.DataFrame] = []
    for uid, g in candidates.groupby(USER_ID_COL):
        positives = positives_by_user.get(int(uid), set())
        if not positives:
            continue
        # Tie-break on movie id so the result is deterministic.
        ranked = g.sort_values([score_col, MOVIE_ID_COL], ascending=[False, True], kind="mergesort")
        ids = ranked[MOVIE_ID_COL].astype(int).tolist()
        rows.append(precision_recall_ndcg_at_k(ids, positives, k))
        top_parts.append(pd.DataFrame({USER_ID_COL: int(uid), MOVIE_ID_COL: ids[:k]}))
    if not rows:
        return {col: float("nan") for col in RANKING_METRIC_COLS}, pd.DataFrame(columns=[USER_ID_COL, MOVIE_ID_COL])
    return pd.DataFrame(rows)[RANKING_METRIC_COLS].mean().to_dict(), pd.concat(top_parts, ignore_index=True)


def evaluate_ranking_baselines(
    candidates: pd.DataFrame,
    positives_by_user: Dict[int, set],
    model_summary: Dict[str, float],
    model_top_k: pd.DataFrame,
    experiment_name: str,
    dirs: OutputDirs,
) -> None:
    popularity = candidates.drop_duplicates(MOVIE_ID_COL).set_index(MOVIE_ID_COL)["movie_history_count"]
    rows: List[Dict[str, Any]] = [{
        "method": f"model: {experiment_name}",
        **model_summary,
        **recommendation_diversity_metrics(model_top_k, popularity),
    }]

    draws = []
    for draw in range(RANDOM_BASELINE_DRAWS):
        scored = candidates[[USER_ID_COL, MOVIE_ID_COL]].copy()
        scored["score"] = np.random.RandomState(RANDOM_STATE + draw).rand(len(scored))
        metrics, top_k = ranking_metrics_for_scores(scored, "score", positives_by_user, RANKING_EVAL_K)
        draws.append({**metrics, **recommendation_diversity_metrics(top_k, popularity)})
    rows.append({"method": f"random (mean of {RANDOM_BASELINE_DRAWS} draws)", **pd.DataFrame(draws).mean().to_dict()})

    for name, col in POPULARITY_BASELINES.items():
        if col not in candidates.columns:
            continue
        metrics, top_k = ranking_metrics_for_scores(candidates, col, positives_by_user, RANKING_EVAL_K)
        rows.append({"method": name, **metrics, **recommendation_diversity_metrics(top_k, popularity)})

    comparison = pd.DataFrame(rows)
    random_row = comparison[comparison["method"].str.startswith("random")].iloc[0]
    for col in RANKING_METRIC_COLS:
        comparison[f"{col}_lift_vs_random"] = comparison[col] / random_row[col] if random_row[col] else np.nan
    write_csv(comparison, dirs.metrics / "ranking_baselines_comparison.csv")
    for _, r in comparison.iterrows():
        log(
            f"Ranking@{RANKING_EVAL_K} {r['method']}: ndcg={r['ndcg_at_k']:.4f} precision={r['precision_at_k']:.4f} "
            f"hit_rate={r['hit_rate_at_k']:.4f} coverage={r.get('item_coverage', float('nan')):.3f} "
            f"gini={r.get('gini_item_exposure', float('nan')):.3f} personalization={r.get('personalization', float('nan')):.3f}"
        )


def evaluate_classification_baselines(eval_frame: pd.DataFrame, global_like_ratio: float, dirs: OutputDirs) -> pd.DataFrame:
    """Simple score baselines on the test rows. Threshold-free metrics only (no threshold is tuned for them)."""
    y_true = eval_frame[TARGET_COL].astype(int).values
    scores: Dict[str, np.ndarray] = {
        "constant_global_like_ratio": np.full(len(y_true), global_like_ratio),
        "user_like_ratio": eval_frame["user_history_like_ratio"].values,
        "movie_like_ratio": eval_frame["movie_history_like_ratio"].values,
        "user_plus_movie_bias": global_like_ratio + eval_frame["user_bias_like"].values + eval_frame["movie_bias_like"].values,
    }
    rows = []
    for name, raw in scores.items():
        proba = np.clip(np.asarray(raw, dtype=float), 0.0, 1.0)
        rows.append({
            "baseline": name,
            "roc_auc": float(roc_auc_score(y_true, proba)),
            "average_precision": float(average_precision_score(y_true, proba)),
            "brier_score": float(brier_score_loss(y_true, proba)),
            "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1.0 - 1e-6))),
            "test_rows": int(len(y_true)),
        })
    out = pd.DataFrame(rows)
    write_csv(out, dirs.metrics / "baseline_classification_metrics.csv")
    for _, r in out.iterrows():
        log(f"Baseline {r['baseline']}: roc_auc={r['roc_auc']:.4f} average_precision={r['average_precision']:.4f}")
    return out


def evaluate_ranking(
    model: Pipeline,
    experiment_name: str,
    train_ratings: pd.DataFrame,
    test_ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    train_behavior_aggregates: Dict[str, Any],
    train_content_aggregates: Dict[str, Any],
    feature_cols: List[str],
    dirs: OutputDirs,
) -> None:
    log("Evaluating top-K ranking for best model")
    positive_test = test_ratings[test_ratings[TARGET_COL] == 1].copy()
    eval_users = sorted(positive_test[USER_ID_COL].unique().astype(int).tolist())
    users_eval = users[users[USER_ID_COL].isin(eval_users)].copy()

    candidates = make_candidate_frame(users_eval, movies, train_ratings[[USER_ID_COL, MOVIE_ID_COL]])
    if candidates.empty:
        write_csv(pd.DataFrame(), dirs.metrics / "ranking_metrics_by_user.csv")
        write_csv(pd.DataFrame(columns=["metric", "mean"]), dirs.metrics / "ranking_metrics_summary.csv")
        return

    use_affinity = experiment_uses_affinity(experiment_name)
    cand_features = prepare_candidate_features(candidates, users, movies, train_behavior_aggregates, train_content_aggregates, use_affinity)
    cand_features["predicted_like_probability"] = predict_probability(model, cand_features[feature_cols])

    positives_by_user = positive_test.groupby(USER_ID_COL)[MOVIE_ID_COL].apply(lambda s: set(s.astype(int))).to_dict()
    movie_title_map = movies.set_index(MOVIE_ID_COL)["main_title"].to_dict() if "main_title" in movies.columns else {}

    metric_rows: List[Dict[str, Any]] = []
    rec_rows: List[Dict[str, Any]] = []

    for uid, g in cand_features.groupby(USER_ID_COL):
        positives = positives_by_user.get(int(uid), set())
        if not positives:
            continue
        ranked = g.sort_values("predicted_like_probability", ascending=False)
        recommended_ids = ranked[MOVIE_ID_COL].astype(int).tolist()
        metrics = precision_recall_ndcg_at_k(recommended_ids, positives, RANKING_EVAL_K)
        metrics.update({
            USER_ID_COL: int(uid),
            "positive_test_items": int(len(positives)),
            "candidate_items": int(len(ranked)),
            "k": int(RANKING_EVAL_K),
        })
        metric_rows.append(metrics)

        for rank, (_, r) in enumerate(ranked.head(RANKING_EVAL_K).iterrows(), start=1):
            mid = int(r[MOVIE_ID_COL])
            rec_rows.append({
                USER_ID_COL: int(uid),
                "rank": rank,
                MOVIE_ID_COL: mid,
                "main_title": movie_title_map.get(mid, ""),
                "predicted_like_probability": float(r["predicted_like_probability"]),
                "is_heldout_positive": int(mid in positives),
            })

    by_user = pd.DataFrame(metric_rows)
    write_csv(by_user, dirs.metrics / "ranking_metrics_by_user.csv")
    write_csv(pd.DataFrame(rec_rows), dirs.predictions / "ranking_eval_top_k_recommendations.csv")

    if not by_user.empty:
        summary = by_user[["precision_at_k", "recall_at_k", "hit_rate_at_k", "ndcg_at_k"]].mean().to_frame("mean").reset_index()
        summary = summary.rename(columns={"index": "metric"})
    else:
        summary = pd.DataFrame(columns=["metric", "mean"])
    write_csv(summary, dirs.metrics / "ranking_metrics_summary.csv")

    if not by_user.empty:
        evaluate_ranking_baselines(
            cand_features,
            positives_by_user,
            by_user[RANKING_METRIC_COLS].mean().to_dict(),
            pd.DataFrame(rec_rows)[[USER_ID_COL, MOVIE_ID_COL]],
            experiment_name,
            dirs,
        )


def train_final_best_model_on_all_data(
    best_model_name: str,
    best_balancing_method: str,
    best_experiment: str,
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    user_feature_cols: List[str],
    best_features: List[str],
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    full_behavior_aggregates = fit_behavior_aggregates(ratings)
    full_content_aggregates = fit_content_affinity_aggregates(ratings, movies)
    # Training rows get out-of-fold features; the full aggregates above are only used for the candidates.
    full_df = build_oof_train_frame(ratings, users, movies, with_affinity=experiment_uses_affinity(best_experiment))

    balanced_df, sample_weight, _ = resample_training_frame(full_df, best_balancing_method)
    models = make_models()
    if best_model_name not in models:
        raise RuntimeError(f"Best model '{best_model_name}' is not available.")

    final_model = fit_pipeline(models[best_model_name], balanced_df[best_features], balanced_df[TARGET_COL].astype(int).values, sample_weight)
    return final_model, full_behavior_aggregates, full_content_aggregates


def build_final_recommendations(
    model: Pipeline,
    experiment_name: str,
    all_ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    behavior_aggregates: Dict[str, Any],
    content_aggregates: Dict[str, Any],
    feature_cols: List[str],
    dirs: OutputDirs,
) -> pd.DataFrame:
    log("Building final top-N recommendations")
    candidates = make_candidate_frame(users, movies, all_ratings[[USER_ID_COL, MOVIE_ID_COL]])
    if candidates.empty:
        recs = pd.DataFrame()
        write_csv(recs, dirs.recommendations / "top_n_recommendations.csv")
        return recs

    use_affinity = experiment_uses_affinity(experiment_name)
    cand_features = prepare_candidate_features(candidates, users, movies, behavior_aggregates, content_aggregates, use_affinity)
    cand_features["predicted_like_probability"] = predict_probability(model, cand_features[feature_cols])

    keep_cols = [
        USER_ID_COL,
        MOVIE_ID_COL,
        "main_title",
        "original_title",
        DIRECTOR_FAME_COL,
        ACTORS_FAME_COL,
        "predicted_like_probability",
    ]
    for col in ["user_director_fame_liked_mean", "user_actor_fame_liked_mean", "same_director_like_ratio_mean", "same_actor_like_ratio_mean"]:
        if col in cand_features.columns:
            keep_cols.append(col)
    keep_cols = [c for c in keep_cols if c in cand_features.columns]

    parts: List[pd.DataFrame] = []
    for uid, g in cand_features.groupby(USER_ID_COL):
        top = g.sort_values("predicted_like_probability", ascending=False).head(TOP_N_RECOMMENDATIONS).copy()
        top["rank"] = range(1, len(top) + 1)
        parts.append(top[[USER_ID_COL, "rank"] + [c for c in keep_cols if c != USER_ID_COL]])

    recs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    write_csv(recs, dirs.recommendations / "top_n_recommendations.csv")

    if not recs.empty:
        popularity = all_ratings[MOVIE_ID_COL].value_counts().reindex(movies[MOVIE_ID_COL].astype(int).values).fillna(0.0)
        diversity = recommendation_diversity_metrics(recs, popularity)
        write_csv(pd.DataFrame([diversity]), dirs.metrics / "final_recommendations_diversity.csv")
        log(
            f"Final top-{TOP_N_RECOMMENDATIONS} diversity: unique_items={diversity['unique_items_recommended']} "
            f"coverage={diversity['item_coverage']:.3f} gini={diversity['gini_item_exposure']:.3f} "
            f"top1_item_user_share={diversity['top1_item_user_share']:.3f} personalization={diversity['personalization']:.3f}"
        )
    return recs


MODEL_BUNDLE_NAME = "final_model_bundle.joblib"


def save_model_bundle(
    model: Pipeline,
    experiment_name: str,
    feature_cols: List[str],
    threshold: float,
    behavior_aggregates: Dict[str, Any],
    content_aggregates: Dict[str, Any],
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    dirs: OutputDirs,
) -> Path:
    """Everything a UI needs to score movies for a user without retraining.

    Load it with recommender_service.load_bundle(). The file is a pickle: only load bundles you created yourself.
    """
    bundle = {
        "bundle_version": 1,
        "created_at": datetime.now().isoformat(),
        "experiment": experiment_name,
        "feature_cols": list(feature_cols),
        "decision_threshold": float(threshold),
        "top_n": TOP_N_RECOMMENDATIONS,
        "model": model,
        "behavior_aggregates": behavior_aggregates,
        "content_aggregates": content_aggregates,
        "users": users,
        "movies": movies,
        "seen_by_user": ratings.groupby(USER_ID_COL)[MOVIE_ID_COL].apply(lambda s: set(s.astype(int))).to_dict(),
        "environment": collect_environment_info(),
    }
    path = dirs.model / MODEL_BUNDLE_NAME
    joblib.dump(bundle, path, compress=3)
    log(f"Saved final model bundle: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


# =============================================================================
# REPORTS AND FIGURES
# =============================================================================


def save_rate_distribution(ratings: pd.DataFrame, dirs: OutputDirs) -> None:
    counts = ratings[RATE_COL].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title("Rating distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(dirs.figures / "rating_distribution.png", dpi=160)
    plt.close()

    class_counts = ratings[TARGET_COL].value_counts().sort_index()
    plt.figure(figsize=(6, 5))
    plt.bar(["not_liked", "liked"], [class_counts.get(0, 0), class_counts.get(1, 0)])
    plt.title(f"Binary target distribution: rate > {LIKE_THRESHOLD}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(dirs.figures / "target_class_distribution.png", dpi=160)
    plt.close()


def save_llm_feature_report(ratings: pd.DataFrame, movies: pd.DataFrame, dirs: OutputDirs) -> None:
    df = ratings.merge(movies[[MOVIE_ID_COL, DIRECTOR_FAME_COL, ACTORS_FAME_COL]], on=MOVIE_ID_COL, how="left")
    rows = []
    for col in [DIRECTOR_FAME_COL, ACTORS_FAME_COL]:
        rows.append({
            "feature": col,
            "mean_all": float(df[col].mean()),
            "mean_liked_1": float(df.loc[df[TARGET_COL] == 1, col].mean()),
            "mean_not_liked_0": float(df.loc[df[TARGET_COL] == 0, col].mean()),
            "corr_with_binary_liked": float(df[[col, TARGET_COL]].corr().iloc[0, 1]),
            "corr_with_rate": float(df[[col, RATE_COL]].corr().iloc[0, 1]),
            "missing_count": int(df[col].isna().sum()),
        })
    write_csv(pd.DataFrame(rows), dirs.metrics / "llm_fame_feature_report.csv")


def save_model_metric_figures(metrics_df: pd.DataFrame, dirs: OutputDirs) -> None:
    if metrics_df.empty:
        return
    for metric in ["accuracy", "f1", "roc_auc", "average_precision", "balanced_accuracy", "recall", "precision"]:
        if metric not in metrics_df.columns:
            continue
        plot_df = metrics_df.sort_values(metric, ascending=False).copy()
        labels = plot_df["experiment"] + "\n" + plot_df["model"] + "\n" + plot_df["balancing_method"]
        plt.figure(figsize=(12, 6))
        plt.bar(labels, plot_df[metric])
        plt.xticks(rotation=65, ha="right", fontsize=8)
        plt.ylim(0, 1)
        plt.title(f"Model comparison - {metric}")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(dirs.figures / f"model_comparison_{metric}.png", dpi=160)
        plt.close()


def save_confusion_matrix_figure(y_true: np.ndarray, y_pred: np.ndarray, dirs: OutputDirs, name: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    cm_df = pd.DataFrame(cm, index=["actual_liked", "actual_not_liked"], columns=["pred_liked", "pred_not_liked"])
    write_csv(cm_df.reset_index(names="label"), dirs.metrics / f"confusion_matrix_{name}.csv")

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion matrix - {name}")
    plt.colorbar()
    plt.xticks(np.arange(2), ["pred liked", "pred not liked"], rotation=30, ha="right")
    plt.yticks(np.arange(2), ["actual liked", "actual not liked"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center")
    plt.tight_layout()
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(dirs.figures / f"confusion_matrix_{name}.png", dpi=160)
    plt.close()


def save_roc_pr_figures(all_predictions: pd.DataFrame, dirs: OutputDirs) -> None:
    if all_predictions.empty:
        return

    plt.figure(figsize=(8, 7))
    for (exp_name, model_name, balancing_method), g in all_predictions.groupby(["experiment", "model", "balancing_method"]):
        y_true = g[TARGET_COL].astype(int).values
        y_proba = g["predicted_like_probability"].astype(float).values
        if len(set(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{exp_name}/{model_name}/{balancing_method} ({auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC curves - tested final configs")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(dirs.figures / "roc_curves_tested_final_configs.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 7))
    for (exp_name, model_name, balancing_method), g in all_predictions.groupby(["experiment", "model", "balancing_method"]):
        y_true = g[TARGET_COL].astype(int).values
        y_proba = g["predicted_like_probability"].astype(float).values
        if len(set(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        plt.plot(recall, precision, label=f"{exp_name}/{model_name}/{balancing_method} ({ap:.3f})")
    plt.title("Precision-Recall curves - tested final configs")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(dirs.figures / "precision_recall_curves_tested_final_configs.png", dpi=160)
    plt.close()


def save_feature_importance(model: Pipeline, feature_cols: List[str], dirs: OutputDirs, name: str) -> None:
    estimator = model.named_steps.get("model")
    if not hasattr(estimator, "feature_importances_"):
        return
    values = estimator.feature_importances_
    importance = pd.DataFrame({"feature": feature_cols, "importance": values}).sort_values("importance", ascending=False)
    write_csv(importance, dirs.metrics / f"feature_importance_{name}.csv")

    top = importance.head(25)
    plt.figure(figsize=(10, 8))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.title(f"Top feature importance - {name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(dirs.figures / f"feature_importance_{name}.png", dpi=160)
    plt.close()


def save_feature_columns(experiments: Dict[str, Dict[str, Any]], dirs: OutputDirs) -> None:
    rows = []
    for exp_name, exp in experiments.items():
        for feature in exp["features"]:
            rows.append({"experiment": exp_name, "feature": feature, "experiment_note": exp.get("note", "")})
    write_csv(pd.DataFrame(rows), dirs.metrics / "feature_columns_by_experiment.csv")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    dirs = make_output_dirs()
    report: Dict[str, Any] = {
        "run_started_at": datetime.now().isoformat(),
        "base_dir": str(BASE_DIR),
        "movies_csv": str(MOVIES_CSV),
        "ratings_csv": str(RATINGS_CSV),
        "user_profiles_csv": str(USER_PROFILES_CSV),
        "like_threshold": LIKE_THRESHOLD,
        "test_size": TEST_SIZE,
        "validation_size_inside_train": VALIDATION_SIZE,
        "balancing_methods": BALANCING_METHODS,
        "top_validation_configs_for_final_test": TOP_VALIDATION_CONFIGS_FOR_FINAL_TEST,
        "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
        "llm_fame_columns": [DIRECTOR_FAME_COL, ACTORS_FAME_COL],
        "method_note": (
            "V3 fast affinity pipeline: reduced model grid, no oversampling/undersampling, "
            "raw LLM fame ablation, personalized LLM fame preference, and train-only director/actor affinity features."
        ),
    }

    report["environment"] = collect_environment_info()
    report["leakage_fix"] = {
        "method": "out_of_fold_features_for_training_rows",
        "oof_folds": OOF_FOLDS,
        "note": "Training-row features exclude the row's own label; eval rows use all training data.",
    }

    run_start = time.perf_counter()

    log("Reading input files")
    with stage("read_inputs"):
        raw_movies = read_csv(MOVIES_CSV)
        raw_ratings = read_csv(RATINGS_CSV)
        raw_users = read_csv(USER_PROFILES_CSV)

    log("Cleaning movies, users, and ratings")
    with stage("clean_data_and_basic_figures"):
        movies = clean_movies(raw_movies, report)
        users, user_feature_cols = clean_users(raw_users, report)
        ratings = clean_ratings(raw_ratings, movies, users, report)
        save_rate_distribution(ratings, dirs)
        save_llm_feature_report(ratings, movies, dirs)

    log("Creating final per-user temporal train/test split")
    with stage("temporal_splits"):
        train_ratings, test_ratings = per_user_temporal_split(ratings, TEST_SIZE)
        if train_ratings.empty or test_ratings.empty:
            raise RuntimeError("Train or test set is empty. Lower MIN_USER_INTERACTIONS_FOR_TEST or check your data.")

        log("Creating inner train/validation split for model and threshold selection")
        inner_train_ratings, validation_ratings = per_user_temporal_split(train_ratings, VALIDATION_SIZE)
        if inner_train_ratings.empty or validation_ratings.empty:
            raise RuntimeError("Inner train or validation set is empty. Lower MIN_USER_INTERACTIONS_FOR_TEST or check your data.")

    split_dist = pd.DataFrame([
        class_distribution(ratings, "all_clean"),
        class_distribution(train_ratings, "outer_train"),
        class_distribution(test_ratings, "outer_test"),
        class_distribution(inner_train_ratings, "inner_train"),
        class_distribution(validation_ratings, "validation"),
    ])
    write_csv(split_dist, dirs.metrics / "class_distribution_by_split.csv")

    report.update({
        "train_rows": int(len(train_ratings)),
        "test_rows": int(len(test_ratings)),
        "validation_rows": int(len(validation_ratings)),
        "inner_train_rows": int(len(inner_train_ratings)),
        "train_users": int(train_ratings[USER_ID_COL].nunique()),
        "test_users": int(test_ratings[USER_ID_COL].nunique()),
        "train_movies": int(train_ratings[MOVIE_ID_COL].nunique()),
        "test_movies": int(test_ratings[MOVIE_ID_COL].nunique()),
    })

    with stage("validation_search_total"):
        validation_metrics = train_validate_choose_thresholds(inner_train_ratings, validation_ratings, users, movies, user_feature_cols, dirs)

    with stage("final_test_evaluation_total"):
        (
            best_experiment,
            best_model_name,
            best_balancing_method,
            best_threshold,
            best_model,
            best_features,
            metrics_df,
            all_predictions,
            train_behavior_aggregates,
            train_content_aggregates,
        ) = train_and_evaluate_on_test(train_ratings, test_ratings, users, movies, user_feature_cols, validation_metrics, dirs)

    report.update({
        "best_experiment": best_experiment,
        "best_model": best_model_name,
        "best_balancing_method": best_balancing_method,
        "best_selected_threshold": best_threshold,
        "best_feature_count": int(len(best_features)),
        "best_features": best_features,
    })

    # Save final feature map for the selected focused experiments.
    final_experiments = build_experiment_frames(
        train_ratings,
        test_ratings,
        users,
        movies,
        user_feature_cols,
        train_behavior_aggregates,
        train_content_aggregates,
        skip_train_features=True,
    )
    save_feature_columns(final_experiments, dirs)

    save_model_metric_figures(metrics_df, dirs)
    save_roc_pr_figures(all_predictions, dirs)

    best_pred = all_predictions[
        (all_predictions["experiment"] == best_experiment)
        & (all_predictions["model"] == best_model_name)
        & (all_predictions["balancing_method"] == best_balancing_method)
    ]
    safe_name = f"{best_experiment}_{best_model_name}_{best_balancing_method}".replace("/", "_")
    save_confusion_matrix_figure(
        best_pred[TARGET_COL].astype(int).values,
        best_pred["predicted_liked"].astype(int).values,
        dirs,
        safe_name,
    )
    save_feature_importance(best_model, best_features, dirs, safe_name)

    if RUN_RANKING_EVAL:
        with stage("ranking_evaluation_with_baselines"):
            evaluate_ranking(
                best_model,
                best_experiment,
                train_ratings,
                test_ratings,
                users,
                movies,
                train_behavior_aggregates,
                train_content_aggregates,
                best_features,
                dirs,
            )

    if RUN_FINAL_RECOMMENDATIONS:
        log("Training final best model on all clean interactions")
        final_start = time.perf_counter()
        final_model, full_behavior_aggregates, full_content_aggregates = train_final_best_model_on_all_data(
            best_model_name,
            best_balancing_method,
            best_experiment,
            ratings,
            users,
            movies,
            user_feature_cols,
            best_features,
        )
        final_recs = build_final_recommendations(
            final_model,
            best_experiment,
            ratings,
            users,
            movies,
            full_behavior_aggregates,
            full_content_aggregates,
            best_features,
            dirs,
        )
        report["final_recommendation_rows"] = int(len(final_recs))

        bundle_path = save_model_bundle(
            final_model,
            best_experiment,
            best_features,
            best_threshold,
            full_behavior_aggregates,
            full_content_aggregates,
            ratings,
            users,
            movies,
            dirs,
        )
        report["model_bundle"] = str(bundle_path)
        STAGE_TIMINGS.append({"stage": "final_model_recommendations_and_bundle", "seconds": round(time.perf_counter() - final_start, 2)})

    STAGE_TIMINGS.append({"stage": "TOTAL_RUN", "seconds": round(time.perf_counter() - run_start, 2)})
    write_csv(pd.DataFrame(STAGE_TIMINGS), dirs.metrics / "stage_timings.csv")
    report["stage_timings_seconds"] = {r["stage"]: r["seconds"] for r in STAGE_TIMINGS}

    report["run_finished_at"] = datetime.now().isoformat()
    save_json(report, dirs.metrics / "pipeline_report.json")

    log(f"DONE. Outputs saved in: {dirs.root}")
    log("Main files: 03_metrics/best_model_summary.csv, model_metrics.csv, llm_fame_feature_report.csv")


if __name__ == "__main__":
    main()
