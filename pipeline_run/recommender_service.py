"""Scoring service for the UI: load the saved model bundle and recommend movies for a user.

Usage:
    from recommender_service import load_bundle, recommend_for_user

    bundle = load_bundle("recommender_training_outputs/run_.../08_model/final_model_bundle.joblib")
    top = recommend_for_user(bundle, user_id=103007, top_n=10)

The bundle is a pickle written by train_llm_recommender_pipeline_v4.py. Only load bundles you created yourself.
A user id that was not in the training data still works: the user/affinity features fall back to global
averages, so the result is then close to a popularity ranking.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd

from train_llm_recommender_pipeline_v4 import (
    MOVIE_ID_COL,
    USER_ID_COL,
    experiment_uses_affinity,
    predict_probability,
    prepare_candidate_features,
)


def load_bundle(path: Union[str, Path]) -> Dict[str, Any]:
    return joblib.load(path)


def recommend_for_user(
    bundle: Dict[str, Any],
    user_id: int,
    top_n: Optional[int] = None,
    exclude_movie_ids: Optional[set] = None,
) -> pd.DataFrame:
    """Top-N unseen movies for a user, ranked by predicted probability of liking (rate > 7).

    exclude_movie_ids: extra movies to hide, e.g. ones the user just rated in the UI.
    """
    movies = bundle["movies"]
    seen = set(bundle["seen_by_user"].get(int(user_id), set())) | set(exclude_movie_ids or ())

    candidates = movies.loc[~movies[MOVIE_ID_COL].isin(seen), [MOVIE_ID_COL]].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["rank", MOVIE_ID_COL, "main_title", "original_title", "predicted_like_probability"])
    candidates[USER_ID_COL] = int(user_id)

    features = prepare_candidate_features(
        candidates,
        bundle["users"],
        movies,
        bundle["behavior_aggregates"],
        bundle["content_aggregates"],
        experiment_uses_affinity(bundle["experiment"]),
    )
    features["predicted_like_probability"] = predict_probability(bundle["model"], features[bundle["feature_cols"]])

    n = int(top_n or bundle.get("top_n", 10))
    top = features.sort_values(["predicted_like_probability", MOVIE_ID_COL], ascending=[False, True]).head(n).copy()
    top["rank"] = range(1, len(top) + 1)
    columns = ["rank", MOVIE_ID_COL, "main_title", "original_title", "predicted_like_probability"]
    return top[[c for c in columns if c in top.columns]].reset_index(drop=True)
