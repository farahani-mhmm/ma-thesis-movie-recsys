"""Automated checks for train_llm_recommender_pipeline_v4.py (leakage, splits, features, metrics).

Run from the pipeline_run folder, in the environment that has scikit-learn and lightgbm installed:

    python -m unittest discover -s tests -v          # or:  python -m pytest tests -v

The tests use small synthetic data in which the label is pure noise (no user, movie or person has a real
effect on it). On such data an honest feature cannot predict the label, so any feature that does is leaking.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_llm_recommender_pipeline_v4 as p  # noqa: E402


def auc(y: np.ndarray, score: np.ndarray) -> float:
    ranks = pd.Series(score).rank().values
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def make_data(seed: int = 0, n_users: int = 60, n_movies: int = 80, per_user: int = 50):
    """Synthetic data. Many distinct directors/actors, so each (user, person) pair is rare, as in the real data."""
    rng = np.random.RandomState(seed)
    movies = pd.DataFrame({p.MOVIE_ID_COL: np.arange(1, n_movies + 1)})
    movies["main_title"] = [f"Movie {i}" for i in movies[p.MOVIE_ID_COL]]
    movies["original_title"] = movies["main_title"]
    movies[p.DIRECTORS_COL] = ["|".join(f"Dir{d}" for d in rng.choice(40, size=rng.randint(1, 3), replace=False)) for _ in range(n_movies)]
    movies[p.ACTORS_COL] = ["|".join(f"Act{a}" for a in rng.choice(150, size=rng.randint(3, 6), replace=False)) for _ in range(n_movies)]
    movies["directors_tokens"] = movies[p.DIRECTORS_COL].apply(p.split_people)
    movies["actors_tokens"] = movies[p.ACTORS_COL].apply(p.split_people)
    movies[p.DIRECTOR_FAME_COL] = rng.uniform(1, 10, n_movies)
    movies[p.ACTORS_FAME_COL] = rng.uniform(1, 10, n_movies)

    users = pd.DataFrame({p.USER_ID_COL: np.arange(1, n_users + 1), "followers_count": rng.randint(0, 5000, n_users)})

    rows = []
    for uid in users[p.USER_ID_COL]:
        for mid in rng.choice(movies[p.MOVIE_ID_COL], size=per_user, replace=False):
            liked = int(rng.rand() < 0.4)                      # pure noise label
            rate = int(rng.randint(8, 11) if liked else rng.randint(1, 8))
            rows.append((uid, mid, rate, pd.Timestamp("2015-01-01") + pd.Timedelta(days=int(rng.randint(0, 2000))), liked))
    ratings = pd.DataFrame(rows, columns=[p.USER_ID_COL, p.MOVIE_ID_COL, p.RATE_COL, "date", p.TARGET_COL])
    return movies, users, ratings


class SplitTests(unittest.TestCase):
    def test_temporal_split_is_disjoint_and_ordered(self):
        _, _, ratings = make_data()
        train, test = p.per_user_temporal_split(ratings, 0.2)
        pairs_train = set(zip(train[p.USER_ID_COL], train[p.MOVIE_ID_COL]))
        pairs_test = set(zip(test[p.USER_ID_COL], test[p.MOVIE_ID_COL]))
        self.assertFalse(pairs_train & pairs_test, "a (user, movie) pair is in both train and test")
        self.assertEqual(len(train) + len(test), len(ratings))
        last_train = train.groupby(p.USER_ID_COL)["date"].max()
        first_test = test.groupby(p.USER_ID_COL)["date"].min()
        joined = pd.concat([last_train, first_test], axis=1, keys=["last_train", "first_test"]).dropna()
        self.assertTrue((joined["last_train"] <= joined["first_test"]).all(), "test ratings are not later than train ratings")
        self.assertAlmostEqual(len(test) / len(ratings), 0.2, delta=0.02)


class LeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.movies, cls.users, ratings = make_data()
        cls.train, cls.test = p.per_user_temporal_split(ratings, 0.2)
        cls.cols = ["followers_count"]
        cls.beh = p.fit_behavior_aggregates(cls.train)
        cls.con = p.fit_content_affinity_aggregates(cls.train, cls.movies)

    def _naive_train(self):
        base = p.merge_base_features(self.train, self.users, self.movies)
        return p.apply_content_affinity_features(p.apply_behavior_features(base, self.beh), self.con)

    def _eval(self):
        base = p.merge_base_features(self.test, self.users, self.movies)
        return p.apply_content_affinity_features(p.apply_behavior_features(base, self.beh), self.con)

    def test_oof_frame_keeps_every_training_row_once(self):
        oof = p.build_oof_train_frame(self.train, self.users, self.movies, with_affinity=True)
        key = lambda d: sorted(zip(d[p.USER_ID_COL], d[p.MOVIE_ID_COL]))
        self.assertEqual(key(oof), key(self.train))
        self.assertEqual(int(oof[p.TARGET_COL].sum()), int(self.train[p.TARGET_COL].sum()))

    def test_naive_training_features_leak_the_label(self):
        """Documents the old bug: features computed on the rows they were fitted on carry the label."""
        naive = self._naive_train()
        y = naive[p.TARGET_COL].values
        for col in ("same_actor_like_ratio_mean", "same_director_like_ratio_mean"):
            self.assertGreater(auc(y, naive[col].values), 0.65, f"{col} should leak on naive training features")

    def test_oof_training_features_do_not_leak(self):
        oof = p.build_oof_train_frame(self.train, self.users, self.movies, with_affinity=True)
        ev = self._eval()
        for col in ("same_actor_like_ratio_mean", "same_director_like_ratio_mean", "user_history_like_ratio", "movie_history_like_ratio"):
            train_auc = auc(oof[p.TARGET_COL].values, oof[col].values)
            eval_auc = auc(ev[p.TARGET_COL].values, ev[col].values)
            self.assertAlmostEqual(train_auc, 0.5, delta=0.08, msg=f"{col}: OOF train AUC {train_auc:.3f} is far from chance on noise labels")
            self.assertAlmostEqual(eval_auc, 0.5, delta=0.10, msg=f"{col}: eval AUC {eval_auc:.3f} is far from chance on noise labels")

    def test_oof_matches_eval_distribution_of_unseen_pairs(self):
        """Eval rows often have a never-seen (user, director) pair; OOF training rows must too (naive: never)."""
        naive = self._naive_train()
        oof = p.build_oof_train_frame(self.train, self.users, self.movies, with_affinity=True)
        ev = self._eval()
        zero = lambda d: float((d["same_director_rated_count_log"] == 0).mean())
        self.assertEqual(zero(naive), 0.0, "naive training rows always contain themselves in the pair count")
        self.assertGreater(zero(oof), 0.05)
        self.assertGreater(zero(ev), 0.05)
        self.assertLess(abs(zero(oof) - zero(ev)), 0.10, "OOF training rows should look like eval rows")

    def test_experiment_frames_expose_all_declared_features(self):
        exps = p.build_experiment_frames(self.train, self.test, self.users, self.movies, self.cols, self.beh, self.con)
        self.assertTrue(exps)
        for name, exp in exps.items():
            self.assertTrue(exp["features"], f"{name} has no features")
            for col in exp["features"]:
                self.assertIn(col, exp["train"].columns)
                self.assertIn(col, exp["eval"].columns)
            self.assertEqual(len(exp["train"]), len(self.train))
            self.assertEqual(len(exp["eval"]), len(self.test))


class MetricTests(unittest.TestCase):
    def test_precision_recall_ndcg_perfect_and_worst(self):
        perfect = p.precision_recall_ndcg_at_k([1, 2, 3, 4], {1, 2}, 2)
        self.assertEqual((perfect["precision_at_k"], perfect["hit_rate_at_k"], round(perfect["ndcg_at_k"], 6)), (1.0, 1.0, 1.0))
        worst = p.precision_recall_ndcg_at_k([3, 4, 1, 2], {1, 2}, 2)
        self.assertEqual((worst["precision_at_k"], worst["hit_rate_at_k"], worst["ndcg_at_k"]), (0.0, 0.0, 0.0))

    def test_diversity_extremes(self):
        popularity = pd.Series(np.ones(100), index=np.arange(1, 101))
        same = pd.DataFrame({p.USER_ID_COL: np.repeat(np.arange(20), 5), p.MOVIE_ID_COL: np.tile(np.arange(1, 6), 20)})
        d = p.recommendation_diversity_metrics(same, popularity)
        self.assertAlmostEqual(d["personalization"], 0.0)
        self.assertAlmostEqual(d["top1_item_user_share"], 1.0)
        self.assertAlmostEqual(d["item_coverage"], 0.05)
        distinct = pd.DataFrame({p.USER_ID_COL: np.repeat(np.arange(20), 5), p.MOVIE_ID_COL: np.arange(1, 101)})
        d = p.recommendation_diversity_metrics(distinct, popularity)
        self.assertAlmostEqual(d["personalization"], 1.0)
        self.assertAlmostEqual(d["item_coverage"], 1.0)
        self.assertLess(d["gini_item_exposure"], 0.05)

    def test_ranking_scores_find_signal(self):
        rows = [(u, m, float(m)) for u in (1, 2) for m in range(1, 21)]          # score = movie id, larger is better
        cand = pd.DataFrame(rows, columns=[p.USER_ID_COL, p.MOVIE_ID_COL, "score"])
        positives = {1: {19, 20}, 2: {18, 20}}
        metrics, top = p.ranking_metrics_for_scores(cand, "score", positives, 5)
        self.assertEqual(metrics["hit_rate_at_k"], 1.0)
        self.assertEqual(len(top), 10)


if __name__ == "__main__":
    unittest.main()
