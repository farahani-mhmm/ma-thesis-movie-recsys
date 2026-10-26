"""Step 2: decide, once and for all, which users are test users in which fold, and in which order each
user's ratings are revealed. No model is trained here; only pandas and numpy are needed.

Design
  * 5 folds x 3 repeats. In every repeat each user belongs to exactly ONE fold (user-grouped split):
    when a fold is the test fold, its users and all their ratings are absent from training.
  * Folds are balanced by activity: users are cut into 4 groups by number of ratings and each group is
    spread evenly over the 5 folds.
  * For every (repeat, user) the user's ratings get a random order, position 0, 1, 2, ...
        revealed to the model at level N : positions 0 .. N-1   (nested: N=5 contains N=3 contains N=1)
        evaluation pool (same for all N) : positions >= EVAL_POOL_START (= 50)
  * Only users with at least MIN_RATINGS_FOR_EVAL (= 60) ratings are eligible as test users.
    The other users stay in the training pool.

Inputs : outputs/01_data/ratings_clean.csv
Outputs (outputs/02_splits/):
    users_folds.csv         user_id, n_ratings, stratum, eligible, fold_r0, fold_r1, fold_r2
    reveal_positions.csv    user_id, movie_id, pos_r0, pos_r1, pos_r2
    split_report.txt        checks and fold sizes
    split_config.json       the settings used
"""
import json
import sys

import numpy as np
import pandas as pd

from config import (CSV_ENCODING, EVAL_POOL_START, MIN_POOL_SIZE, MIN_RATINGS_FOR_EVAL, N_FOLDS, N_GRID,
                    N_REPEATS, OUT_DIR, RANDOM_STATE, ensure_dir)

N_STRATA = 4


def main():
    src = OUT_DIR / "01_data" / "ratings_clean.csv"
    if not src.exists():
        sys.exit(f"Run step 1 first. Missing: {src}")

    ratings = pd.read_csv(src, encoding=CSV_ENCODING, usecols=["user_id", "movie_id"])
    ratings = ratings.sort_values(["user_id", "movie_id"], kind="mergesort").reset_index(drop=True)

    users = ratings.groupby("user_id").size().rename("n_ratings").reset_index()
    users["eligible"] = users["n_ratings"] >= MIN_RATINGS_FOR_EVAL
    # activity strata (ties broken by order, so every stratum has the same number of users +-1)
    users["stratum"] = pd.qcut(users["n_ratings"].rank(method="first"), N_STRATA, labels=False).astype(int)

    # ---------------- folds ----------------
    for r in range(N_REPEATS):
        rng = np.random.default_rng([RANDOM_STATE, r])
        fold = np.empty(len(users), dtype=int)
        for s in range(N_STRATA):
            idx = rng.permutation(np.flatnonzero(users["stratum"].to_numpy() == s))
            offset = int(rng.integers(0, N_FOLDS))
            fold[idx] = (np.arange(len(idx)) + offset) % N_FOLDS
        users[f"fold_r{r}"] = fold

    # ---------------- reveal order ----------------
    uid_arr = ratings["user_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(uid_arr[1:] != uid_arr[:-1]) + 1]
    ends = np.r_[starts[1:], len(ratings)]
    for r in range(N_REPEATS):
        pos = np.empty(len(ratings), dtype=np.int32)
        for s, e in zip(starts, ends):
            rng = np.random.default_rng([RANDOM_STATE, r, int(uid_arr[s])])
            pos[s:e] = rng.permutation(e - s)
        ratings[f"pos_r{r}"] = pos

    # ---------------- checks (stop if anything is wrong) ----------------
    problems = []
    for r in range(N_REPEATS):
        pos = ratings[f"pos_r{r}"].to_numpy()
        for s, e in zip(starts, ends):
            if not np.array_equal(np.sort(pos[s:e]), np.arange(e - s)):
                problems.append(f"repeat {r}: positions of user {uid_arr[s]} are not a permutation")
                break
        if users[f"fold_r{r}"].isna().any() or not users[f"fold_r{r}"].between(0, N_FOLDS - 1).all():
            problems.append(f"repeat {r}: some user has no valid fold")
    if users["user_id"].duplicated().any():
        problems.append("duplicate user ids in users table")
    if problems:
        sys.exit("CHECKS FAILED:\n" + "\n".join(problems))

    # ---------------- report ----------------
    lines = ["STEP 2 - SPLIT REPORT", ""]
    lines.append(f"users: {len(users)} | eligible as test users (>= {MIN_RATINGS_FOR_EVAL} ratings): "
                 f"{int(users['eligible'].sum())} | never tested: {int((~users['eligible']).sum())}")
    lines.append(f"N levels revealed: {N_GRID} | evaluation pool = positions >= {EVAL_POOL_START} "
                 f"(at least {MIN_POOL_SIZE} ratings)")
    pool = (users["n_ratings"] - EVAL_POOL_START)[users["eligible"]]
    lines.append(f"evaluation pool size per eligible user: min {int(pool.min())}, median {int(pool.median())}, "
                 f"max {int(pool.max())}")
    lines.append(f"strata (by number of ratings): sizes {users['stratum'].value_counts().sort_index().tolist()}")
    lines.append("")
    for r in range(N_REPEATS):
        lines.append(f"repeat {r}: users / eligible users / mean ratings per user in each test fold")
        for f in range(N_FOLDS):
            part = users[users[f"fold_r{r}"] == f]
            lines.append(f"   fold {f}: {len(part):4d} users | {int(part['eligible'].sum()):4d} eligible | "
                         f"mean ratings {part['n_ratings'].mean():7.1f}")
    lines.append("")
    lines.append("checks passed: every user has one fold per repeat; every user's positions are a permutation "
                 "0..n-1 in every repeat.")
    text = "\n".join(lines)

    out = ensure_dir(OUT_DIR / "02_splits")
    users.to_csv(out / "users_folds.csv", index=False, encoding=CSV_ENCODING)
    ratings.to_csv(out / "reveal_positions.csv", index=False, encoding=CSV_ENCODING)
    (out / "split_report.txt").write_text(text, encoding="utf-8")
    (out / "split_config.json").write_text(json.dumps({
        "random_state": RANDOM_STATE, "n_folds": N_FOLDS, "n_repeats": N_REPEATS, "n_grid": N_GRID,
        "eval_pool_start": EVAL_POOL_START, "min_pool_size": MIN_POOL_SIZE,
        "min_ratings_for_eval": MIN_RATINGS_FOR_EVAL, "n_strata": N_STRATA}, indent=2), encoding="utf-8")
    print(text)
    print(f"\nfiles written to: {out}")


if __name__ == "__main__":
    main()
