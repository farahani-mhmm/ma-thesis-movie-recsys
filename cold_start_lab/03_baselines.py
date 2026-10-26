"""Step 3: the baselines that every later model must beat. Only pandas and numpy are needed.

Methods (none of them trains a model, and none uses the N revealed ratings, so they do not depend on N):
    B0_majority   : everybody gets the same score, so every ranking is a tie (per-user AUC is exactly 0.5).
                    We also report the ACCURACY of "always predict the majority class", because the old paper
                    reported accuracy and this shows what accuracy a model must beat.
    B1_pop_like   : score of a film = its like ratio among TRAINING users of the fold
    B1_pop_mean   : score of a film = its mean rating among TRAINING users
    B1_pop_count  : score of a film = how many TRAINING users rated it

Evaluation (the contract made in step 2):
    for every repeat r and fold f: the test users are the ELIGIBLE users of fold f; all other users are the
    training users. A test user is judged on their evaluation pool (positions >= EVAL_POOL_START in the user's
    random order), the same pool for every N. Film statistics come from training users only.
    Main metric: AUC computed separately for every test user, then averaged over users.
    A user's value = mean of the user's AUCs over the repeats where it is defined.
    Confidence interval: 95% percentile bootstrap over USERS (2000 resamples).

Both liked rules (gt7 and ge7) are always reported.

Inputs : outputs/01_data/ratings_clean.csv, outputs/02_splits/users_folds.csv, reveal_positions.csv
Outputs (outputs/03_baselines/):
    per_user_auc.csv         one row per (user, repeat, rule, method)
    baseline_summary.csv     mean AUC, 95% interval, per-repeat means, number of users
    paired_differences.csv   differences between methods with their 95% interval
    majority_accuracy.csv    accuracy of the majority guess per repeat and fold
    baseline_report.txt      the same, for people
"""
import sys

import numpy as np
import pandas as pd

from config import (CSV_ENCODING, EVAL_POOL_START, LIKE_RULES, N_FOLDS, N_REPEATS, OUT_DIR, RANDOM_STATE,
                    ensure_dir)

METHODS = ["B0_majority", "B1_pop_like", "B1_pop_mean", "B1_pop_count"]
PAIRS = [("B1_pop_like", "B0_majority"), ("B1_pop_mean", "B0_majority"), ("B1_pop_count", "B0_majority"),
         ("B1_pop_like", "B1_pop_mean"), ("B1_pop_like", "B1_pop_count")]
N_BOOT = 2000


def auc_binary(scores, y):
    """AUC with ties counted as half. NaN when only one class is present."""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(np.asarray(scores, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    d1, d2 = OUT_DIR / "01_data", OUT_DIR / "02_splits"
    for p in (d1 / "ratings_clean.csv", d2 / "users_folds.csv", d2 / "reveal_positions.csv"):
        if not p.exists():
            sys.exit(f"Run steps 1 and 2 first. Missing: {p}")

    rules = list(LIKE_RULES)
    ratings = pd.read_csv(d1 / "ratings_clean.csv", encoding=CSV_ENCODING,
                          usecols=["user_id", "movie_id", "rate"] + [f"liked_{k}" for k in rules])
    users = pd.read_csv(d2 / "users_folds.csv", encoding=CSV_ENCODING)
    pos = pd.read_csv(d2 / "reveal_positions.csv", encoding=CSV_ENCODING)
    data = ratings.merge(pos, on=["user_id", "movie_id"], validate="one_to_one")
    if len(data) != len(ratings):
        sys.exit("Ratings and reveal positions do not match")
    eligible = users[users["eligible"]]

    rows, acc_rows = [], []
    for r in range(N_REPEATS):
        data["fold"] = data["user_id"].map(users.set_index("user_id")[f"fold_r{r}"])
        pos_col = f"pos_r{r}"
        for f in range(N_FOLDS):
            train = data[data["fold"] != f]
            test_users = set(eligible.loc[eligible[f"fold_r{r}"] == f, "user_id"])
            if test_users & set(train["user_id"]):
                sys.exit(f"LEAK: a test user is also a training user (repeat {r}, fold {f})")
            pool = data[(data["fold"] == f) & data["user_id"].isin(test_users) & (data[pos_col] >= EVAL_POOL_START)]

            film = train.groupby("movie_id").agg(n=("rate", "size"), mean_rate=("rate", "mean"))
            glob_mean = float(train["rate"].mean())
            pool = pool.merge(film, left_on="movie_id", right_index=True, how="left")
            pool["n"] = pool["n"].fillna(0)
            pool["mean_rate"] = pool["mean_rate"].fillna(glob_mean)

            for rule in rules:
                lab = f"liked_{rule}"
                glob_like = float(train[lab].mean())
                like_by_film = train.groupby("movie_id")[lab].mean()
                pool["score_like"] = pool["movie_id"].map(like_by_film).fillna(glob_like)
                majority = int(glob_like >= 0.5)
                acc_rows.append({"repeat": r, "fold": f, "rule": rule, "majority_prediction": majority,
                                 "train_like_rate": glob_like,
                                 "accuracy": float((pool[lab] == majority).mean()),
                                 "pool_ratings": int(len(pool))})
                for uid, g in pool.groupby("user_id"):
                    y = g[lab].to_numpy()
                    base = {"user_id": int(uid), "repeat": r, "fold": f, "rule": rule,
                            "n_pool": int(len(g)), "pool_like_rate": float(y.mean())}
                    scores = {"B0_majority": np.zeros(len(g)), "B1_pop_like": g["score_like"].to_numpy(),
                              "B1_pop_mean": g["mean_rate"].to_numpy(), "B1_pop_count": g["n"].to_numpy()}
                    for m in METHODS:
                        rows.append({**base, "method": m, "auc": auc_binary(scores[m], y)})
        print(f"repeat {r} done", flush=True)

    df = pd.DataFrame(rows)
    acc = pd.DataFrame(acc_rows)

    # every eligible user must be tested exactly once per repeat
    for r in range(N_REPEATS):
        seen = df[(df["repeat"] == r) & (df["rule"] == rules[0]) & (df["method"] == METHODS[0])]["user_id"]
        if seen.duplicated().any() or len(seen) != len(eligible):
            sys.exit(f"CHECK FAILED: repeat {r} tested {len(seen)} users, expected {len(eligible)}")

    # ---------------- summaries ----------------
    rng = np.random.default_rng(RANDOM_STATE)
    summary, diffs, undefined = [], [], []
    for rule in rules:
        sub = df[df["rule"] == rule]
        for r in range(N_REPEATS):
            und = sub[(sub["repeat"] == r) & (sub["method"] == METHODS[0])]["auc"].isna().sum()
            undefined.append({"rule": rule, "repeat": r, "users_with_undefined_auc": int(und)})
        wide = sub.groupby(["user_id", "method"])["auc"].mean().unstack("method").dropna()
        n_users = len(wide)
        idx = rng.integers(0, n_users, size=(N_BOOT, n_users))
        per_rep = sub.groupby(["method", "repeat"])["auc"].mean().unstack("repeat")
        for m in METHODS:
            v = wide[m].to_numpy()
            lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
            row = {"rule": rule, "method": m, "mean_auc": float(v.mean()), "ci_low": float(lo), "ci_high": float(hi),
                   "users": int(n_users)}
            for r in range(N_REPEATS):
                row[f"repeat{r}_mean"] = float(per_rep.loc[m, r])
            summary.append(row)
        for a, b in PAIRS:
            d = (wide[a] - wide[b]).to_numpy()
            lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
            diffs.append({"rule": rule, "A": a, "B": b, "mean_difference_A_minus_B": float(d.mean()),
                          "ci_low": float(lo), "ci_high": float(hi), "users": int(n_users)})

    summary = pd.DataFrame(summary)
    diffs = pd.DataFrame(diffs)
    undefined = pd.DataFrame(undefined)
    acc_sum = acc.groupby("rule").agg(majority_prediction=("majority_prediction", "mean"),
                                      train_like_rate=("train_like_rate", "mean"),
                                      accuracy=("accuracy", "mean")).reset_index()

    # ---------------- save ----------------
    out = ensure_dir(OUT_DIR / "03_baselines")
    df.to_csv(out / "per_user_auc.csv", index=False, encoding=CSV_ENCODING)
    summary.to_csv(out / "baseline_summary.csv", index=False, encoding=CSV_ENCODING)
    diffs.to_csv(out / "paired_differences.csv", index=False, encoding=CSV_ENCODING)
    acc.to_csv(out / "majority_accuracy.csv", index=False, encoding=CSV_ENCODING)

    lines = ["STEP 3 - BASELINES", f"test users per repeat: {len(eligible)} | bootstrap resamples: {N_BOOT}", ""]
    for rule in rules:
        lines.append(f"=== label rule: {rule} ===")
        for _, s in summary[summary["rule"] == rule].iterrows():
            lines.append(f"  {s['method']:<14} mean per-user AUC {s['mean_auc']:.4f}  95% CI [{s['ci_low']:.4f}, {s['ci_high']:.4f}]"
                         f"  per repeat: {s['repeat0_mean']:.4f} {s['repeat1_mean']:.4f} {s['repeat2_mean']:.4f}  (users {s['users']})")
        lines.append("  paired differences (A - B):")
        for _, d in diffs[diffs["rule"] == rule].iterrows():
            lines.append(f"     {d['A']} - {d['B']}: {d['mean_difference_A_minus_B']:+.4f}  95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")
        a = acc_sum[acc_sum["rule"] == rule].iloc[0]
        lines.append(f"  majority guess: predicts {'LIKED' if a['majority_prediction'] >= 0.5 else 'NOT LIKED'} for everyone "
                     f"(training like rate {a['train_like_rate']:.3f}); its accuracy on the test pool = {a['accuracy']:.4f}")
        und = undefined[undefined["rule"] == rule]["users_with_undefined_auc"].tolist()
        lines.append(f"  test users with undefined AUC (all liked or none liked in their pool), per repeat: {und}")
        lines.append("")
    lines.append(f"files written to: {out}")
    text = "\n".join(lines)
    (out / "baseline_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
