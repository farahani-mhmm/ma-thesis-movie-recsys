"""Helpers shared by steps 4, 5 and 6 (no model is trained in this file except inside fit_predict)."""
import argparse
import sys
import warnings

import numpy as np
import pandas as pd

from config import CSV_ENCODING, LIKE_RULES, N_FOLDS, N_GRID, N_REPEATS, OUT_DIR, RANDOM_STATE

K_INNER = 5          # inner user folds used to build training features without target leakage
PRIMARY_LEARNER = "hgb"


# ----------------------------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------------------------
def parse_args(description="", final=False):
    """final=True gives the defaults of the final protocol (step 14): rule gt7, N = 0/5/20/50, M = 0/5,
    learners logreg/hgb_c/rf, Sonnet fame columns, 200 training rows per user."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--scenario", default="all", help="step 14: A (new user), B (new film), C (both new) or all")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 1 repeat, 1 fold, rule gt7, N in 0/5/50, small samples (a few minutes)")
    ap.add_argument("--repeats", default=None, help="comma list, e.g. 0,1,2")
    ap.add_argument("--folds", default=None, help="comma list, e.g. 0,1,2,3,4")
    ap.add_argument("--rules", default=None, help="comma list, e.g. gt7,ge7")
    ap.add_argument("--n-list", default=None, help="comma list of N levels, e.g. 0,5,50")
    ap.add_argument("--with-rf", action="store_true", help="also run random forests (slow)")
    ap.add_argument("--force", action="store_true", help="recompute even if outputs exist (used by run_all.py)")
    ap.add_argument("--train-per-user", type=int, default=None,
                    help="training rows sampled per user in steps 5 and 7 (default 40 in step 5, 80 in step 7)")
    ap.add_argument("--out-tag", default="", help="suffix for the output folders of steps 5, 6 and 7, e.g. _tpu200")
    ap.add_argument("--m-list", default=None, help="step 7: comma list of film-history sizes M, e.g. 0,5,20,50")
    ap.add_argument("--with-uu", action="store_true",
                    help="step 5: add the user-user collaborative-filtering feature and the experiments E1U, E1PU, E4U")
    ap.add_argument("--learners", default=None,
                    help="step 8: comma list from logreg,logreg_c,hgb,hgb_b,hgb_c,rf (default logreg,hgb)")
    ap.add_argument("--interact", action="store_true",
                    help="step 8: add explicit user x film product columns and the experiments D5n, D5, D6n, D6")
    ap.add_argument("--fuzzy", action="store_true",
                    help="step 8: add the fuzzified profile (R2, C1, F1, F2, F3 from step 9) and the experiments Z*")
    ap.add_argument("--llm", default=None, choices=["gpt4o", "sonnet", "sonnet_max", "sonnet_mean"],
                    help="which LLM fame columns the film features use: gpt4o (original), sonnet = director mean + "
                         "mean of the first 3 billed actors (declared primary), sonnet_max = max over the cast, "
                         "sonnet_mean = mean over the whole cast (like for like with the GPT-4o column)")
    ap.add_argument("--llm-csv", default=None, help="movies_with_sonnet_fame.csv (default: pipeline_run/llm_claude_outputs/full_sonnet55)")
    a = ap.parse_args()

    def ints(s, default):
        return [int(x) for x in s.split(",")] if s else default

    q = a.quick
    tag_quick = "_quick" if q else ""
    if final:
        rules_default = ["gt7"]
        n_default = [0, 5] if q else [0, 5, 20, 50]
        m_default = [0, 5]
        learners_default = ["logreg", "hgb_c", "rf"]
        tpu_default = 15 if q else 200
        llm_default = "sonnet"
    else:
        rules_default = ["gt7"] if q else list(LIKE_RULES)
        n_default = [0, 5, 50] if q else list(N_GRID)
        m_default = [0, 20] if q else [0, 5, 20, 50]
        learners_default = None
        tpu_default = None
        llm_default = "gpt4o"
    return {
        "quick": q,
        "tag_quick": tag_quick,
        "tag": tag_quick + a.out_tag,
        "scenario": a.scenario.upper(),
        "repeats": ints(a.repeats, [0] if q else list(range(N_REPEATS))),
        "folds": ints(a.folds, [0] if q else list(range(N_FOLDS))),
        "rules": a.rules.split(",") if a.rules else rules_default,
        "n_list": ints(a.n_list, n_default),
        "m_list": ints(a.m_list, m_default),
        "with_rf": a.with_rf,
        "with_uu": a.with_uu,
        "learners": a.learners.split(",") if a.learners else learners_default,
        "interact": a.interact,
        "fuzzy": a.fuzzy,
        "llm": a.llm or llm_default,
        "llm_csv": a.llm_csv,
        "hgb_iter": 30 if q else 100,
        "train_per_user": a.train_per_user if a.train_per_user else (tpu_default or (15 if q else 40)),
        "train_per_user_b": a.train_per_user if a.train_per_user else (tpu_default or (15 if q else 80)),
        "perfilm_cap": 120 if q else None,
        "force": a.force,
    }


def progress(done, total, t_start, label="folds"):
    """Print a progress bar with elapsed time and a rough estimate of the time left."""
    import time
    elapsed = time.time() - t_start
    eta = elapsed / done * (total - done) if done else 0.0
    width = 24
    filled = int(width * done / max(total, 1))
    print(f"[{'#' * filled}{'-' * (width - filled)}] {done}/{total} {label} | elapsed {elapsed / 60:.1f} min | "
          f"about {eta / 60:.1f} min left (rough)", flush=True)


def require_sklearn():
    try:
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit("scikit-learn is not installed in this Python.\n"
                 "Install it first, for example:\n"
                 f'  "{sys.executable}" -m pip install scikit-learn matplotlib')


def quiet():
    warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def apply_llm_source(movies, cfg):
    """Replace the two GPT-4o fame columns by Sonnet columns when --llm sonnet* is given (same shape, nothing else changes)."""
    if not cfg or cfg.get("llm", "gpt4o") == "gpt4o":
        return movies
    from config import PROJECT_DIR
    from pathlib import Path
    path = Path(cfg["llm_csv"]) if cfg.get("llm_csv") else (
        PROJECT_DIR / "pipeline_run" / "llm_claude_outputs" / "full_sonnet55" / "movies_with_sonnet_fame.csv")
    if not path.exists():
        sys.exit(f"Sonnet film file not found: {path}")
    son = pd.read_csv(path, encoding=CSV_ENCODING, low_memory=False)
    actor_col = {"sonnet": "actors_fame_top3_mean_sonnet", "sonnet_max": "actors_fame_max_sonnet",
                 "sonnet_mean": "actors_fame_score_sonnet"}[cfg["llm"]]
    m = movies.drop(columns=["director_fame_score_llm", "actors_fame_score_llm"]).merge(
        son[["id", "director_fame_score_sonnet", actor_col]].rename(
            columns={"id": "movie_id", "director_fame_score_sonnet": "director_fame_score_llm",
                     actor_col: "actors_fame_score_llm"}), on="movie_id", how="left", validate="one_to_one")
    print(f"LLM columns: {cfg['llm']} (director mean + {actor_col}); missing: "
          f"{int(m['director_fame_score_llm'].isna().sum())} director, {int(m['actors_fame_score_llm'].isna().sum())} actor films",
          flush=True)
    return m


def load_base(cfg=None):
    d1, d2 = OUT_DIR / "01_data", OUT_DIR / "02_splits"
    for p in (d1 / "ratings_clean.csv", d1 / "users_clean.csv", d1 / "movies_clean.csv",
              d2 / "users_folds.csv", d2 / "reveal_positions.csv"):
        if not p.exists():
            sys.exit(f"Run steps 1 and 2 first. Missing: {p}")
    ratings = pd.read_csv(d1 / "ratings_clean.csv", encoding=CSV_ENCODING,
                          usecols=["user_id", "movie_id", "rate"] + [f"liked_{k}" for k in LIKE_RULES])
    pos = pd.read_csv(d2 / "reveal_positions.csv", encoding=CSV_ENCODING)
    users_f = pd.read_csv(d2 / "users_folds.csv", encoding=CSV_ENCODING)
    prof = pd.read_csv(d1 / "users_clean.csv", encoding=CSV_ENCODING)
    movies = pd.read_csv(d1 / "movies_clean.csv", encoding=CSV_ENCODING)
    data = ratings.merge(pos, on=["user_id", "movie_id"], validate="one_to_one")
    if len(data) != len(ratings):
        sys.exit("Ratings and reveal positions do not match")
    return data, users_f, prof, apply_llm_source(movies.drop_duplicates("movie_id"), cfg)


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def auc_binary(scores, y):
    """AUC with ties counted as half. NaN when only one class is present."""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(np.asarray(scores, dtype=float)).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def add_auc_rows(rows, user_ids, y, score_dict, base):
    """One AUC per (user, method) from flat arrays; appended to `rows`."""
    user_ids = np.asarray(user_ids)
    y = np.asarray(y)
    order = np.argsort(user_ids, kind="mergesort")
    u = user_ids[order]
    bounds = np.flatnonzero(np.r_[True, u[1:] != u[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        yy = y[idx]
        for name, sc in score_dict.items():
            rows.append({**base, "user_id": int(u[a]), "method": name, "auc": auc_binary(np.asarray(sc)[idx], yy)})


def summarize(df, keys, pairs, n_boot=2000, seed=RANDOM_STATE):
    """Mean per-user AUC with a 95% bootstrap interval over users, and paired differences.

    df columns: user_id, method, auc and every column in `keys` (e.g. ["rule", "N"]).
    A user's value is the mean of its AUCs over repeats (undefined AUCs are skipped).
    """
    rng = np.random.default_rng(seed)
    summary, diffs = [], []
    for key_vals, sub in df.groupby(keys):
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        key = dict(zip(keys, key_vals))
        wide = sub.groupby(["user_id", "method"])["auc"].mean().unstack("method")
        for m in wide.columns:
            v = wide[m].dropna().to_numpy()
            if len(v) < 5:
                continue
            idx = rng.integers(0, len(v), size=(n_boot, len(v)))
            lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
            summary.append({**key, "method": m, "mean_auc": float(v.mean()), "ci_low": float(lo),
                            "ci_high": float(hi), "users": int(len(v))})
        for a, b in pairs:
            if a in wide.columns and b in wide.columns:
                d = (wide[a] - wide[b]).dropna().to_numpy()
                if len(d) < 5:
                    continue
                idx = rng.integers(0, len(d), size=(n_boot, len(d)))
                lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
                diffs.append({**key, "A": a, "B": b, "mean_difference_A_minus_B": float(d.mean()),
                              "ci_low": float(lo), "ci_high": float(hi), "users": int(len(d))})
    return pd.DataFrame(summary), pd.DataFrame(diffs)


# ----------------------------------------------------------------------------------------------
# film statistics without target leakage
# ----------------------------------------------------------------------------------------------
def film_stats(df, lab):
    g = df.groupby("movie_id")
    return pd.DataFrame({"f_like": g[lab].mean(), "f_mean": g["rate"].mean(), "f_logc": np.log1p(g.size())})


def map_film_features(df, stats, glob):
    mid = df["movie_id"]
    return np.column_stack([
        mid.map(stats["f_like"]).fillna(glob[0]).to_numpy(),
        mid.map(stats["f_mean"]).fillna(glob[1]).to_numpy(),
        mid.map(stats["f_logc"]).fillna(0.0).to_numpy(),
    ]).astype(np.float32)


def oof_film_features(train, lab, seed_parts):
    """Film like ratio / mean rating / log count for TRAINING rows, built from the other inner folds of users,
    so a row's own rating never enters its own feature. Returns (features, inner_fold_of_each_row)."""
    users = np.sort(train["user_id"].unique())
    rng = np.random.default_rng([RANDOM_STATE, *seed_parts, 11])
    inner = dict(zip(users.tolist(), (rng.permutation(len(users)) % K_INNER).tolist()))
    ki = train["user_id"].map(inner).to_numpy()
    glob = (float(train[lab].mean()), float(train["rate"].mean()))
    out = np.zeros((len(train), 3), dtype=np.float32)
    for k in range(K_INNER):
        m = ki == k
        stats = film_stats(train[~m], lab)
        out[m] = map_film_features(train[m], stats, glob)
    return out, ki


# ----------------------------------------------------------------------------------------------
# learners
# ----------------------------------------------------------------------------------------------
def fit_predict(kind, Xtr, ytr, Xpo, cfg, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(ytr)) < 2:
        return np.full(len(Xpo), float(np.mean(ytr)))
    if kind == "logreg":
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=1.0, max_iter=300)
        m.fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xpo))[:, 1]
    if kind == "logreg_c":      # same model, 10 times stronger L2 regularisation (useful with many columns)
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=0.1, max_iter=300)
        m.fit(sc.transform(Xtr), ytr)
        return m.predict_proba(sc.transform(Xpo))[:, 1]
    if kind == "hgb":
        m = HistGradientBoostingClassifier(max_iter=cfg["hgb_iter"], learning_rate=0.1, max_leaf_nodes=15,
                                           min_samples_leaf=40, early_stopping=False, random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xpo)[:, 1]
    if kind == "hgb_b":         # more regularised gradient boosting
        m = HistGradientBoostingClassifier(max_iter=2 * cfg["hgb_iter"], learning_rate=0.05, max_leaf_nodes=8,
                                           min_samples_leaf=200, l2_regularization=10.0, early_stopping=False,
                                           random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xpo)[:, 1]
    if kind == "hgb_c":         # very regularised gradient boosting (small trees, slow learning)
        m = HistGradientBoostingClassifier(max_iter=3 * cfg["hgb_iter"], learning_rate=0.03, max_leaf_nodes=4,
                                           min_samples_leaf=400, l2_regularization=10.0, early_stopping=False,
                                           random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xpo)[:, 1]
    if kind == "rf":            # random forest (the method of the CSP 2023 paper); each tree sees 25% of the rows for speed
        m = RandomForestClassifier(n_estimators=60, min_samples_leaf=30, max_samples=0.25, n_jobs=-1, random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xpo)[:, 1]
    raise ValueError(kind)


# ----------------------------------------------------------------------------------------------
# yes/no decisions: threshold learned on training users only, then per-user confusion counts
# ----------------------------------------------------------------------------------------------
def best_f1_threshold(y, p):
    """Score value above which a row is called 'liked' so that F1 of the liked class is largest on (y, p)."""
    from sklearn.metrics import precision_recall_curve
    y = np.asarray(y)
    if y.sum() == 0 or y.sum() == len(y) or np.ptp(p) == 0:
        return float(np.min(p))
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def fit_predict_thr(kind, Xtr, ytr, utr, Xpo, cfg, seed):
    """Scores of the test rows from a model fitted on ALL training users, and the decision threshold.

    The threshold is learned without touching test users: a second model is fitted on 80% of the training users
    and the threshold that maximises F1 of the liked class on the other 20% is used."""
    p = fit_predict(kind, Xtr, ytr, Xpo, cfg, seed)
    users = np.unique(utr)
    rng = np.random.default_rng([RANDOM_STATE, seed, 99])
    hold_users = users[rng.random(len(users)) < 0.2]
    hold = np.isin(utr, hold_users)
    if hold.sum() < 100 or (~hold).sum() < 100 or len(np.unique(ytr[~hold])) < 2 or len(np.unique(ytr[hold])) < 2:
        return p, float(np.quantile(p, 1.0 - float(np.mean(ytr))))
    ph = fit_predict(kind, Xtr[~hold], ytr[~hold], Xtr[hold], cfg, seed + 1)
    return p, best_f1_threshold(ytr[hold], ph)


def add_confusion_rows(rows, user_ids, y, score_dict, thr_dict, base):
    """Per (user, method) counts tp, fp, fn, tn at the method's threshold; appended to `rows`."""
    user_ids = np.asarray(user_ids)
    y = np.asarray(y).astype(bool)
    uniq, inv = np.unique(user_ids, return_inverse=True)
    for name, thr in thr_dict.items():
        pred = np.asarray(score_dict[name]) >= thr
        tp = np.bincount(inv, weights=(pred & y), minlength=len(uniq))
        fp = np.bincount(inv, weights=(pred & ~y), minlength=len(uniq))
        fn = np.bincount(inv, weights=(~pred & y), minlength=len(uniq))
        tn = np.bincount(inv, weights=(~pred & ~y), minlength=len(uniq))
        for i, u in enumerate(uniq):
            rows.append({**base, "user_id": int(u), "method": name, "tp": int(tp[i]), "fp": int(fp[i]),
                         "fn": int(fn[i]), "tn": int(tn[i])})
