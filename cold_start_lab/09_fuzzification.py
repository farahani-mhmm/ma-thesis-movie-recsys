"""Step 9 (stage 2): FUZZIFICATION of the user's social-network profile, for a NEW USER (new user, known films).

Question: does turning the 12 profile columns into fuzzy membership degrees help to rank films for a new user,
compared with (a) no profile, (b) the raw profile, (c) a log/circular transform of the profile without fuzziness and
(d) a CRISP version of the very same sets (each variable put in its single best set)?  Only (d) isolates "fuzziness".

Profile representations (every one is fitted on the TRAINING users of the fold only):
    R1  raw profile, standardised                                         (control: what we had before)
    R2  log of the 4 counts, sin/cos of hour and weekday, standardised     (control: transform only, nothing fuzzy)
    C1  crisp: argmax of the F1 sets, one-hot                              (control: same sets as F1 but hard)
    F1  linguistic fuzzy sets: Low/Medium/High (shoulder + triangle at the training 10/50/90 percent quantiles) for
        year and the 4 log counts; 4 circular sets for the hour (night/morning/afternoon/evening); 7 circular sets
        for the weekday; the 5 yes/no flags unchanged
    F2  fuzzy c-means (6 clusters, fuzzifier m=1.5) on R2: the user's degree of belonging to each cluster
    F3  expert fuzzy rules on the F1 sets (AND = min, OR = max): Influencer, Broadcaster, Newcomer, Veteran,
        NightOwl, Social, Lurker (7 degrees in [0,1])

Two ways to use a representation (a "global" model and a "similar users" model):
    global model   X = P + H + representation     the learner finds interactions with the film columns itself
    kNN model      X = P + H + KP_rep             KP = taste score taken from the 50 training users whose representation is
                                                  most similar (cosine): sum_k sim_k * (liked_k - film popularity) /
                                                  (sum_k sim_k * rated_k + 1), over the neighbours who rated the film
Experiments (global): B (no profile), R1, R2, C1, F1, F2, F3.   (kNN): KR1, KR2, KC1, KF1, KF2, KF3.
P = film popularity from training users, H = history of the N revealed ratings (as in step 5).

Primary learner decided before seeing results: hgb_b (regularised gradient boosting); checks: logreg, hgb_c.
No leakage: as in step 5 (inner user folds for training rows, test users never in training, fuzzy sets/clusters fitted
on training users only).  No selection among the three fuzzifiers: all are reported, the intervals are not adjusted.

Run:  python 09_fuzzification.py --quick      then      python 09_fuzzification.py --train-per-user 200
Output (outputs/09_fuzzification[_tag]/): per_user_auc.csv, final_summary.csv, final_pairs.csv, final_report.txt
"""
import importlib
import json
import sys
import time

import numpy as np
import pandas as pd

from config import EVAL_POOL_START, LIKE_RULES, OUT_DIR, PROFILE_COLS, RANDOM_STATE, ensure_dir
from labutils import (K_INNER, add_auc_rows, fit_predict, load_base, parse_args, progress, quiet, require_sklearn,
                      summarize)

h5 = importlib.import_module("05_history_models")      # film context, history features (steps 5 reuse)

PRIMARY = "hgb_b"
KNN_K = 50
KNN_LAMBDA = 1.0
FCM_C = 6
FCM_M = 1.5
REPS = ["R1", "R2", "C1", "F1", "F2", "F3"]
COUNTS = ["friends_count", "favourites_count", "followers_count", "number_of_tweets"]
FLAGS = ["weekend_tweeter", "early_bird", "night_owl", "geo_enabled", "week_tweeter"]
EXPERIMENTS = {"B": ["P", "H"], **{r: ["P", "H", r] for r in REPS}, **{f"K{r}": ["P", "H", f"KP_{r}"] for r in REPS}}


# ----------------------------------------------------------------------------------------------
# fuzzy sets
# ----------------------------------------------------------------------------------------------
def three_sets(x, q10, q50, q90):
    """Low / Medium / High memberships (they add up to 1): shoulders at the ends, a triangle at the median."""
    d1, d2 = max(q50 - q10, 1e-6), max(q90 - q50, 1e-6)
    low = np.clip((q50 - x) / d1, 0, 1)
    high = np.clip((x - q50) / d2, 0, 1)
    med = np.clip(np.minimum((x - q10) / d1, (q90 - x) / d2), 0, 1)
    return low, med, high


def circular_sets(x, centers, period, halfwidth):
    cols = []
    for c in centers:
        d = np.abs(x - c)
        d = np.minimum(d, period - d)
        cols.append(np.clip(1 - d / halfwidth, 0, 1))
    return np.column_stack(cols)


def fcm_fit(X, c, m, seed, iters=150):
    rng = np.random.default_rng(seed)
    C = X[[int(rng.integers(len(X)))]]                      # k-means++ start: far-apart real users as first centres
    while len(C) < c:
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        C = np.vstack([C, X[int(rng.choice(len(X), p=d2 / d2.sum()))]])
    U = fcm_predict(X, C, m)
    for _ in range(iters):
        Um = U ** m
        C = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-12)
        U_new = fcm_predict(X, C, m)
        if np.abs(U_new - U).max() < 1e-5:
            break
        U = U_new
    return C


def fcm_predict(X, C, m):
    D = np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)) + 1e-9
    inv = D ** (-2.0 / (m - 1))
    return inv / inv.sum(axis=1, keepdims=True)


def build_reps(prof, train_ids, seed):
    """All six representations for every user in `prof` (rows aligned with prof.index); fitted on train_ids only."""
    from sklearn.preprocessing import StandardScaler
    tr = prof.index.isin(train_ids)
    reps = {}
    raw = prof[PROFILE_COLS].to_numpy(dtype=float)
    reps["R1"] = StandardScaler().fit(raw[tr]).transform(raw)

    log = np.column_stack([np.log1p(prof[c].to_numpy(dtype=float)) for c in COUNTS])
    year = prof["account_creation_year"].to_numpy(dtype=float)[:, None]
    hr = prof["preferred_hour"].to_numpy(dtype=float)
    wd = prof["preferred_weekday"].to_numpy(dtype=float)
    cont = np.hstack([year, log])                                               # 5 continuous variables
    r2 = np.hstack([cont, np.sin(2 * np.pi * hr / 24)[:, None], np.cos(2 * np.pi * hr / 24)[:, None],
                    np.sin(2 * np.pi * wd / 7)[:, None], np.cos(2 * np.pi * wd / 7)[:, None],
                    prof[FLAGS].to_numpy(dtype=float)])
    reps["R2"] = StandardScaler().fit(r2[tr]).transform(r2)

    # F1: fuzzy linguistic sets
    names = ["year", "friends", "favourites", "followers", "tweets"]
    groups, sets = [], {}
    for j, nm in enumerate(names):
        q10, q50, q90 = np.quantile(cont[tr, j], [0.1, 0.5, 0.9])
        lo, me, hi = three_sets(cont[:, j], q10, q50, q90)
        sets[nm] = (lo, me, hi)
        groups.append(np.column_stack([lo, me, hi]))
    hour_sets = circular_sets(hr, [3, 9, 15, 21], 24, 6)
    wday_sets = circular_sets(wd, list(range(7)), 7, 2)
    groups += [hour_sets, wday_sets]
    flags = prof[FLAGS].to_numpy(dtype=float)
    reps["F1"] = np.hstack(groups + [flags])
    # C1: the same sets, hard (each variable group keeps only its best set)
    hard = []
    for g in groups:
        h = np.zeros_like(g)
        h[np.arange(len(g)), g.argmax(axis=1)] = 1.0
        hard.append(h)
    reps["C1"] = np.hstack(hard + [flags])
    # F2: fuzzy c-means on R2
    C = fcm_fit(reps["R2"][tr], FCM_C, FCM_M, seed)
    reps["F2"] = fcm_predict(reps["R2"], C, FCM_M)
    # F3: expert rules (AND = min, OR = max)
    lo_y, _, hi_y = sets["year"]                       # year high = recent account
    lo_f, _, hi_f = sets["friends"]
    lo_v, _, hi_v = sets["favourites"]
    lo_w, _, hi_w = sets["followers"]
    lo_t, me_t, hi_t = sets["tweets"]
    night = hour_sets[:, 0]
    night_owl = np.maximum(night, flags[:, FLAGS.index("night_owl")])
    reps["F3"] = np.column_stack([
        np.minimum(hi_w, np.maximum(me_t, hi_t)),      # Influencer
        np.minimum(hi_t, lo_w),                        # Broadcaster
        np.minimum(hi_y, lo_t),                        # Newcomer
        np.minimum(lo_y, hi_t),                        # Veteran
        night_owl,                                     # NightOwl
        np.minimum(hi_f, hi_v),                        # Social
        np.minimum(lo_t, lo_v),                        # Lurker
    ])
    return {k: v.astype(np.float32) for k, v in reps.items()}


# ----------------------------------------------------------------------------------------------
# rows
# ----------------------------------------------------------------------------------------------
def prepare_ctx(ctx, uids, Rn, uid_row):
    ctx.rows = np.array([uid_row[int(u)] for u in uids])
    ctx.Qn = {k: Rn[k][ctx.rows] for k in REPS}
    return ctx


def kp_feature(ctx, rule, name, q, T):
    s = ctx.Qn[name] @ q
    K = min(KNN_K, len(s))
    top = np.argpartition(-s, K - 1)[:K]
    top = top[s[top] > 0]
    if len(top) == 0:
        return np.zeros(len(T), dtype=np.float32)
    sw = s[top]
    num = sw @ ctx.Rv[rule][np.ix_(top, T)]
    den = sw @ ctx.Mv[np.ix_(top, T)] + KNN_LAMBDA
    return (num / den).astype(np.float32)


def build_rows(users, N, ctx_of_user, rule, UA, F, reps, Rn, uid_row, mode, r, cfg):
    parts = {k: [] for k in ["P", "H"] + REPS + [f"KP_{x}" for x in REPS]}
    ys, us = [], []
    for uid in users:
        films_u, liked_by_rule = UA[uid]
        liked_u = liked_by_rule[rule]
        n = len(films_u)
        if mode == "train":
            cand = np.arange(N, n)
            if len(cand) == 0:
                continue
            S = cfg["train_per_user"]
            if len(cand) > S:
                rng = np.random.default_rng([RANDOM_STATE, r, int(uid), N, 5])
                tidx = np.sort(rng.choice(cand, S, replace=False))
            else:
                tidx = cand
        else:
            tidx = np.arange(EVAL_POOL_START, n)
            if len(tidx) == 0:
                continue
        ctx = ctx_of_user(uid)
        P, H, _, _, _ = h5.feats_for_user(ctx, rule, films_u, liked_u, N, tidx, F, False)
        parts["P"].append(P)
        parts["H"].append(H)
        row = uid_row[int(uid)]
        T = films_u[tidx]
        for k in REPS:
            parts[k].append(np.repeat(reps[k][row][None, :], len(tidx), axis=0))
            parts[f"KP_{k}"].append(kp_feature(ctx, rule, k, Rn[k][row], T).reshape(-1, 1))
        ys.append(liked_u[tidx])
        us.append(np.full(len(tidx), uid))
    blocks = {k: np.vstack(v).astype(np.float32) for k, v in parts.items()}
    return blocks, np.concatenate(ys), np.concatenate(us)


def make_pairs():
    pairs = []
    for x in ["R1", "R2", "C1", "F1", "F2", "F3"]:
        pairs.append((x, "B"))
        pairs.append((f"K{x}", "B"))
    for x in ["F1", "F2", "F3"]:
        pairs += [(x, "R1"), (x, "R2"), (x, "C1"), (f"K{x}", "KR1"), (f"K{x}", "KR2"), (f"K{x}", "KC1")]
    pairs += [("R2", "R1"), ("KR2", "KR1"), ("C1", "R1"), ("KC1", "KR1")]
    return pairs


def verdict(row):
    if row["ci_low"] > 0:
        return "better"
    if row["ci_high"] < 0:
        return "worse"
    return "no clear difference"


def main():
    cfg = parse_args(__doc__)
    require_sklearn()
    quiet()
    n_levels = cfg["n_list"]
    if n_levels == [0, 1, 3, 5, 10, 20, 50]:
        n_levels = [0, 5, 20]
    elif n_levels == [0, 5, 50]:
        n_levels = [0, 5]
    learners = cfg["learners"] or ["logreg", PRIMARY, "hgb_c"]
    out = ensure_dir(OUT_DIR / f"09_fuzzification{cfg['tag']}")
    if (out / "per_user_auc.csv").exists() and not cfg["force"]:
        print(f"Outputs already exist in {out}. Use --force to recompute.")
        return

    data, users_f, prof, movies = load_base()
    film_ids = np.sort(data["movie_id"].unique())
    M = len(film_ids)
    fidx = {m: i for i, m in enumerate(film_ids.tolist())}
    data["fi"] = data["movie_id"].map(fidx).astype(int)
    F = h5.build_films(movies, film_ids)
    prof = prof.set_index("user_id").sort_index()
    uid_row = {int(u): i for i, u in enumerate(prof.index)}
    eligible = users_f[users_f["eligible"]]
    rules = cfg["rules"]

    rows = []
    t_start = time.time()
    folds_done, folds_total = 0, len(cfg["repeats"]) * len(cfg["folds"])
    for r in cfg["repeats"]:
        d = data.sort_values(["user_id", f"pos_r{r}"], kind="mergesort")
        UA = {}
        for uid, g in d.groupby("user_id", sort=False):
            UA[int(uid)] = (g["fi"].to_numpy(), {k: g[f"liked_{k}"].to_numpy(dtype=np.float32) for k in LIKE_RULES})
        data["fold"] = data["user_id"].map(users_f.set_index("user_id")[f"fold_r{r}"])
        for f in cfg["folds"]:
            t0 = time.time()
            train = data[data["fold"] != f]
            train_users = np.sort(train["user_id"].unique())
            test_users = np.sort(eligible.loc[eligible[f"fold_r{r}"] == f, "user_id"].to_numpy())
            if set(test_users.tolist()) & set(train_users.tolist()):
                sys.exit(f"LEAK: a test user is also a training user (repeat {r}, fold {f})")
            reps = build_reps(prof, train_users, RANDOM_STATE + 100 * r + f)
            Rn = {k: v / (np.sqrt((v ** 2).sum(axis=1, keepdims=True)) + 1e-9) for k, v in reps.items()}

            rng = np.random.default_rng([RANDOM_STATE, r, f, 21])
            inner_of = dict(zip(train_users.tolist(), (rng.permutation(len(train_users)) % K_INNER).tolist()))
            inner_arr = train["user_id"].map(inner_of).to_numpy()
            sub_all = train
            ctx_all = prepare_ctx(h5.make_ctx(sub_all, rules, M, True), pd.factorize(sub_all["user_id"])[1], Rn, uid_row)
            ctx_inner = []
            for k in range(K_INNER):
                sub = train[inner_arr != k]
                ctx_inner.append(prepare_ctx(h5.make_ctx(sub, rules, M, True), pd.factorize(sub["user_id"])[1], Rn, uid_row))

            for rule in rules:
                for N in n_levels:
                    Btr, ytr, _ = build_rows(train_users.tolist(), N, lambda u: ctx_inner[inner_of[int(u)]], rule,
                                             UA, F, reps, Rn, uid_row, "train", r, cfg)
                    Bpo, ypo, upo = build_rows(test_users.tolist(), N, lambda u: ctx_all, rule,
                                               UA, F, reps, Rn, uid_row, "pool", r, cfg)
                    scores = {}
                    seed = RANDOM_STATE + 1000 * r + 10 * f + N
                    for exp, groups in EXPERIMENTS.items():
                        Xtr = np.hstack([Btr[g] for g in groups])
                        Xpo = np.hstack([Bpo[g] for g in groups])
                        for kind in learners:
                            scores[f"{exp}_{kind}"] = fit_predict(kind, Xtr, ytr, Xpo, cfg, seed)
                    add_auc_rows(rows, upo, ypo, scores, {"repeat": r, "fold": f, "rule": rule, "N": N})
            print(f"repeat {r} fold {f} done in {time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)", flush=True)
            folds_done += 1
            progress(folds_done, folds_total, t_start)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_user_auc.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({**cfg, "n_levels": n_levels, "learners": learners}, indent=2),
                                         encoding="utf-8")
    methods = set(df["method"].unique())
    pairs = [(f"{a}_{k}", f"{b}_{k}") for k in learners for a, b in make_pairs()]
    pairs = [(a, b) for a, b in pairs if a in methods and b in methods]
    summary, diffs = summarize(df, ["rule", "N"], pairs)
    summary.to_csv(out / "final_summary.csv", index=False)
    diffs["verdict"] = diffs.apply(verdict, axis=1)
    diffs.to_csv(out / "final_pairs.csv", index=False)

    ns = sorted(df["N"].unique().tolist())
    lines = ["STEP 9 - FUZZIFICATION OF THE PROFILE (new user, known films)", f"revealed ratings N: {ns}",
             f"learners: {learners}; primary learner: {PRIMARY} (decided before seeing results)",
             "B = film popularity + history only (no profile). R1 raw, R2 log/circular, C1 crisp, F1 fuzzy sets, "
             "F2 fuzzy c-means, F3 fuzzy rules; K* = same representation used to find similar users (kNN).", ""]
    for rule in sorted(summary["rule"].unique()):
        sub = summary[summary["rule"] == rule]
        lines.append(f"=== label rule: {rule} (about {int(sub['users'].median())} test users) ===")
        for k in learners:
            lines.append(f"-- learner {k}: mean per-user AUC [95% interval]")
            lines.append("  " + "method".ljust(10) + "".join(f"N={n}".rjust(24) for n in ns))
            for exp in EXPERIMENTS:
                m = f"{exp}_{k}"
                cells = []
                for n in ns:
                    s = sub[(sub["method"] == m) & (sub["N"] == n)]
                    cells.append(f"{s['mean_auc'].iloc[0]:.4f} [{s['ci_low'].iloc[0]:.3f},{s['ci_high'].iloc[0]:.3f}]"
                                 .rjust(24) if len(s) else "-".rjust(24))
                lines.append("  " + exp.ljust(10) + "".join(cells))
            lines.append("")
        for k in learners:
            lines.append(f"-- paired differences, learner {k}: A - B  (positive = A better)")
            for a, b in make_pairs():
                dd = diffs[(diffs["rule"] == rule) & (diffs["A"] == f"{a}_{k}") & (diffs["B"] == f"{b}_{k}")].sort_values("N")
                if dd.empty:
                    continue
                cells = "  ".join(f"N={int(x['N'])}: {x['mean_difference_A_minus_B']:+.4f} "
                                  f"[{x['ci_low']:+.4f},{x['ci_high']:+.4f}] {x['verdict'].replace('no clear difference', 'none')}"
                                  for _, x in dd.iterrows())
                lines.append(f"  {a} - {b}:".ljust(14) + cells)
            lines.append("")
    text = "\n".join(lines)
    (out / "final_report.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nfiles written to: {out}")


if __name__ == "__main__":
    main()
