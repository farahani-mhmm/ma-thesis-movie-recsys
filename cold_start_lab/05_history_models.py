"""Step 5: the experiments that use the N revealed ratings of a new user: E1, E1+P, E2, E3, E4 (and a control).

All features are computed once per (repeat, fold, rule, N) and every experiment uses a subset of them.

Feature groups (all for a pair "user u, film m"):
    P   film popularity from TRAINING users: like ratio, mean rating, log count
    H   history of the user's N revealed ratings:
          h_n      N itself
          h_like   smoothed like ratio of the revealed films
          h_bias   mean of (liked - film popularity) over the revealed films ("how picky is this user")
          h_cf     taste score from similar films: sum over revealed films j of sim(m,j) * (liked_j - popularity_j),
                   divided by sum |sim| + 1; sim = cosine of centred ratings between films (training users only)
    L   the two GPT-4o fame columns of the film (director fame, actors fame) from movies.csv
    A   affinity: how the user rated REVEALED films that share a director / an actor with film m
          (count, smoothed like ratio) and the distance between the film's fame and the fame of films the user liked
    PR  the 12 raw profile columns (standardised on training users)

Experiments:  G0h = P (control)   E1 = P+H   E1P = P+H+PR   E2 = P+H+L   E3 = P+H+L+A   E4 = P+H+L+A+PR
Learners:     hgb (primary, decided before seeing results), logreg, and rf with --with-rf.

No leakage:  training rows of a fold get their film statistics and similarities from the other inner folds of
training users (K_INNER = 5); the test pool uses all training users. Test users are never in the training users.
Training rows: for every training user the first N ratings (in the random order of step 2) are revealed and up to
`train_per_user` of the REMAINING ratings are the training targets, so the model learns with the same N it will see.

Needs: pandas, numpy, scikit-learn.
Run  : python 05_history_models.py            (all repeats, folds, rules and N levels)
       python 05_history_models.py --quick    (smoke test, a few minutes)
Output (outputs/05_history_models[_quick]/): per_user_auc.csv, mean_auc.txt, run_config.json
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from config import EVAL_POOL_START, LIKE_RULES, OUT_DIR, PROFILE_COLS, RANDOM_STATE, ensure_dir
from labutils import (K_INNER, PRIMARY_LEARNER, add_auc_rows, fit_predict, load_base, parse_args, progress, quiet,
                      require_sklearn)

INVALID_NAMES = {"NO_CREDITED_ACTORS", "Varios", "NaN", "", "N/A", "Unknown", "None", "No Credit", "Uncredited"}
H_SMOOTH = 5.0       # smoothing of the user's like ratio toward the global like ratio
A_SMOOTH = 2.0       # smoothing of affinity like ratios toward the user's own like ratio
MAX_ACTORS = 8       # the first cast members carry the signal (same limit as the old pipeline)
EXPERIMENTS = {
    "G0h": ["P"],
    "E1": ["P", "H"],
    "E1P": ["P", "H", "PR"],
    "E2": ["P", "H", "L"],
    "E3": ["P", "H", "L", "A"],
    "E4": ["P", "H", "L", "A", "PR"],
}
# --with-uu adds a user-user collaborative-filtering feature UU: the revealed ratings of the new user are compared
# with those of every training user on the films both rated (cosine of "liked minus film popularity", shrunk
# toward 0 when the overlap is small); the UU_K most similar training users who rated the target film vote.
UU_EXPERIMENTS = {"E1U": ["P", "H", "UU"], "E1PU": ["P", "H", "PR", "UU"], "E4U": ["P", "H", "L", "A", "PR", "UU"]}
UU_K = 30
UU_SHRINK = 10.0
UU_MIN_OVERLAP = 3
UU_LAMBDA = 1.0


# ----------------------------------------------------------------------------------------------
# film attributes
# ----------------------------------------------------------------------------------------------
def tokens(cell, limit=None):
    if pd.isna(cell):
        return []
    out = [t.strip() for t in str(cell).split("|")]
    out = [t for t in out if t and t not in INVALID_NAMES]
    return out[:limit] if limit else out


def share_matrix(token_lists):
    """M x M float matrix: 1 when two films share at least one token (director or actor)."""
    vocab, rows, cols = {}, [], []
    for i, toks in enumerate(token_lists):
        for t in set(toks):
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
    A = np.zeros((len(token_lists), max(len(vocab), 1)), dtype=np.float32)
    if rows:
        A[rows, cols] = 1.0
    S = (A @ A.T) > 0
    np.fill_diagonal(S, False)
    return S.astype(np.float32)


class Films:
    pass


def build_films(movies, film_ids):
    mv = movies.drop_duplicates("movie_id").set_index("movie_id").loc[film_ids]
    F = Films()
    fd = pd.to_numeric(mv["director_fame_score_llm"], errors="coerce")
    fa = pd.to_numeric(mv["actors_fame_score_llm"], errors="coerce")
    F.mean_fame_dir = float(fd.mean())
    F.mean_fame_act = float(fa.mean())
    F.fame_dir = fd.fillna(F.mean_fame_dir).to_numpy(dtype=np.float32)
    F.fame_act = fa.fillna(F.mean_fame_act).to_numpy(dtype=np.float32)
    F.dir_share = share_matrix([tokens(x) for x in mv["directors"]])
    F.act_share = share_matrix([tokens(x, MAX_ACTORS) for x in mv["actors"]])
    return F


# ----------------------------------------------------------------------------------------------
# contexts: film statistics + film-film similarity built from a set of users
# ----------------------------------------------------------------------------------------------
class Ctx:
    pass


def make_ctx(df_ctx, rules, M, with_uu=False):
    c = Ctx()
    fi = df_ctx["fi"].to_numpy()
    rate = df_ctx["rate"].to_numpy(dtype=np.float32)
    cnt = np.bincount(fi, minlength=M).astype(np.float64)
    gmean = float(rate.mean())
    c.f_mean = np.where(cnt > 0, np.bincount(fi, weights=rate, minlength=M) / np.maximum(cnt, 1), gmean).astype(np.float32)
    c.f_logc = np.log1p(cnt).astype(np.float32)
    c.f_like, c.g_like = {}, {}
    for rule in rules:
        lk = df_ctx[f"liked_{rule}"].to_numpy(dtype=np.float32)
        g = float(lk.mean())
        c.f_like[rule] = np.where(cnt > 0, np.bincount(fi, weights=lk, minlength=M) / np.maximum(cnt, 1), g).astype(np.float32)
        c.g_like[rule] = g
    uix = pd.factorize(df_ctx["user_id"])[0]
    Z = np.zeros((int(uix.max()) + 1, M), dtype=np.float32)
    Z[uix, fi] = rate - c.f_mean[fi]
    nrm = np.sqrt((Z ** 2).sum(axis=0)) + 1e-6
    sim = (Z.T @ Z) / (nrm[:, None] * nrm[None, :])
    np.fill_diagonal(sim, 0.0)
    c.sim = sim.astype(np.float32)
    if with_uu:
        # residual matrices of the context users: liked minus film popularity, zero where not rated
        nU = int(uix.max()) + 1
        c.Mv = np.zeros((nU, M), dtype=np.float32)
        c.Mv[uix, fi] = 1.0
        c.Rv, c.Rv2 = {}, {}
        for rule in rules:
            lk = df_ctx[f"liked_{rule}"].to_numpy(dtype=np.float32)
            Rv = np.zeros((nU, M), dtype=np.float32)
            Rv[uix, fi] = lk - c.f_like[rule][fi]
            c.Rv[rule] = Rv
            c.Rv2[rule] = Rv ** 2
    return c


# ----------------------------------------------------------------------------------------------
# features of one user
# ----------------------------------------------------------------------------------------------
def uu_feature(ctx, rule, R, resid, T):
    """User-user collaborative filtering score for the target films T (see UU_EXPERIMENTS)."""
    Rv, Rv2, Mv = ctx.Rv[rule], ctx.Rv2[rule], ctx.Mv
    nM = Rv.shape[1]
    x = np.zeros(nM, dtype=np.float32)
    x[R] = resid
    ind = np.zeros(nM, dtype=np.float32)
    ind[R] = 1.0
    dot = Rv @ x
    q = Rv2 @ ind
    nov = Mv @ ind
    xn = float(np.sqrt((x ** 2).sum()))
    sim = dot / (np.sqrt(q) * xn + 1e-6) * (nov / (nov + UU_SHRINK))
    sim[nov < UU_MIN_OVERLAP] = 0.0
    top = np.argsort(-sim)[:UU_K]
    top = top[sim[top] > 0]
    if len(top) == 0:
        return np.zeros(len(T), dtype=np.float32)
    sw = sim[top]
    P = Rv[np.ix_(top, T)]
    Mk = Mv[np.ix_(top, T)]
    return ((sw @ P) / (np.abs(sw) @ Mk + UU_LAMBDA)).astype(np.float32)


def feats_for_user(ctx, rule, films_u, liked_u, N, tidx, F, with_uu=False):
    R = films_u[:N]
    lR = liked_u[:N]
    T = films_u[tidx]
    nT = len(T)
    n = len(R)
    fl = ctx.f_like[rule]
    g = ctx.g_like[rule]
    uu = np.zeros(nT, dtype=np.float32)
    P = np.column_stack([fl[T], ctx.f_mean[T], ctx.f_logc[T]])
    if n == 0:
        h_like, h_bias = g, 0.0
        cf = np.zeros(nT, dtype=np.float32)
        a_dir_cnt = np.zeros(nT, dtype=np.float32)
        a_act_cnt = np.zeros(nT, dtype=np.float32)
        a_dir_like = np.full(nT, h_like, dtype=np.float32)
        a_act_like = np.full(nT, h_like, dtype=np.float32)
        pref_d, pref_a = F.mean_fame_dir, F.mean_fame_act
    else:
        h_like = (float(lR.sum()) + H_SMOOTH * g) / (n + H_SMOOTH)
        resid = lR - fl[R]
        h_bias = float(resid.mean())
        if with_uu:
            uu = uu_feature(ctx, rule, R, resid, T)
        W = ctx.sim[np.ix_(T, R)]
        cf = (W @ resid) / (np.abs(W).sum(axis=1) + 1.0)
        Dm = F.dir_share[np.ix_(T, R)]
        Am = F.act_share[np.ix_(T, R)]
        a_dir_cnt = Dm.sum(axis=1)
        a_act_cnt = Am.sum(axis=1)
        a_dir_like = (Dm @ lR + A_SMOOTH * h_like) / (a_dir_cnt + A_SMOOTH)
        a_act_like = (Am @ lR + A_SMOOTH * h_like) / (a_act_cnt + A_SMOOTH)
        liked_mask = lR > 0.5
        pref_d = float(F.fame_dir[R][liked_mask].mean()) if liked_mask.any() else F.mean_fame_dir
        pref_a = float(F.fame_act[R][liked_mask].mean()) if liked_mask.any() else F.mean_fame_act
    H = np.column_stack([np.full(nT, n), np.full(nT, h_like), np.full(nT, h_bias), cf])
    L = np.column_stack([F.fame_dir[T], F.fame_act[T]])
    A = np.column_stack([a_dir_cnt, a_dir_like, a_act_cnt, a_act_like,
                         np.abs(F.fame_dir[T] - pref_d), np.abs(F.fame_act[T] - pref_a)])
    return (P.astype(np.float32), H.astype(np.float32), L.astype(np.float32), A.astype(np.float32),
            uu.reshape(-1, 1).astype(np.float32))


def build_block(users, N, ctx_of_user, rule, UA, F, prof_vec, mode, r, cfg):
    parts = {"P": [], "H": [], "L": [], "A": [], "PR": [], "UU": []}
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
        P, H, L, A, UU = feats_for_user(ctx_of_user(uid), rule, films_u, liked_u, N, tidx, F, cfg["with_uu"])
        for key, arr in (("P", P), ("H", H), ("L", L), ("A", A), ("UU", UU)):
            parts[key].append(arr)
        parts["PR"].append(np.repeat(prof_vec[uid][None, :], len(tidx), axis=0))
        ys.append(liked_u[tidx])
        us.append(np.full(len(tidx), uid))
    blocks = {k: np.vstack(v).astype(np.float32) for k, v in parts.items()}
    return blocks, np.concatenate(ys), np.concatenate(us)


def main():
    cfg = parse_args(__doc__)
    require_sklearn()
    quiet()
    from sklearn.preprocessing import StandardScaler

    out = ensure_dir(OUT_DIR / f"05_history_models{cfg['tag']}")
    if (out / "per_user_auc.csv").exists() and not cfg["force"]:
        print(f"Outputs already exist in {out}. Use --force to recompute.")
        return

    data, users_f, prof, movies = load_base(cfg)
    film_ids = np.sort(data["movie_id"].unique())
    M = len(film_ids)
    fidx = {m: i for i, m in enumerate(film_ids.tolist())}
    data["fi"] = data["movie_id"].map(fidx).astype(int)
    F = build_films(movies, film_ids)
    prof = prof.set_index("user_id")
    eligible = users_f[users_f["eligible"]]
    learners = ["logreg", PRIMARY_LEARNER] + (["rf"] if cfg["with_rf"] else [])
    rules = cfg["rules"]

    rows = []
    t_start = time.time()
    folds_done, folds_total = 0, len(cfg["repeats"]) * len(cfg["folds"])
    for r in cfg["repeats"]:
        pos_col = f"pos_r{r}"
        d = data.sort_values(["user_id", pos_col], kind="mergesort")
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

            scaler = StandardScaler().fit(prof.loc[train_users, PROFILE_COLS])
            prof_vec = {int(u): v.astype(np.float32)
                        for u, v in zip(prof.index, scaler.transform(prof[PROFILE_COLS]))}

            rng = np.random.default_rng([RANDOM_STATE, r, f, 21])
            inner_of = dict(zip(train_users.tolist(), (rng.permutation(len(train_users)) % K_INNER).tolist()))
            inner_arr = train["user_id"].map(inner_of).to_numpy()
            ctx_all = make_ctx(train, rules, M, cfg["with_uu"])
            ctx_inner = [make_ctx(train[inner_arr != k], rules, M, cfg["with_uu"]) for k in range(K_INNER)]

            for rule in rules:
                lab = f"liked_{rule}"
                for N in cfg["n_list"]:
                    Btr, ytr, _ = build_block(train_users.tolist(), N, lambda u: ctx_inner[inner_of[int(u)]], rule,
                                              UA, F, prof_vec, "train", r, cfg)
                    Bpo, ypo, upo = build_block(test_users.tolist(), N, lambda u: ctx_all, rule,
                                                UA, F, prof_vec, "pool", r, cfg)
                    scores = {}
                    exps = dict(EXPERIMENTS)
                    if cfg["with_uu"]:
                        exps.update(UU_EXPERIMENTS)
                    for exp, groups in exps.items():
                        Xtr = np.hstack([Btr[g] for g in groups])
                        Xpo = np.hstack([Bpo[g] for g in groups])
                        for kind in learners:
                            seed = RANDOM_STATE + 1000 * r + 10 * f + N
                            scores[f"{exp}_{kind}"] = fit_predict(kind, Xtr, ytr, Xpo, cfg, seed)
                    add_auc_rows(rows, upo, ypo, scores, {"repeat": r, "fold": f, "rule": rule, "N": N})
            print(f"repeat {r} fold {f} done in {time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)", flush=True)
            folds_done += 1
            progress(folds_done, folds_total, t_start)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_user_auc.csv", index=False)
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    lines = ["STEP 5 - history experiments - mean per-user AUC by N (confidence intervals come in step 6)", ""]
    for rule in rules:
        lines.append(f"=== label rule: {rule} ===")
        tab = df[df["rule"] == rule].groupby(["method", "N"])["auc"].mean().unstack("N")
        header = "  " + "method".ljust(14) + "".join(f"N={n}".rjust(9) for n in tab.columns)
        lines.append(header)
        for m, rr in tab.iterrows():
            lines.append("  " + m.ljust(14) + "".join(f"{v:9.4f}" for v in rr.to_numpy()))
        ref = OUT_DIR / "03_baselines" / "per_user_auc.csv"
        if ref.exists():
            b = pd.read_csv(ref)
            b = b[(b["rule"] == rule) & (b["method"] == "B1_pop_like") & (b["repeat"].isin(cfg["repeats"]))]
            if len(b):
                lines.append(f"  (reference, any N) B1_pop_like from step 3: {b['auc'].mean():.4f}")
        lines.append("")
    text = "\n".join(lines)
    (out / "mean_auc.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nfiles written to: {out}")


if __name__ == "__main__":
    main()
