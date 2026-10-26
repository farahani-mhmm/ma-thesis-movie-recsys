"""Step 14: the FINAL PROTOCOL. One script, three research questions, one set of choices.

Question 1  scenario A, NEW USER      does the fuzzified Twitter profile help beyond film popularity and history?
Question 2  scenario B, NEW FILM      does the LLM fame of director and actors help a film nobody has rated?
Question 3  scenario C, BOTH NEW      what helps when the user and the film are both new?

Fixed choices: label rule gt7 (rate > 7; ge7 only with --rules gt7,ge7), Sonnet 5.5 fame columns, ONE fuzzifier (F1,
linguistic sets) against the controls raw (R1) and crisp (C1), learners logreg / hgb_c / rf, N = 0, 5, 20, 50
revealed ratings of the new user, M = 0, 5 known ratings of the new film, user-grouped 5 folds x 3 repeats.

Information blocks (the names of the experiments are built from them):
    P       film popularity from training users (like ratio, mean rating, log count)           [scenario A]
    H       history of the user: like ratio of the revealed (A: also bias and taste) ratings
    T       track record: how well training films with the same director / actor were liked   [B, C]
    L       LLM fame of the director and of the actors (Sonnet 5.5)
    A       affinity: how the user rated revealed films with the same director / actor
    FH      film history: like ratio of the M known ratings of the new film                   [B]
    PRraw   the 12 profile columns, standardised      PRcrisp  the same, crisp sets      PRfuzzy  fuzzy sets F1

Experiments: add-one chains (P, P+H, P+PRfuzzy, T, T+L, ...) AND leave-one-block-out from the full model
(FULL, FULL-P, FULL-L, ...). Every experiment is run for every learner, except rf which runs on a smaller list.

For every model: per-user AUC (as before) and per-user counts of true/false positives/negatives at a threshold that
maximises F1 on a held-out 20% of the TRAINING users (test users are never used for it).
Results are appended fold by fold to outputs/14_final[_quick]/<scenario>/partial_*.csv, so a stopped run resumes.

Run  : python 14_final_protocol.py --quick           (smoke test)
       python 14_final_protocol.py                   (everything; long)
       python 14_final_protocol.py --scenario C      (one scenario)
Then : python 15_final_report.py
"""
import importlib
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from config import EVAL_POOL_START, LIKE_RULES, N_FOLDS, OUT_DIR, PROFILE_COLS, RANDOM_STATE, ensure_dir
from labfilms import build_films
from labutils import (K_INNER, add_auc_rows, add_confusion_rows, fit_predict_thr, load_base, parse_args, progress,
                      quiet, require_sklearn)

h5 = importlib.import_module("05_history_models")
fz = importlib.import_module("09_fuzzification")

S_SMOOTH = 2.0
H_SMOOTH = 5.0
PR_REPS = {"PRraw": "R1", "PRcrisp": "C1", "PRfuzzy": "F1"}


# ----------------------------------------------------------------------------------------------
# experiments and pairs (the names are made of the blocks)
# ----------------------------------------------------------------------------------------------
def leave_one_out(full, groups):
    """FULL and FULL-<label> for every label; groups = {label: [blocks that leave together]}."""
    out = {"FULL": list(full)}
    for label, drop in groups.items():
        rest = [b for b in full if b not in drop]
        if len(rest) < len(full):
            out[f"FULL-{label}"] = rest
    return out


A_FULL = ["P", "H", "L", "A", "PRfuzzy"]
A_EXPS = {
    "P": ["P"], "P+H": ["P", "H"],
    "P+PRraw": ["P", "PRraw"], "P+PRcrisp": ["P", "PRcrisp"], "P+PRfuzzy": ["P", "PRfuzzy"],
    "P+H+PRraw": ["P", "H", "PRraw"], "P+H+PRcrisp": ["P", "H", "PRcrisp"], "P+H+PRfuzzy": ["P", "H", "PRfuzzy"],
    "P+H+L": ["P", "H", "L"], "P+H+L+A": ["P", "H", "L", "A"],
    **leave_one_out(A_FULL, {"P": ["P"], "H": ["H"], "L": ["L"], "A": ["A"], "PR": ["PRfuzzy"]}),
}
A_PAIRS = [("P", "B0_const"), ("P+H", "P"), ("P+PRraw", "P"), ("P+PRcrisp", "P"), ("P+PRfuzzy", "P"),
           ("P+PRfuzzy", "P+PRraw"), ("P+PRfuzzy", "P+PRcrisp"),
           ("P+H+PRraw", "P+H"), ("P+H+PRcrisp", "P+H"), ("P+H+PRfuzzy", "P+H"),
           ("P+H+PRfuzzy", "P+H+PRraw"), ("P+H+PRfuzzy", "P+H+PRcrisp"),
           ("P+H+L", "P+H"), ("P+H+L+A", "P+H+L"), ("FULL", "B0_const")] + \
          [("FULL", f"FULL-{x}") for x in ["P", "H", "L", "A", "PR"]]
A_RF = {"P", "P+H", "P+PRraw", "P+PRfuzzy", "P+H+PRfuzzy", "FULL"}

B_PAIRS = [("T", "B0_const"), ("L", "B0_const"), ("T+L", "T"), ("T+A+H", "T"), ("FULL", "FH"),
           ("FULL", "FULL-T"), ("FULL", "FULL-L"), ("FULL", "FULL-AH"), ("FULL", "FULL-FH")]
B_RF = {"T", "T+L", "FULL"}

C_PAIRS = [("T", "B0_const"), ("T+L", "T"), ("T+PRraw", "T"), ("T+PRcrisp", "T"), ("T+PRfuzzy", "T"),
           ("T+PRfuzzy", "T+PRraw"), ("T+PRfuzzy", "T+PRcrisp"),
           ("T+L+PRraw", "T+L"), ("T+L+PRcrisp", "T+L"), ("T+L+PRfuzzy", "T+L"),
           ("T+L+PRfuzzy", "T+PRfuzzy"), ("T+L+PRfuzzy", "T+L+PRraw"), ("T+L+PRfuzzy", "T+L+PRcrisp"),
           ("T+A+H", "T"), ("T+L+A+H", "T+A+H"), ("FULL", "B0_const"),
           ("FULL", "FULL-T"), ("FULL", "FULL-L"), ("FULL", "FULL-PR"), ("FULL", "FULL-AH")]
C_RF = {"T", "T+L", "T+PRfuzzy", "T+L+PRfuzzy", "FULL"}


def film_folds(cnt, r, n_films):
    """Film folds balanced by popularity: films sorted by rating count, each block of N_FOLDS spread over the folds."""
    rng = np.random.default_rng([RANDOM_STATE, r, 77])
    order = np.argsort(-cnt, kind="stable")
    fold = np.zeros(n_films, dtype=int)
    for s in range(0, n_films, N_FOLDS):
        blk = order[s:s + N_FOLDS]
        fold[blk] = rng.permutation(N_FOLDS)[:len(blk)]
    return fold


def affinity_block(F, cand, ruf, lu, h_u):
    """A: for each candidate film, count and like ratio of the user's known films with the same director / actor,
    and the distance between the film's fame and the fame of films the user liked."""
    Dm, Am = F.dir_share[cand], F.act_share[cand]
    a_dir_cnt, a_act_cnt = Dm @ ruf, Am @ ruf
    a_dir_like = (Dm @ lu + S_SMOOTH * h_u) / (a_dir_cnt + S_SMOOTH)
    a_act_like = (Am @ lu + S_SMOOTH * h_u) / (a_act_cnt + S_SMOOTH)
    liked_mask = lu > 0.5
    pref_d = float(F.fame_dir[liked_mask].mean()) if liked_mask.any() else F.mean_fame_dir
    pref_a = float(F.fame_act[liked_mask].mean()) if liked_mask.any() else F.mean_fame_act
    return np.column_stack([a_dir_cnt, a_dir_like, a_act_cnt, a_act_like,
                            np.abs(F.fame_dir[cand] - pref_d), np.abs(F.fame_act[cand] - pref_a)]).astype(np.float32)


def track_features(F, f_like, g, tr_f):
    """T: for every film, number and smoothed like ratio of TRAINING films with the same director / actor."""
    trf = tr_f.astype(np.float32)
    Wd, Wa = F.dir_share * trf[None, :], F.act_share * trf[None, :]
    dir_n, act_n = Wd.sum(axis=1), Wa.sum(axis=1)
    dir_like = (Wd @ f_like + S_SMOOTH * g) / (dir_n + S_SMOOTH)
    act_like = (Wa @ f_like + S_SMOOTH * g) / (act_n + S_SMOOTH)
    return np.column_stack([dir_n, dir_like, act_n, act_like]).astype(np.float32)


class Matrices:
    """Dense user x film arrays shared by scenarios B and C."""

    def __init__(self, cfg):
        data, users_f, prof, movies = load_base(cfg)
        self.users = np.sort(data["user_id"].unique())
        uidx = {u: i for i, u in enumerate(self.users.tolist())}
        self.film_ids = np.sort(data["movie_id"].unique())
        fidx = {m: i for i, m in enumerate(self.film_ids.tolist())}
        self.U, self.M = len(self.users), len(self.film_ids)
        data["ui"] = data["user_id"].map(uidx).to_numpy()
        data["fi"] = data["movie_id"].map(fidx).to_numpy()
        self.data = data
        self.Rated = np.zeros((self.U, self.M), dtype=bool)
        self.Rated[data["ui"].to_numpy(), data["fi"].to_numpy()] = True
        self.Liked = {}
        for rule in LIKE_RULES:
            self.Liked[rule] = np.zeros((self.U, self.M), dtype=np.float32)
            self.Liked[rule][data["ui"].to_numpy(), data["fi"].to_numpy()] = data[f"liked_{rule}"].to_numpy(dtype=np.float32)
        self.F = build_films(movies, self.film_ids)
        self.cnt = self.Rated.sum(axis=0)
        self.uf = users_f.set_index("user_id").loc[self.users]
        self.eligible = self.uf["eligible"].to_numpy()
        self.prof_all = prof.set_index("user_id").loc[self.users, PROFILE_COLS]
        self.Lfeat = np.column_stack([self.F.fame_dir, self.F.fame_act]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# scenario A: new user
# ----------------------------------------------------------------------------------------------
class ScenA:
    name, rf_exps, pairs, level_key = "A_new_user", A_RF, A_PAIRS, "N"

    def __init__(self, cfg):
        self.cfg = cfg
        data, users_f, prof, movies = load_base(cfg)
        self.film_ids = np.sort(data["movie_id"].unique())
        self.M = len(self.film_ids)
        fidx = {m: i for i, m in enumerate(self.film_ids.tolist())}
        data["fi"] = data["movie_id"].map(fidx).astype(int)
        self.data = data
        self.users_f = users_f
        self.F = build_films(movies, self.film_ids)
        self.prof = prof.set_index("user_id")
        self.eligible = users_f[users_f["eligible"]]
        self.uid_row = {int(u): i for i, u in enumerate(self.prof.index)}
        self.UA = {}

    def levels(self):
        return self.cfg["n_list"]

    def exps(self, N):
        return A_EXPS

    def setup(self, r, f):
        cfg, data, M = self.cfg, self.data, self.M
        if r not in self.UA:
            d = data.sort_values(["user_id", f"pos_r{r}"], kind="mergesort")
            self.UA[r] = {int(uid): (g["fi"].to_numpy(), {k: g[f"liked_{k}"].to_numpy(dtype=np.float32) for k in LIKE_RULES})
                          for uid, g in d.groupby("user_id", sort=False)}
        fold_of = self.users_f.set_index("user_id")[f"fold_r{r}"]
        train = data[data["user_id"].map(fold_of).to_numpy() != f]
        train_users = np.sort(train["user_id"].unique())
        test_users = np.sort(self.eligible.loc[self.eligible[f"fold_r{r}"] == f, "user_id"].to_numpy())
        if set(test_users.tolist()) & set(train_users.tolist()):
            sys.exit(f"LEAK: a test user is also a training user (repeat {r}, fold {f})")
        rng = np.random.default_rng([RANDOM_STATE, r, f, 21])
        inner_of = dict(zip(train_users.tolist(), (rng.permutation(len(train_users)) % K_INNER).tolist()))
        inner_arr = train["user_id"].map(inner_of).to_numpy()
        rules = cfg["rules"]
        return SimpleNamespace(
            r=r, f=f, UA=self.UA[r], train_users=train_users, test_users=test_users, inner_of=inner_of,
            ctx_all=h5.make_ctx(train, rules, M, False),
            ctx_inner=[h5.make_ctx(train[inner_arr != k], rules, M, False) for k in range(K_INNER)],
            reps=fz.build_reps(self.prof, train_users, RANDOM_STATE + 100 * r + f))

    def rows(self, su, users, N, ctx_of_user, rule, mode):
        parts = {k: [] for k in ["P", "H", "L", "A", *PR_REPS]}
        ys, us = [], []
        for uid in users:
            films_u, liked_by_rule = su.UA[uid]
            liked_u = liked_by_rule[rule]
            n = len(films_u)
            if mode == "train":
                cand = np.arange(N, n)
                if len(cand) == 0:
                    continue
                S = self.cfg["train_per_user"]
                if len(cand) > S:
                    rng = np.random.default_rng([RANDOM_STATE, su.r, int(uid), N, 5])
                    tidx = np.sort(rng.choice(cand, S, replace=False))
                else:
                    tidx = cand
            else:
                tidx = np.arange(EVAL_POOL_START, n)
                if len(tidx) == 0:
                    continue
            P, H, L, A, _ = h5.feats_for_user(ctx_of_user(uid), rule, films_u, liked_u, N, tidx, self.F, False)
            for key, arr in (("P", P), ("H", H), ("L", L), ("A", A)):
                parts[key].append(arr)
            row = self.uid_row[int(uid)]
            for key, rep in PR_REPS.items():
                parts[key].append(np.repeat(su.reps[rep][row][None, :], len(tidx), axis=0))
            ys.append(liked_u[tidx])
            us.append(np.full(len(tidx), uid))
        return {k: np.vstack(v).astype(np.float32) for k, v in parts.items()}, np.concatenate(ys), np.concatenate(us)

    def blocks(self, su, rule, N):
        Btr, ytr, utr = self.rows(su, su.train_users.tolist(), N, lambda u: su.ctx_inner[su.inner_of[int(u)]], rule, "train")
        Bpo, ypo, upo = self.rows(su, su.test_users.tolist(), N, lambda u: su.ctx_all, rule, "pool")
        return Btr, ytr, utr, Bpo, ypo, upo, {"S_pop": Bpo["P"][:, 0]}


# ----------------------------------------------------------------------------------------------
# scenario B: new film
# ----------------------------------------------------------------------------------------------
class ScenB:
    name, rf_exps, pairs, level_key = "B_new_film", B_RF, B_PAIRS, "M"

    def __init__(self, cfg):
        self.cfg = cfg
        self.m = Matrices(cfg)

    def levels(self):
        return self.cfg["m_list"]

    def exps(self, M):
        full = ["T", "L", "A", "H"] + (["FH"] if M > 0 else [])
        e = {"L": ["L"], "T": ["T"], "T+L": ["T", "L"], "T+A+H": ["T", "A", "H"]}
        if M > 0:
            e["FH"] = ["FH"]
        e.update(leave_one_out(full, {"T": ["T"], "L": ["L"], "AH": ["A", "H"], "FH": ["FH"]}))
        return e

    def setup(self, r, f):
        m = self.m
        ufold = m.uf[f"fold_r{r}"].to_numpy()
        te_f = film_folds(m.cnt, r, m.M) == f
        return SimpleNamespace(r=r, f=f, te_f=te_f, tr_f=~te_f, tr_users=np.flatnonzero(ufold != f),
                               te_users=np.flatnonzero((ufold == f) & m.eligible))

    def blocks(self, su, rule, Mlev):
        m, F, cfg = self.m, self.m.F, self.cfg
        r, f, tr_f, te_f = su.r, su.f, su.tr_f, su.te_f
        L = m.Liked[rule]
        Rt, Lt = m.Rated[su.tr_users], L[su.tr_users]
        c = Rt.sum(axis=0).astype(np.float32)
        s_like = (Lt * Rt).sum(axis=0)
        g = float(s_like[tr_f].sum() / max(c[tr_f].sum(), 1))
        f_like = np.where(c > 0, s_like / np.maximum(c, 1), g).astype(np.float32)
        Tfeat = track_features(F, f_like, g, tr_f)
        rng = np.random.default_rng([RANDOM_STATE, r, f, Mlev, 31])
        fh_like = np.full(m.M, g, dtype=np.float32)
        fh_logc = np.zeros(m.M, dtype=np.float32)
        InR = np.zeros((m.U, m.M), dtype=bool)
        if Mlev > 0:
            for j in range(m.M):
                cand = su.tr_users[m.Rated[su.tr_users, j]]
                if len(cand) == 0:
                    continue
                pick = rng.choice(cand, min(Mlev, len(cand)), replace=False)
                InR[pick, j] = True
                fh_like[j] = (L[pick, j].sum() + S_SMOOTH * g) / (len(pick) + S_SMOOTH)
                fh_logc[j] = np.log1p(len(pick))
        FH = np.column_stack([fh_like, fh_logc]).astype(np.float32)

        def build(user_list, mode):
            parts = {"L": [], "T": [], "A": [], "H": [], "FH": []}
            ys, us = [], []
            for u in user_list:
                ru = m.Rated[u] & tr_f
                if mode == "train":
                    cand = np.flatnonzero(ru & ~InR[u]) if Mlev > 0 else np.flatnonzero(ru)
                    S = cfg["train_per_user_b"]
                    if len(cand) > S:
                        rr = np.random.default_rng([RANDOM_STATE, r, f, Mlev, 41, int(u)])
                        cand = np.sort(rr.choice(cand, S, replace=False))
                else:
                    cand = np.flatnonzero(m.Rated[u] & te_f)
                if len(cand) == 0:
                    continue
                ruf = ru.astype(np.float32)
                lu = L[u] * ruf
                h_u = (float(lu.sum()) + H_SMOOTH * g) / (float(ruf.sum()) + H_SMOOTH)
                parts["L"].append(m.Lfeat[cand])
                parts["T"].append(Tfeat[cand])
                parts["A"].append(affinity_block(F, cand, ruf, lu, h_u))
                parts["H"].append(np.full((len(cand), 1), h_u, dtype=np.float32))
                parts["FH"].append(FH[cand])
                ys.append(L[u, cand])
                us.append(np.full(len(cand), m.users[u]))
            return {k: np.vstack(v).astype(np.float32) for k, v in parts.items()}, np.concatenate(ys), np.concatenate(us)

        Btr, ytr, utr = build(su.tr_users.tolist(), "train")
        Bpo, ypo, upo = build(su.te_users.tolist(), "test")
        simple = {"S_fame_sum": Bpo["L"][:, 0] + Bpo["L"][:, 1], "S_track_dir": Bpo["T"][:, 1]}
        if Mlev > 0:
            simple["S_fh_like"] = Bpo["FH"][:, 0]
        return Btr, ytr, utr, Bpo, ypo, upo, simple


# ----------------------------------------------------------------------------------------------
# scenario C: both new
# ----------------------------------------------------------------------------------------------
class ScenC:
    name, rf_exps, pairs, level_key = "C_both_new", C_RF, C_PAIRS, "N"

    def __init__(self, cfg):
        self.cfg = cfg
        self.m = Matrices(cfg)
        self.order_of = {}

    def levels(self):
        return self.cfg["n_list"]

    def exps(self, N):
        e = {"T": ["T"], "T+L": ["T", "L"]}
        for p in PR_REPS:
            e[f"T+{p}"] = ["T", p]
            e[f"T+L+{p}"] = ["T", "L", p]
        if N > 0:
            e["T+A+H"] = ["T", "A", "H"]
            e["T+L+A+H"] = ["T", "L", "A", "H"]
        full = ["T", "L", "PRfuzzy"] + (["A", "H"] if N > 0 else [])
        e.update(leave_one_out(full, {"T": ["T"], "L": ["L"], "PR": ["PRfuzzy"], "AH": ["A", "H"]}))
        return e

    def setup(self, r, f):
        m = self.m
        if r not in self.order_of:
            d = m.data.sort_values(["ui", f"pos_r{r}"], kind="mergesort")
            self.order_of[r] = {int(u): g["fi"].to_numpy() for u, g in d.groupby("ui", sort=False)}
        ufold = m.uf[f"fold_r{r}"].to_numpy()
        te_f = film_folds(m.cnt, r, m.M) == f
        tr_users = np.flatnonzero(ufold != f)
        return SimpleNamespace(r=r, f=f, te_f=te_f, tr_f=~te_f, tr_users=tr_users,
                               te_users=np.flatnonzero((ufold == f) & m.eligible), order_of=self.order_of[r],
                               reps=fz.build_reps(m.prof_all, m.users[tr_users], RANDOM_STATE + 100 * r + f))

    def blocks(self, su, rule, N):
        m, F, cfg = self.m, self.m.F, self.cfg
        r, f, tr_f, te_f = su.r, su.f, su.tr_f, su.te_f
        L = m.Liked[rule]
        Rt, Lt = m.Rated[su.tr_users], L[su.tr_users]
        c = Rt.sum(axis=0).astype(np.float32)
        s_like = (Lt * Rt).sum(axis=0)
        g = float(s_like[tr_f].sum() / max(c[tr_f].sum(), 1))
        f_like = np.where(c > 0, s_like / np.maximum(c, 1), g).astype(np.float32)
        Tfeat = track_features(F, f_like, g, tr_f)

        def build(user_list, mode):
            parts = {"L": [], "T": [], "A": [], "H": [], **{k: [] for k in PR_REPS}}
            ys, us = [], []
            for u in user_list:
                ords = su.order_of[int(u)]
                ords = ords[tr_f[ords]]                 # the user's ratings of OLD films, in random order
                rev = ords[:N]
                if mode == "train":
                    cand = ords[N:]
                    S = cfg["train_per_user_b"]
                    if len(cand) > S:
                        rr = np.random.default_rng([RANDOM_STATE, r, f, N, 43, int(u)])
                        cand = np.sort(rr.choice(cand, S, replace=False))
                else:
                    cand = np.flatnonzero(m.Rated[u] & te_f)
                if len(cand) == 0:
                    continue
                ruf = np.zeros(m.M, dtype=np.float32)
                ruf[rev] = 1.0
                lu = L[u] * ruf
                h_u = (float(lu.sum()) + H_SMOOTH * g) / (float(ruf.sum()) + H_SMOOTH)
                parts["L"].append(m.Lfeat[cand])
                parts["T"].append(Tfeat[cand])
                parts["A"].append(affinity_block(F, cand, ruf, lu, h_u))
                parts["H"].append(np.full((len(cand), 1), h_u, dtype=np.float32))
                for key, rep in PR_REPS.items():
                    parts[key].append(np.repeat(su.reps[rep][u][None, :], len(cand), axis=0))
                ys.append(L[u, cand])
                us.append(np.full(len(cand), m.users[u]))
            return {k: np.vstack(v).astype(np.float32) for k, v in parts.items()}, np.concatenate(ys), np.concatenate(us)

        Btr, ytr, utr = build(su.tr_users.tolist(), "train")
        Bpo, ypo, upo = build(su.te_users.tolist(), "test")
        simple = {"S_fame_sum": Bpo["L"][:, 0] + Bpo["L"][:, 1], "S_track_dir": Bpo["T"][:, 1]}
        return Btr, ytr, utr, Bpo, ypo, upo, simple


# ----------------------------------------------------------------------------------------------
# the common loop
# ----------------------------------------------------------------------------------------------
def append_csv(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def run(scen, cfg):
    out = ensure_dir(OUT_DIR / f"14_final{cfg['tag']}" / scen.name)
    auc_path, cls_path = out / "partial_auc.csv", out / "partial_cls.csv"
    if cfg["force"]:
        for p in (auc_path, cls_path):
            if p.exists():
                p.unlink()
    done = set()
    if auc_path.exists():
        prev = pd.read_csv(auc_path, usecols=["repeat", "fold"]).drop_duplicates()
        done = {tuple(x) for x in prev.to_numpy().tolist()}
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    learners = cfg["learners"]
    t_start = time.time()
    total = len(cfg["repeats"]) * len(cfg["folds"])
    count = 0
    for r in cfg["repeats"]:
        for f in cfg["folds"]:
            count += 1
            if (r, f) in done:
                print(f"[{scen.name}] repeat {r} fold {f} already done, skipped", flush=True)
                continue
            t0 = time.time()
            su = scen.setup(r, f)
            rows_auc, rows_cls = [], []
            for rule in cfg["rules"]:
                for lv in scen.levels():
                    Btr, ytr, utr, Bpo, ypo, upo, simple = scen.blocks(su, rule, lv)
                    scores, thrs = {"B0_const": np.zeros(len(ypo)), **simple}, {}
                    seed = RANDOM_STATE + 1000 * r + 10 * f + lv
                    for exp, blocks in scen.exps(lv).items():
                        Xtr = np.hstack([Btr[k] for k in blocks])
                        Xpo = np.hstack([Bpo[k] for k in blocks])
                        for kind in learners:
                            if kind == "rf" and exp not in scen.rf_exps:
                                continue
                            p, thr = fit_predict_thr(kind, Xtr, ytr, utr, Xpo, cfg, seed)
                            scores[f"{exp}|{kind}"], thrs[f"{exp}|{kind}"] = p, thr
                    base = {"repeat": r, "fold": f, "rule": rule, "level": lv}
                    add_auc_rows(rows_auc, upo, ypo, scores, base)
                    add_confusion_rows(rows_cls, upo, ypo, scores, thrs, base)
                    print(f"  [{scen.name}] repeat {r} fold {f} rule {rule} {scen.level_key}={lv}: "
                          f"{len(ytr)} training rows, {len(ypo)} test rows, {time.time() - t0:.0f}s", flush=True)
            append_csv(cls_path, rows_cls)
            append_csv(auc_path, rows_auc)          # written last: it marks the fold as done
            print(f"[{scen.name}] repeat {r} fold {f} done in {time.time() - t0:.0f}s", flush=True)
            progress(count, total, t_start)
    print(f"[{scen.name}] files in {out}")


def main():
    cfg = parse_args(__doc__, final=True)
    require_sklearn()
    quiet()
    wanted = {"A": [ScenA], "B": [ScenB], "C": [ScenC], "ALL": [ScenA, ScenB, ScenC]}.get(cfg["scenario"])
    if wanted is None:
        sys.exit("--scenario must be A, B, C or all")
    for cls in wanted:
        run(cls(cfg), cfg)


if __name__ == "__main__":
    main()
