"""
STEP 19 - benchmark against the two CSP papers, inside OUR user-held-out protocol (scenario A, new user, N = 0).

Two published methods are re-implemented from their papers and run on exactly the same training users, test users and
test rows as step 14 (user-grouped folds, 3 repeats x 5 folds, test rows = positions >= 50 of each test user):

  CSP23  = the model of the dataset paper (Information 2023, 14, 19):
           content part   - for each of 10 binary user features and its value, the average rating of every film among the
                            training users with that value; a test user gets the mean over his own 10 feature values
           affinity part  - training users who share >= 8 of the 10 binary features with the test user; their average
                            normalised rating per film
           total          - mean of both parts (normalised to 0-1)
           Simplifications (the paper does not give them): the binary versions of counts and of the account year are cut at
           the MEDIAN of the training users; the movie-feature projection step of the paper is left out; where a film has no
           high-affinity rater the content part alone is used.
  RF2020 = the model of Herce-Zelaya et al. (Information Sciences 536, 2020): one model per film, 12 raw Twitter-profile
           columns, default RandomForest (100 trees) and DecisionTree, label "rating >= 7" (the paper's rule; we also run gt7).
           The paper's films have > 480 ratings; no film of our cleaned data has that many, so the 15 MOST RATED films are used.

Both are compared with the trivial guesses (film popularity / majority per film / all yes / all no) and with our hgb_c FULL model
at N = 0. Ranking metrics follow the CSP 2023 paper (top-5 average real rating, improvement over the user's average, MRR) plus
our per-user AUC. The paper does not define its "accuracy 60%" and gives only a loose description of its MRR; the MRR here is
the reciprocal rank of the first liked film in the predicted ranking (our reading, say so in the text).

Run :  python 19_benchmark_csp.py            (about 20-40 minutes)  |  python 19_benchmark_csp.py --report   (only rebuild the report)
Output: outputs/19_benchmark/ranking_users.csv, film15_rows.csv, benchmark_report.txt
"""
import importlib
import sys
import time

import numpy as np
import pandas as pd

REPORT_ONLY = "--report" in sys.argv
if REPORT_ONLY:
    sys.argv.remove("--report")
sys.argv = [sys.argv[0], "--scenario", "A", "--n-list", "0", "--rules", "gt7,ge7", "--learners", "hgb_c"]

from config import OUT_DIR, PROFILE_COLS, RANDOM_STATE, ensure_dir  # noqa: E402
import labutils as lu  # noqa: E402

OUT = ensure_dir(OUT_DIR / "19_benchmark")
RANK_CSV, FILM_CSV, REPORT = OUT / "ranking_users.csv", OUT / "film15_rows.csv", OUT / "benchmark_report.txt"
TOP_FILMS, TOPN, AFFINITY_MIN = 15, 5, 8


def binary_features(prof, train_users):
    """The 10 binary user features of the CSP 2023 paper; thresholds = median of the training users."""
    tr = prof.loc[train_users]
    f = pd.DataFrame(index=prof.index)
    f["early_bird"] = prof["early_bird"].astype(int)
    f["night_owl"] = prof["night_owl"].astype(int)
    f["geo_enabled"] = prof["geo_enabled"].astype(int)
    f["week_tweeter"] = prof["week_tweeter"].astype(int)
    f["weekend_tweeter"] = prof["weekend_tweeter"].astype(int)
    f["early_user"] = (prof["account_creation_year"] <= tr["account_creation_year"].median()).astype(int)
    for col, name in (("friends_count", "many_friends"), ("followers_count", "many_followers"),
                      ("favourites_count", "many_favourites"), ("number_of_tweets", "heavy_tweeter")):
        f[name] = (prof[col] > tr[col].median()).astype(int)
    return f.to_numpy()


def mean_ignore_nan(a, axis=0):
    ok = ~np.isnan(a)
    s = np.where(ok, a, 0.0).sum(axis=axis)
    c = ok.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def csp23_scores(R, F, train_rows, test_row):
    """Content, affinity and total predictions (0-1 scale) of one test user for every film."""
    fv = F[test_row]
    Rtr, Ftr = R[train_rows], F[train_rows]
    pop = mean_ignore_nan(Rtr) / 10.0
    glob = np.nanmean(Rtr) / 10.0
    pop = np.where(np.isnan(pop), glob, pop)
    tabs = np.stack([mean_ignore_nan(Rtr[Ftr[:, k] == fv[k]]) for k in range(F.shape[1])])
    content = mean_ignore_nan(tabs) / 10.0
    near = (Ftr == fv[None, :]).sum(axis=1) >= AFFINITY_MIN
    aff = mean_ignore_nan(Rtr[near]) / 10.0 if near.any() else np.full(R.shape[1], np.nan)
    content = np.where(np.isnan(content), pop, content)
    total = np.where(np.isnan(aff), content, (content + aff) / 2.0)
    aff_f = np.where(np.isnan(aff), pop, aff)
    return {"CSP23_total": total, "CSP23_content": content, "CSP23_affinity": aff_f, "POP_mean": pop}


def user_metrics(score, rate, y):
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    top = order[:TOPN]
    pos = np.flatnonzero(y[order] == 1)
    return {"auc": lu.auc_binary(score, y), "top5_avg": float(rate[top].mean()), "user_avg": float(rate.mean()),
            "p5": float(y[top].mean()), "mrr": float(1.0 / (pos[0] + 1)) if len(pos) else np.nan}


def run():
    m14 = importlib.import_module("14_final_protocol")
    cfg = lu.parse_args(__doc__, final=True)
    lu.require_sklearn()
    lu.quiet()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    scen = m14.ScenA(cfg)
    prof = scen.prof
    users_all = prof.index.to_numpy()
    uidx = {int(u): i for i, u in enumerate(users_all)}
    data = scen.data
    R = np.full((len(users_all), scen.M), np.nan)
    R[data["user_id"].map(uidx).to_numpy(), data["fi"].to_numpy()] = data["rate"].to_numpy(dtype=float)
    counts = data.groupby("fi").size().sort_values(ascending=False)
    top_films = counts.index[:TOP_FILMS].to_numpy()
    print(f"15 most rated films: ratings {counts.iloc[0]} ... {counts.iloc[TOP_FILMS - 1]}", flush=True)
    profX = prof[PROFILE_COLS].astype(float)
    liked_tab = {k: np.full_like(R, np.nan) for k in lu.LIKE_RULES}
    for k in lu.LIKE_RULES:
        liked_tab[k][data["user_id"].map(uidx).to_numpy(), data["fi"].to_numpy()] = data[f"liked_{k}"].to_numpy(dtype=float)
    done = set()
    if RANK_CSV.exists() and FILM_CSV.exists():
        d = pd.read_csv(RANK_CSV, usecols=["repeat", "fold"]).drop_duplicates()
        done = {tuple(x) for x in d.to_numpy().tolist()}
    t_start = time.time()
    count = 0
    for r in cfg["repeats"]:
        for f in cfg["folds"]:
            count += 1
            if (r, f) in done:
                print(f"repeat {r} fold {f} already done", flush=True)
                continue
            t0 = time.time()
            su = scen.setup(r, f)
            train_rows = np.array([uidx[int(u)] for u in su.train_users])
            F = binary_features(prof, su.train_users)
            rank_rows, film_rows = [], []
            for rule in cfg["rules"]:
                Btr, ytr, utr, Bpo, ypo, upo, simple = scen.blocks(su, rule, 0)
                blocks = scen.exps(0)["FULL"]
                Xtr = np.hstack([Btr[k] for k in blocks])
                Xpo = np.hstack([Bpo[k] for k in blocks])
                hgb, hgb_thr = lu.fit_predict_thr("hgb_c", Xtr, ytr, utr, Xpo, cfg, RANDOM_STATE + 1000 * r + 10 * f)
                films_row = np.concatenate([su.UA[int(u)][0][scen.pool:] for u in su.test_users])
                users_row = np.concatenate([np.full(len(su.UA[int(u)][0]) - scen.pool, int(u)) for u in su.test_users])
                if len(films_row) != len(ypo) or not np.array_equal(users_row, upo):
                    sys.exit("test rows do not line up with the protocol rows")
                rate_row = R[[uidx[int(u)] for u in users_row], films_row]
                if not np.array_equal((rate_row > 7) if rule == "gt7" else (rate_row >= 7), ypo.astype(bool)):
                    sys.exit("rating and liked label do not line up")
                csp = {name: np.zeros(len(ypo)) for name in ("CSP23_total", "CSP23_content", "CSP23_affinity", "POP_mean")}
                for u in su.test_users:
                    mask = users_row == int(u)
                    sc = csp23_scores(R, F, train_rows, uidx[int(u)])
                    for name, arr in sc.items():
                        csp[name][mask] = arr[films_row[mask]]
                score_dict = {**csp, "S_pop": simple["S_pop"], "HGB_FULL": hgb}
                order = np.argsort(upo, kind="mergesort")
                u_sorted = upo[order]
                bounds = np.flatnonzero(np.r_[True, u_sorted[1:] != u_sorted[:-1], True])
                for a, b in zip(bounds[:-1], bounds[1:]):
                    idx = order[a:b]
                    for name, sc in score_dict.items():
                        rank_rows.append({"repeat": r, "fold": f, "rule": rule, "user_id": int(u_sorted[a]), "method": name,
                                          **user_metrics(sc[idx], rate_row[idx], ypo[idx])})
                # ---- per-film models of the 2020 paper on the 15 most rated films
                for j in top_films:
                    tr_users = [int(u) for u in su.train_users if not np.isnan(R[uidx[int(u)], j])]
                    mask = films_row == j
                    if len(tr_users) < 20 or mask.sum() == 0:
                        continue
                    Xj = profX.loc[tr_users].to_numpy()
                    yj = liked_tab[rule][[uidx[u] for u in tr_users], j]
                    Xt = profX.loc[users_row[mask]].to_numpy()
                    maj = int(yj.mean() >= 0.5)
                    if len(np.unique(yj)) < 2:
                        rf_p = np.full(mask.sum(), maj)
                        dt_p = rf_p.copy()
                    else:
                        rf = RandomForestClassifier(random_state=RANDOM_STATE + r * 100 + f).fit(Xj, yj)
                        dt = DecisionTreeClassifier(random_state=RANDOM_STATE + r * 100 + f).fit(Xj, yj)
                        rf_p, dt_p = rf.predict(Xt).astype(int), dt.predict(Xt).astype(int)
                    film_rows.append(pd.DataFrame({
                        "repeat": r, "fold": f, "rule": rule, "user_id": users_row[mask], "film": int(scen.film_ids[j]),
                        "y": ypo[mask].astype(int), "RF2020": rf_p, "DT2020": dt_p, "MAJ": maj,
                        "HGB_FULL": (hgb[mask] > hgb_thr).astype(int)}))
            pd.DataFrame(rank_rows).to_csv(RANK_CSV, mode="a", header=not RANK_CSV.exists(), index=False)
            pd.concat(film_rows).to_csv(FILM_CSV, mode="a", header=not FILM_CSV.exists(), index=False)
            print(f"repeat {r} fold {f} done in {time.time() - t0:.0f}s", flush=True)
            lu.progress(count, len(cfg["repeats"]) * len(cfg["folds"]), t_start)


def boot_ci(v, n_boot=2000, seed=RANDOM_STATE):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return v.mean(), lo, hi


def report():
    rk = pd.read_csv(RANK_CSV)
    fm = pd.read_csv(FILM_CSV)
    out = []
    methods = ["POP_mean", "S_pop", "CSP23_content", "CSP23_affinity", "CSP23_total", "HGB_FULL"]
    for rule in ("gt7", "ge7"):
        d = rk[rk.rule == rule]
        out.append("=" * 110)
        out.append(f"PART 1  new user (A), N=0, rule {rule}: ranking of each test user's own rated films, {d.user_id.nunique()} users, top-{TOPN}")
        out.append(f"{'method':16s}{'AUC [95% interval]':26s}{'P@5':>8s}{'top-5 real avg':>16s}{'user avg':>10s}{'improvement':>13s}{'MRR*':>8s}")
        per_user = d.groupby(["user_id", "method"])[["auc", "top5_avg", "user_avg", "p5", "mrr"]].mean().reset_index()
        for m in methods:
            u = per_user[per_user.method == m]
            a, lo, hi = boot_ci(u.auc)
            imp = u.top5_avg.mean() / u.user_avg.mean() - 1.0
            out.append(f"{m:16s}{a:7.4f} [{lo:.3f},{hi:.3f}]       {u.p5.mean():8.3f}{u.top5_avg.mean():16.2f}{u.user_avg.mean():10.2f}"
                       f"{100 * imp:12.1f}%{u.mrr.mean():8.3f}")
        out.append("paired AUC differences (A minus B), 95% bootstrap interval over users:")
        wide = per_user.pivot(index="user_id", columns="method", values="auc")
        for a_, b_ in (("CSP23_total", "POP_mean"), ("CSP23_content", "POP_mean"), ("CSP23_total", "CSP23_content"),
                       ("HGB_FULL", "POP_mean"), ("HGB_FULL", "CSP23_total")):
            diff = (wide[a_] - wide[b_]).dropna()
            m_, lo, hi = boot_ci(diff)
            out.append(f"  {a_} - {b_}: {m_:+.4f} [{lo:+.4f},{hi:+.4f}]  {'clear' if lo > 0 or hi < 0 else 'no clear difference'}")
    out.append("*MRR = reciprocal rank of the first liked film in the predicted ranking (our reading; the paper's wording is loose).")
    out.append("")
    methods2 = ["RF2020", "DT2020", "MAJ", "HGB_FULL", "ALL_YES", "ALL_NO"]
    for rule in ("ge7", "gt7"):
        d = fm[fm.rule == rule].copy()
        d["ALL_YES"], d["ALL_NO"] = 1, 0
        out.append("=" * 110)
        out.append(f"PART 2  per-film models of the 2020 paper, 15 most rated films, rule {rule}"
                   f"{' (the paper rule)' if rule == 'ge7' else ''}; metrics computed per film on pooled test rows, then averaged over films")
        out.append(f"{'method':10s}{'accuracy':>10s}{'F1 liked':>10s}{'F1 weighted':>13s}{'precision':>11s}{'RMSE':>8s}   95% interval of accuracy / F1 liked (over users)")
        users = np.sort(d.user_id.unique())
        uid_pos = {u: i for i, u in enumerate(users)}
        d["ui"] = d.user_id.map(uid_pos)
        rng = np.random.default_rng(RANDOM_STATE)
        W = np.stack([np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users)) for _ in range(1000)])
        for m in methods2:
            tp = {}
            for fl, g in d.groupby("film"):
                p = g[m].to_numpy()
                y = g.y.to_numpy()
                ui = g.ui.to_numpy()
                tp[fl] = np.stack([np.bincount(ui, weights=((p == 1) & (y == 1)).astype(float), minlength=len(users)),
                                   np.bincount(ui, weights=((p == 1) & (y == 0)).astype(float), minlength=len(users)),
                                   np.bincount(ui, weights=((p == 0) & (y == 1)).astype(float), minlength=len(users)),
                                   np.bincount(ui, weights=((p == 0) & (y == 0)).astype(float), minlength=len(users))])

            def metrics(weights):
                accs, f1s, f1w, precs = [], [], [], []
                for fl, c in tp.items():
                    TP, FP, FN, TN = (c * weights[None, :]).sum(axis=1)
                    n = TP + FP + FN + TN
                    if n == 0:
                        continue
                    prec = TP / (TP + FP) if TP + FP else 0.0
                    rec = TP / (TP + FN) if TP + FN else 0.0
                    f1p = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                    prec0 = TN / (TN + FN) if TN + FN else 0.0
                    rec0 = TN / (TN + FP) if TN + FP else 0.0
                    f10 = 2 * prec0 * rec0 / (prec0 + rec0) if prec0 + rec0 else 0.0
                    accs.append((TP + TN) / n)
                    f1s.append(f1p)
                    precs.append(prec)
                    f1w.append(((TP + FN) * f1p + (TN + FP) * f10) / n)
                return np.mean(accs), np.mean(f1s), np.mean(f1w), np.mean(precs)

            base = metrics(np.ones(len(users)))
            boots = np.array([metrics(w) for w in W[:300]])
            lo_a, hi_a = np.percentile(boots[:, 0], [2.5, 97.5])
            lo_f, hi_f = np.percentile(boots[:, 1], [2.5, 97.5])
            out.append(f"{m:10s}{base[0]:10.3f}{base[1]:10.3f}{base[2]:13.3f}{base[3]:11.3f}{np.sqrt(1 - base[0]):8.3f}"
                       f"   acc [{lo_a:.3f},{hi_a:.3f}]  F1 [{lo_f:.3f},{hi_f:.3f}]")
        out.append("RMSE of 0/1 labels = sqrt(1 - accuracy). MAJ = per film, the class most training users gave. HGB_FULL = our hgb_c with the F1-optimal threshold learned on training users.")
    text = "\n".join(out)
    REPORT.write_text(text, encoding="utf-8")
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))


if __name__ == "__main__":
    if not REPORT_ONLY:
        run()
    report()
