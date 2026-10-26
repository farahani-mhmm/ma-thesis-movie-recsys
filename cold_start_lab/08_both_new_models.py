"""Step 8: scenario C, BOTH NEW: a new user AND a new film (the hardest, most realistic cold start).

Nobody has rated the film in the training data, and the user is new: all we know about the user is the 12 profile
columns and (for N > 0) N revealed ratings, which are ratings of OLD (training) films. Nothing is known about the
user's taste for the new film except what can be inferred from those.

Split (per repeat, per fold f): the eligible users of user-fold f are the test users, the films of film-fold f are the
test films (balanced by popularity). Training = other users x other films. A test user is judged on their ratings of the
test films, one AUC per user (95% bootstrap interval over users, both label rules).

Feature groups for a pair (user u, new film m):
    PR  the 12 raw profile columns (standardised on training users)
    L   the two GPT-4o fame scores of the film
    T   track record from TRAINING films that share a director / an actor with m (data-derived content popularity)
    A+U affinity and overall like ratio of the user, computed only from the N revealed ratings

Experiments (n = no LLM columns):
    D0n = T          D0 = L+T         D2n = PR+T        D2 = PR+L+T
    D3n = T+A+U      D3 = L+T+A+U     D4n = PR+T+A+U    D4 = PR+L+T+A+U     (D3 and D4 only for N > 0)
Simple scores without a learner: B0_const, S_fame_sum, S_track_dir.
Effects: LLM = D0-D0n, D2-D2n, D3-D3n, D4-D4n;  profile = D2n-D0n, D2-D0, D4-D3;  revealed ratings = D3-D0.

Needs: pandas, numpy, scikit-learn (matplotlib optional).
Run  : python 08_both_new_models.py --quick   then   python 08_both_new_models.py
Output (outputs/08_both_new_models[_quick]/): per_user_auc.csv, final_summary.csv, final_pairs.csv, final_report.txt, curves_<rule>.png
"""
import importlib
import json
import sys
import time

import numpy as np
import pandas as pd

from config import LIKE_RULES, N_FOLDS, N_GRID, OUT_DIR, PROFILE_COLS, RANDOM_STATE, ensure_dir
from labfilms import build_films
from labutils import (PRIMARY_LEARNER, add_auc_rows, fit_predict, load_base, parse_args, progress, quiet,
                      require_sklearn, summarize)

S_SMOOTH = 2.0
H_SMOOTH = 5.0
EXPS = {"D0n": ["T"], "D0": ["L", "T"], "D2n": ["PR", "T"], "D2": ["PR", "L", "T"],
        "D3n": ["T", "A", "U"], "D3": ["L", "T", "A", "U"], "D4n": ["PR", "T", "A", "U"], "D4": ["PR", "L", "T", "A", "U"]}
# only with --interact: X* = explicit products (profile column) x (film column), so even a linear model can use the profile
EXPS_X = {"D5n": ["PR", "T", "XT"], "D5": ["PR", "L", "T", "XT", "XL"],
          "D6n": ["PR", "T", "A", "U", "XT"], "D6": ["PR", "L", "T", "A", "U", "XT", "XL"]}


FUZ = ["R2", "C1", "F1", "F2", "F3"]
# only with --fuzzy: Z<rep>n = representation + T, Z<rep> = representation + L + T   (profile representations of step 9)
EXPS_Z = {**{f"Z{x}n": [f"Z{x}", "T"] for x in FUZ}, **{f"Z{x}": [f"Z{x}", "L", "T"] for x in FUZ}}


def film_folds(cnt, r, n_films):
    rng = np.random.default_rng([RANDOM_STATE, r, 77])
    order = np.argsort(-cnt, kind="stable")
    fold = np.zeros(n_films, dtype=int)
    for s in range(0, n_films, N_FOLDS):
        blk = order[s:s + N_FOLDS]
        fold[blk] = rng.permutation(N_FOLDS)[:len(blk)]
    return fold


def make_pairs(learners):
    pairs = []
    for k in learners:
        pairs += [(f"D0_{k}", f"D0n_{k}"), (f"D2_{k}", f"D2n_{k}"), (f"D3_{k}", f"D3n_{k}"), (f"D4_{k}", f"D4n_{k}"),
                  (f"D2n_{k}", f"D0n_{k}"), (f"D2_{k}", f"D0_{k}"), (f"D4_{k}", f"D3_{k}"), (f"D4n_{k}", f"D3n_{k}"),
                  (f"D3_{k}", f"D0_{k}"), (f"D3n_{k}", f"D0n_{k}"), (f"D0n_{k}", "S_track_dir"), (f"D0_{k}", "B0_const"),
                  (f"D4_{k}", "B0_const"),
                  (f"D5n_{k}", f"D2n_{k}"), (f"D5_{k}", f"D2_{k}"), (f"D5_{k}", f"D5n_{k}"), (f"D5n_{k}", f"D0n_{k}"),
                  (f"D5_{k}", f"D0_{k}"), (f"D6n_{k}", f"D4n_{k}"), (f"D6_{k}", f"D4_{k}"), (f"D6_{k}", f"D6n_{k}")]
        for x in FUZ:
            pairs += [(f"Z{x}n_{k}", f"D2n_{k}"), (f"Z{x}_{k}", f"D2_{k}"), (f"Z{x}n_{k}", f"D0n_{k}"),
                      (f"Z{x}_{k}", f"D0_{k}"), (f"Z{x}_{k}", f"Z{x}n_{k}")]
        for x in ["F1", "F2", "F3"]:
            pairs += [(f"Z{x}n_{k}", f"ZC1n_{k}"), (f"Z{x}_{k}", f"ZC1_{k}"), (f"Z{x}n_{k}", f"ZR2n_{k}"),
                      (f"Z{x}_{k}", f"ZR2_{k}")]
    pairs +=[("S_fame_sum", "B0_const"), ("S_track_dir", "B0_const")]
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
    from sklearn.preprocessing import StandardScaler
    n_levels = cfg["n_list"]
    if n_levels == list(N_GRID):
        n_levels = [0, 5, 20]
    elif n_levels == [0, 5, 50]:
        n_levels = [0, 5]
    out = ensure_dir(OUT_DIR / f"08_both_new_models{cfg['tag']}")
    if (out / "per_user_auc.csv").exists() and not cfg["force"]:
        print(f"Outputs already exist in {out}. Use --force to recompute.")
        return

    data, users_f, prof, movies = load_base(cfg)
    users = np.sort(data["user_id"].unique())
    uidx = {u: i for i, u in enumerate(users.tolist())}
    film_ids = np.sort(data["movie_id"].unique())
    fidx = {m: i for i, m in enumerate(film_ids.tolist())}
    U, M = len(users), len(film_ids)
    data["ui"] = data["user_id"].map(uidx).to_numpy()
    data["fi"] = data["movie_id"].map(fidx).to_numpy()
    Rated = np.zeros((U, M), dtype=bool)
    Rated[data["ui"].to_numpy(), data["fi"].to_numpy()] = True
    Liked = {}
    for rule in LIKE_RULES:
        Liked[rule] = np.zeros((U, M), dtype=np.float32)
        Liked[rule][data["ui"].to_numpy(), data["fi"].to_numpy()] = data[f"liked_{rule}"].to_numpy(dtype=np.float32)
    F = build_films(movies, film_ids)
    cnt = Rated.sum(axis=0)
    uf = users_f.set_index("user_id").loc[users]
    eligible_u = uf["eligible"].to_numpy()
    prof_all = prof.set_index("user_id").loc[users, PROFILE_COLS]
    Lfeat = np.column_stack([F.fame_dir, F.fame_act]).astype(np.float32)
    learners = cfg["learners"] or (["logreg", PRIMARY_LEARNER] + (["rf"] if cfg["with_rf"] else []))
    exps = {**EXPS, **(EXPS_X if cfg["interact"] else {}), **(EXPS_Z if cfg["fuzzy"] else {})}
    if cfg["fuzzy"]:
        fz = importlib.import_module("09_fuzzification")
    Lz = (Lfeat - Lfeat.mean(axis=0)) / (Lfeat.std(axis=0) + 1e-6)

    rows, diag = [], []
    t_start = time.time()
    folds_done, folds_total = 0, len(cfg["repeats"]) * len(cfg["folds"])
    for r in cfg["repeats"]:
        ufold = uf[f"fold_r{r}"].to_numpy()
        ffold = film_folds(cnt, r, M)
        d = data.sort_values(["ui", f"pos_r{r}"], kind="mergesort")
        order_of = {int(u): g["fi"].to_numpy() for u, g in d.groupby("ui", sort=False)}
        for f in cfg["folds"]:
            t0 = time.time()
            te_f = ffold == f
            tr_f = ~te_f
            tr_users = np.flatnonzero(ufold != f)
            te_users = np.flatnonzero((ufold == f) & eligible_u)
            scaler = StandardScaler().fit(prof_all.iloc[tr_users])
            prof_std = scaler.transform(prof_all).astype(np.float32)
            reps = fz.build_reps(prof_all, users[tr_users], RANDOM_STATE + 100 * r + f) if cfg["fuzzy"] else None
            for rule in cfg["rules"]:
                L = Liked[rule]
                Rt, Lt = Rated[tr_users], L[tr_users]
                c = Rt.sum(axis=0).astype(np.float32)
                s_like = (Lt * Rt).sum(axis=0)
                g = float(s_like[tr_f].sum() / max(c[tr_f].sum(), 1))
                f_like = np.where(c > 0, s_like / np.maximum(c, 1), g).astype(np.float32)
                trf = tr_f.astype(np.float32)
                Wd, Wa = F.dir_share * trf[None, :], F.act_share * trf[None, :]
                dir_n, act_n = Wd.sum(axis=1), Wa.sum(axis=1)
                dir_like = (Wd @ f_like + S_SMOOTH * g) / (dir_n + S_SMOOTH)
                act_like = (Wa @ f_like + S_SMOOTH * g) / (act_n + S_SMOOTH)
                Tfeat = np.column_stack([dir_n, dir_like, act_n, act_like]).astype(np.float32)
                Tc = (Tfeat[:, [1, 3]] - g)
                Tc = (Tc / (Tc.std(axis=0) + 1e-6)).astype(np.float32)       # centred, unit-scale track-record likes

                for N in n_levels:
                    def build(user_list, mode):
                        parts = {"PR": [], "L": [], "T": [], "A": [], "U": [], "XT": [], "XL": [], **{f"Z{x}": [] for x in FUZ}}
                        ys, us = [], []
                        for u in user_list:
                            ords = order_of[int(u)]
                            ords = ords[tr_f[ords]]                 # the user's ratings of OLD films, in random order
                            rev = ords[:N]
                            if mode == "train":
                                cand = ords[N:]
                                S = cfg["train_per_user_b"]
                                if len(cand) > S:
                                    rr = np.random.default_rng([RANDOM_STATE, r, f, N, 43, int(u)])
                                    cand = np.sort(rr.choice(cand, S, replace=False))
                            else:
                                cand = np.flatnonzero(Rated[u] & te_f)
                            if len(cand) == 0:
                                continue
                            ruf = np.zeros(M, dtype=np.float32)
                            ruf[rev] = 1.0
                            lu = L[u] * ruf
                            n_u = float(ruf.sum())
                            h_u = (float(lu.sum()) + H_SMOOTH * g) / (n_u + H_SMOOTH)
                            Dm, Am = F.dir_share[cand], F.act_share[cand]
                            a_dir_cnt, a_act_cnt = Dm @ ruf, Am @ ruf
                            a_dir_like = (Dm @ lu + S_SMOOTH * h_u) / (a_dir_cnt + S_SMOOTH)
                            a_act_like = (Am @ lu + S_SMOOTH * h_u) / (a_act_cnt + S_SMOOTH)
                            liked_mask = lu > 0.5
                            pref_d = float(F.fame_dir[liked_mask].mean()) if liked_mask.any() else F.mean_fame_dir
                            pref_a = float(F.fame_act[liked_mask].mean()) if liked_mask.any() else F.mean_fame_act
                            A = np.column_stack([a_dir_cnt, a_dir_like, a_act_cnt, a_act_like,
                                                 np.abs(F.fame_dir[cand] - pref_d), np.abs(F.fame_act[cand] - pref_a)])
                            parts["PR"].append(np.repeat(prof_std[u][None, :], len(cand), axis=0))
                            parts["L"].append(Lfeat[cand])
                            for x in FUZ:
                                parts[f"Z{x}"].append(np.repeat(reps[x][u][None, :], len(cand), axis=0) if cfg["fuzzy"]
                                                      else np.zeros((len(cand), 0), dtype=np.float32))
                            if cfg["interact"]:
                                pu = prof_std[u]
                                parts["XT"].append((pu[:, None, None] * Tc[cand][None, :, :]).transpose(1, 0, 2)
                                                   .reshape(len(cand), -1))
                                parts["XL"].append((pu[:, None, None] * Lz[cand][None, :, :]).transpose(1, 0, 2)
                                                   .reshape(len(cand), -1))
                            else:
                                parts["XT"].append(np.zeros((len(cand), 0), dtype=np.float32))
                                parts["XL"].append(np.zeros((len(cand), 0), dtype=np.float32))
                            parts["T"].append(Tfeat[cand])
                            parts["A"].append(A.astype(np.float32))
                            parts["U"].append(np.full((len(cand), 1), h_u, dtype=np.float32))
                            ys.append(L[u, cand])
                            us.append(np.full(len(cand), u))
                        blocks = {k: np.vstack(v).astype(np.float32) for k, v in parts.items()}
                        return blocks, np.concatenate(ys), np.concatenate(us)

                    Btr, ytr, _ = build(tr_users.tolist(), "train")
                    Bpo, ypo, upo = build(te_users.tolist(), "test")
                    seed = RANDOM_STATE + 1000 * r + 10 * f + N
                    scores = {"B0_const": np.zeros(len(ypo)), "S_fame_sum": Bpo["L"][:, 0] + Bpo["L"][:, 1],
                              "S_track_dir": Bpo["T"][:, 1]}
                    for exp, groups in exps.items():
                        if N == 0 and exp in ("D3n", "D3", "D4n", "D4", "D6n", "D6"):
                            continue
                        Xtr = np.hstack([Btr[k] for k in groups])
                        Xpo = np.hstack([Bpo[k] for k in groups])
                        for kind in learners:
                            scores[f"{exp}_{kind}"] = fit_predict(kind, Xtr, ytr, Xpo, cfg, seed)
                    add_auc_rows(rows, upo, ypo, scores, {"repeat": r, "fold": f, "rule": rule, "N": N})
                    diag.append({"repeat": r, "fold": f, "rule": rule, "N": N, "test_users": len(te_users),
                                 "test_films": int(te_f.sum()), "train_rows": len(ytr), "test_rows": len(ypo)})
            print(f"repeat {r} fold {f} done in {time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)", flush=True)
            folds_done += 1
            progress(folds_done, folds_total, t_start)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_user_auc.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "diagnostics.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({**cfg, "n_levels": n_levels}, indent=2), encoding="utf-8")

    methods = set(df["method"].unique())
    pairs = [(a, b) for a, b in make_pairs(learners) if a in methods and b in methods]
    summary, diffs = summarize(df, ["rule", "N"], pairs)
    summary.to_csv(out / "final_summary.csv", index=False)
    diffs["verdict"] = diffs.apply(verdict, axis=1)
    diffs.to_csv(out / "final_pairs.csv", index=False)

    L_ = PRIMARY_LEARNER if PRIMARY_LEARNER in learners else learners[0]
    show = ["B0_const", "S_fame_sum", "S_track_dir"] + [f"{e}_{k}" for k in learners for e in exps]
    show = [m for m in show if m in methods]
    ns = sorted(df["N"].unique().tolist())
    lines = ["STEP 8 - BOTH NEW (new user and new film)", f"revealed ratings N: {ns}",
             f"learners: {learners}; interactions: {cfg['interact']}; primary learner: {PRIMARY_LEARNER} "
             "(decided before seeing results)", ""]

    def questions(k):
        return [(f"[{k}] Does the LLM add to the track record when the user is new? (D0 - D0n)", f"D0_{k}", f"D0n_{k}"),
                (f"[{k}] Does the raw profile help? Without LLM columns (D2n - D0n)", f"D2n_{k}", f"D0n_{k}"),
                (f"[{k}] Does the raw profile help? With LLM columns (D2 - D0)", f"D2_{k}", f"D0_{k}"),
                (f"[{k}] Does the LLM add on top of the profile? (D2 - D2n)", f"D2_{k}", f"D2n_{k}"),
                (f"[{k}] Do the N revealed ratings help? (D3 - D0)", f"D3_{k}", f"D0_{k}"),
                (f"[{k}] Does the LLM add once the revealed ratings are used? (D3 - D3n)", f"D3_{k}", f"D3n_{k}"),
                (f"[{k}] Does the profile still help once ratings are revealed? (D4 - D3)", f"D4_{k}", f"D3_{k}"),
                (f"[{k}] Is the best full model better than nothing? (D4 - B0_const)", f"D4_{k}", "B0_const"),
                (f"[{k}] Do explicit profile x film products help? No LLM (D5n - D2n)", f"D5n_{k}", f"D2n_{k}"),
                (f"[{k}] Do explicit profile x film products help? With LLM (D5 - D2)", f"D5_{k}", f"D2_{k}"),
                (f"[{k}] With products, does the profile help? No LLM (D5n - D0n)", f"D5n_{k}", f"D0n_{k}"),
                (f"[{k}] With products, does the profile help? With LLM (D5 - D0)", f"D5_{k}", f"D0_{k}"),
                (f"[{k}] With products, does the LLM add? (D5 - D5n)", f"D5_{k}", f"D5n_{k}"),
                (f"[{k}] Products plus revealed ratings (D6 - D4)", f"D6_{k}", f"D4_{k}"),
                (f"[{k}] With products and revealed ratings, does the LLM add? (D6 - D6n)", f"D6_{k}", f"D6n_{k}")] + [
                (f"[{k}] Z{x}n - D2n (representation {x} vs raw profile, no LLM)", f"Z{x}n_{k}", f"D2n_{k}") for x in FUZ] + [
                (f"[{k}] Z{x}n - D0n (representation {x} vs no profile, no LLM)", f"Z{x}n_{k}", f"D0n_{k}") for x in FUZ] + [
                (f"[{k}] Z{x} - D2 (representation {x} vs raw profile, with LLM)", f"Z{x}_{k}", f"D2_{k}") for x in FUZ] + [
                (f"[{k}] Z{x} - D0 (representation {x} vs no profile, with LLM)", f"Z{x}_{k}", f"D0_{k}") for x in FUZ] + [
                (f"[{k}] Z{x} - ZC1 (fuzzy vs crisp, with LLM)", f"Z{x}_{k}", f"ZC1_{k}") for x in ["F1", "F2", "F3"]] + [
                (f"[{k}] Z{x}n - ZC1n (fuzzy vs crisp, no LLM)", f"Z{x}n_{k}", f"ZC1n_{k}") for x in ["F1", "F2", "F3"]] + [
                (f"[{k}] Z{x}n - ZR2n (fuzzy vs log/circular, no LLM)", f"Z{x}n_{k}", f"ZR2n_{k}") for x in ["F1", "F2", "F3"]]
    for rule in sorted(summary["rule"].unique()):
        sub = summary[summary["rule"] == rule]
        lines.append(f"=== label rule: {rule} (about {int(sub['users'].median())} test users) ===")
        lines.append("mean per-user AUC [95% interval]")
        lines.append("  " + "method".ljust(18) + "".join(f"N={n}".rjust(24) for n in ns))
        for m in show:
            cells = []
            for n in ns:
                s = sub[(sub["method"] == m) & (sub["N"] == n)]
                cells.append(f"{s['mean_auc'].iloc[0]:.4f} [{s['ci_low'].iloc[0]:.3f},{s['ci_high'].iloc[0]:.3f}]".rjust(24)
                             if len(s) else "-".rjust(24))
            lines.append("  " + m.ljust(18) + "".join(cells))
        lines.append("")
        for k in learners:
            for question, a, b in questions(k):
                d = diffs[(diffs["rule"] == rule) & (diffs["A"] == a) & (diffs["B"] == b)].sort_values("N")
                if d.empty:
                    continue
                lines.append(question)
                for _, x in d.iterrows():
                    lines.append(f"    N={int(x['N']):>3}: {x['mean_difference_A_minus_B']:+.4f}  "
                                 f"[{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]  {x['verdict']}")
                lines.append("")
    text = "\n".join(lines)
    (out / "final_report.txt").write_text(text, encoding="utf-8")
    print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        curve = [m for m in ["B0_const", "S_track_dir", f"D0n_{L_}", f"D0_{L_}", f"D2n_{L_}", f"D2_{L_}", f"D3_{L_}", f"D4_{L_}"]
                 if m in methods]
        for rule in sorted(summary["rule"].unique()):
            sub = summary[summary["rule"] == rule]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for m in curve:
                s = sub[sub["method"] == m].sort_values("N")
                if s.empty:
                    continue
                x = [ns.index(v) for v in s["N"]]
                ax.plot(x, s["mean_auc"], marker="o", label=m)
                ax.fill_between(x, s["ci_low"], s["ci_high"], alpha=0.15)
            ax.set_xticks(range(len(ns)))
            ax.set_xticklabels(ns)
            ax.set_xlabel("N = revealed ratings of the new user (on old films)")
            ax.set_ylabel("mean per-user AUC")
            ax.set_title(f"Both new (new user, new film), label rule {rule}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"curves_{rule}.png", dpi=150)
            plt.close(fig)
        print("curves written")
    except ImportError:
        print("matplotlib is not installed: tables were written, curves were skipped")
    print(f"\nfiles written to: {out}")


if __name__ == "__main__":
    main()
