"""Step 7: scenario B, a NEW FILM. The user is known, the film has (almost) no ratings.

This is where the LLM columns can matter: for a film nobody has rated, film popularity is unavailable, and the
only film information left is its content (director, actors, the GPT-4o fame scores).

Split (per repeat, per fold f): users of user-fold f (eligible ones) are the test users, films of film-fold f are the
test films. Film folds are balanced by popularity. Models are trained only on TRAINING users x TRAINING films.
A test user is judged on their ratings of the test films; every test film is new for every test user.
Per-user AUC, 95% bootstrap interval over users, both label rules, as in the other steps.

M = how many ratings of the new film are already known (taken from random TRAINING users, never from test users;
M = 0 means a completely new film).

Feature groups for a pair (user u, new film m):
    L   the two GPT-4o fame scores of the film
    T   track record from TRAINING films: how many other training films share a director / an actor with m and how
        well they were liked (data-derived content popularity that does not need the LLM)
    A   affinity of the user for films that share a director / actor with m (from the user's ratings of training
        films) and the distance between the film's fame and the fame of films the user liked
    U   the user's overall like ratio (training films)
    FH  film history: like ratio and log count of the M known ratings of the film

Experiments:  C1=L   C2=T   C3=L+T   C6=T+A+U (no LLM)   C4=L+T+A+U   F1=FH   F2=FH+L+T   F3n=FH+T+A+U   F3=FH+L+T+A+U
Simple scores without a learner: S_fame_dir, S_fame_act, S_fame_sum, S_track_dir, S_fh_like (M>0), B0_const.
The effect of the LLM is C3-C2, C4-C6 (no film ratings) and F3-F3n (with M known ratings).

Needs: pandas, numpy, scikit-learn (matplotlib optional).
Run  : python 07_new_film_models.py --quick   (smoke test)   then   python 07_new_film_models.py
Output (outputs/07_new_film_models[_quick]/): per_user_auc.csv, final_summary.csv, final_pairs.csv,
        final_report.txt, curves_<rule>.png
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from config import LIKE_RULES, N_FOLDS, OUT_DIR, RANDOM_STATE, ensure_dir
from labfilms import build_films
from labutils import (PRIMARY_LEARNER, add_auc_rows, fit_predict, load_base, parse_args, progress, quiet,
                      require_sklearn, summarize)

S_SMOOTH = 2.0       # smoothing of track-record, affinity and film-history like ratios
H_SMOOTH = 5.0       # smoothing of the user's overall like ratio
EXPS = {"C1": ["L"], "C2": ["T"], "C3": ["L", "T"], "C6": ["T", "A", "U"], "C4": ["L", "T", "A", "U"],
        "F1": ["FH"], "F2": ["FH", "L", "T"], "F3n": ["FH", "T", "A", "U"], "F3": ["FH", "L", "T", "A", "U"]}


def film_folds(cnt, r, n_films):
    """Film folds balanced by popularity: films sorted by rating count, each block of 5 spread over the 5 folds."""
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
        pairs += [(f"C1_{k}", "B0_const"), (f"C2_{k}", "B0_const"), (f"C1_{k}", f"C2_{k}"), (f"C3_{k}", f"C2_{k}"),
                  (f"C3_{k}", f"C1_{k}"), (f"C6_{k}", f"C2_{k}"), (f"C4_{k}", f"C6_{k}"), (f"C4_{k}", f"C3_{k}"),
                  (f"F1_{k}", "S_fh_like"), (f"F2_{k}", f"F1_{k}"), (f"F3_{k}", f"F3n_{k}"), (f"F3_{k}", f"F2_{k}"),
                  (f"F3_{k}", "S_fh_like"), (f"C1_{k}", "S_fame_sum")]
    pairs += [("S_fame_sum", "B0_const"), ("S_fame_dir", "B0_const"), ("S_fame_act", "B0_const"),
              ("S_track_dir", "B0_const"), ("S_fame_sum", "S_track_dir")]
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
    out = ensure_dir(OUT_DIR / f"07_new_film_models{cfg['tag']}")
    if (out / "per_user_auc.csv").exists() and not cfg["force"]:
        print(f"Outputs already exist in {out}. Use --force to recompute.")
        return

    data, users_f, prof, movies = load_base(cfg)
    users = np.sort(data["user_id"].unique())
    uidx = {u: i for i, u in enumerate(users.tolist())}
    film_ids = np.sort(data["movie_id"].unique())
    fidx = {m: i for i, m in enumerate(film_ids.tolist())}
    U, M = len(users), len(film_ids)
    ui = data["user_id"].map(uidx).to_numpy()
    fi = data["movie_id"].map(fidx).to_numpy()
    Rated = np.zeros((U, M), dtype=bool)
    Rated[ui, fi] = True
    Liked = {}
    for rule in LIKE_RULES:
        Liked[rule] = np.zeros((U, M), dtype=np.float32)
        Liked[rule][ui, fi] = data[f"liked_{rule}"].to_numpy(dtype=np.float32)
    F = build_films(movies, film_ids)
    cnt = Rated.sum(axis=0)
    uf = users_f.set_index("user_id").loc[users]
    eligible_u = uf["eligible"].to_numpy()
    Lfeat = np.column_stack([F.fame_dir, F.fame_act]).astype(np.float32)
    learners = ["logreg", PRIMARY_LEARNER] + (["rf"] if cfg["with_rf"] else [])

    rows, diag = [], []
    t_start = time.time()
    folds_done, folds_total = 0, len(cfg["repeats"]) * len(cfg["folds"])
    for r in cfg["repeats"]:
        ufold = uf[f"fold_r{r}"].to_numpy()
        ffold = film_folds(cnt, r, M)
        for f in cfg["folds"]:
            t0 = time.time()
            te_f = ffold == f
            tr_f = ~te_f
            tr_users = np.flatnonzero(ufold != f)
            te_users = np.flatnonzero((ufold == f) & eligible_u)
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

                for Mlev in cfg["m_list"]:
                    rng = np.random.default_rng([RANDOM_STATE, r, f, Mlev, 31])
                    fh_like = np.full(M, g, dtype=np.float32)
                    fh_logc = np.zeros(M, dtype=np.float32)
                    InR = np.zeros((U, M), dtype=bool)
                    if Mlev > 0:
                        for m in range(M):
                            cand = tr_users[Rated[tr_users, m]]
                            if len(cand) == 0:
                                continue
                            pick = rng.choice(cand, min(Mlev, len(cand)), replace=False)
                            InR[pick, m] = True
                            fh_like[m] = (L[pick, m].sum() + S_SMOOTH * g) / (len(pick) + S_SMOOTH)
                            fh_logc[m] = np.log1p(len(pick))
                    FH = np.column_stack([fh_like, fh_logc]).astype(np.float32)

                    def build(user_list, mode):
                        parts = {"L": [], "T": [], "A": [], "U": [], "FH": []}
                        ys, us = [], []
                        for u in user_list:
                            ru = Rated[u] & tr_f
                            if mode == "train":
                                cand = np.flatnonzero(ru & ~InR[u]) if Mlev > 0 else np.flatnonzero(ru)
                                S = cfg["train_per_user_b"]
                                if len(cand) > S:
                                    rr = np.random.default_rng([RANDOM_STATE, r, f, Mlev, 41, int(u)])
                                    cand = np.sort(rr.choice(cand, S, replace=False))
                            else:
                                cand = np.flatnonzero(Rated[u] & te_f)
                            if len(cand) == 0:
                                continue
                            ruf = ru.astype(np.float32)
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
                            parts["L"].append(Lfeat[cand])
                            parts["T"].append(Tfeat[cand])
                            parts["A"].append(A.astype(np.float32))
                            parts["U"].append(np.full((len(cand), 1), h_u, dtype=np.float32))
                            parts["FH"].append(FH[cand])
                            ys.append(L[u, cand])
                            us.append(np.full(len(cand), u))
                        blocks = {k: np.vstack(v).astype(np.float32) for k, v in parts.items()}
                        return blocks, np.concatenate(ys), np.concatenate(us)

                    Btr, ytr, _ = build(tr_users.tolist(), "train")
                    Bpo, ypo, upo = build(te_users.tolist(), "test")
                    seed = RANDOM_STATE + 1000 * r + 10 * f + Mlev
                    scores = {"B0_const": np.zeros(len(ypo)), "S_fame_dir": Bpo["L"][:, 0], "S_fame_act": Bpo["L"][:, 1],
                              "S_fame_sum": Bpo["L"][:, 0] + Bpo["L"][:, 1], "S_track_dir": Bpo["T"][:, 1]}
                    if Mlev > 0:
                        scores["S_fh_like"] = Bpo["FH"][:, 0]
                    for exp, groups in EXPS.items():
                        if exp.startswith("F") and Mlev == 0:
                            continue
                        Xtr = np.hstack([Btr[k] for k in groups])
                        Xpo = np.hstack([Bpo[k] for k in groups])
                        for kind in learners:
                            scores[f"{exp}_{kind}"] = fit_predict(kind, Xtr, ytr, Xpo, cfg, seed)
                    add_auc_rows(rows, upo, ypo, scores, {"repeat": r, "fold": f, "rule": rule, "M": Mlev})
                    diag.append({"repeat": r, "fold": f, "rule": rule, "M": Mlev, "test_users": len(te_users),
                                 "test_films": int(te_f.sum()), "train_rows": len(ytr), "test_rows": len(ypo)})
            print(f"repeat {r} fold {f} done in {time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)", flush=True)
            folds_done += 1
            progress(folds_done, folds_total, t_start)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_user_auc.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "diagnostics.csv", index=False)
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    methods = set(df["method"].unique())
    pairs = [(a, b) for a, b in make_pairs(learners) if a in methods and b in methods]
    summary, diffs = summarize(df, ["rule", "M"], pairs)
    summary.to_csv(out / "final_summary.csv", index=False)
    diffs["verdict"] = diffs.apply(verdict, axis=1)
    diffs.to_csv(out / "final_pairs.csv", index=False)

    L_ = PRIMARY_LEARNER
    show = ["B0_const", "S_fame_dir", "S_fame_act", "S_fame_sum", "S_track_dir", "S_fh_like", f"C1_{L_}", f"C2_{L_}",
            f"C3_{L_}", f"C6_{L_}", f"C4_{L_}", f"F1_{L_}", f"F2_{L_}", f"F3n_{L_}", f"F3_{L_}",
            "C3_logreg", "C4_logreg", "F3_logreg"]
    show = [m for m in show if m in methods]
    ms = sorted(df["M"].unique().tolist())
    lines = ["STEP 7 - NEW FILM (scenario B)", f"known ratings of the new film M: {ms}",
             f"primary learner: {L_} (decided before seeing results)", ""]
    qs = [("Is the LLM fame score alone better than nothing? (S_fame_sum - B0_const)", "S_fame_sum", "B0_const"),
          ("Is the data-derived track record alone better than nothing? (S_track_dir - B0_const)", "S_track_dir", "B0_const"),
          ("LLM fame vs data track record (S_fame_sum - S_track_dir)", "S_fame_sum", "S_track_dir"),
          ("Does the LLM add to the track record? (C3 - C2)", f"C3_{L_}", f"C2_{L_}"),
          ("Does the user's affinity help? (C6 - C2)", f"C6_{L_}", f"C2_{L_}"),
          ("Does the LLM still add once user affinity is used? (C4 - C6)", f"C4_{L_}", f"C6_{L_}"),
          ("With M known ratings: do they beat the plain known like ratio? (F1 - S_fh_like)", f"F1_{L_}", "S_fh_like"),
          ("With M known ratings: does the LLM still add? (F3 - F3n)", f"F3_{L_}", f"F3n_{L_}")]
    for rule in sorted(summary["rule"].unique()):
        sub = summary[summary["rule"] == rule]
        lines.append(f"=== label rule: {rule} (about {int(sub['users'].median())} test users) ===")
        lines.append("mean per-user AUC [95% interval]")
        lines.append("  " + "method".ljust(14) + "".join(f"M={m}".rjust(24) for m in ms))
        for m in show:
            cells = []
            for mm in ms:
                s = sub[(sub["method"] == m) & (sub["M"] == mm)]
                cells.append(f"{s['mean_auc'].iloc[0]:.4f} [{s['ci_low'].iloc[0]:.3f},{s['ci_high'].iloc[0]:.3f}]".rjust(24)
                             if len(s) else "-".rjust(24))
            lines.append("  " + m.ljust(14) + "".join(cells))
        lines.append("")
        for question, a, b in qs:
            d = diffs[(diffs["rule"] == rule) & (diffs["A"] == a) & (diffs["B"] == b)].sort_values("M")
            if d.empty:
                continue
            lines.append(question)
            for _, x in d.iterrows():
                lines.append(f"    M={int(x['M']):>3}: {x['mean_difference_A_minus_B']:+.4f}  "
                             f"[{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]  {x['verdict']}")
            lines.append("")
    text = "\n".join(lines)
    (out / "final_report.txt").write_text(text, encoding="utf-8")
    print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        curve = [m for m in ["B0_const", "S_fame_sum", "S_track_dir", "S_fh_like", f"C3_{L_}", f"C4_{L_}", f"F2_{L_}",
                             f"F3n_{L_}", f"F3_{L_}"] if m in methods]
        for rule in sorted(summary["rule"].unique()):
            sub = summary[summary["rule"] == rule]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for m in curve:
                s = sub[sub["method"] == m].sort_values("M")
                if s.empty:
                    continue
                x = [ms.index(v) for v in s["M"]]
                ax.plot(x, s["mean_auc"], marker="o", label=m)
                ax.fill_between(x, s["ci_low"], s["ci_high"], alpha=0.15)
            ax.set_xticks(range(len(ms)))
            ax.set_xticklabels(ms)
            ax.set_xlabel("M = known ratings of the new film")
            ax.set_ylabel("mean per-user AUC")
            ax.set_title(f"New-film curves, label rule {rule}")
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
