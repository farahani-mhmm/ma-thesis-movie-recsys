"""Step 4: E0, the raw Twitter profile (12 columns, no fuzzification yet), compared with the baselines.

Why two versions (see the step-3 report): per-user AUC only looks at the order of films for ONE user, so a
model that sees only the user's profile gives every film the same score and gets exactly 0.5. A profile can
only help if it is combined with information about the films.

    E0a  one model per film (this is what the old paper did): "does a user with this profile like THIS film?"
         trained on the training users who rated that film. Films with too few likes or dislikes fall back to
         the film's like ratio.   learners: logistic regression (and random forest with --with-rf)
    G0   control: ONE global model that sees only film statistics (like ratio, mean rating, log count) taken
         from training users, no profile.      learners: logistic regression, hist gradient boosting
    E0b  the same global model plus the 12 profile columns.     learners: logistic regression, hist gradient boosting

    The effect of the profile is E0b - G0 (same learner).

Film statistics of TRAINING rows are built out-of-fold over users (no row sees its own rating).
Reference from step 3 is B1_pop_like (film like ratio only).

Needs: pandas, numpy, scikit-learn.
Run  : python 04_profile_models.py            (all repeats and folds)
       python 04_profile_models.py --quick    (smoke test, a few minutes)
Output (outputs/04_profile_models[_quick]/): per_user_auc.csv, diagnostics.csv, mean_auc.txt
"""
import json
import sys
import time

import numpy as np
import pandas as pd

from config import EVAL_POOL_START, OUT_DIR, PROFILE_COLS, RANDOM_STATE, ensure_dir
from labutils import (PRIMARY_LEARNER, add_auc_rows, fit_predict, film_stats, load_base, map_film_features,
                      oof_film_features, parse_args, progress, quiet, require_sklearn)

MIN_CLASS = 5       # a per-film model needs at least this many likes and dislikes, else the film falls back


def perfilm_scores(train, pool, Xtr_p, Xpo_p, ytr, fallback, glob_like, make_model, cap_films=None):
    """Score every pool row with the model of its own film. Returns (scores, n_films, n_fallback)."""
    scores = np.zeros(len(pool), dtype=float)
    tr_idx = train.groupby("movie_id").indices
    po_idx = pool.groupby("movie_id").indices
    n_films = n_fallback = 0
    for k, (mid, pidx) in enumerate(po_idx.items()):
        n_films += 1
        base = float(fallback.get(mid, glob_like))
        tidx = tr_idx.get(mid)
        if tidx is None:
            scores[pidx] = base
            n_fallback += 1
            continue
        y = ytr[tidx]
        n1 = int(y.sum())
        if n1 < MIN_CLASS or (len(y) - n1) < MIN_CLASS:
            scores[pidx] = base
            n_fallback += 1
            continue
        m = make_model()
        m.fit(Xtr_p[tidx], y)
        scores[pidx] = m.predict_proba(Xpo_p[pidx])[:, 1]
    return scores, n_films, n_fallback


def main():
    cfg = parse_args(__doc__)
    require_sklearn()
    quiet()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    out = ensure_dir(OUT_DIR / f"04_profile_models{cfg['tag_quick']}")
    if (out / "per_user_auc.csv").exists() and not cfg["force"]:
        print(f"Outputs already exist in {out}. Use --force to recompute.")
        return

    data, users_f, prof, movies = load_base()
    data = data.merge(prof, on="user_id", validate="many_to_one")
    eligible = users_f[users_f["eligible"]]

    rows, diag = [], []
    t_start = time.time()
    folds_done, folds_total = 0, len(cfg["repeats"]) * len(cfg["folds"])
    for r in cfg["repeats"]:
        data["fold"] = data["user_id"].map(users_f.set_index("user_id")[f"fold_r{r}"])
        pos_col = f"pos_r{r}"
        for f in cfg["folds"]:
            t0 = time.time()
            train = data[data["fold"] != f].reset_index(drop=True)
            test_users = set(eligible.loc[eligible[f"fold_r{r}"] == f, "user_id"])
            if test_users & set(train["user_id"]):
                sys.exit(f"LEAK: a test user is also a training user (repeat {r}, fold {f})")
            pool = data[(data["fold"] == f) & data["user_id"].isin(test_users)
                        & (data[pos_col] >= EVAL_POOL_START)].reset_index(drop=True)
            if cfg["perfilm_cap"]:
                keep = sorted(pool["movie_id"].unique())[:cfg["perfilm_cap"]]
                pool = pool[pool["movie_id"].isin(keep)].reset_index(drop=True)

            scaler = StandardScaler().fit(train.drop_duplicates("user_id")[PROFILE_COLS])
            Xtr_p = scaler.transform(train[PROFILE_COLS]).astype(np.float32)
            Xpo_p = scaler.transform(pool[PROFILE_COLS]).astype(np.float32)

            for rule in cfg["rules"]:
                lab = f"liked_{rule}"
                ytr = train[lab].to_numpy()
                ypo = pool[lab].to_numpy()
                glob = (float(train[lab].mean()), float(train["rate"].mean()))
                stats_all = film_stats(train, lab)
                Fpo = map_film_features(pool, stats_all, glob)
                Ftr, _ = oof_film_features(train, lab, (r, f))
                seed = RANDOM_STATE + 100 * r + f
                scores = {}

                # ---- E0a: one model per film ----
                s, nf, nfb = perfilm_scores(train, pool, Xtr_p, Xpo_p, ytr, stats_all["f_like"], glob[0],
                                            lambda: LogisticRegression(C=0.1, max_iter=200))
                scores["E0a_logreg"] = s
                diag.append({"repeat": r, "fold": f, "rule": rule, "method": "E0a_logreg", "films": nf, "fallback": nfb})
                if cfg["with_rf"]:
                    s, nf, nfb = perfilm_scores(train, pool, Xtr_p, Xpo_p, ytr, stats_all["f_like"], glob[0],
                                                lambda: RandomForestClassifier(n_estimators=60, min_samples_leaf=5,
                                                                               random_state=seed, n_jobs=1))
                    scores["E0a_rf"] = s
                    diag.append({"repeat": r, "fold": f, "rule": rule, "method": "E0a_rf", "films": nf, "fallback": nfb})

                # ---- G0 (films only) and E0b (films + profile): one global model each ----
                Xb_tr = np.hstack([Xtr_p, Ftr])
                Xb_po = np.hstack([Xpo_p, Fpo])
                for kind in ("logreg", PRIMARY_LEARNER):
                    scores[f"G0_{kind}"] = fit_predict(kind, Ftr, ytr, Fpo, cfg, seed)
                    scores[f"E0b_{kind}"] = fit_predict(kind, Xb_tr, ytr, Xb_po, cfg, seed)

                add_auc_rows(rows, pool["user_id"].to_numpy(), ypo, scores,
                             {"repeat": r, "fold": f, "rule": rule})
            print(f"repeat {r} fold {f} done in {time.time() - t0:.0f}s (total {time.time() - t_start:.0f}s)", flush=True)
            folds_done += 1
            progress(folds_done, folds_total, t_start)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_user_auc.csv", index=False)
    pd.DataFrame(diag).to_csv(out / "diagnostics.csv", index=False)
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    lines = ["STEP 4 - E0 (raw profile) - mean per-user AUC (confidence intervals come in step 6)", ""]
    ref = OUT_DIR / "03_baselines" / "per_user_auc.csv"
    for rule in cfg["rules"]:
        lines.append(f"=== label rule: {rule} ===")
        sub = df[df["rule"] == rule].groupby("method")["auc"].mean()
        for m, v in sub.items():
            lines.append(f"  {m:<14} {v:.4f}")
        if ref.exists():
            b = pd.read_csv(ref)
            b = b[(b["rule"] == rule) & (b["method"] == "B1_pop_like") & (b["repeat"].isin(cfg["repeats"]))]
            if len(b):
                lines.append(f"  (reference) B1_pop_like from step 3 on all folds: {b['auc'].mean():.4f}")
        lines.append("")
    d = pd.DataFrame(diag)
    if len(d):
        g = d.groupby("method")[["films", "fallback"]].sum()
        for m, rr in g.iterrows():
            lines.append(f"per-film models {m}: {int(rr['fallback'])} of {int(rr['films'])} film-fold cases used the "
                         f"popularity fallback (too few likes or dislikes)")
    text = "\n".join(lines)
    (out / "mean_auc.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\nfiles written to: {out}")


if __name__ == "__main__":
    main()
