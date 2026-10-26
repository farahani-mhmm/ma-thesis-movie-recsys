"""Step 11: collect the headline numbers of steps 3-9 and export a small demo model for the presentation page.

No cross-validation is repeated here. The numbers are READ from the existing result files. The demo model is ONE
logistic regression (scenario "both new": the film has no ratings, the user is new) fitted on all users and films,
only to show how its inputs turn into a probability; its quality figures come from the cross-validated runs.

Output: outputs/11_demo/results.json
Run   : python 11_export_demo.py
"""
import json

import numpy as np
import pandas as pd

from config import LIKE_RULES, OUT_DIR, RANDOM_STATE, ensure_dir
from labfilms import build_films
from labutils import load_base

S_SMOOTH = 2.0


def val(path, rule, method, **key):
    f = pd.read_csv(OUT_DIR / path)
    f = f[(f["rule"] == rule) & (f["method"] == method)]
    for k, v in key.items():
        f = f[f[k] == v]
    if f.empty:
        return None
    r = f.iloc[0]
    return {"auc": round(float(r["mean_auc"]), 4), "lo": round(float(r["ci_low"]), 4), "hi": round(float(r["ci_high"]), 4)}


def pair(path, rule, a, b, **key):
    f = pd.read_csv(OUT_DIR / path)
    f = f[(f["rule"] == rule) & (f["A"] == a) & (f["B"] == b)]
    for k, v in key.items():
        f = f[f[k] == v]
    if f.empty:
        return None
    r = f.iloc[0]
    return {"diff": round(float(r["mean_difference_A_minus_B"]), 4), "lo": round(float(r["ci_low"]), 4),
            "hi": round(float(r["ci_high"]), 4)}


def collect():
    res = {}
    base = pd.read_csv(OUT_DIR / "03_baselines" / "per_user_auc.csv")
    for rule in LIKE_RULES:
        b = base[(base["rule"] == rule) & (base["method"] == "B1_pop_like")]
        pop = round(float(b["auc"].mean()), 4)
        r = {"popularity": pop}
        r["new_user"] = {
            "history": {int(n): val("06_report_tpu200/final_summary.csv", rule, "E1_hgb", N=n)
                        for n in (0, 5, 20, 50)},
            "control": {int(n): val("06_report_tpu200/final_summary.csv", rule, "G0h_hgb", N=n)
                        for n in (0, 5, 20, 50)},
            "profile_effect": pair("06_report_tpu200/final_pairs.csv", rule, "E1P_hgb", "E1_hgb", N=0),
        }
        s9 = "09_fuzzification/final_summary.csv"
        r["fuzzy_new_user"] = {m: val(s9, rule, f"{m}_hgb_b", N=0) for m in ("B", "R1", "R2", "C1", "F1", "F2", "F3")}
        s7 = "07_new_film_models_sonnet/final_summary.csv"
        s7g = "07_new_film_models/final_summary.csv"
        r["new_film"] = {
            "fame_only": val(s7, rule, "C1_logreg", M=0),
            "track_record": val(s7, rule, "C2_logreg", M=0),
            "track_plus_fame": val(s7, rule, "C3_logreg", M=0),
            "llm_effect_hgb": {"gpt4o": pair("07_new_film_models/final_pairs.csv", rule, "C3_hgb", "C2_hgb", M=0),
                               "sonnet": pair("07_new_film_models_sonnet/final_pairs.csv", rule, "C3_hgb", "C2_hgb", M=0)},
            "with_known_ratings": {int(m): val(s7, rule, "F3_logreg", M=m) for m in (5, 20, 50)},
        }
        s8 = "08_both_new_models_sonnet/final_summary.csv"
        p8 = "08_both_new_models_sonnet/final_pairs.csv"
        s8g = "08_both_new_models_strong/final_summary.csv"
        p8g = "08_both_new_models_strong/final_pairs.csv"
        r["both_new"] = {
            "track_only": val(s8, rule, "D0n_logreg", N=0),
            "track_plus_fame": val(s8, rule, "D0_logreg", N=0),
            "best_model_sonnet": val(s8, rule, "D2_hgb_c", N=0),
            "best_model_gpt4o": val(s8g, rule, "D2_hgb_c", N=0),
            "llm_effect_logreg": {"gpt4o": pair(p8g, rule, "D0_logreg", "D0n_logreg", N=0),
                                  "sonnet": pair(p8, rule, "D0_logreg", "D0n_logreg", N=0)},
            "llm_effect_hgb_c": {"gpt4o": pair(p8g, rule, "D0_hgb_c", "D0n_hgb_c", N=0),
                                 "sonnet": pair(p8, rule, "D0_hgb_c", "D0n_hgb_c", N=0)},
            "profile_effect_hgb_c": {"gpt4o": pair(p8g, rule, "D2n_hgb_c", "D0n_hgb_c", N=0),
                                     "sonnet": pair(p8, rule, "D2n_hgb_c", "D0n_hgb_c", N=0)},
        }
        s8f = "08_both_new_models_fuzzy/final_summary.csv"
        r["fuzzy_both_new"] = {m: val(s8f, rule, f"{m}_hgb_c", N=0) for m in ("D0n", "D2n", "ZR2n", "ZC1n", "ZF1n", "ZF2n", "ZF3n")}
        res[rule] = r
    return res


def demo_model():
    data, users_f, prof, movies = load_base({"llm": "sonnet", "llm_csv": None})
    users = np.sort(data["user_id"].unique())
    uidx = {u: i for i, u in enumerate(users.tolist())}
    film_ids = np.sort(data["movie_id"].unique())
    fidx = {m: i for i, m in enumerate(film_ids.tolist())}
    U, M = len(users), len(film_ids)
    ui, fi = data["user_id"].map(uidx).to_numpy(), data["movie_id"].map(fidx).to_numpy()
    F = build_films(movies, film_ids)
    mv = movies.drop_duplicates("movie_id").set_index("movie_id").loc[film_ids]
    rule = "gt7"
    liked = data[f"liked_{rule}"].to_numpy(dtype=np.float32)
    cnt = np.bincount(fi, minlength=M).astype(np.float32)
    f_like = np.bincount(fi, weights=liked, minlength=M) / np.maximum(cnt, 1)
    g = float(liked.mean())
    f_like = np.where(cnt > 0, f_like, g).astype(np.float32)
    dir_n, act_n = F.dir_share.sum(axis=1), F.act_share.sum(axis=1)
    dir_like = (F.dir_share @ f_like + S_SMOOTH * g) / (dir_n + S_SMOOTH)
    act_like = (F.act_share @ f_like + S_SMOOTH * g) / (act_n + S_SMOOTH)
    X_film = np.column_stack([dir_n, dir_like, act_n, act_like, F.fame_dir, F.fame_act]).astype(np.float32)
    names = ["dir_n", "dir_like", "act_n", "act_like", "fame_dir", "fame_act"]

    rng = np.random.default_rng(RANDOM_STATE)
    keep = np.zeros(len(data), dtype=bool)
    for u in range(U):
        idx = np.flatnonzero(ui == u)
        keep[rng.choice(idx, min(len(idx), 80), replace=False)] = True
    X, y = X_film[fi[keep]], liked[keep]
    mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-9
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1.0, max_iter=300).fit((X - mean) / scale, y)
    p = lr.predict_proba((X_film - mean) / scale)[:, 1]

    title = mv["original_title"] if "original_title" in mv.columns else mv.iloc[:, 0]
    order = np.argsort(p)
    pick = [order[int(q * (M - 1))] for q in np.linspace(0.02, 0.98, 12)]
    films = []
    for j in pick:
        films.append({"title": str(title.iloc[j]), "directors": str(mv["directors"].iloc[j]).split("|")[0][:40],
                      **{n: round(float(X_film[j, k]), 3) for k, n in enumerate(names)},
                      "observed_like_share": round(float(f_like[j]), 3), "n_ratings": int(cnt[j])})
    ranges = {n: [round(float(np.percentile(X_film[:, k], 2)), 2), round(float(np.percentile(X_film[:, k], 98)), 2)]
              for k, n in enumerate(names)}
    return {"label_rule": rule, "features": names, "mean": [round(float(v), 5) for v in mean],
            "scale": [round(float(v), 5) for v in scale], "coef": [round(float(v), 5) for v in lr.coef_[0]],
            "intercept": round(float(lr.intercept_[0]), 5), "base_rate": round(g, 4), "ranges": ranges,
            "example_films": films}


def main():
    out = ensure_dir(OUT_DIR / "11_demo")
    res = {"status": "Sonnet 5.5 fame columns (director mean, mean of the first 3 billed actors); GPT-4o shown for comparison",
           "results": collect(), "demo_model": demo_model()}
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written: {out / 'results.json'}")


if __name__ == "__main__":
    main()
