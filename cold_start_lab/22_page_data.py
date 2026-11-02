"""
STEP 22 - numbers for the meeting page, read from the result files (nothing typed by hand).
Output: outputs/22_page/page_data.json  (used by the page builder)
Run   : python 22_page_data.py
"""
import json

import numpy as np
import pandas as pd

from config import OUT_DIR, RANDOM_STATE, ensure_dir

OUT = ensure_dir(OUT_DIR / "22_page")
L = "hgb_c"


def rd(folder, sc, kind="summary_auc"):
    d = pd.read_csv(OUT_DIR / folder / sc / f"{kind}.csv")
    return d[d.rule == "gt7"]


def auc(folder, sc, level, method, with_ci=False):
    d = rd(folder, sc)
    q = d[(d.level == level) & ((d.method == f"{method}|{L}") | (d.method == method))]
    if q.empty:
        return None
    r = q.iloc[0]
    return [round(float(r.mean_auc), 4), round(float(r.ci_low), 4), round(float(r.ci_high), 4)] if with_ci else round(float(r.mean_auc), 4)


def pair(folder, sc, level, a, b):
    d = rd(folder, sc, "pairs_auc")
    q = d[(d.level == level) & (d.A == f"{a}|{L}") & (d.B == f"{b}|{L}")]
    if q.empty:
        return None
    r = q.iloc[0]
    return [round(float(r.mean_difference_A_minus_B), 4), round(float(r.ci_low), 4), round(float(r.ci_high), 4), str(r.verdict)]


def boot(v, n=2000, seed=RANDOM_STATE):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return [round(float(v.mean()), 4), round(float(lo), 4), round(float(hi), 4)]


D = {}
SC = {"A": "A_new_user", "B": "B_new_film", "C": "C_both_new"}

# ---------------------------------------------------------------- TMDB
D["tmdb"] = {s: {m: auc("14_final_tmdb", SC[s], 0, m) for m in ("FULL", "FULLK", "FULLKP")} for s in "ABC"}
D["tmdb"]["A"]["base"] = auc("14_final_tmdb", SC["A"], 0, "S_pop")
D["tmdb"]["B"]["base"] = auc("14_final_tmdb", SC["B"], 0, "S_track_dir")
D["tmdb"]["C"]["base"] = auc("14_final_tmdb", SC["C"], 0, "S_track_dir")
D["tmdb"]["pairs"] = {s: {"K": pair("14_final_tmdb", SC[s], 0, "FULLK", "FULL"), "PUB": pair("14_final_tmdb", SC[s], 0, "FULLKP", "FULLK")} for s in "ABC"}

# ---------------------------------------------------------------- ablation (FULLK minus a block)
blocks = {"A": ["P", "H", "L", "A", "PR", "K"], "B": ["T", "L", "AH", "K"], "C": ["T", "L", "PR", "AH", "K"]}
D["abl"] = {s: {b: pair("14_final_ablk", SC[s], 0, "FULLK", f"FULLK-{b}") for b in blocks[s]} for s in "ABC"}
D["abl"]["B5"] = {b: pair("14_final_ablk", SC["B"], 5, "FULLK", f"FULLK-{b}") for b in ["T", "L", "AH", "FH", "K"]}
D["fullk"] = {s: {str(lv): auc("14_final_ablk", SC[s], lv, "FULLK") for lv in ([0, 5] if s == "B" else [0, 5, 20, 50])} for s in "ABC"}

# ---------------------------------------------------------------- fuzzy x LLM (scenario C, base run)
d = pd.read_csv(OUT_DIR / "14_final" / "C_both_new" / "partial_auc.csv")
d = d[d.rule == "gt7"]


def user_auc(m, lv):
    return d[(d.level == lv) & (d.method == f"{m}|{L}")].groupby("user_id").auc.mean()


def fx(a, b, lv):
    z = (user_auc(a, lv) - user_auc(b, lv)).dropna().to_numpy()
    r = boot(z)
    return r + ["better" if r[1] > 0 else ("worse" if r[2] < 0 else "no clear difference")]


cells = {"00": "T+PRraw", "01": "T+L+PRraw", "10": "T+PRfuzzy", "11": "T+L+PRfuzzy"}
D["fxl"] = {"levels": [0, 20],
            "mean": {k: {str(lv): round(float(user_auc(m, lv).mean()), 4) for lv in (0, 20)} for k, m in cells.items()},
            "eff": {"fuzzy_noLLM": {str(lv): fx("T+PRfuzzy", "T+PRraw", lv) for lv in (0, 20)},
                    "fuzzy_LLM": {str(lv): fx("T+L+PRfuzzy", "T+L+PRraw", lv) for lv in (0, 20)},
                    "llm_noFuzzy": {str(lv): fx("T+L+PRraw", "T+PRraw", lv) for lv in (0, 20)},
                    "llm_fuzzy": {str(lv): fx("T+L+PRfuzzy", "T+PRfuzzy", lv) for lv in (0, 20)}}}
# delete LLM (final model with TMDB and base model without TMDB)
D["delL"] = {"FULLK": {s: {str(lv): pair("14_final_ablk", SC[s], lv, "FULLK", "FULLK-L") for lv in ([0, 5] if s == "B" else [0, 20])} for s in "ABC"},
             "FULL": {s: {str(lv): pair("14_final", SC[s], lv, "FULL", "FULL-L") for lv in ([0, 5] if s == "B" else [0, 20])} for s in "ABC"}}

# ---------------------------------------------------------------- sparsity
TAGS = [(100, "_sp100"), (50, "_sp050"), (20, "_sp020"), (10, "_sp010"), (5, "_sp005")]
spm = {"A": ["P", "P+K"], "B": ["T", "T+L", "T+K", "FULL", "FULLK"], "C": ["T", "T+L", "T+K", "FULL", "FULLK"]}
D["sparse"] = {"keep": [k for k, _ in TAGS],
               "auc": {s: {m: [auc("14_final" + t, SC[s], 0, m) for _, t in TAGS] for m in spm[s]} for s in "ABC"},
               "effK": {s: [pair("14_final" + t, SC[s], 0, "FULLK" if s != "A" else "P+K", "FULL" if s != "A" else "P")[0] for _, t in TAGS] for s in "ABC"},
               "effL": {s: [pair("14_final" + t, SC[s], 0, "T+L", "T")[0] for _, t in TAGS] for s in "BC"}}

# ---------------------------------------------------------------- number of known ratings
nl = [0, 5, 20, 50, 100, 200, 400]
D["curve"] = {"N": nl,
              "A_BK": [auc("14_final_taste", SC["A"], n, "BK") for n in nl], "A_pop": auc("14_final_taste", SC["A"], 0, "S_pop"),
              "C_BK": [auc("14_final_taste", SC["C"], n, "BK") for n in nl], "C_base": auc("14_final_taste", SC["C"], 0, "S_track_dir"),
              "B_M": [0, 5], "B_FULLK": [auc("14_final_ablk", SC["B"], m, "FULLK") for m in (0, 5)],
              "B_FULL": [auc("14_final", SC["B"], m, "FULL") for m in (0, 5)], "B_T": [auc("14_final", SC["B"], m, "T") for m in (0, 5)],
              "A_P50": [auc("14_final", SC["A"], n, "P") for n in (0, 5, 20, 50)], "A_FULLK50": [auc("14_final_ablk", SC["A"], n, "FULLK") for n in (0, 5, 20, 50)]}

# ---------------------------------------------------------------- benchmark
rk = pd.read_csv(OUT_DIR / "19_benchmark" / "ranking_users.csv")
rk = rk[rk.rule == "gt7"]
per = rk.groupby(["user_id", "method"])[["auc", "top5_avg", "user_avg"]].mean().reset_index()
bm = {}
for m in ("POP_mean", "CSP23_content", "CSP23_affinity", "CSP23_total", "HGB_FULL"):
    u = per[per.method == m]
    bm[m] = {"auc": boot(u.auc), "top5": round(float(u.top5_avg.mean()), 2), "imp": round(float(u.top5_avg.mean() / u.user_avg.mean() - 1) * 100, 1)}
wide = per.pivot(index="user_id", columns="method", values="auc")
bm["diff_csp_pop"] = boot((wide["CSP23_total"] - wide["POP_mean"]).dropna())
bm["diff_ours_pop"] = boot((wide["HGB_FULL"] - wide["POP_mean"]).dropna())
bm["users"] = int(wide.shape[0])
fm = pd.read_csv(OUT_DIR / "19_benchmark" / "film15_rows.csv")
fm = fm[fm.rule == "ge7"]
rf = {}
for m in ("RF2020", "MAJ"):
    accs, f1s = [], []
    for _, g in fm.groupby("film"):
        p, y = g[m].to_numpy(), g.y.to_numpy()
        tp = ((p == 1) & (y == 1)).sum()
        fp = ((p == 1) & (y == 0)).sum()
        fn = ((p == 0) & (y == 1)).sum()
        accs.append((p == y).mean())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    rf[m] = [round(float(np.mean(accs)), 3), round(float(np.mean(f1s)), 3)]
bm["rf2020_ge7"] = rf
D["bench"] = bm

# ---------------------------------------------------------------- F1 and accuracy
cl = {}
for s in "ABC":
    d_ = pd.read_csv(OUT_DIR / "14_final_ablk" / SC[s] / "summary_cls.csv")
    d_ = d_[(d_.rule == "gt7") & (d_.level == 0)]
    row = {}
    for m, key in ((f"FULLK|{L}", "model"), ("ALL_NO", "no"), ("ALL_YES", "yes")):
        r = d_[d_.method == m].iloc[0]
        row[key] = {"acc": round(float(r.accuracy), 3), "f1": round(float(r.f1_liked), 3), "f1w": round(float(r.f1_weighted), 3)}
    cl[s] = row
D["cls"] = cl

# ---------------------------------------------------------------- Top-10 NDCG
tk = pd.read_csv(OUT_DIR / "20_topk" / "ranking_users.csv")
nd = {}
for N in (0, 20):
    q = tk[tk.N == N].groupby(["user_id", "model"])[["ndcg10", "p10", "avg10"]].mean().reset_index()
    nd[str(N)] = {m: {"ndcg": boot(q[q.model == m].ndcg10), "p10": round(float(q[q.model == m].p10.mean()), 3), "avg10": round(float(q[q.model == m].avg10.mean()), 2)}
                  for m in ("RANDOM", "POP", "FULLK_noL", "FULLK")}
    w = q.pivot(index="user_id", columns="model", values="ndcg10")
    nd[str(N)]["diff_pop"] = boot((w["FULLK"] - w["POP"]).dropna())
nd["users"] = int(q.user_id.nunique())
D["topk"] = nd

# ---------------------------------------------------------------- learners
lr = {}
for s, lv in (("A", 0), ("B", 0), ("C", 0)):
    lr[s] = {k: auc("14_final", SC[s], lv, "FULL" if True else "", False) for k in ()}
    d_ = rd("14_final", SC[s])
    lr[s] = {k: round(float(d_[(d_.level == lv) & (d_.method == f"FULL|{k}")].mean_auc.iloc[0]), 4) for k in ("logreg", "hgb_c", "rf")}
    dd = rd("14_final_deep", SC[s])
    lr[s]["mlp_BK"] = round(float(dd[(dd.level == lv) & (dd.method == "BK|mlp")].mean_auc.iloc[0]), 4)
    lr[s]["hgb_BK"] = round(float(dd[(dd.level == lv) & (dd.method == "BK|hgb_c")].mean_auc.iloc[0]), 4)
D["learners"] = lr

(OUT / "page_data.json").write_text(json.dumps(D, ensure_ascii=False), encoding="utf-8")
print("written", OUT / "page_data.json")
