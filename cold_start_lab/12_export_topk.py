"""Step 12: export a small Top-10 recommender (new user who has liked / disliked a few films) for the presentation page.

The model is the step-5 experiment E3 (film popularity P, history of the revealed ratings H, LLM fame L, affinity A)
fitted once as a logistic regression with N = 5 revealed ratings, rule gt7, Sonnet fame columns. A variant without
any LLM column is exported too, so the page can show what the LLM columns change in the list.
Training rows use inner-fold user contexts (no leakage); serving statistics use all users.

Output: outputs/12_topk/topk_data.json  and  a python check list printed at the end (used to test the JavaScript copy).
Run   : python 12_export_topk.py
"""
import base64
import importlib
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import OUT_DIR, RANDOM_STATE, ensure_dir
from labutils import K_INNER, load_base

h5 = importlib.import_module("05_history_models")
RULE, N = "gt7", 5


def clean(s):
    return str(s).replace("�", "").strip()


def main():
    data, users_f, prof, movies = load_base({"llm": "sonnet", "llm_csv": None})
    film_ids = np.sort(data["movie_id"].unique())
    M = len(film_ids)
    fidx = {m: i for i, m in enumerate(film_ids.tolist())}
    data["fi"] = data["movie_id"].map(fidx).astype(int)
    F = h5.build_films(movies, film_ids)
    mv = movies.drop_duplicates("movie_id").set_index("movie_id").loc[film_ids]

    d = data.sort_values(["user_id", "pos_r0"], kind="mergesort")
    UA = {int(u): (g["fi"].to_numpy(), {k: g[f"liked_{k}"].to_numpy(dtype=np.float32) for k in ("gt7", "ge7")})
          for u, g in d.groupby("user_id", sort=False)}
    users = np.sort(data["user_id"].unique())
    rng = np.random.default_rng([RANDOM_STATE, 21])
    inner_of = dict(zip(users.tolist(), (rng.permutation(len(users)) % K_INNER).tolist()))
    inner_arr = data["user_id"].map(inner_of).to_numpy()
    ctx_inner = [h5.make_ctx(data[inner_arr != k], [RULE], M, False) for k in range(K_INNER)]
    ctx_all = h5.make_ctx(data, [RULE], M, False)
    prof_vec = {int(u): np.zeros(12, dtype=np.float32) for u in users}
    cfg = {"train_per_user": 200, "with_uu": False}
    B, y, _ = h5.build_block(users.tolist(), N, lambda u: ctx_inner[inner_of[int(u)]], RULE, UA, F, prof_vec,
                             "train", 0, cfg)

    out = {"rule": RULE, "N": N, "models": {}}
    sets = {"llm": ["P", "H", "L", "A"], "no_llm": ["P", "H", "A4"]}
    for name, groups in sets.items():
        cols = [B["A"][:, :4] if g == "A4" else B[g] for g in groups]
        X = np.hstack(cols)
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(X), y)
        out["models"][name] = {"groups": groups, "mean": sc.mean_.round(6).tolist(), "scale": sc.scale_.round(6).tolist(),
                               "coef": lr.coef_[0].round(6).tolist(), "intercept": round(float(lr.intercept_[0]), 6)}
        print(name, "rows", X.shape, "coef", np.round(lr.coef_[0], 3).tolist())

    out["g_like"] = round(float(ctx_all.g_like[RULE]), 5)
    out["mean_fame"] = [round(F.mean_fame_dir, 4), round(F.mean_fame_act, 4)]
    out["films"] = [[clean(mv["original_title"].iloc[i]), clean(str(mv["directors"].iloc[i]).split("|")[0])[:36],
                     round(float(F.fame_dir[i]), 3), round(float(F.fame_act[i]), 3),
                     round(float(ctx_all.f_like[RULE][i]), 4), round(float(ctx_all.f_mean[i]), 3),
                     round(float(ctx_all.f_logc[i]), 3)] for i in range(M)]
    out["sim"] = base64.b64encode(np.round(np.clip(ctx_all.sim, -1, 1) * 127).astype(np.int8).tobytes()).decode()
    out["dir_share"] = [np.flatnonzero(F.dir_share[i]).tolist() for i in range(M)]
    out["act_share"] = [np.flatnonzero(F.act_share[i]).tolist() for i in range(M)]

    # python reference for the JavaScript copy: user 1 revealed = first 5 films (liked as in the data)
    uid = users[10]
    films_u, liked = UA[int(uid)]
    R, lR = films_u[:N], liked[RULE][:N]
    allT = np.arange(M)
    P, H, L, A, _ = h5.feats_for_user(ctx_all, RULE, np.concatenate([R, np.setdiff1d(allT, R)]),
                                      np.concatenate([lR, np.zeros(M - N, dtype=np.float32)]), N, np.arange(N, M), F, False)
    m = out["models"]["llm"]
    X = np.hstack([P, H, L, A])
    z = m["intercept"] + ((X - np.array(m["mean"])) / np.array(m["scale"])) @ np.array(m["coef"])
    top = np.argsort(-z)[:10]
    cand = np.concatenate([R, np.setdiff1d(allT, R)])[N:]
    out["check"] = {"revealed": R.tolist(), "liked": lR.tolist(), "top10": cand[top].tolist(),
                    "p": (1 / (1 + np.exp(-z[top]))).round(4).tolist()}
    print("check top10:", [out["films"][i][0] for i in out["check"]["top10"]])
    path = ensure_dir(OUT_DIR / "12_topk") / "topk_data.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("written", path, round(path.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
