"""
STEP 21 - real data for the interactive simulator of the meeting page.

Everything in the simulator comes from this file; nothing is invented. One fold (repeat 0, fold 0) of the final protocol is fitted:
  * scenario A (new user): 10 test users, N = 0 and 20 revealed ratings. For every film the user rated (positions >= 50) the scores of
    nine models (RANDOM, POP, P, P+H, P+H+A, P+H+A+L, +PR, FULLK = +TMDB content K, FULLK without LLM) and CSP23 (dataset paper) are stored,
    plus the real rating of the user and the raw values of the blocks P and L for the explainer.
  * scenario B (new film): 8 films that are NOT in the training films; for every user who rated such a film the predicted chance that
    the user likes it (models T, T+L, K, T+K, FULLK, FULLK without LLM, RANDOM) and the real answer.
The demo users and films are chosen by a fixed seed, not by looking at the result.

Run   : python 21_simulator_data.py        (about 10-15 minutes)
Output: outputs/21_sim/sim_data.json
"""
import importlib
import json
import sys

import numpy as np
import pandas as pd

from config import OUT_DIR, RANDOM_STATE, ensure_dir

OUT = ensure_dir(OUT_DIR / "21_sim")
ARGV = [sys.argv[0], "--learners", "hgb_c", "--rules", "gt7", "--tmdb", "--repeats", "0", "--folds", "0"]

sys.argv = ARGV + ["--scenario", "A", "--n-list", "0,20"]
import labutils as lu  # noqa: E402

m14 = importlib.import_module("14_final_protocol")
cfgA = lu.parse_args(__doc__, final=True)
lu.require_sklearn()
lu.quiet()
GENRES = m14.GENRES
A_BLOCKS = {"POP": None, "P": ["P"], "P+H": ["P", "H"], "P+H+A": ["P", "H", "A"], "P+H+A+L": ["P", "H", "A", "L"],
            "+PR": ["P", "H", "A", "L", "PRfuzzy"], "FULLK": ["P", "H", "A", "L", "PRfuzzy", "K"],
            "FULLK_noL": ["P", "H", "A", "PRfuzzy", "K"]}


def clean(s):
    return str(s).replace("�", "").strip()


def mean_ignore_nan(a, axis=0):
    ok = ~np.isnan(a)
    s = np.where(ok, a, 0.0).sum(axis=axis)
    c = ok.sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def csp23_user(R, F, train_rows, test_row):
    fv = F[test_row]
    Rtr, Ftr = R[train_rows], F[train_rows]
    pop = mean_ignore_nan(Rtr) / 10.0
    pop = np.where(np.isnan(pop), np.nanmean(Rtr) / 10.0, pop)
    tabs = np.stack([mean_ignore_nan(Rtr[Ftr[:, k] == fv[k]]) for k in range(F.shape[1])])
    content = mean_ignore_nan(tabs) / 10.0
    content = np.where(np.isnan(content), pop, content)
    near = (Ftr == fv[None, :]).sum(axis=1) >= 8
    aff = mean_ignore_nan(Rtr[near]) / 10.0 if near.any() else np.full(R.shape[1], np.nan)
    return np.where(np.isnan(aff), content, (content + aff) / 2.0)


def binary_features(prof, train_users):
    tr = prof.loc[train_users]
    f = pd.DataFrame(index=prof.index)
    for c in ("early_bird", "night_owl", "geo_enabled", "week_tweeter", "weekend_tweeter"):
        f[c] = prof[c].astype(int)
    f["early_user"] = (prof["account_creation_year"] <= tr["account_creation_year"].median()).astype(int)
    for col, name in (("friends_count", "mf"), ("followers_count", "mo"), ("favourites_count", "mv"), ("number_of_tweets", "ht")):
        f[name] = (prof[col] > tr[col].median()).astype(int)
    return f.to_numpy()


def ndcg10(score, rate):
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    disc = 1.0 / np.log2(np.arange(2, 12))
    k = min(10, len(rate))
    return float((rate[order[:k]] * disc[:k]).sum() / (np.sort(rate)[::-1][:k] * disc[:k]).sum())


# ----------------------------------------------------------------------------- scenario A
scen = m14.ScenA(cfgA)
data = scen.data
users_all = scen.prof.index.to_numpy()
uidx = {int(u): i for i, u in enumerate(users_all)}
R = np.full((len(users_all), scen.M), np.nan)
R[data["user_id"].map(uidx).to_numpy(), data["fi"].to_numpy()] = data["rate"].to_numpy(dtype=float)
_, _, _, movies = lu.load_base(cfgA)
mv = movies.drop_duplicates("movie_id").set_index("movie_id").loc[scen.film_ids]
tm = pd.read_csv(m14.TMDB_FILE).drop_duplicates("csp_id").set_index("csp_id").reindex(scen.film_ids)
films = [{"t": clean(mv["original_title"].iloc[i]), "d": clean(str(mv["directors"].iloc[i]).split("|")[0]),
          "y": int(tm["year"].iloc[i]) if pd.notna(tm["year"].iloc[i]) else None,
          "g": [g for g in str(tm["genres"].iloc[i]).split("|")[:3] if g and g != "nan"]} for i in range(scen.M)]
su = scen.setup(0, 0)
train_rows = np.array([uidx[int(u)] for u in su.train_users])
F = binary_features(scen.prof, su.train_users)
rng = np.random.default_rng([RANDOM_STATE, 21, 1])
films_row = None
out = {"films": films, "genres": GENRES, "A": {"users": [], "models": list(A_BLOCKS) + ["RANDOM", "CSP23"]}}
res = {}
for N in cfgA["n_list"]:
    Btr, ytr, utr, Bpo, ypo, upo, simple = scen.blocks(su, "gt7", N)
    films_row = np.concatenate([su.UA[int(u)][0][scen.pool:] for u in su.test_users])
    users_row = np.concatenate([np.full(len(su.UA[int(u)][0]) - scen.pool, int(u)) for u in su.test_users])
    if len(films_row) != len(ypo) or not np.array_equal(users_row, upo):
        sys.exit("rows do not line up")
    score = {"POP": simple["S_pop"], "RANDOM": np.random.default_rng([RANDOM_STATE, 21, N]).random(len(ypo))}
    for name, blk in A_BLOCKS.items():
        if blk is None:
            continue
        score[name] = lu.fit_predict("hgb_c", np.hstack([Btr[k] for k in blk]), ytr, np.hstack([Bpo[k] for k in blk]), cfgA,
                                     RANDOM_STATE + N)
    csp = np.zeros(len(ypo))
    for u in su.test_users:
        mask = users_row == int(u)
        csp[mask] = csp23_user(R, F, train_rows, uidx[int(u)])[films_row[mask]]
    score["CSP23"] = csp
    res[N] = {"score": score, "P": Bpo["P"], "L": Bpo["L"], "users_row": users_row}
    print("scenario A, N =", N, "done", flush=True)

users_row = res[0]["users_row"]
big = [int(u) for u in su.test_users if (users_row == u).sum() >= 150]
chosen = [int(u) for u in rng.choice(big, 10, replace=False)]
for k, u in enumerate(chosen):
    idx = np.flatnonzero(users_row == u)
    ent = {"id": k + 1, "n": int(len(idx)), "rows": [], "ndcg": {}}
    rate = R[uidx[u], films_row[idx]]
    for i, ix in enumerate(idx):
        row = {"f": int(films_row[ix]), "r": int(rate[i]), "P": [round(float(x), 3) for x in res[0]["P"][ix]],
               "L": [round(float(x), 2) for x in res[0]["L"][ix]], "s": {}}
        for N in cfgA["n_list"]:
            row["s"][str(N)] = {m: round(float(s[ix]), 4) for m, s in res[N]["score"].items()}
        ent["rows"].append(row)
    for N in cfgA["n_list"]:
        ent["ndcg"][str(N)] = {m: round(ndcg10(s[idx], rate), 4) for m, s in res[N]["score"].items()}
    out["A"]["users"].append(ent)

# ----------------------------------------------------------------------------- scenario B
sys.argv = ARGV + ["--scenario", "B", "--m-list", "0"]
cfgB = lu.parse_args(__doc__, final=True)
sb = m14.ScenB(cfgB)
mb = sb.m
suB = sb.setup(0, 0)
BtrB, ytrB, utrB, BpoB, ypoB, upoB, _ = sb.blocks(suB, "gt7", 0)
rows_f, rows_u = [], []
for u in suB.te_users.tolist():
    cand = np.flatnonzero(mb.Rated[u] & suB.te_f)
    rows_f.append(cand)
    rows_u.append(np.full(len(cand), mb.users[u]))
rows_f, rows_u = np.concatenate(rows_f), np.concatenate(rows_u)
if len(rows_f) != len(ypoB):
    sys.exit("scenario B rows do not line up")
B_BLOCKS = {"T": ["T"], "T+L": ["T", "L"], "K": ["K"], "T+K": ["T", "K"], "FULLK": ["T", "L", "A", "H", "K"], "FULLK_noL": ["T", "A", "H", "K"]}
scoreB = {"RANDOM": np.random.default_rng([RANDOM_STATE, 21, 77]).random(len(ypoB))}
for name, blk in B_BLOCKS.items():
    scoreB[name] = lu.fit_predict("hgb_c", np.hstack([BtrB[k] for k in blk]), ytrB, np.hstack([BpoB[k] for k in blk]), cfgB, RANDOM_STATE + 5)
movies_B = movies.drop_duplicates("movie_id").set_index("movie_id")
fid_index = {int(f): i for i, f in enumerate(scen.film_ids)}
cnt = pd.Series(rows_f).value_counts()
good = [int(f) for f in cnt.index if cnt[f] >= 25]
rngB = np.random.default_rng([RANDOM_STATE, 21, 2])
chosenB = [int(f) for f in rngB.choice(good, 8, replace=False)]
out["B"] = {"models": list(scoreB), "films": []}
for f in chosenB:
    idx = np.flatnonzero(rows_f == f)
    fi = fid_index[int(mb.film_ids[f])]
    ent = {"f": fi, "n": int(len(idx)), "users": []}
    for j, ix in enumerate(idx):
        ent["users"].append({"u": j + 1, "y": int(ypoB[ix]), "s": {m: round(float(s[ix]), 4) for m, s in scoreB.items()}})
    out["B"]["films"].append(ent)
(OUT / "sim_data.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
print("written", OUT / "sim_data.json", (OUT / "sim_data.json").stat().st_size // 1024, "KB", flush=True)
