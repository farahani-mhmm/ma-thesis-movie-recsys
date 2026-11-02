"""
STEP 20 - Top-10 recommendations for a new user (scenario A) and NDCG@10 over all test users.

For every test user (user-held-out folds, 3 repeats x 5 folds, the same test rows as step 14) the films the user rated are
ranked by the model score and the first 10 are the recommendation. Because every film in the list has a real rating from that
user, the list can be scored:

  NDCG@10   gain = the real rating (1-10); 1 = the best possible order of that user's own rated films
  P@10      share of the 10 films that the user really liked (rating > 7)
  top-10 avg  mean real rating of the 10 films (the CSP 2023 paper reports the same kind of number)

Models: RANDOM, POP (film popularity), P (popularity block through hgb_c), FULL (all blocks, no TMDB), FULLK (all blocks + TMDB
content K = the model used in the thesis, PUB is NOT used) and FULLK_noL (FULLK without the LLM fame block L, "delete LLM").
N = number of revealed ratings of the new user (0 = no rating at all, 20).

Run   : python 20_topk.py                (about 20-30 minutes)   |   python 20_topk.py --quick   |   python 20_topk.py --report
Output: outputs/20_topk/ranking_users.csv, topk_report.txt, samples.json, topk_demo_fa.html
"""
import html
import importlib
import json
import sys
import time

import numpy as np
import pandas as pd

QUICK = "--quick" in sys.argv
REPORT_ONLY = "--report" in sys.argv
sys.argv = [sys.argv[0], "--scenario", "A", "--n-list", "0,20", "--rules", "gt7", "--learners", "hgb_c", "--tmdb"]
if QUICK:
    sys.argv += ["--repeats", "0", "--folds", "0"]

from config import OUT_DIR, RANDOM_STATE, ensure_dir  # noqa: E402
import labutils as lu  # noqa: E402

OUT = ensure_dir(OUT_DIR / ("20_topk_quick" if QUICK else "20_topk"))
RANK_CSV, REPORT, SAMPLES, PAGE = OUT / "ranking_users.csv", OUT / "topk_report.txt", OUT / "samples.json", OUT / "topk_demo_fa.html"
K_TOP = 10
MODELS = ["RANDOM", "POP", "P", "FULL", "FULLK_noL", "FULLK"]
FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa(x, nd=3):
    return f"{x:.{nd}f}".replace(".", "٫").replace("-", "−").translate(FA)


def clean(s):
    return str(s).replace("�", "").strip()


def metrics(score, rate, y):
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    ideal = np.sort(rate)[::-1]
    disc = 1.0 / np.log2(np.arange(2, K_TOP + 2))
    k = min(K_TOP, len(rate))
    dcg = float((rate[order[:k]] * disc[:k]).sum())
    idcg = float((ideal[:k] * disc[:k]).sum())
    return {"ndcg10": dcg / idcg, "p10": float(y[order[:k]].mean()), "avg10": float(rate[order[:k]].mean()),
            "user_avg": float(rate.mean()), "n": len(rate)}


def run():
    m14 = importlib.import_module("14_final_protocol")
    cfg = lu.parse_args(__doc__, final=True)
    lu.require_sklearn()
    lu.quiet()
    scen = m14.ScenA(cfg)
    data = scen.data
    users_all = scen.prof.index.to_numpy()
    uidx = {int(u): i for i, u in enumerate(users_all)}
    R = np.full((len(users_all), scen.M), np.nan)
    R[data["user_id"].map(uidx).to_numpy(), data["fi"].to_numpy()] = data["rate"].to_numpy(dtype=float)
    _, _, _, movies = lu.load_base(cfg)
    mv = movies.drop_duplicates("movie_id").set_index("movie_id").loc[scen.film_ids]
    titles = [clean(t) for t in mv["original_title"].to_numpy()]
    directors = [clean(str(d).split("|")[0]) for d in mv["directors"].fillna("").to_numpy()]
    done = set()
    if RANK_CSV.exists():
        d = pd.read_csv(RANK_CSV, usecols=["repeat", "fold"]).drop_duplicates()
        done = {tuple(x) for x in d.to_numpy().tolist()}
    samples = {}
    t_start, count = time.time(), 0
    for r in cfg["repeats"]:
        for f in cfg["folds"]:
            count += 1
            if (r, f) in done:
                continue
            t0 = time.time()
            su = scen.setup(r, f)
            rows = []
            for N in cfg["n_list"]:
                Btr, ytr, utr, Bpo, ypo, upo, simple = scen.blocks(su, "gt7", N)
                exps = scen.exps(N)
                blocks = {"P": exps["P"], "FULL": exps["FULL"], "FULLK": exps["FULLK"],
                          "FULLK_noL": [b for b in exps["FULLK"] if b != "L"]}
                films_row = np.concatenate([su.UA[int(u)][0][scen.pool:] for u in su.test_users])
                users_row = np.concatenate([np.full(len(su.UA[int(u)][0]) - scen.pool, int(u)) for u in su.test_users])
                if len(films_row) != len(ypo) or not np.array_equal(users_row, upo):
                    sys.exit("test rows do not line up with the protocol rows")
                rate_row = R[[uidx[int(u)] for u in users_row], films_row]
                seed = RANDOM_STATE + 1000 * r + 10 * f + N
                score = {"POP": simple["S_pop"], "RANDOM": np.random.default_rng([seed, 5]).random(len(ypo))}
                for name, blk in blocks.items():
                    Xtr = np.hstack([Btr[k] for k in blk])
                    Xpo = np.hstack([Bpo[k] for k in blk])
                    score[name] = lu.fit_predict("hgb_c", Xtr, ytr, Xpo, cfg, seed)
                order = np.argsort(upo, kind="mergesort")
                u_sorted = upo[order]
                bounds = np.flatnonzero(np.r_[True, u_sorted[1:] != u_sorted[:-1], True])
                for a, b in zip(bounds[:-1], bounds[1:]):
                    idx = order[a:b]
                    for name in MODELS:
                        rows.append({"repeat": r, "fold": f, "N": N, "user_id": int(u_sorted[a]), "model": name,
                                     **metrics(score[name][idx], rate_row[idx], ypo[idx])})
                if r == 0 and f == 0:                      # demo users: fixed by the seed, not chosen by the result
                    rng = np.random.default_rng([RANDOM_STATE, 20])
                    big = [int(u) for u in su.test_users if (users_row == u).sum() >= 150]
                    for u in rng.choice(big, 3, replace=False).tolist():
                        idx = np.flatnonzero(users_row == u)
                        entry = {"user_id": u, "n_test_films": int(len(idx)), "N": N, "lists": {}}
                        for name in ("FULLK", "FULLK_noL", "POP"):
                            top = idx[np.argsort(-score[name][idx], kind="mergesort")[:K_TOP]]
                            entry["lists"][name] = [{"title": titles[films_row[i]], "director": directors[films_row[i]],
                                                     "score": float(score[name][i]), "rating": float(rate_row[i]),
                                                     "liked": int(ypo[i])} for i in top]
                        samples.setdefault(str(u), {})[str(N)] = entry
            pd.DataFrame(rows).to_csv(RANK_CSV, mode="a", header=not RANK_CSV.exists(), index=False)
            print(f"repeat {r} fold {f} done in {time.time() - t0:.0f}s", flush=True)
            lu.progress(count, len(cfg["repeats"]) * len(cfg["folds"]), t_start)
    if samples:
        SAMPLES.write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")


def boot(v, n_boot=2000, seed=RANDOM_STATE):
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n_boot, len(v)))
    lo, hi = np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    return v.mean(), lo, hi


def report():
    d = pd.read_csv(RANK_CSV)
    out = []
    for N in sorted(d.N.unique()):
        q = d[d.N == N]
        per = q.groupby(["user_id", "model"])[["ndcg10", "p10", "avg10", "user_avg"]].mean().reset_index()
        out.append("=" * 100)
        out.append(f"TOP-{K_TOP}, new user (A), N={N} revealed ratings, hgb_c, rule gt7, {per.user_id.nunique()} users (mean per user over repeats)")
        out.append(f"{'model':12s}{'NDCG@10 [95% interval]':28s}{'P@10':>8s}{'top-10 avg rating':>20s}{'user avg':>10s}")
        for m in MODELS:
            u = per[per.model == m]
            a, lo, hi = boot(u.ndcg10)
            out.append(f"{m:12s}{a:7.4f} [{lo:.3f},{hi:.3f}]       {u.p10.mean():8.3f}{u.avg10.mean():20.2f}{u.user_avg.mean():10.2f}")
        wide = per.pivot(index="user_id", columns="model", values="ndcg10")
        out.append("paired NDCG@10 differences (A minus B), 95% bootstrap interval over users:")
        for a_, b_ in (("POP", "RANDOM"), ("FULLK", "POP"), ("FULL", "POP"), ("FULLK", "FULL"), ("FULLK", "FULLK_noL")):
            diff = (wide[a_] - wide[b_]).dropna()
            m_, lo, hi = boot(diff)
            out.append(f"  {a_} - {b_}: {m_:+.4f} [{lo:+.4f},{hi:+.4f}]  {'clear' if lo > 0 or hi < 0 else 'no clear difference'}")
    text = "\n".join(out)
    REPORT.write_text(text, encoding="utf-8")
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))


def page():
    if not SAMPLES.exists():
        return
    s = json.loads(SAMPLES.read_text(encoding="utf-8"))
    uid = sorted(s)[0]
    body = ['<h1>پیشنهاد ۱۰ فیلم برای یک کاربر جدید (Recommendation: get user)</h1>',
            '<p class="sub">کاربر زیر از مجموعهٔ <b>کاربران آزمون</b> است و مدل او را هرگز در آموزش ندیده است. هر ۱۰ فیلم فهرست را این کاربر واقعاً امتیاز داده است، پس می‌توان درستی فهرست را سنجید. '
            'فیلم‌ها بر اساس نمرهٔ مدل مرتب شده‌اند. «پسندیده» یعنی نمرهٔ واقعی کاربر بیشتر از ۷ است.</p>']
    for N in ("0", "20"):
        e = s[uid].get(N)
        if not e:
            continue
        what = "بدون هیچ نمره‌ای از خود کاربر (فقط پروفایل توییتری و اطلاعات فیلم)" if N == "0" else f"با {int(N)} نمرهٔ آشکارشدهٔ کاربر"
        body.append(f"<h2>{what}</h2>")
        body.append(f"<p>کاربر شمارهٔ {e['user_id']}؛ {e['n_test_films']} فیلم نمره‌دار از او برای انتخاب وجود دارد.</p>".translate(FA))
        for name, label in (("FULLK", "مدل نهایی (با LLM و محتوای TMDB)"), ("FULLK_noL", "همان مدل بدون LLM"), ("POP", "فقط محبوبیت فیلم")):
            lst = e["lists"][name]
            hit = sum(x["liked"] for x in lst)
            rows = "".join(f"<tr><td>{(i + 1)}</td><td class='ltr'>{html.escape(x['title'])}</td><td class='ltr'>{html.escape(x['director'])}</td>"
                           f"<td>{fa(x['score'], 2)}</td><td>{str(int(x['rating'])).translate(FA)}</td><td>{'✔' if x['liked'] else '✘'}</td></tr>"
                           for i, x in enumerate(lst))
            body.append(f"<h3>{label} — از ۱۰ فیلم، {str(hit).translate(FA)} تا واقعاً پسندیده شد</h3>"
                        f"<div class='scroll'><table><tr><th>#</th><th>فیلم</th><th>کارگردان</th><th>نمرهٔ مدل</th><th>نمرهٔ واقعی کاربر</th><th>پسندید؟</th></tr>{rows}</table></div>")
    body.append("<p class='warn'>این یک نمونهٔ ثابت است (کاربر با بذر ثابت انتخاب شد، نه با نگاه به نتیجه). برای قضاوت دربارهٔ کیفیت، میانگین NDCG@10 روی همهٔ کاربران آزمون را ببینید (topk_report.txt).</p>")
    css = ("body{font-family:Tahoma,'Vazirmatn',sans-serif;direction:rtl;text-align:right;max-width:1000px;margin:24px auto;padding:0 16px;line-height:1.9;background:#fff;color:#1d1d1f}"
           "h1{font-size:1.5rem}h2{font-size:1.2rem;margin-top:1.8rem;border-bottom:1px solid #ddd}h3{font-size:1rem}.sub{color:#555}"
           "table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:4px 8px;text-align:center}th{background:#f2f2f2}td.ltr{direction:ltr;text-align:left}"
           ".warn{background:#fff8e6;border-radius:8px;padding:8px 14px}.scroll{overflow-x:auto}"
           "@media (prefers-color-scheme:dark){body{background:#1c1c1e;color:#eee}th{background:#2c2c2e}th,td,h2{border-color:#444}.warn{background:#3a321f}.sub{color:#aaa}}")
    PAGE.write_text(f'<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>پیشنهاد ۱۰ فیلم</title><style>{css}</style></head><body>{"".join(body)}</body></html>', encoding="utf-8")


if __name__ == "__main__":
    if not REPORT_ONLY:
        run()
    report()
    page()
