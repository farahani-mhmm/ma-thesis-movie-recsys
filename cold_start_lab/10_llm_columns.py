"""Step 10: BETTER LLM FILM COLUMNS without new API calls (and a validity check of the GPT-4o scores).

Problem found in steps 7-8: the GPT-4o actor column is the MEAN fame over the WHOLE cast (up to 95 people), so a film
with many small parts gets a low score whatever its stars are (cast size vs actor fame: Spearman about -0.20).
The GPT-4o score of every single person was saved by llm.py only in person_fame_scores_checkpoint.json (drive I:),
which is not in this folder, so max / top-3 over the cast can only be built when that file is given.

What this step does (no API, no learner, nothing is fitted on ratings):
  A  GPT-4o director scores per person: a film with one director carries exactly that director's score;
     films with several directors are solved when all but one co-director are already known.
  B  Validity of the director scores against the 4 human raters (40 pilot directors in "Director fame rating"):
     Spearman of GPT-4o with each rater, the human ceiling (each rater vs the mean of the others), and whether
     directors that humans do NOT know get lower GPT-4o scores ("fame = being known").
  C  Film columns (outputs/10_llm_columns/films_llm_features.csv, one row per film, used by later steps):
        dir_gpt, act_gpt_meanall         the two old columns
        n_actors, n_directors            cast / director counts
        act_gpt_adj                      old actor column with the cast-size effect removed (linear in log(1+n_actors),
                                         fitted on films only, no ratings)
     with --gpt-person-json (the old checkpoint): act_gpt_{mean8,max8,top3}_{skip,u1}, act_gpt_missing8, dir_gpt_max
     with --sonnet-person-csv (person_scores_sonnet.csv of pipeline_run/llm_claude.py): the same with _son
        _skip = a person the LLM does not know is left out;  _u1 = such a person counts as fame 1 (obscure);
        *_missing8 / *_unknown8 = share of the first 8 billed actors without a score.
  D  Screening: every column used ALONE as a score, mean per-user AUC over all users (both label rules), with a
     bootstrap interval over users and the paired difference to the old actor column. Descriptive only: whether a
     column adds anything on top of popularity is tested by the learners in the next step.

Run:  python 10_llm_columns.py          (looks for the two person files by itself; pass the paths to override)
      python 10_llm_columns.py --gpt-person-json "<path>/person_fame_scores_checkpoint.json"
Output (outputs/10_llm_columns/): films_llm_features.csv, directors_gpt4o.csv, director_validation.csv,
                                  single_column_auc.csv, report.txt
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import CSV_ENCODING, LIKE_RULES, OUT_DIR, PROJECT_DIR, RANDOM_STATE, ensure_dir
from labfilms import tokens

RATER_DIR = PROJECT_DIR / "Director fame rating"
PILOT_FILE = PROJECT_DIR / "pipeline_run" / "pilot_directors_40.txt"
TOP_ACTORS = 8          # same limit as labfilms.MAX_ACTORS
N_BOOT = 2000
REFERENCE = "act_gpt_meanall"

lines = []


def say(msg=""):
    print(msg, flush=True)
    lines.append(msg)


def norm(name):
    return re.sub(r"\s+", " ", str(name).strip())


# ----------------------------------------------------------------------------------------------
# A  director scores per person
# ----------------------------------------------------------------------------------------------
def recover_directors(mv):
    known, source = {}, {}
    films = [([norm(n) for n in tokens(d)], s) for d, s in zip(mv["directors"], mv["director_fame_score_llm"]) if pd.notna(s)]
    conflicts = 0
    for names, s in films:
        if len(names) == 1:
            n = names[0]
            if n in known and abs(known[n] - s) > 1e-6:
                conflicts += 1
            known.setdefault(n, float(s))
            source.setdefault(n, "single")
    changed = True
    while changed:                       # several directors: solve the one that is still missing
        changed = False
        for names, s in films:
            missing = [n for n in set(names) if n not in known]
            if len(names) > 1 and len(missing) == 1 and len(set(names)) == len(names):
                rest = sum(known[n] for n in names if n != missing[0])
                val = s * len(names) - rest
                if 1 - 1e-6 <= val <= 10 + 1e-6:
                    known[missing[0]] = round(float(val), 4)
                    source[missing[0]] = "solved"
                    changed = True
    nfilms = pd.Series([n for names, _ in films for n in set(names)]).value_counts()
    out = pd.DataFrame({"name": list(known), "score": list(known.values())})
    out["source"] = out["name"].map(source)
    out["n_films"] = out["name"].map(nfilms).fillna(0).astype(int)
    all_dirs = {n for names, _ in films for n in names}
    return out.sort_values("name").reset_index(drop=True), len(all_dirs), conflicts


# ----------------------------------------------------------------------------------------------
# B  validity against human raters
# ----------------------------------------------------------------------------------------------
def read_raters():
    rows = []
    for f in sorted(RATER_DIR.glob("*.txt")):
        for ln in f.read_text(encoding="utf-8-sig").splitlines():
            parts = ln.strip().split()
            if len(parts) < 3 or parts[0].lower() == "rater":
                continue
            val = parts[-1]
            score = float(val) if re.fullmatch(r"\d+(\.\d+)?", val) else np.nan
            rows.append({"rater": f.stem, "name": norm(" ".join(parts[1:-1])), "score": score})
    return pd.DataFrame(rows)


def boot_spearman(x, y, rng):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    est = spearmanr(x, y).statistic
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            bs.append(spearmanr(x[i], y[i]).statistic)
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


def validate(llm_scores, label, rat, pilot, rng):
    """llm_scores: dict name -> score. Returns rows for director_validation.csv and writes to the report."""
    wide = rat.pivot_table(index="name", columns="rater", values="score", aggfunc="first").reindex(pilot)
    raters = list(wide.columns)
    s = pd.Series({n: llm_scores.get(n, np.nan) for n in pilot})
    out = []
    say(f"  [{label}] pilot directors with an LLM score: {int(s.notna().sum())}/{len(pilot)}")
    for r in raters:
        m = wide[r].notna() & s.notna()
        est, lo, hi = boot_spearman(s[m], wide.loc[m, r], rng)
        others = wide.drop(columns=r).mean(axis=1)
        mh = wide[r].notna() & others.notna()
        hest, hlo, hhi = boot_spearman(wide.loc[mh, r], others[mh], rng)
        out.append({"llm": label, "rater": r, "n_known": int(wide[r].notna().sum()), "n_both": int(m.sum()),
                    "spearman_llm": est, "llm_lo": lo, "llm_hi": hi,
                    "spearman_rater_vs_others": hest, "human_lo": hlo, "human_hi": hhi})
        say(f"    {r:10s} knows {int(wide[r].notna().sum()):2d}/40   LLM vs rater rho = {est:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            f"   rater vs other raters (ceiling) = {hest:+.2f} [{hlo:+.2f}, {hhi:+.2f}]")
    mean_h = wide.mean(axis=1)
    m = mean_h.notna() & s.notna() & (wide.notna().sum(axis=1) >= 2)
    est, lo, hi = boot_spearman(s[m], mean_h[m], rng)
    say(f"    LLM vs mean of raters (directors known by >= 2 raters, n={int(m.sum())}): rho = {est:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    k = wide.notna().sum(axis=1)
    m = s.notna()
    est, lo, hi = boot_spearman(s[m], k[m], rng)
    say(f"    LLM score vs number of raters who know the director: rho = {est:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    for grp, mask in (("known by nobody", k == 0), ("known by 1-2", k.between(1, 2)), ("known by 3-4", k >= 3)):
        mm = mask & s.notna()
        say(f"      mean LLM score, {grp:15s} (n={int(mm.sum()):2d}): {s[mm].mean():.2f}")
    out.append({"llm": label, "rater": "MEAN_OF_RATERS", "n_both": int(m.sum()), "spearman_llm": est,
                "llm_lo": lo, "llm_hi": hi})
    return out


# ----------------------------------------------------------------------------------------------
# C  film columns
# ----------------------------------------------------------------------------------------------
def person_columns(mv, score, tag, unknown=frozenset()):
    """score: dict name -> fame (NaN or absent = no score). Builds max / top-3 / mean over the first 8 billed actors."""
    cols = {f"act_{tag}_{s}_{v}": [] for s in ("mean8", "max8", "top3") for v in ("skip", "u1")}
    cols[f"act_{tag}_missing8"], cols[f"dir_{tag}_max"], cols[f"dir_{tag}_mean"], cols[f"act_{tag}_meanall_check"] = [], [], [], []
    for d_cell, a_cell in zip(mv["directors"], mv["actors"]):
        acts = [norm(n) for n in tokens(a_cell, TOP_ACTORS)]
        vals = [score.get(n, np.nan) for n in acts]
        vals = [np.nan if n in unknown else v for n, v in zip(acts, vals)]
        for v in ("skip", "u1"):
            vv = [x for x in vals if pd.notna(x)] if v == "skip" else [1.0 if pd.isna(x) else x for x in vals]
            t3 = vals[:3]
            t3 = [x for x in t3 if pd.notna(x)] if v == "skip" else [1.0 if pd.isna(x) else x for x in t3]
            cols[f"act_{tag}_mean8_{v}"].append(np.mean(vv) if vv else np.nan)
            cols[f"act_{tag}_max8_{v}"].append(max(vv) if vv else np.nan)
            cols[f"act_{tag}_top3_{v}"].append(np.mean(t3) if t3 else np.nan)
        cols[f"act_{tag}_missing8"].append(np.mean([pd.isna(x) for x in vals]) if vals else np.nan)
        ds = [score.get(norm(n), np.nan) for n in tokens(d_cell)]
        ds = [x for x in ds if pd.notna(x)]
        cols[f"dir_{tag}_max"].append(max(ds) if ds else np.nan)
        cols[f"dir_{tag}_mean"].append(np.mean(ds) if ds else np.nan)
        allv = [score.get(norm(n), np.nan) for n in tokens(a_cell)]
        allv = [x for x in allv if pd.notna(x)]
        cols[f"act_{tag}_meanall_check"].append(np.mean(allv) if allv else np.nan)
    return pd.DataFrame(cols, index=mv.index)


def film_table(mv, gpt_json, sonnet_csv):
    F = pd.DataFrame({"movie_id": mv["movie_id"].to_numpy()}, index=mv.index)
    F["dir_gpt"] = pd.to_numeric(mv["director_fame_score_llm"], errors="coerce")
    F["act_gpt_meanall"] = pd.to_numeric(mv["actors_fame_score_llm"], errors="coerce")
    F["n_actors"] = mv["actors"].map(lambda c: len(tokens(c)))
    F["n_directors"] = mv["directors"].map(lambda c: len(tokens(c)))
    x = np.log1p(F["n_actors"].to_numpy(float))
    y = F["act_gpt_meanall"].to_numpy(float)
    ok = np.isfinite(y)
    slope, icpt = np.polyfit(x[ok], y[ok], 1)
    F["act_gpt_adj"] = y - slope * (x - x[ok].mean())
    say(f"  cast-size correction: act_gpt_meanall = {icpt:.3f} {slope:+.3f} * log(1+n_actors)  ->  act_gpt_adj")
    say(f"  Spearman with n_actors: before {spearmanr(y[ok], x[ok]).statistic:+.3f}, "
        f"after {spearmanr(F['act_gpt_adj'][ok], x[ok]).statistic:+.3f}")

    if gpt_json:
        raw = json.loads(Path(gpt_json).read_text(encoding="utf-8"))
        score = {norm(k): float(v) for k, v in raw.items()}
        P = person_columns(mv, score, "gpt")
        diff = (P["act_gpt_meanall_check"] - F["act_gpt_meanall"]).abs()
        say(f"  GPT-4o person file: {len(score)} people. Rebuilt whole-cast mean vs movies.csv: "
            f"max |difference| = {diff.max():.4f} (should be ~0; otherwise the file is from another run)")
        n3 = sum(1 for v in score.values() if v == 3.0)
        say(f"  people with exactly 3.0 (also llm.py's default for failed or missing answers): {n3}")
        F = F.join(P.drop(columns=["act_gpt_meanall_check"]))
    if sonnet_csv:
        ps = pd.read_csv(sonnet_csv, encoding=CSV_ENCODING)
        score = {norm(n): s for n, s in zip(ps["name"], ps["score_median"]) if pd.notna(s)}
        unknown = frozenset(norm(n) for n, s in zip(ps["name"], ps["score_median"]) if pd.isna(s))
        P = person_columns(mv, score, "son", unknown).rename(columns={"act_son_missing8": "act_son_unknown8"})
        say(f"  Sonnet person file: {len(score)} scored, {len(unknown)} unknown or failed")
        F = F.join(P.drop(columns=["act_son_meanall_check"]))
    return F.reset_index(drop=True)


# ----------------------------------------------------------------------------------------------
# D  screening: each column alone as a score, per-user AUC
# ----------------------------------------------------------------------------------------------
def per_user_auc(df, col, label):
    s = df[col].fillna(df[col].mean())
    r = s.groupby(df["user_id"]).rank(method="average")
    y = df[label]
    g = df.assign(_r=r * y).groupby("user_id")
    n1, n = g[label].sum(), g[label].size()
    n0 = n - n1
    auc = (g["_r"].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return auc[(n1 > 0) & (n0 > 0)]


def screen(F, ratings, rng):
    feat = [c for c in F.columns if c not in ("movie_id",)]
    df = ratings.merge(F, on="movie_id", how="inner")
    rows = []
    for rule in LIKE_RULES:
        label = f"liked_{rule}"
        aucs = pd.DataFrame({c: per_user_auc(df, c, label) for c in feat}).dropna()
        idx = rng.integers(0, len(aucs), (N_BOOT, len(aucs)))
        A = aucs.to_numpy()
        ref = aucs.columns.get_loc(REFERENCE)
        say(f"\n  label rule {rule}: {len(aucs)} users, mean per-user AUC of each column used alone")
        say(f"  {'column':24s} {'AUC':>7s} {'95% interval':>18s}   {'minus ' + REFERENCE:>22s} {'95% interval':>18s}")
        for j, c in enumerate(aucs.columns):
            bm = A[idx, j].mean(axis=1)
            bd = (A[idx, j] - A[idx, ref]).mean(axis=1)
            d = A[:, j].mean() - A[:, ref].mean()
            rows.append({"rule": rule, "column": c, "auc": A[:, j].mean(), "lo": np.percentile(bm, 2.5),
                         "hi": np.percentile(bm, 97.5), "diff_vs_ref": d, "diff_lo": np.percentile(bd, 2.5),
                         "diff_hi": np.percentile(bd, 97.5)})
            r_ = rows[-1]
            say(f"  {c:24s} {r_['auc']:7.4f} [{r_['lo']:.4f}, {r_['hi']:.4f}]   {d:+22.4f} [{r_['diff_lo']:+.4f}, {r_['diff_hi']:+.4f}]")
    say("  (AUC below 0.5 = the column ranks films the wrong way round; counts are included only as a reference)")
    return pd.DataFrame(rows)


def find_file(name, extra_roots=()):
    """Look for `name` in the project folder, the desktop, downloads and (if present) drive I:, a few levels deep."""
    home = Path.home()
    roots = [PROJECT_DIR, PROJECT_DIR.parent, home / "Desktop", home / "Downloads", home / "Documents",
             Path("I:/"), *extra_roots]
    for root in roots:
        try:
            if not root.exists():
                continue
            for depth in range(0, 5):
                hits = list(root.glob("/".join(["*"] * depth + [name])))
                if hits:
                    return hits[0]
        except OSError:
            continue
    return None


def main():
    ap = argparse.ArgumentParser(description="Step 10: better LLM film columns without API calls")
    ap.add_argument("--gpt-person-json", help="llm.py checkpoint person_fame_scores_checkpoint.json (if found)")
    ap.add_argument("--sonnet-person-csv", help="person_scores_sonnet.csv from pipeline_run/llm_claude.py (optional)")
    args = ap.parse_args()
    if not args.gpt_person_json:
        args.gpt_person_json = find_file("person_fame_scores_checkpoint.json")
        print(f"GPT-4o person file {'found automatically: ' + str(args.gpt_person_json) if args.gpt_person_json else 'not found (max / top-3 columns will be skipped)'}")
    if not args.sonnet_person_csv:
        args.sonnet_person_csv = find_file("person_scores_sonnet.csv")
        if args.sonnet_person_csv:
            print(f"Sonnet person file found automatically: {args.sonnet_person_csv}")
    for p in (args.gpt_person_json, args.sonnet_person_csv):
        if p and not Path(p).exists():
            sys.exit(f"File not found: {p}")

    out = ensure_dir(OUT_DIR / "10_llm_columns")
    d1 = OUT_DIR / "01_data"
    mv = pd.read_csv(d1 / "movies_clean.csv", encoding=CSV_ENCODING).drop_duplicates("movie_id").reset_index(drop=True)
    ratings = pd.read_csv(d1 / "ratings_clean.csv", encoding=CSV_ENCODING)
    rng = np.random.default_rng(RANDOM_STATE)

    say("STEP 10 - LLM FILM COLUMNS (no API calls)")
    say(f"films: {len(mv)}, ratings: {len(ratings)}, users: {ratings['user_id'].nunique()}")

    say("\n=== A. GPT-4o director score per person (recovered from the film column) ===")
    dirs, n_all, conflicts = recover_directors(mv)
    dirs.to_csv(out / "directors_gpt4o.csv", index=False, encoding=CSV_ENCODING)
    say(f"  recovered {len(dirs)} of {n_all} directors ({(dirs['source'] == 'single').sum()} from one-director films, "
        f"{(dirs['source'] == 'solved').sum()} solved from co-directed films); conflicting values: {conflicts}")
    say(f"  share of recovered directors with exactly 3.0 (llm.py default on failure, or a real 3): "
        f"{(dirs['score'] == 3.0).mean():.3f}")
    big = mv[mv["directors"].map(lambda c: len(tokens(c))) >= 10]
    if len(big):
        say(f"  note: {len(big)} films list 10 or more 'directors' (check the data), e.g. {big['original_title'].head(3).tolist()}")

    say("\n=== B. Validity against the 4 human raters (40 pilot directors) ===")
    rat = read_raters()
    pilot = [norm(n) for n in PILOT_FILE.read_text(encoding="utf-8").splitlines() if n.strip()]
    say(f"  raters: {sorted(rat['rater'].unique())}; rows read: {len(rat)}; "
        f"names not in the pilot list: {sorted(set(rat['name']) - set(pilot))[:5]}")
    val = validate(dict(zip(dirs["name"], dirs["score"])), "gpt4o", rat, pilot, rng)
    if args.sonnet_person_csv:
        ps = pd.read_csv(args.sonnet_person_csv, encoding=CSV_ENCODING)
        val += validate(dict(zip(ps["name"].map(norm), ps["score_median"])), "sonnet", rat, pilot, rng)
    pd.DataFrame(val).to_csv(out / "director_validation.csv", index=False, encoding=CSV_ENCODING)

    say("\n=== C. Film columns ===")
    F = film_table(mv, args.gpt_person_json, args.sonnet_person_csv)
    F.to_csv(out / "films_llm_features.csv", index=False, encoding=CSV_ENCODING)
    say(f"  columns written: {[c for c in F.columns if c != 'movie_id']}")
    say(f"  missing values per column: {F.isna().sum()[F.isna().sum() > 0].to_dict()}")
    if not args.gpt_person_json:
        say("  (no --gpt-person-json: max / top-3 over the cast cannot be built from the GPT-4o film means)")

    say("\n=== D. Screening: each column alone (descriptive, all users, no learner) ===")
    screen(F, ratings, rng).to_csv(out / "single_column_auc.csv", index=False, encoding=CSV_ENCODING)

    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
