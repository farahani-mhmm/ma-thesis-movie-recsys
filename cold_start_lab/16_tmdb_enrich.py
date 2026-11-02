"""Step 16: enrich the CSP films with public content from TMDB (genres, year, runtime, keywords, overview, ...).

For every film the script searches TMDB by the original title, fetches the top candidates with their credits and keeps
the candidate whose directors and billed actors match the CSP row (a title alone is not trusted). Two groups of columns
are written, so the thesis can report them separately:
  content columns (known before release):  year, runtime, genres, keywords, overview, tagline, language, countries,
                                            budget, in_collection
  public-rating columns (known only after release, prefix pub_): vote_average, vote_count, popularity, revenue
Every TMDB answer is cached on disk, so a stopped run continues and nothing is requested twice.

Needs the environment variable TMDB_API_KEY (the v3 API key). The key is never written to any file.
TMDB terms do not allow redistribution, so outputs/16_tmdb/ is git-ignored.

Run  : python 16_tmdb_enrich.py [--limit 20]
Output: outputs/16_tmdb/movies_tmdb.csv, match_report.txt, cache/
"""
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from config import CSV_ENCODING, DATA_DIR, OUT_DIR, ensure_dir

API = "https://api.themoviedb.org/3"
PAUSE = 0.03               # seconds between requests of one worker
WORKERS = 8                # parallel films (TMDB allows about 50 requests per second)
N_CANDIDATES = 5
N_ACTORS_CHECK = 10        # billed CSP actors compared with the TMDB cast
REJECT = {(640188, 1762506), (640188, 1335757)}   # found by reading the match report: CSP film id, wrong TMDB id (Band of Brothers: Legacy, and its making-of)
MIN_SCORE = 2.0            # director match = 2, each matched actor = 1


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def clean_title(t):
    """Title without a trailing '(TV Series)', '(S)', '(TV)' ...; and whether it is a series."""
    m = re.search(r"\(([^)]*)\)\s*$", t)
    tag = m.group(1) if m else ""
    is_series = bool(re.search(r"serie", tag, re.I)) and "TV" in tag
    return (t[:m.start()].strip() if m and re.fullmatch(r"(TV Series|TV Mini Series|Serie de TV|TV|S|C|V|VG|TV Short|Short)", tag) else t), is_series


def title_variants(title):
    """The title itself, the part before a parenthesis ('Amour (Love)'), the parenthesis, and a letters-only form
    ('WALL-E' damaged by a bad encoding becomes 'wall e')."""
    v = {title, re.sub(r"\s*\(.*?\)", "", title).strip()}
    v |= set(re.findall(r"\(([^)]+)\)", title))
    return [x for x in v if x]


def episode_series(title):
    """'Black Mirror: Nosedive (TV)' -> 'Black Mirror' (an episode is matched to its series, flagged as series level)."""
    m = re.match(r"^(.+?)\s*(?::|\s-\s)\s*.+$", title)
    return m.group(1).strip() if m else None


class Tmdb:
    def __init__(self, key, cache_dir):
        self.key, self.dir = key, cache_dir
        self.n_calls = 0

    def get(self, path, **params):
        tag = hashlib.md5((path + json.dumps(params, sort_keys=True)).encode()).hexdigest()[:16]
        f = self.dir / f"{path.strip('/').replace('/', '_')}_{tag}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        url = f"{API}{path}?{urllib.parse.urlencode({**params, 'api_key': self.key})}"
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    time.sleep(float(e.headers.get("Retry-After", 2)) + 0.5)
                    continue
                if e.code in (401, 403):
                    sys.exit(f"TMDB refused the key (HTTP {e.code}). Check TMDB_API_KEY.")
                if e.code == 404:
                    data = None
                    break
                time.sleep(1 + attempt)
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1 + attempt)
        else:
            sys.exit(f"TMDB request failed repeatedly: {path}")
        f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        self.n_calls += 1
        time.sleep(PAUSE)
        return data


def score_candidate(det, directors, actors, kind):
    """2 for a matching director, plus 1 per matching billed actor."""
    credits = (det or {}).get("credits") or {}
    crew = {norm(c["name"]) for c in credits.get("crew", []) if c.get("job") in ("Director", "Creator", "Executive Producer")}
    if kind == "tv":
        crew |= {norm(c["name"]) for c in det.get("created_by", [])}
    cast = {norm(c["name"]) for c in credits.get("cast", [])}
    d = 2.0 if any(norm(x) in crew for x in directors) else 0.0
    a = sum(norm(x) in cast for x in actors[:N_ACTORS_CHECK])
    return d + a


def row_from(det, kind):
    gen = "|".join(g["name"] for g in det.get("genres", []))
    kw = (det.get("keywords") or {})
    kws = "|".join(k["name"] for k in (kw.get("keywords") or kw.get("results") or []))
    date = det.get("release_date") or det.get("first_air_date") or ""
    runtime = det.get("runtime") or (det.get("episode_run_time") or [None])[0]
    return {
        "year": int(date[:4]) if date[:4].isdigit() else None,
        "runtime": runtime,
        "genres": gen,
        "keywords": kws,
        "overview": (det.get("overview") or "").replace("\n", " "),
        "tagline": det.get("tagline") or "",
        "language": det.get("original_language"),
        "countries": "|".join(c.get("iso_3166_1", "") for c in det.get("production_countries", [])) or "|".join(det.get("origin_country", [])),
        "budget": det.get("budget") if kind == "movie" else None,
        "in_collection": int(bool(det.get("belongs_to_collection"))) if kind == "movie" else 0,
        "pub_vote_average": det.get("vote_average"),
        "pub_vote_count": det.get("vote_count"),
        "pub_popularity": det.get("popularity"),
        "pub_revenue": det.get("revenue") if kind == "movie" else None,
    }


def main():
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        sys.exit("TMDB_API_KEY is not set. Run  setx TMDB_API_KEY ...  and restart the terminal.")
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    out = ensure_dir(OUT_DIR / "16_tmdb")
    api = Tmdb(key, ensure_dir(out / "cache"))
    movies = pd.read_csv(DATA_DIR / "movies.csv", encoding=CSV_ENCODING)
    if limit:
        movies = movies.head(limit)

    def process(m):
        directors = [d.strip() for d in str(m.directors).split("|") if d.strip()]
        actors = [a.strip() for a in str(m.actors).split("|") if a.strip()]
        title, is_series = clean_title(str(m.original_title))
        alt, _ = clean_title(str(m.main_title))
        kinds = ["tv"] if is_series else ["movie", "tv"]
        best = (0.0, None, None)
        wanted = {norm(x) for t in (title, alt) for x in title_variants(t)}
        queries = [(q, lang) for q, lang in [(title, "en-US"), (alt, "es-ES")]]
        queries += [(v, "en-US") for v in title_variants(title) if v != title and norm(v)] + [(re.sub(r"[^A-Za-z0-9]+", " ", title).strip(), "en-US")]
        for kind in kinds:
            for q, lang in queries:
                if not q:
                    continue
                res = api.get(f"/search/{kind}", query=q, language=lang, include_adult="false")
                for c in ((res or {}).get("results") or [])[:N_CANDIDATES]:
                    det = api.get(f"/{kind}/{c['id']}", append_to_response="credits,keywords", language="en-US")
                    if not det:
                        continue
                    sc = score_candidate(det, directors, actors, kind)
                    names = {norm(det.get("title") or det.get("name") or ""), norm(det.get("original_title") or det.get("original_name") or "")}
                    # a title must also agree unless the evidence is strong, and a series must match a series
                    if sc < 6 and not (names & wanted):
                        continue
                    if (is_series and kind != "tv") or (m.id, c["id"]) in REJECT:
                        continue
                    if sc > best[0]:
                        best = (sc, det, kind)
                if best[0] >= MIN_SCORE + 2:
                    break
            if best[0] >= MIN_SCORE + 2:
                break
        series_level = 0
        if best[0] < MIN_SCORE and episode_series(title):          # an episode: use the series
            res = api.get("/search/tv", query=episode_series(title), language="en-US", include_adult="false")
            for c in ((res or {}).get("results") or [])[:3]:
                if norm(c.get("name", "")) == norm(episode_series(title)) or norm(c.get("original_name", "")) == norm(episode_series(title)):
                    det = api.get(f"/tv/{c['id']}", append_to_response="credits,keywords", language="en-US")
                    best, series_level = (MIN_SCORE, det, "tv"), 1
                    break
        row = {"csp_id": m.id, "csp_title": m.original_title, "matched": int(best[0] >= MIN_SCORE), "match_score": best[0],
               "series_level": series_level}
        if best[1] is not None and best[0] >= MIN_SCORE:
            row.update({"tmdb_id": best[1]["id"], "kind": best[2], "tmdb_title": best[1].get("title") or best[1].get("name"),
                        **row_from(best[1], best[2])})
        return row

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:       # the network, not the computer, is the slow part
        for i, row in enumerate(pool.map(process, movies.itertuples(index=False)), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(movies):
                print(f"{i}/{len(movies)}  matched so far {sum(r['matched'] for r in rows)}  new requests {api.n_calls}  {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / ("movies_tmdb_sample.csv" if limit else "movies_tmdb.csv"), index=False, encoding="utf-8")
    ok = df[df["matched"] == 1]
    lines = [f"films: {len(df)}   matched: {len(ok)} ({len(ok) / len(df):.1%})   unmatched: {len(df) - len(ok)}",
             f"tv series among matched: {int((ok['kind'] == 'tv').sum())}   matched only at series level (episodes): {int(ok['series_level'].sum())}",
             f"with overview: {int((ok['overview'].fillna('') != '').sum())}   with genres: {int((ok['genres'].fillna('') != '').sum())}   with keywords: {int((ok['keywords'].fillna('') != '').sum())}",
             "match score (matched): " + str(ok["match_score"].describe().round(2).to_dict()),
             "", "unmatched titles:"] + [f"  {r.csp_id}  {r.csp_title}  (best score {r.match_score:.0f})" for r in df[df['matched'] == 0].itertuples()]
    (out / ("match_report_sample.txt" if limit else "match_report.txt")).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:5]))
    print(f"files written to: {out}")


if __name__ == "__main__":
    main()
