"""Film-level scores from Claude Sonnet 5.5: how famous and how well liked is each FILM (not its people).

Why: for a film nobody in the system has rated, the only knowledge about the film itself is its content (TMDB) and
the fame of its people. This asks the model what it knows about the film. It is NOT a TMDB or IMDb lookup, but the
model has read about these films, so the score carries public opinion, like the TMDB public rating does. The thesis
text must say that. Only the title, the year and the first director are sent (no user data, no rating of the dataset).

Each film is scored 3 times (Sonnet 5.5 accepts no temperature) and the median is used. 10 films per request.

    python llm_film_scores.py                # dry run: how many requests, the price estimate, nothing is sent
    python llm_film_scores.py --yes          # the real run; safe to stop and start again (finished requests are kept)

The key comes from the environment (ANTHROPIC_API_KEY, and ANTHROPIC_BASE_URL for Liara); it is never written in code.
Liara (a base URL with "liara" in it, or LIARA_BASE_URL + LIARA_API_KEY) speaks the OpenAI format, so the openai package
is used there, with Liara's model name (anthropic/claude-sonnet-5.5).
Output: llm_claude_outputs/film_scores/film_scores.csv  (movie_id, film_known, film_pop_sonnet, film_quality_sonnet)
"""
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "llm_claude_outputs" / "film_scores"
TMDB = BASE.parent / "cold_start_lab" / "outputs" / "16_tmdb" / "movies_tmdb.csv"
MODEL = "claude-sonnet-5-5"
BATCH, RUNS, WORKERS = 10, 3, 4
_anth_url = os.environ.get("ANTHROPIC_BASE_URL", "")
LIARA_URL = os.environ.get("LIARA_BASE_URL") or (_anth_url if "liara" in _anth_url else "")
LIARA_KEY = os.environ.get("LIARA_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
LIARA_MODELS = {"claude-sonnet-5-5": "anthropic/claude-sonnet-5.5", "claude-opus-5-5": "anthropic/claude-opus-5.5"}
_use_schema = True

SYSTEM = """You are a film historian. For each numbered film give two integer scores from 1 to 10:
- popularity: how widely known the film is among general audiences worldwide (10 = known almost everywhere, 1 = almost nobody has heard of it).
- quality: how well liked the film is by audiences and critics together (10 = widely loved, 5 = mixed or average, 1 = widely disliked).
Rules: rate the film itself, not its stars. If you do not recognise the film, set known to false and both scores to 0. Do not guess.
Return exactly one entry for every number in the list, with the same id."""
SCHEMA = {"type": "object", "properties": {"films": {"type": "array", "items": {"type": "object", "properties": {
    "id": {"type": "integer"}, "known": {"type": "boolean"}, "popularity": {"type": "integer"}, "quality": {"type": "integer"}},
    "required": ["id", "known", "popularity", "quality"], "additionalProperties": False}}},
    "required": ["films"], "additionalProperties": False}


def films():
    mv = pd.read_csv(BASE / "movies.csv", encoding="utf-8-sig")
    tm = pd.read_csv(TMDB).drop_duplicates("csp_id").set_index("csp_id")
    mv = mv.drop_duplicates("id" if "id" in mv.columns else "movie_id")
    idc = "id" if "id" in mv.columns else "movie_id"
    title = mv["original_title"] if "original_title" in mv.columns else mv["main_title"]
    year = mv[idc].map(tm["year"]).fillna(0).astype(int)
    first_dir = mv["directors"].fillna("").astype(str).str.split("|").str[0].str.strip()
    return pd.DataFrame({"movie_id": mv[idc], "title": title.astype(str), "year": year, "director": first_dir}).reset_index(drop=True)


def message(batch):
    lines = [f'{i}. "{r.title}"' + (f" ({r.year})" if r.year else "") + (f", directed by {r.director}" if r.director else "")
             for i, r in enumerate(batch.itertuples(), 1)]
    return "Rate these films:\n" + "\n".join(lines)


def ask(batch):
    """One request: (text, input tokens, output tokens). Liara speaks the OpenAI format, Anthropic its own."""
    global _use_schema
    if LIARA_URL:
        from openai import OpenAI
        kw = {"model": LIARA_MODELS.get(MODEL, MODEL), "max_tokens": 4000,
              "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": message(batch)}]}
        if _use_schema:
            kw["response_format"] = {"type": "json_schema", "json_schema": {"name": "films", "schema": SCHEMA, "strict": True}}
        else:
            kw["messages"][-1]["content"] += ('\n\nReturn ONLY JSON like '
                                              '{"films": [{"id": 1, "known": true, "popularity": 7, "quality": 6}]}.')
        try:
            r = OpenAI(base_url=LIARA_URL, api_key=LIARA_KEY).chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            if _use_schema and ("response_format" in str(e) or "json_schema" in str(e)):
                print("json_schema not accepted by the endpoint; continuing with a JSON instruction in the prompt", flush=True)
                _use_schema = False
                return ask(batch)
            raise
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (r.choices[0].message.content or "").strip())
        return text, r.usage.prompt_tokens, r.usage.completion_tokens
    import anthropic
    m = anthropic.Anthropic().messages.create(model=MODEL, max_tokens=4000, system=SYSTEM,
                                              messages=[{"role": "user", "content": message(batch)}],
                                              output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
    return next(b.text for b in m.content if getattr(b, "type", "") == "text"), m.usage.input_tokens, m.usage.output_tokens


def one_request(args):
    run, bi, batch = args
    err = ""
    for attempt in range(4):
        try:
            text, tok_in, tok_out = ask(batch)
            items = {int(x["id"]): x for x in json.loads(text)["films"]}
            rows = []
            for i, r in enumerate(batch.itertuples(), 1):
                x = items.get(i)
                ok = x is not None and 0 <= x["popularity"] <= 10 and 0 <= x["quality"] <= 10
                rows.append({"run": run, "batch": bi, "movie_id": int(r.movie_id), "ok": bool(ok),
                             "known": bool(x["known"]) if ok else False,
                             "popularity": x["popularity"] if ok else np.nan, "quality": x["quality"] if ok else np.nan})
            return rows, (tok_in, tok_out, "")
        except Exception as e:  # noqa: BLE001 - retry on anything, report at the end
            err = repr(e)[:300]
            if "402" in err or "balance" in err.lower():
                raise SystemExit("STOP: the balance is too low (402). Top up, then run the same command again; it resumes.")
            time.sleep(5 * (attempt + 1))
    return [{"run": run, "batch": bi, "movie_id": int(r.movie_id), "ok": False, "known": False, "popularity": np.nan,
             "quality": np.nan} for r in batch.itertuples()], (0, 0, err)


def endpoint_check(out):
    """Is Liara really Claude? One tiny request (no JSON schema, fixed text): print the model and id fields of the reply,
    and, when ANTHROPIC_DIRECT_KEY is set, compare Liara's input token count with Anthropic's free token counter for the
    same text (Claude's tokenizer differs from other families). Saved to endpoint_check.json for the thesis."""
    from openai import OpenAI
    system, user = "You are a film historian.", 'In one short sentence: who directed the film "Ex Machina" (2014)?'
    r = OpenAI(base_url=LIARA_URL, api_key=LIARA_KEY).chat.completions.create(
        model=LIARA_MODELS.get(MODEL, MODEL), max_tokens=50,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    info = {"checked_at": time.strftime("%Y-%m-%d %H:%M"), "asked_model": LIARA_MODELS.get(MODEL, MODEL),
            "reply_model": r.model, "reply_id": r.id, "liara_input_tokens": r.usage.prompt_tokens,
            "liara_output_tokens": r.usage.completion_tokens, "answer": (r.choices[0].message.content or "").strip()[:200]}
    direct = os.environ.get("ANTHROPIC_DIRECT_KEY")
    if direct and not direct.startswith("sk-ant-"):
        print("ANTHROPIC_DIRECT_KEY is not an Anthropic key (those start with sk-ant-); token count check skipped", flush=True)
        direct = None
    if direct:
        import anthropic
        try:
            c = anthropic.Anthropic(api_key=direct, base_url="https://api.anthropic.com").messages.count_tokens(
                model=MODEL, system=system, messages=[{"role": "user", "content": user}])
            info["anthropic_count_tokens"] = c.input_tokens
        except Exception as e:  # noqa: BLE001
            info["anthropic_count_tokens_error"] = repr(e)[:300]
    (out / "endpoint_check.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("endpoint check:", json.dumps(info, ensure_ascii=False), flush=True)


def main():
    global MODEL, OUT, RUNS
    ap =argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="really send the requests (spends money)")
    ap.add_argument("--model", default=MODEL, help="model id; a non-default model writes to its own folder")
    ap.add_argument("--limit", type=int, default=0, help="only the first N films (a pilot); writes to its own folder")
    ap.add_argument("--runs", type=int, default=RUNS, help="repeats per film (the median is used)")
    a = ap.parse_args()
    MODEL, RUNS = a.model, a.runs
    if a.model != "claude-sonnet-5-5" or a.limit:
        OUT = OUT.parent / ("film_scores_" + a.model + (f"_first{a.limit}" if a.limit else ""))
    F = films()
    if a.limit:
        F = F.iloc[:a.limit].reset_index(drop=True)
    batches = [F.iloc[i:i + BATCH] for i in range(0, len(F), BATCH)]
    OUT.mkdir(parents=True, exist_ok=True)
    long_path = OUT / "film_scores_long.csv"
    done = set()
    if long_path.exists():
        d = pd.read_csv(long_path)
        done = {(int(r), int(b)) for r, b in d.loc[d.ok, ["run", "batch"]].drop_duplicates().to_numpy()}
    todo = [(r, bi, b) for r in range(RUNS) for bi, b in enumerate(batches) if (r, bi) not in done]
    est_in, est_out = 600 + 60 * BATCH, 40 * BATCH + 300        # rough tokens per request
    price = len(todo) * (est_in * 2.0 + est_out * 10.0) / 1e6
    print("endpoint:", f"Liara ({LIARA_MODELS.get(MODEL, MODEL)})" if LIARA_URL else f"Anthropic ({MODEL})")
    print(f"films: {len(F)}, requests still to send: {len(todo)} (of {RUNS * len(batches)}), rough price: about ${price:.2f}")
    if not (LIARA_KEY if LIARA_URL else os.environ.get("ANTHROPIC_API_KEY")):
        sys.exit("No key in this terminal window. For Liara set ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL first.")
    if not a.yes:
        print("dry run: nothing was sent. Add --yes to run.")
        return
    if LIARA_URL:
        endpoint_check(OUT)
    tot, errors = [0, 0], []
    if todo:                                   # the first request alone: if it fails, stop before spending more
        rows, usage = one_request(todo[0])
        if usage[2] or not any(x["ok"] for x in rows):
            sys.exit(f"STOP: the first request failed, nothing else was sent. Error: {usage[2] or 'no valid scores in the reply'}")
        pd.DataFrame(rows).to_csv(long_path, mode="a", header=not long_path.exists(), index=False)
        tot = [usage[0], usage[1]]
        print(f"first request OK: tokens in/out {usage[0]}/{usage[1]}, about ${(usage[0] * 2 + usage[1] * 10) / 1e6 * len(todo):.2f} "
              f"for all {len(todo)} requests at $2/$10 per million", flush=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        for k, (rows, usage) in enumerate(pool.map(one_request, todo[1:]), 2):
            pd.DataFrame(rows).to_csv(long_path, mode="a", header=not long_path.exists(), index=False)
            tot[0] += usage[0]
            tot[1] += usage[1]
            if usage[2]:
                errors.append(usage[2])
            if k % 20 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} requests, tokens in/out so far {tot[0]}/{tot[1]}", flush=True)
    if errors:
        print(f"{len(errors)} requests failed after 4 tries. First error: {errors[0]}", flush=True)
    d = pd.read_csv(long_path)
    d = d[d.ok]
    g = d.groupby("movie_id")
    res = pd.DataFrame({"film_known": g["known"].mean() > 0.5,
                        "film_pop_sonnet": d[d.known].groupby("movie_id")["popularity"].median(),
                        "film_quality_sonnet": d[d.known].groupby("movie_id")["quality"].median()}).reset_index()
    res.to_csv(OUT / "film_scores.csv", index=False)
    print("written:", OUT / "film_scores.csv", "| films with a score:", int(res.film_quality_sonnet.notna().sum()), "of", len(F))
    sys.exit(0)


if __name__ == "__main__":
    main()
