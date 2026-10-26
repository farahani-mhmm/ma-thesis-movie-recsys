"""Fame scores for directors and actors with Claude Sonnet 5.5.

A Claude version of llm.py (which uses the OpenAI SDK and is left untouched). It scores every unique
person once per run, repeats the run several times and takes the median (Sonnet 5.5 does not accept a
temperature, so repeats are how the noise is averaged out), then builds movie-level columns.

Typical use (each command is a dry run until you add --yes, because --yes spends money):

    pip install -U anthropic
    # the API key comes from the environment (ANTHROPIC_API_KEY) or `ant auth login`; never write it in code

    python llm_claude.py --pilot-file pilot_directors_40.txt --runs 3 --mode sync            # plan only
    python llm_claude.py --pilot-file pilot_directors_40.txt --runs 3 --mode sync --yes      # real pilot
    python llm_claude.py --runs 3 --mode batch --yes                                         # all people, Batches API
    python llm_claude.py --resume llm_claude_outputs/<run folder> --mode batch --yes         # continue a batch

Outputs (in llm_claude_outputs/<model>_<timestamp>/):
    people.csv, batches.json          what was asked (same batches are reused on --resume)
    runs_long.csv                     one row per (person, run): score, status
    person_scores_sonnet.csv          per person: median score, min, max, how often unknown
    movies_with_sonnet_fame.csv       movies.csv + the new movie-level columns (GPT columns are kept)
    run_metadata.json                 model, prompt hash, status counts, token usage, package versions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
MOVIES_CSV = BASE_DIR / "movies.csv"
OUTPUT_ROOT = BASE_DIR / "llm_claude_outputs"
CSV_ENCODING = "utf-8-sig"

MODEL = "claude-sonnet-5-5"          # exact id, no date suffix
EFFORT = "low"                        # thinking depth; fame lookup needs little reasoning
MAX_TOKENS = 4000                     # thinking tokens count toward this, so leave room
DEFAULT_BATCH_SIZE = 10               # people per request, as in llm.py
DEFAULT_RUNS = 3                      # repeats; the final score is the median
POLL_SECONDS = 30

INVALID_NAMES = {"NO_CREDITED_ACTORS", "Varios", "NaN", "", "N/A", "Unknown", "None", "No Credit", "Uncredited"}
MAX_ACTORS_PER_MOVIE_FOR_SCORING: Optional[int] = None   # None = score everybody, as llm.py did

# Rough price table for the dry-run estimate only (USD per million tokens, Claude Sonnet 5.5).
PRICE_INPUT_PER_M = 2.0
PRICE_OUTPUT_PER_M = 10.0
BATCH_DISCOUNT = 0.5

SYSTEM_PROMPT = """You are a film historian who rates how famous film people are.

For each numbered person, give an integer fame score from 1 to 10:
- 9-10: global megastar or legendary filmmaker, recognised everywhere.
- 7-8: very famous, several major mainstream hits, wide international recognition.
- 5-6: well known; recognised by film fans and by audiences in some regions.
- 3-4: niche; known mainly by dedicated fans or within one genre.
- 1-2: obscure, local, or almost unknown.

Rules:
1. Rate GLOBAL fame, not fame in a single country.
2. Directors and actors use the same scale.
3. A person known mainly in a non-English-speaking film industry is still rated on international recognition.
4. The film named after each person only tells you which person is meant. Do not rate the film.
5. If you do not recognise the person, set known to false and score to 0. Do not guess.
6. Return exactly one entry for every number in the list, using the same number as id."""

# Plain types only (integer, boolean), so every platform accepts the schema. The 1..10 range is checked in code.
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "known": {"type": "boolean"},
                    "score": {"type": "integer"},
                },
                "required": ["id", "known", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# =============================================================================
# PEOPLE AND BATCHES
# =============================================================================

def clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().strip("|").strip())


def extract_names(cell: Any, limit: Optional[int] = None) -> List[str]:
    if pd.isna(cell) or str(cell).strip() == "":
        return []
    names = [clean_name(n) for n in str(cell).split("|")]
    names = [n for n in names if n and n not in INVALID_NAMES]
    return names[:limit] if limit else names


def build_people(movies: pd.DataFrame, only: Optional[List[str]] = None) -> pd.DataFrame:
    """One row per unique person: name, role and one film (to tell same-named people apart)."""
    seen: Dict[str, Tuple[str, str]] = {}
    director_film: Dict[str, str] = {}
    title_col = "original_title" if "original_title" in movies.columns else "main_title"
    for _, row in movies.iterrows():
        film = str(row.get(title_col, "") or "")
        for role, col, limit in (("director", "directors", None), ("actor", "actors", MAX_ACTORS_PER_MOVIE_FOR_SCORING)):
            for name in extract_names(row.get(col), limit):
                seen.setdefault(name, (role, film))
                if role == "director":
                    director_film.setdefault(name, film)
    if only is not None:
        # Pilot list = directors (the human face-validity test). Always ask as director, with a film they directed.
        people = pd.DataFrame({"name": only, "role": "director", "film": [director_film.get(n, "") for n in only]})
    else:
        people = pd.DataFrame([(n, r, f) for n, (r, f) in seen.items()], columns=["name", "role", "film"])
    return people.sort_values("name", kind="mergesort").reset_index(drop=True)


def make_batches(people: pd.DataFrame, batch_size: int) -> List[List[Dict[str, str]]]:
    rows = people.to_dict("records")
    return [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]


def user_message(batch: List[Dict[str, str]]) -> str:
    lines = []
    for i, p in enumerate(batch, start=1):
        film = f' of "{p["film"]}"' if p.get("film") else ""
        lines.append(f"{i}. {p['name']} ({p['role']}{film})")
    return "Rate these people:\n" + "\n".join(lines)


def request_params(batch: List[Dict[str, str]]) -> Dict[str, Any]:
    # No temperature (not accepted by Sonnet 5.5), no `thinking` field (it runs adaptive; "disabled" is a 400).
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": user_message(batch)}],
        "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    }


def custom_id(run: int, batch_index: int) -> str:
    return f"r{run:02d}_b{batch_index:05d}"


def parse_custom_id(cid: str) -> Tuple[int, int]:
    m = re.fullmatch(r"r(\d+)_b(\d+)", cid)
    if not m:
        raise ValueError(f"unexpected custom_id: {cid}")
    return int(m.group(1)), int(m.group(2))


# =============================================================================
# PARSING
# =============================================================================

NAN = float("nan")


def parse_message(message: Any, n: int) -> Dict[int, Tuple[float, str]]:
    """Map id (1..n) -> (score, status). Status: ok, unknown, invalid_score, missing, bad_json, no_text, stop_<reason>."""
    expected = list(range(1, n + 1))
    reason = getattr(message, "stop_reason", None)
    if reason in ("max_tokens", "refusal"):
        return {i: (NAN, f"stop_{reason}") for i in expected}
    text = next((b.text for b in getattr(message, "content", []) if getattr(b, "type", "") == "text"), None)
    if text is None:
        return {i: (NAN, "no_text") for i in expected}
    try:
        items = json.loads(text)["scores"]
    except (ValueError, KeyError, TypeError):
        return {i: (NAN, "bad_json") for i in expected}

    out: Dict[int, Tuple[float, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        i = item.get("id")
        if not isinstance(i, int) or i not in expected or i in out:
            continue
        score = item.get("score")
        if item.get("known") is False:
            out[i] = (NAN, "unknown")
        elif isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 10:
            out[i] = (float(score), "ok")
        else:
            out[i] = (NAN, "invalid_score")
    for i in expected:
        out.setdefault(i, (NAN, "missing"))
    return out


def rows_from_message(message: Any, batch: List[Dict[str, str]], run: int) -> List[Dict[str, Any]]:
    parsed = parse_message(message, len(batch))
    return [
        {"name": p["name"], "run": run, "score": parsed[i][0], "status": parsed[i][1]}
        for i, p in enumerate(batch, start=1)
    ]


def usage_of(message: Any) -> Tuple[int, int]:
    u = getattr(message, "usage", None)
    return int(getattr(u, "input_tokens", 0) or 0), int(getattr(u, "output_tokens", 0) or 0)


# =============================================================================
# RUNNING (sync and Batches API)
# =============================================================================

def make_client():
    try:
        import anthropic
    except ImportError:
        sys.exit("The 'anthropic' package is not installed. Run: pip install -U anthropic")
    return anthropic.Anthropic()  # key from ANTHROPIC_API_KEY or an `ant auth login` profile


def run_sync(client, batches, runs: int) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    rows: List[Dict[str, Any]] = []
    tokens_in = tokens_out = 0
    total = runs * len(batches)
    done = 0
    for run in range(1, runs + 1):
        for b_idx, batch in enumerate(batches):
            message = client.messages.create(**request_params(batch))
            rows.extend(rows_from_message(message, batch, run))
            a, b = usage_of(message)
            tokens_in += a
            tokens_out += b
            done += 1
            if done % 20 == 0 or done == total:
                log(f"sync progress: {done}/{total} requests")
    return rows, (tokens_in, tokens_out)


def run_batch_api(client, batches, runs: int, state_path: Path) -> Tuple[List[Dict[str, Any]], Tuple[int, int]]:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    if state_path.exists():
        batch_id = json.loads(state_path.read_text(encoding="utf-8"))["batch_id"]
        log(f"Resuming existing batch {batch_id}")
    else:
        requests = [
            Request(custom_id=custom_id(run, b_idx), params=MessageCreateParamsNonStreaming(**request_params(batch)))
            for run in range(1, runs + 1)
            for b_idx, batch in enumerate(batches)
        ]
        if len(requests) > 100_000:
            sys.exit(f"{len(requests)} requests is more than one batch allows (100,000).")
        created = client.messages.batches.create(requests=requests)
        batch_id = created.id
        state_path.write_text(json.dumps({"batch_id": batch_id, "created_at": datetime.now().isoformat(),
                                          "n_requests": len(requests)}, indent=2), encoding="utf-8")
        log(f"Created batch {batch_id} with {len(requests)} requests (saved to {state_path.name})")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        log(f"batch {batch.processing_status}: processing={counts.processing} succeeded={counts.succeeded} errored={counts.errored}")
        if batch.processing_status == "ended":
            break
        time.sleep(POLL_SECONDS)

    rows: List[Dict[str, Any]] = []
    tokens_in = tokens_out = 0
    got = set()
    for result in client.messages.batches.results(batch_id):
        run, b_idx = parse_custom_id(result.custom_id)
        got.add((run, b_idx))
        names = batches[b_idx]
        if result.result.type == "succeeded":
            message = result.result.message
            rows.extend(rows_from_message(message, names, run))
            a, b = usage_of(message)
            tokens_in += a
            tokens_out += b
        else:
            rows.extend({"name": p["name"], "run": run, "score": NAN, "status": f"batch_{result.result.type}"} for p in names)
    for run in range(1, runs + 1):  # requests that never came back
        for b_idx, names in enumerate(batches):
            if (run, b_idx) not in got:
                rows.extend({"name": p["name"], "run": run, "score": NAN, "status": "batch_missing"} for p in names)
    return rows, (tokens_in, tokens_out)


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_people(runs_long: pd.DataFrame, people: pd.DataFrame) -> pd.DataFrame:
    """Median over the valid runs. A person judged 'unknown' in as many runs as 'ok' runs gets no score."""
    g = runs_long.groupby("name")
    summary = pd.DataFrame({
        "n_runs": g.size(),
        "n_ok": g["status"].apply(lambda s: int((s == "ok").sum())),
        "n_unknown": g["status"].apply(lambda s: int((s == "unknown").sum())),
        "score_min": g["score"].min(),
        "score_max": g["score"].max(),
        "score_median": g["score"].median(),
    }).reset_index()
    summary.loc[summary["n_ok"] <= summary["n_unknown"], "score_median"] = np.nan
    return people.merge(summary, on="name", how="left")


def movie_features(movies: pd.DataFrame, person_scores: pd.DataFrame) -> pd.DataFrame:
    score = dict(zip(person_scores["name"], person_scores["score_median"]))
    out = movies.copy()
    dir_mean, act_mean, act_max, act_top3, nd_all, nd_ok, na_all, na_ok = [], [], [], [], [], [], [], []
    for _, row in movies.iterrows():
        d = extract_names(row.get("directors"))
        a = extract_names(row.get("actors"), MAX_ACTORS_PER_MOVIE_FOR_SCORING)
        ds = [score[n] for n in d if n in score and pd.notna(score[n])]
        as_ = [score[n] for n in a if n in score and pd.notna(score[n])]
        top3 = [score[n] for n in a[:3] if n in score and pd.notna(score[n])]
        dir_mean.append(round(float(np.mean(ds)), 4) if ds else np.nan)
        act_mean.append(round(float(np.mean(as_)), 4) if as_ else np.nan)
        act_max.append(max(as_) if as_ else np.nan)
        act_top3.append(round(float(np.mean(top3)), 4) if top3 else np.nan)
        nd_all.append(len(d)); nd_ok.append(len(ds)); na_all.append(len(a)); na_ok.append(len(as_))
    out["director_fame_score_sonnet"] = dir_mean
    out["actors_fame_score_sonnet"] = act_mean
    out["actors_fame_max_sonnet"] = act_max
    out["actors_fame_top3_mean_sonnet"] = act_top3
    out["n_directors"], out["n_directors_scored"] = nd_all, nd_ok
    out["n_actors"], out["n_actors_scored"] = na_all, na_ok
    return out


# =============================================================================
# PLAN / MAIN
# =============================================================================

def estimate(batches, runs: int, mode: str) -> Dict[str, float]:
    n_requests = len(batches) * runs
    prompt_chars = len(SYSTEM_PROMPT) + sum(len(user_message(b)) for b in batches) / max(len(batches), 1)
    in_per_req = prompt_chars / 3.5 + 150                         # rough chars-to-tokens, plus schema overhead
    out_per_req = (sum(len(b) for b in batches) / max(len(batches), 1)) * 14 + 300   # JSON entries plus a little thinking
    cost = (n_requests * in_per_req * PRICE_INPUT_PER_M + n_requests * out_per_req * PRICE_OUTPUT_PER_M) / 1e6
    if mode == "batch":
        cost *= BATCH_DISCOUNT
    return {"requests": n_requests, "input_tokens": n_requests * in_per_req, "output_tokens": n_requests * out_per_req, "usd": cost}


def package_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata
    out = {}
    for name in ("anthropic", "pandas", "numpy"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--movies", type=Path, default=MOVIES_CSV)
    ap.add_argument("--pilot-file", type=Path, help="text file with one person name per line; scores only those people")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--mode", choices=["sync", "batch"], default="batch")
    ap.add_argument("--out-dir", type=Path, help="output folder (default: a new timestamped folder)")
    ap.add_argument("--resume", type=Path, help="existing output folder to continue (batch mode)")
    ap.add_argument("--yes", action="store_true", help="actually call the API (this spends money). Without it: plan only.")
    args = ap.parse_args(argv)

    movies = pd.read_csv(args.movies, encoding=CSV_ENCODING, low_memory=False)

    if args.resume:
        out_dir = args.resume
        batches_raw = json.loads((out_dir / "batches.json").read_text(encoding="utf-8"))
        batches = batches_raw["batches"]
        runs = batches_raw["runs"]
        people = pd.read_csv(out_dir / "people.csv", encoding=CSV_ENCODING).fillna({"film": ""})
    else:
        only = None
        if args.pilot_file:
            only = [ln.strip() for ln in args.pilot_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        people = build_people(movies, only)
        batches = make_batches(people, args.batch_size)
        runs = args.runs
        out_dir = args.out_dir or OUTPUT_ROOT / f"{MODEL}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    est = estimate(batches, runs, args.mode)
    log(f"People to score: {len(people)} ({(people['role'] == 'director').sum()} first seen as directors, "
        f"{(people['role'] == 'actor').sum()} as actors)")
    log(f"Plan: {len(batches)} batches x {runs} runs = {est['requests']} requests, mode={args.mode}, model={MODEL}")
    log(f"Rough cost estimate: about ${est['usd']:.2f} (input ~{est['input_tokens'] / 1e6:.2f}M tokens, output ~{est['output_tokens'] / 1e6:.2f}M tokens). "
        "This is an estimate from character counts, not a measurement.")
    if not args.yes:
        log("Dry run only. Add --yes to call the API.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        people.to_csv(out_dir / "people.csv", index=False, encoding=CSV_ENCODING)
        (out_dir / "batches.json").write_text(json.dumps({"runs": runs, "batches": batches}, ensure_ascii=False), encoding="utf-8")

    client = make_client()
    started = datetime.now().isoformat()
    if args.mode == "sync":
        rows, (tok_in, tok_out) = run_sync(client, batches, runs)
    else:
        rows, (tok_in, tok_out) = run_batch_api(client, batches, runs, out_dir / "batch_state.json")

    runs_long = pd.DataFrame(rows)
    runs_long.to_csv(out_dir / "runs_long.csv", index=False, encoding=CSV_ENCODING)
    person_scores = aggregate_people(runs_long, people)
    person_scores.to_csv(out_dir / "person_scores_sonnet.csv", index=False, encoding=CSV_ENCODING)
    movie_features(movies, person_scores).to_csv(out_dir / "movies_with_sonnet_fame.csv", index=False, encoding=CSV_ENCODING)

    status_counts = runs_long["status"].value_counts().to_dict()
    actual_cost = (tok_in * PRICE_INPUT_PER_M + tok_out * PRICE_OUTPUT_PER_M) / 1e6 * (BATCH_DISCOUNT if args.mode == "batch" else 1)
    meta = {
        "model": MODEL, "effort": EFFORT, "mode": args.mode, "runs": runs, "batch_size": args.batch_size,
        "started_at": started, "finished_at": datetime.now().isoformat(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "schema": SCHEMA, "n_people": int(len(people)), "n_requests": est["requests"],
        "status_counts": status_counts, "input_tokens": tok_in, "output_tokens": tok_out,
        "approx_cost_usd": round(actual_cost, 4), "package_versions": package_versions(),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Status counts: {status_counts}")
    log(f"Tokens in/out: {tok_in}/{tok_out} (~${actual_cost:.2f}). Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
