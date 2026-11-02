"""Fame scores for directors and actors with Gemini (default gemini-3.8-flash).

The Gemini copy of llm_claude.py. Everything except the model is taken from llm_claude.py so the two runs can be
compared: the same system prompt, the same JSON schema, the same parsing, the same aggregation (median per person,
"unknown" handling) and the same movie-level columns. By default it also reuses the exact people list and batches of
the full Sonnet run (llm_claude_outputs/full_sonnet55), with 1 run, like that run.

The free Gemini tier has a requests-per-minute and requests-per-day limit, so every finished request is appended to
partial_rows.jsonl right away. Starting the same command again continues where it stopped.

Typical use (a dry run until you add --yes):

    pip install -U google-genai
    # the key comes from the environment (GEMINI_API_KEY); never write it in code

    python llm_gemini.py --pilot-file pilot_directors_40.txt --out-dir llm_gemini_outputs/pilot40            # plan only
    python llm_gemini.py --pilot-file pilot_directors_40.txt --out-dir llm_gemini_outputs/pilot40 --yes      # pilot
    python llm_gemini.py --yes                                                                                # full run
    python llm_gemini.py --yes                                                                                # again = resume

Outputs (default folder llm_gemini_outputs/gemini38/):
    people.csv, batches.json          what was asked (copied from the Sonnet run unless --new-batches)
    partial_rows.jsonl                one line per finished request (used to resume)
    runs_long.csv                     one row per (person, run): score, status
    person_scores_gemini.csv          per person: median score, min, max, how often unknown
    movies_with_gemini_fame.csv       movies.csv + the movie-level columns. The columns keep the *_sonnet names that
                                      14_final_protocol.py reads (--llm-csv), but they hold the Gemini scores.
    run_metadata.json                 model, prompt hash, status counts, token usage, package versions
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import llm_claude as lc

# =============================================================================
# CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "llm_gemini_outputs"
SONNET_RUN = BASE_DIR / "llm_claude_outputs" / "full_sonnet55"
CSV_ENCODING = lc.CSV_ENCODING

MODEL = "gemini-3.8-flash"           # fixed id (not a *-latest alias), so the run can be repeated
THINKING = "low"                     # same idea as EFFORT = "low" in llm_claude.py
MAX_TOKENS = lc.MAX_TOKENS
DEFAULT_RUNS = 1                     # the full Sonnet run has 1 run, so the comparison uses 1 as well
SLEEP_SECONDS = 4.0                  # pause between requests (free tier: limited requests per minute)
RETRY_WAITS = [30, 60, 120, 300]     # waits after a 429 / 5xx before giving up on this invocation

FINISH_TO_STATUS = {"MAX_TOKENS": "stop_max_tokens", "SAFETY": "stop_refusal", "PROHIBITED_CONTENT": "stop_refusal",
                    "BLOCKLIST": "stop_refusal", "RECITATION": "stop_recitation"}


# =============================================================================
# GEMINI CALL
# =============================================================================

def make_client():
    try:
        from google import genai
    except ImportError:
        sys.exit("The 'google-genai' package is not installed. Run: pip install -U google-genai")
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        sys.exit("GEMINI_API_KEY is not set in this terminal window. Set it first (see the commands given in chat).")
    return genai.Client()  # key from GEMINI_API_KEY (or GOOGLE_API_KEY)


def make_config(thinking: str):
    from google.genai import types
    kw: Dict[str, Any] = dict(system_instruction=lc.SYSTEM_PROMPT, response_mime_type="application/json",
                              response_json_schema=lc.SCHEMA, max_output_tokens=MAX_TOKENS)
    if thinking != "default":
        kw["thinking_config"] = types.ThinkingConfig(thinking_level=thinking.upper())
    return types.GenerateContentConfig(**kw)


def as_message(resp: Any) -> SimpleNamespace:
    """Wrap a Gemini response like an Anthropic message, so lc.parse_message (the Sonnet parser) is used unchanged."""
    cands = getattr(resp, "candidates", None) or []
    finish = getattr(getattr(cands[0], "finish_reason", None), "name", None) if cands else "NO_CANDIDATE"
    status = FINISH_TO_STATUS.get(finish or "")
    if status:
        return SimpleNamespace(stop_reason=status.removeprefix("stop_"), content=[])
    try:
        text = resp.text
    except Exception:
        text = None
    content = [SimpleNamespace(type="text", text=text)] if text else []
    return SimpleNamespace(stop_reason=None, content=content)


def usage_of(resp: Any) -> Tuple[int, int]:
    u = getattr(resp, "usage_metadata", None)
    out = (getattr(u, "candidates_token_count", 0) or 0) + (getattr(u, "thoughts_token_count", 0) or 0)
    return int(getattr(u, "prompt_token_count", 0) or 0), int(out)


def call(client, config, model: str, batch: List[Dict[str, str]]):
    """One request with waits on rate limits, server errors and dropped connections.
    Returns the response, or None to stop this invocation."""
    import httpx
    from google.genai import errors
    for wait in RETRY_WAITS + [None]:
        try:
            return client.models.generate_content(model=model, contents=lc.user_message(batch), config=config)
        except httpx.TransportError as e:                  # connection dropped, timeout, proxy/VPN hiccup
            if wait is None:
                lc.log(f"network still failing ({type(e).__name__}); stopping. Run the same command later to continue.")
                return None
            lc.log(f"network error ({type(e).__name__}: {str(e)[:100]}); waiting {wait}s")
            time.sleep(wait)
        except errors.APIError as e:
            if e.code == 429 or (e.code or 0) >= 500:
                if wait is None:
                    lc.log(f"still {e.code} after retries; stopping. Run the same command later to continue.")
                    return None
                lc.log(f"{e.code} ({str(e.message)[:120]}); waiting {wait}s")
                time.sleep(wait)
            else:
                sys.exit(f"API error {e.code}: {e.message}")
    return None


# =============================================================================
# MAIN
# =============================================================================

def load_done(partial: Path) -> Tuple[set, List[Dict[str, Any]], int, int]:
    done, rows, tin, tout = set(), [], 0, 0
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done.add((rec["run"], rec["batch"]))
            rows.extend(rec["rows"])
            tin += rec["tokens_in"]
            tout += rec["tokens_out"]
    return done, rows, tin, tout


def package_versions() -> Dict[str, Optional[str]]:
    from importlib import metadata
    out = {}
    for name in ("google-genai", "pandas", "numpy"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--movies", type=Path, default=lc.MOVIES_CSV)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--thinking", default=THINKING, choices=["minimal", "low", "medium", "high", "default"],
                    help="thinking level; 'default' sends no thinking setting")
    ap.add_argument("--pilot-file", type=Path, help="one person name per line; scores only those people (as directors)")
    ap.add_argument("--new-batches", action="store_true", help="build the people list and batches again instead of copying the Sonnet run")
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    ap.add_argument("--batch-size", type=int, default=lc.DEFAULT_BATCH_SIZE, help="only with --new-batches or --pilot-file")
    ap.add_argument("--sleep", type=float, default=SLEEP_SECONDS, help="seconds between requests")
    ap.add_argument("--max-requests", type=int, default=None, help="stop after this many requests in this invocation")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_ROOT / "gemini38")
    ap.add_argument("--yes", action="store_true", help="actually call the API. Without it: plan only.")
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    movies = pd.read_csv(args.movies, encoding=CSV_ENCODING, low_memory=False)

    if (out_dir / "batches.json").exists():                        # resume: keep exactly what was planned
        raw = json.loads((out_dir / "batches.json").read_text(encoding="utf-8"))
        batches, runs = raw["batches"], raw["runs"]
        people = pd.read_csv(out_dir / "people.csv", encoding=CSV_ENCODING).fillna({"film": ""})
        source = "existing plan in the output folder"
    elif args.pilot_file or args.new_batches:
        only = None
        if args.pilot_file:
            only = [ln.strip() for ln in args.pilot_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        people = lc.build_people(movies, only)
        batches, runs = lc.make_batches(people, args.batch_size), args.runs
        source = "pilot file" if args.pilot_file else "new batches"
    else:
        raw = json.loads((SONNET_RUN / "batches.json").read_text(encoding="utf-8"))
        batches, runs = raw["batches"], args.runs
        people = pd.read_csv(SONNET_RUN / "people.csv", encoding=CSV_ENCODING).fillna({"film": ""})
        source = f"copied from {SONNET_RUN.name}"

    partial = out_dir / "partial_rows.jsonl"
    done, rows, tok_in, tok_out = load_done(partial)
    todo = [(r, b) for r in range(1, runs + 1) for b in range(len(batches)) if (r, b) not in done]
    lc.log(f"People: {len(people)}; batches: {len(batches)} x {runs} runs ({source}); model={args.model}, thinking={args.thinking}")
    lc.log(f"Requests finished: {len(done)}; still to do: {len(todo)}")
    if not args.yes:
        lc.log("Dry run only. Add --yes to call the API.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "batches.json").exists():
        people.to_csv(out_dir / "people.csv", index=False, encoding=CSV_ENCODING)
        (out_dir / "batches.json").write_text(json.dumps({"runs": runs, "batches": batches}, ensure_ascii=False), encoding="utf-8")
        if source.startswith("copied"):
            shutil.copy2(SONNET_RUN / "batches.json", out_dir / "batches_sonnet_original.json")

    client = make_client()
    try:
        config = make_config(args.thinking)
    except Exception as e:
        sys.exit(f"Could not build the request settings ({e}). Try --thinking default, or update: pip install -U google-genai")

    started = datetime.now().isoformat()
    sent = 0
    with partial.open("a", encoding="utf-8") as fh:
        for run, b_idx in todo:
            if args.max_requests is not None and sent >= args.max_requests:
                lc.log(f"--max-requests {args.max_requests} reached; run the same command again to continue.")
                break
            batch = batches[b_idx]
            resp = call(client, config, args.model, batch)
            if resp is None:
                break
            new_rows = lc.rows_from_message(as_message(resp), batch, run)
            a, b = usage_of(resp)
            fh.write(json.dumps({"run": run, "batch": b_idx, "rows": new_rows, "tokens_in": a, "tokens_out": b},
                                ensure_ascii=False, allow_nan=True) + "\n")
            fh.flush()
            rows.extend(new_rows)
            tok_in, tok_out, sent = tok_in + a, tok_out + b, sent + 1
            done.add((run, b_idx))
            if sent % 20 == 0 or len(done) == runs * len(batches):
                lc.log(f"progress: {len(done)}/{runs * len(batches)} requests")
            time.sleep(args.sleep)

    if len(done) < runs * len(batches):
        lc.log(f"Not finished: {len(done)}/{runs * len(batches)} requests. Run the same command again later to continue.")
        return

    runs_long = pd.DataFrame(rows)
    runs_long.to_csv(out_dir / "runs_long.csv", index=False, encoding=CSV_ENCODING)
    person_scores = lc.aggregate_people(runs_long, people)
    person_scores.to_csv(out_dir / "person_scores_gemini.csv", index=False, encoding=CSV_ENCODING)
    lc.movie_features(movies, person_scores).to_csv(out_dir / "movies_with_gemini_fame.csv", index=False, encoding=CSV_ENCODING)

    status_counts = runs_long["status"].value_counts().to_dict()
    meta = {
        "model": args.model, "thinking": args.thinking, "runs": runs, "n_batches": len(batches), "batches_source": source,
        "started_at": started, "finished_at": datetime.now().isoformat(),
        "system_prompt_sha256": hashlib.sha256(lc.SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "schema": lc.SCHEMA, "n_people": int(len(people)), "status_counts": status_counts,
        "input_tokens": tok_in, "output_tokens": tok_out, "package_versions": package_versions(),
        "note": "movie columns keep the *_sonnet names read by 14_final_protocol.py; they hold Gemini scores",
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    lc.log(f"Status counts: {status_counts}")
    lc.log(f"Tokens in/out: {tok_in}/{tok_out}. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
