"""Fame scores with Sonnet 5.5 through Liara's OpenAI-compatible endpoint (same prompt, schema, parsing and outputs
as llm_claude.py; only the way the model is called changes: sync requests, no Batches API).

Settings come from environment variables, never from the code:
    LIARA_BASE_URL   https://ai.liara.ir/api/<your id>/v1
    LIARA_API_KEY    key made in the Liara panel
    LIARA_MODEL      optional, default anthropic/claude-sonnet-5.5

Every command is a DRY RUN (plan and cost estimate, no API call) until you add --yes, because --yes spends money:
    python llm_liara.py --pilot-file pilot_directors_40.txt --runs 3                 # plan only
    python llm_liara.py --pilot-file pilot_directors_40.txt --runs 3 --yes           # real pilot (40 directors)
    python llm_liara.py --runs 3 --yes                                               # everybody (directors + actors)
Outputs (a new timestamped folder under llm_claude_outputs/): runs_long.csv, person_scores_sonnet.csv,
movies_with_sonnet_fame.csv, run_metadata.json  (the same files as llm_claude.py, so step 10 reads them).
"""
import json
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime
from types import SimpleNamespace

import llm_claude as lc

BASE_URL = os.environ.get("LIARA_BASE_URL")
API_KEY = os.environ.get("LIARA_API_KEY")
MODEL = os.environ.get("LIARA_MODEL", "anthropic/claude-sonnet-5.5")
RETRIES = 4
_use_schema = True
_token_key = "max_tokens"
MAX_TOKENS = int(os.environ.get("LIARA_MAX_TOKENS", lc.MAX_TOKENS))   # reasoning models also count thinking here


def make_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("The 'openai' package is not installed. Run: python -m pip install -U openai")
    if not BASE_URL or not API_KEY:
        sys.exit("Set LIARA_BASE_URL and LIARA_API_KEY first.")
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def as_message(resp):
    """Wrap an OpenAI-style response so that llm_claude.parse_message / usage_of can read it."""
    choice = resp.choices[0]
    text = (choice.message.content or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    reason = {"length": "max_tokens", "content_filter": "refusal"}.get(choice.finish_reason, choice.finish_reason)
    u = resp.usage
    return SimpleNamespace(
        stop_reason=reason,
        content=[SimpleNamespace(type="text", text=text)] if text else [],
        usage=SimpleNamespace(input_tokens=getattr(u, "prompt_tokens", 0), output_tokens=getattr(u, "completion_tokens", 0)),
    )


def call(client, batch):
    global _use_schema, _token_key
    kwargs = {
        "model": MODEL,
        "messages": [{"role": "system", "content": lc.SYSTEM_PROMPT},
                     {"role": "user", "content": lc.user_message(batch)}],
    }
    for attempt in range(1, RETRIES + 1):
        try:
            kwargs.pop("max_tokens", None)
            kwargs.pop("max_completion_tokens", None)
            kwargs[_token_key] = MAX_TOKENS
            if _use_schema:
                kwargs["response_format"] = {"type": "json_schema",
                                             "json_schema": {"name": "scores", "schema": lc.SCHEMA, "strict": True}}
            else:
                kwargs.pop("response_format", None)
                kwargs["messages"][-1]["content"] = (
                    lc.user_message(batch) + '\n\nReturn ONLY JSON like {"scores": [{"id": 1, "known": true, "score": 7}]}.')
            return client.chat.completions.create(**kwargs)
        except Exception as e:                      # noqa: BLE001
            msg = str(e)
            if "402" in msg or "balance" in msg.lower():
                raise SystemExit("STOP: the Liara balance is too low (402). Top up the account, then run the same command again; it resumes.")
            if _token_key == "max_tokens" and "max_completion_tokens" in msg:
                lc.log("this model wants max_completion_tokens instead of max_tokens; switching")
                _token_key = "max_completion_tokens"
                continue
            if _use_schema and ("response_format" in msg or "json_schema" in msg or "400" in msg):
                lc.log("json_schema not accepted by the endpoint; continuing with a JSON instruction in the prompt")
                _use_schema = False
                continue
            if attempt == RETRIES:
                raise
            lc.log(f"request failed ({msg[:120]}); retry {attempt}/{RETRIES - 1}")
            time.sleep(5 * attempt)


OUT_DIR = None          # set in __main__: partial results are written here so a crash or reboot loses nothing
WORKERS = int(os.environ.get("LIARA_WORKERS", "6"))


def run_sync(client, batches, runs):
    """Parallel requests (WORKERS threads). Every finished request is appended to partial_rows.jsonl; a second run in
    the same folder (--resume <folder>) skips what is already there."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    part = OUT_DIR / "partial_rows.jsonl" if OUT_DIR else None
    done_keys, rows, tok_in, tok_out = set(), [], 0, 0
    if part and part.exists():
        for ln in part.read_text(encoding="utf-8").splitlines():
            rec = json.loads(ln)
            done_keys.add((rec["run"], rec["b"]))
            rows.extend(rec["rows"])
            tok_in += rec["tin"]
            tok_out += rec["tout"]
        lc.log(f"resuming: {len(done_keys)} requests already done")
    jobs = [(run, b) for run in range(1, runs + 1) for b in range(len(batches)) if (run, b) not in done_keys]
    total = runs * len(batches)
    lock = threading.Lock()
    t0, finished = time.time(), 0

    def work(run, b):
        message = as_message(call(client, batches[b]))
        a, c = lc.usage_of(message)
        return run, b, lc.rows_from_message(message, batches[b], run), a, c

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, run, b) for run, b in jobs]
        for fut in as_completed(futs):
            try:
                run, b, r, a, c = fut.result()      # an exception here stops the run; run the same command again to continue
            except BaseException:
                ex.shutdown(wait=False, cancel_futures=True)
                raise
            with lock:
                rows.extend(r)
                tok_in += a
                tok_out += c
                finished += 1
                if part:
                    with part.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"run": run, "b": b, "rows": r, "tin": a, "tout": c}, ensure_ascii=False) + chr(10))
                if finished % 20 == 0 or finished == len(jobs):
                    el = time.time() - t0
                    eta = el / finished * (len(jobs) - finished)
                    cost = (tok_in * lc.PRICE_INPUT_PER_M + tok_out * lc.PRICE_OUTPUT_PER_M) / 1e6
                    lc.log(f"{len(done_keys) + finished}/{total} requests | elapsed {el / 60:.1f} min | about "
                           f"{eta / 60:.1f} min left | ~${cost:.2f} so far")
    return rows, (tok_in, tok_out)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--mode" not in argv:
        argv += ["--mode", "sync"]
    if "--out-dir" not in argv and "--resume" not in argv:
        slug = re.sub(r"[^A-Za-z0-9.]+", "-", MODEL).strip("-")
        argv += ["--out-dir", str(lc.OUTPUT_ROOT / f"liara_{slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")]
    outs = argv[argv.index("--resume") + 1] if "--resume" in argv else argv[argv.index("--out-dir") + 1]
    OUT_DIR = Path(outs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lc.MODEL = MODEL
    lc.make_client = make_client
    lc.run_sync = run_sync
    lc.main(argv)
