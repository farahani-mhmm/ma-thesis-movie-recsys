"""Build the final Sonnet person and film files from partial_rows.jsonl WITHOUT any API call.

Only complete runs are used (by default run 1), so every person is judged under the same rule.
Usage: python finalize_partial.py llm_claude_outputs/full_sonnet55 [--runs 1]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import llm_claude as lc

d = Path(sys.argv[1])
k = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 1
batches = json.loads((d / "batches.json").read_text(encoding="utf-8"))["batches"]
rec = [json.loads(ln) for ln in (d / "partial_rows.jsonl").read_text(encoding="utf-8").splitlines()]
done = {}
for r in rec:
    done.setdefault(r["run"], set()).add(r["b"])
use = [run for run in range(1, k + 1) if len(done.get(run, ())) == len(batches)]
if len(use) < k:
    sys.exit(f"Not all of runs 1..{k} are complete: {({r: len(v) for r, v in done.items()})} of {len(batches)} batches")
rows = [x for r in rec if r["run"] in use for x in r["rows"]]
people = pd.read_csv(d / "people.csv", encoding=lc.CSV_ENCODING).fillna({"film": ""})
runs_long = pd.DataFrame(rows)
ps = lc.aggregate_people(runs_long, people)
ps.to_csv(d / "person_scores_sonnet.csv", index=False, encoding=lc.CSV_ENCODING)
movies = pd.read_csv(lc.MOVIES_CSV, encoding=lc.CSV_ENCODING, low_memory=False)
lc.movie_features(movies, ps).to_csv(d / "movies_with_sonnet_fame.csv", index=False, encoding=lc.CSV_ENCODING)
runs_long.to_csv(d / "runs_long.csv", index=False, encoding=lc.CSV_ENCODING)
print(f"runs used: {use}; people: {len(ps)}; scored: {int(ps['score_median'].notna().sum())}; "
      f"unknown or failed: {int(ps['score_median'].isna().sum())}")
if 2 in done and 1 in use:
    a = pd.DataFrame([x for r in rec if r["run"] == 1 and r["b"] in done[2] for x in r["rows"]]).set_index("name")
    b = pd.DataFrame([x for r in rec if r["run"] == 2 for x in r["rows"]]).set_index("name")
    j = a[["score", "status"]].join(b[["score", "status"]], lsuffix="1", rsuffix="2", how="inner")
    both = j.dropna(subset=["score1", "score2"])
    agree = (j["status1"] == "unknown") == (j["status2"] == "unknown")
    print(f"stability on {len(j)} people scored twice: same known/unknown verdict {agree.mean():.3f}; "
          f"where both scored: exact same score {float((both.score1 == both.score2).mean()):.3f}, "
          f"mean |difference| {float((both.score1 - both.score2).abs().mean()):.3f}, "
          f"Spearman {both.score1.corr(both.score2, method='spearman'):.3f}")
