"""Sparsity test driver (step 14 with --keep-frac): one scenario, five levels of kept ratings, hgb_c only, N = 0,5 / M = 0,5.

usage:  python run_sparsity.py A      (or B, C)   -> outputs/14_final_sp100 ... _sp005 and the log outputs/sparse_run_<S>.log
Safe to start again: finished folds are skipped.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

here = Path(__file__).resolve().parent
scen = sys.argv[1].upper()
assert scen in ("A", "B", "C"), "scenario must be A, B or C"
log = here / "outputs" / f"sparse_run_{scen}.log"
with open(log, "a", encoding="utf-8") as lf:
    for keep in (1.0, 0.5, 0.2, 0.1, 0.05):
        tag = f"_sp{round(keep * 100):03d}"
        lf.write(f"=== scenario {scen} keep {keep} tag {tag} {datetime.now()}\n")
        lf.flush()
        subprocess.run([sys.executable, str(here / "14_final_protocol.py"), "--scenario", scen, "--tmdb", "--learners", "hgb_c",
                        "--n-list", "0,5", "--m-list", "0,5", "--keep-frac", str(keep), "--out-tag", tag],
                       stdout=lf, stderr=subprocess.STDOUT, cwd=str(here))
    lf.write(f"=== finished scenario {scen} {datetime.now()}\n")
