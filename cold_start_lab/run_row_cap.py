"""Row cap learning curve driver (step 14): one scenario, every level of the per-user training cap, nothing else changed.

Level 200 is not run here: it comes from outputs/14_final_filmq and outputs/14_final.
Two experiment lists are needed, because --filmq keeps only the TMDB/Q experiments:
    --filmq     blocks P, H, T, K, Q, PUB, PRfuzzy        -> outputs/14_final_filmq_tpu<level>[_uw]
    no flag     blocks L, A, FH, PRraw, PRcrisp, FULL-x   -> outputs/14_final_tpu<level>[_uw]
"all" means no cap (a value above any user's count); "_uw" adds --user-weight.

usage:  python run_row_cap.py A      (or B, C)   -> log in outputs/rowcap_run_<S>.log
Safe to start again: finished folds are skipped.
"""
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

NO_CAP = 99999                     # the largest user has 1085 ratings
LEVELS = [("all", NO_CAP, False), ("400", 400, False), ("100", 100, False), ("all", NO_CAP, True)]
OMP_THREADS = "5"                  # three scenarios run side by side on 16 logical cores; hgb_c results do not depend on this
BLAS_THREADS = "16"                # keep the default: fewer BLAS threads change logreg in the last digits (checked on one fold)

here = Path(__file__).resolve().parent
scen = sys.argv[1].upper()
assert scen in ("A", "B", "C"), "scenario must be A, B or C"
env = {**os.environ, "OMP_NUM_THREADS": OMP_THREADS, "OPENBLAS_NUM_THREADS": BLAS_THREADS}
log = here / "outputs" / f"rowcap_run_{scen}.log"
with open(log, "a", encoding="utf-8") as lf:
    for flags, prefix in ((["--filmq"], "_filmq"), ([], "")):
        for name, cap, weighted in LEVELS:
            tag = f"{prefix}_tpu{name}" + ("_uw" if weighted else "")
            lf.write(f"=== scenario {scen} cap {cap} weighted {weighted} tag {tag} {datetime.now()}\n")
            lf.flush()
            r = subprocess.run([sys.executable, str(here / "14_final_protocol.py"), "--scenario", scen, *flags, "--train-per-user", str(cap),
                                *(["--user-weight"] if weighted else []), "--out-tag", tag],
                               stdout=lf, stderr=subprocess.STDOUT, cwd=str(here), env=env)
            lf.write(f"=== exit code {r.returncode} tag {tag} {datetime.now()}\n")
            lf.flush()
            if r.returncode != 0:
                sys.exit(f"step 14 failed for {tag}: see {log}")
    lf.write(f"=== finished scenario {scen} {datetime.now()}\n")
