"""Run the two extra experiments of stage 1 in one go:

    1. user-user collaborative filtering (step 5 with --with-uu, then the report)  -> outputs/05_history_models_uu, 06_report_uu
    2. both new: new user AND new film (step 8)                                    -> outputs/08_both_new_models

    python run_extra.py --quick     smoke test of both (minutes). Run this FIRST.
    python run_extra.py             the full runs
    python run_extra.py --force     recompute even if outputs exist

Steps 1-7 must already have been run. The chain stops at the first error.
"""
import subprocess
import sys
import time

from config import LAB_DIR

JOBS = [
    ("05_history_models.py", ["--with-uu", "--out-tag", "_uu"]),
    ("06_report.py", ["--out-tag", "_uu"]),
    ("08_both_new_models.py", []),
]


def main():
    flags = [a for a in sys.argv[1:] if a in ("--quick", "--force", "--with-rf")]
    unknown = [a for a in sys.argv[1:] if a not in ("--quick", "--force", "--with-rf")]
    if unknown:
        sys.exit(f"Unknown arguments: {unknown}. Allowed: --quick --force --with-rf")
    t0 = time.time()
    for script, extra in JOBS:
        args = extra + flags
        print(f"\n===== {script} {' '.join(args)} =====", flush=True)
        r = subprocess.run([sys.executable, script] + args, cwd=str(LAB_DIR))
        if r.returncode != 0:
            sys.exit(f"\nStopped: {script} failed (exit code {r.returncode}). Send me the last lines above.")
        print(f"----- {script} finished, {time.time() - t0:.0f}s so far", flush=True)
    print("\nAll extra experiments finished.")


if __name__ == "__main__":
    main()
