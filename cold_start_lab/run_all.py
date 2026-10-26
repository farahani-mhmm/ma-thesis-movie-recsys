"""Run steps 4, 5 and 6 one after another.

    python run_all.py --quick        smoke test: the whole chain on a small slice (minutes). Run this FIRST.
    python run_all.py                the full run (about an hour or more; each step prints its progress)
    python run_all.py --force        recompute even if outputs exist
    python run_all.py --with-rf      also run random forests (much slower)

The chain stops at the first error. Steps 1-3 must already have been run.
"""
import subprocess
import sys
import time

from config import LAB_DIR

STEPS = ["04_profile_models.py", "05_history_models.py", "06_report.py"]


def main():
    args = sys.argv[1:]
    flags = ("--quick", "--force", "--with-rf")
    valued = ("--train-per-user", "--out-tag")
    passthrough, i = [], 0
    while i < len(args):
        if args[i] in flags:
            passthrough.append(args[i])
            i += 1
        elif args[i] in valued and i + 1 < len(args):
            passthrough += [args[i], args[i + 1]]
            i += 2
        else:
            sys.exit(f"Unknown argument: {args[i]}. Allowed: {' '.join(flags)} --train-per-user N --out-tag TEXT")
    t0 = time.time()
    for step in STEPS:
        # every step accepts all the arguments; step 4 simply ignores --train-per-user and --out-tag
        print(f"\n===== {step} {' '.join(passthrough)} =====", flush=True)
        r = subprocess.run([sys.executable, step] + passthrough, cwd=str(LAB_DIR))
        if r.returncode != 0:
            sys.exit(f"\nStopped: {step} failed (exit code {r.returncode}). Send me the last lines above.")
        print(f"----- {step} finished, {time.time() - t0:.0f}s so far", flush=True)
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
