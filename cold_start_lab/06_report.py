"""Step 6: put everything together: tables with 95% intervals, paired differences, and the cold-start curves.

Reads  : 03_baselines, 04_profile_models, 05_history_models (the "_quick" folders with --quick)
Writes : outputs/06_report[_quick]/
            final_summary.csv    mean per-user AUC and 95% interval for every method, rule and N
            final_pairs.csv      paired differences (A - B) with 95% intervals
            final_report.txt     tables and answers to the research questions
            curves_<rule>.png    AUC against N (only if matplotlib is installed)

Reading guide
    - Methods without N (B0, B1, E0a, G0, E0b) do not use the revealed ratings, so their value is repeated
      for every N (flat lines in the curves).
    - A difference is called "better" only if its 95% interval is entirely above 0, "worse" if entirely below.
Needs: pandas, numpy (matplotlib optional).
"""
import sys

import numpy as np
import pandas as pd

from config import OUT_DIR, ensure_dir
from labutils import PRIMARY_LEARNER, parse_args, summarize

COLS = ["user_id", "repeat", "rule", "N", "method", "auc"]
BASELINES = ["B0_majority", "B1_pop_like", "B1_pop_mean"]


def load_all(cfg):
    tag = cfg["tag"]
    paths = {"03": OUT_DIR / "03_baselines" / "per_user_auc.csv",
             "04": OUT_DIR / f"04_profile_models{cfg['tag_quick']}" / "per_user_auc.csv",
             "05": OUT_DIR / f"05_history_models{tag}" / "per_user_auc.csv"}
    for k, p in paths.items():
        if not p.exists():
            sys.exit(f"Missing results of step {k}: {p}\nRun the earlier steps first.")
    d5 = pd.read_csv(paths["05"])
    keys = d5[["repeat", "fold", "rule"]].drop_duplicates()
    d3 = pd.read_csv(paths["03"]).merge(keys, on=["repeat", "fold", "rule"])
    d3 = d3[d3["method"].isin(BASELINES)]
    d4 = pd.read_csv(paths["04"]).merge(keys, on=["repeat", "fold", "rule"])
    ns = sorted(d5["N"].unique().tolist())

    def expand(df):
        return pd.concat([df.assign(N=n) for n in ns], ignore_index=True)

    allr = pd.concat([expand(d3)[COLS], expand(d4)[COLS], d5[COLS]], ignore_index=True)
    return allr, ns


def make_pairs(methods):
    pairs = []
    for k in ("hgb", "logreg", "rf"):
        pairs += [(f"E1_{k}", f"G0h_{k}"), (f"E1P_{k}", f"E1_{k}"), (f"E2_{k}", f"E1_{k}"), (f"E3_{k}", f"E2_{k}"),
                  (f"E4_{k}", f"E3_{k}"), (f"E4_{k}", f"E1P_{k}"), (f"E0b_{k}", f"G0_{k}"), (f"E0b_{k}", "B1_pop_like"),
                  (f"G0_{k}", "B1_pop_like"), (f"E1_{k}", "B1_pop_like"), (f"E1P_{k}", "B1_pop_like"),
                  (f"E2_{k}", "B1_pop_like"), (f"E3_{k}", "B1_pop_like"), (f"E4_{k}", "B1_pop_like"),
                  (f"E1U_{k}", f"E1_{k}"), (f"E1U_{k}", "B1_pop_like"), (f"E1PU_{k}", f"E1P_{k}"),
                  (f"E4U_{k}", f"E4_{k}"), (f"E4U_{k}", "B1_pop_like")]
    pairs += [("E0a_logreg", "B1_pop_like"), ("E0a_rf", "B1_pop_like")]
    return [(a, b) for a, b in pairs if a in methods and b in methods]


def verdict(row):
    if row["ci_low"] > 0:
        return "better"
    if row["ci_high"] < 0:
        return "worse"
    return "no clear difference"


def main():
    cfg = parse_args(__doc__)
    out = ensure_dir(OUT_DIR / f"06_report{cfg['tag']}")
    allr, ns = load_all(cfg)
    methods = set(allr["method"].unique())
    pairs = make_pairs(methods)
    summary, diffs = summarize(allr, ["rule", "N"], pairs)
    summary.to_csv(out / "final_summary.csv", index=False)
    diffs["verdict"] = diffs.apply(verdict, axis=1)
    diffs.to_csv(out / "final_pairs.csv", index=False)

    L = PRIMARY_LEARNER
    table_methods = ["B0_majority", "B1_pop_like", "B1_pop_mean", "E0a_logreg", f"G0_{L}", f"E0b_{L}",
                     f"G0h_{L}", f"E1_{L}", f"E1P_{L}", f"E2_{L}", f"E3_{L}", f"E4_{L}", f"E1U_{L}", f"E1PU_{L}",
                     f"E4U_{L}", "E1U_logreg"]
    table_methods = [m for m in table_methods if m in methods]
    lines = ["STEP 6 - FINAL REPORT", f"N levels: {ns}", f"primary learner: {L} (decided before seeing results)", ""]
    for rule in sorted(summary["rule"].unique()):
        sub = summary[summary["rule"] == rule]
        users = int(sub["users"].median())
        lines.append(f"=== label rule: {rule} (about {users} test users) ===")
        lines.append("mean per-user AUC [95% interval]")
        lines.append("  " + "method".ljust(14) + "".join(f"N={n}".rjust(24) for n in ns))
        for m in table_methods:
            cells = []
            for n in ns:
                s = sub[(sub["method"] == m) & (sub["N"] == n)]
                cells.append(f"{s['mean_auc'].iloc[0]:.4f} [{s['ci_low'].iloc[0]:.3f},{s['ci_high'].iloc[0]:.3f}]".rjust(24)
                             if len(s) else "n/a".rjust(24))
            lines.append("  " + m.ljust(14) + "".join(cells))
        lines.append("")

        qs = [("Does the raw profile add anything to film popularity? (global model, E0b - G0)", f"E0b_{L}", f"G0_{L}"),
              ("Is the best profile-only model better than plain popularity? (E0a_logreg - B1_pop_like)", "E0a_logreg", "B1_pop_like"),
              ("Do N revealed ratings help? (E1 - film-only control G0h)", f"E1_{L}", f"G0h_{L}"),
              ("Does the profile still help once history is known? (E1P - E1)", f"E1P_{L}", f"E1_{L}"),
              ("Do the LLM fame columns help? (E2 - E1)", f"E2_{L}", f"E1_{L}"),
              ("Does director/actor affinity help? (E3 - E2)", f"E3_{L}", f"E2_{L}"),
              ("Does the profile help on top of affinity? (E4 - E3)", f"E4_{L}", f"E3_{L}"),
              ("Is the best full model better than plain popularity? (E4 - B1_pop_like)", f"E4_{L}", "B1_pop_like"),
              ("Does user-user collaborative filtering help? (E1U - E1)", f"E1U_{L}", f"E1_{L}"),
              ("Is E1U better than plain popularity? (E1U - B1_pop_like)", f"E1U_{L}", "B1_pop_like"),
              ("Does user-user CF help for logistic regression? (E1U - E1)", "E1U_logreg", "E1_logreg")]
        for question, a, b in qs:
            d = diffs[(diffs["rule"] == rule) & (diffs["A"] == a) & (diffs["B"] == b)].sort_values("N")
            if d.empty:
                continue
            lines.append(question)
            for _, x in d.iterrows():
                lines.append(f"    N={int(x['N']):>3}: {x['mean_difference_A_minus_B']:+.4f}  "
                             f"[{x['ci_low']:+.4f}, {x['ci_high']:+.4f}]  {x['verdict']}")
            lines.append("")

    text = "\n".join(lines)
    (out / "final_report.txt").write_text(text, encoding="utf-8")
    print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        curve = ["B1_pop_like", f"E0b_{L}", f"G0h_{L}", f"E1_{L}", f"E1P_{L}", f"E2_{L}", f"E3_{L}", f"E4_{L}"]
        curve = [m for m in curve if m in methods]
        for rule in sorted(summary["rule"].unique()):
            sub = summary[summary["rule"] == rule]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            for m in curve:
                s = sub[sub["method"] == m].sort_values("N")
                if s.empty:
                    continue
                x = [ns.index(n) for n in s["N"]]
                ax.plot(x, s["mean_auc"], marker="o", label=m)
                ax.fill_between(x, s["ci_low"], s["ci_high"], alpha=0.15)
            ax.set_xticks(range(len(ns)))
            ax.set_xticklabels(ns)
            ax.set_xlabel("N = revealed ratings of the new user")
            ax.set_ylabel("mean per-user AUC")
            ax.set_title(f"Cold-start curves, label rule {rule}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / f"curves_{rule}.png", dpi=150)
            plt.close(fig)
        print("curves written")
    except ImportError:
        print("matplotlib is not installed: tables were written, curves were skipped")
    print(f"\nfiles written to: {out}")


if __name__ == "__main__":
    main()
