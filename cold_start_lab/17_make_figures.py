"""Step 17: figures of the final protocol (reads the tables of step 15; no model is trained).

  fig1_auc_vs_level.png          mean per-user AUC against N (or M) for the main experiments of each scenario
  fig2_leave_one_out.png         what the full model loses when one information block is removed (paired, 95% interval)
  fig3_fuzzy_vs_controls.png     fuzzified profile minus raw profile and minus crisp profile (paired, 95% interval)
  fig4_f1_accuracy_baselines.png F1 of the liked class and accuracy of the full model against the two trivial guesses
  fig5_learners.png              the same experiments with the three learners (logistic regression, hgb_c, random forest)
Output: outputs/14_final[_quick][tag]/figures/

Run  : python 17_make_figures.py [--quick] [--tag _ge7]
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUT_DIR, ensure_dir

PRIMARY = "hgb_c"
RULE = "gt7"
SCEN = {"A": ("A_new_user", "N", "A  new user"), "B": ("B_new_film", "M", "B  new film"), "C": ("C_both_new", "N", "C  both new")}
MAIN = {"A": ["P", "P+H", "P+PRfuzzy", "FULL"], "B": ["T", "T+L", "T+A+H", "FH", "FULL"], "C": ["T", "T+L", "T+L+PRfuzzy", "FULL"]}
COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})


def load(base, folder, name):
    return pd.read_csv(base / folder / f"{name}.csv")


def with_learner(df, col):
    df = df.copy()
    df["exp"] = df[col].str.split("|").str[0]
    df["learner"] = df[col].str.split("|").str[1]
    return df


def fig1(base, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (sc, (folder, key, title)) in zip(axes, SCEN.items()):
        s = with_learner(load(base, folder, "summary_auc"), "method")
        s = s[(s.rule == RULE)]
        s = s[s.exp != "B0_const"]
        for color, exp in zip(COLORS, MAIN[sc]):
            q = s[(s.exp == exp) & (s.learner == PRIMARY)].sort_values("level")
            if q.empty:
                continue
            ax.plot(q.level, q.mean_auc, marker="o", ms=4, color=color, label=exp)
            ax.fill_between(q.level, q.ci_low, q.ci_high, color=color, alpha=0.12, lw=0)
        ax.set_xlabel(f"{key} = " + ("revealed ratings of the new user" if key == "N" else "known ratings of the new film"))
        ax.set_ylabel("mean per-user AUC")
        ax.set_title(title)
        ax.set_xticks(sorted(s.level.unique()))
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Mean per-user AUC, 95% bootstrap interval over users ({PRIMARY}, rule {RULE}; a model without information has AUC 0.5)", y=1.02)
    fig.savefig(out / "fig1_auc_vs_level.png")
    plt.close(fig)


def fig2(base, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (sc, (folder, key, title)) in zip(axes, SCEN.items()):
        p = with_learner(load(base, folder, "pairs_auc"), "A")
        p = p[(p.rule == RULE) & (p.learner == PRIMARY) & (p.exp == "FULL") & p.B.str.startswith("FULL-")]
        p["block"] = p.B.str.split("|").str[0].str[5:]
        blocks = list(dict.fromkeys(p.block))
        levels = sorted(p.level.unique())
        w = 0.8 / max(len(levels), 1)
        for j, lv in enumerate(levels):
            q = p[p.level == lv].set_index("block").reindex(blocks)
            y = np.arange(len(blocks)) + (j - (len(levels) - 1) / 2) * w
            ax.errorbar(q.mean_difference_A_minus_B, y, xerr=[q.mean_difference_A_minus_B - q.ci_low, q.ci_high - q.mean_difference_A_minus_B],
                        fmt="o", ms=4, capsize=2, color=COLORS[j % len(COLORS)], label=f"{key}={lv}")
        ax.axvline(0, color="black", lw=1)
        ax.set_xscale("symlog", linthresh=0.002)
        ax.set_xlim(-0.003, float(p.ci_high.max()) * 3)
        ax.set_yticks(range(len(blocks)))
        ax.set_yticklabels([f"remove {b}" for b in blocks])
        ax.invert_yaxis()
        ax.set_xlabel("AUC lost (log scale)")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle(f"Leave-one-block-out from the full model (paired difference, 95% interval, {PRIMARY}, rule {RULE}; the bands in figure 1 overlap because they are not paired)", y=1.02)
    fig.savefig(out / "fig2_leave_one_out.png")
    plt.close(fig)


def fig3(base, out):
    specs = {"A": [("P+PRfuzzy", "P+PRraw"), ("P+PRfuzzy", "P+PRcrisp"), ("P+H+PRfuzzy", "P+H+PRraw"), ("P+H+PRfuzzy", "P+H+PRcrisp")],
             "C": [("T+PRfuzzy", "T+PRraw"), ("T+PRfuzzy", "T+PRcrisp"), ("T+L+PRfuzzy", "T+L+PRraw"), ("T+L+PRfuzzy", "T+L+PRcrisp")]}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, sc in zip(axes, ["A", "C"]):
        folder, key, title = SCEN[sc]
        p = with_learner(load(base, folder, "pairs_auc"), "A")
        p["b"] = p.B.str.split("|").str[0]
        p = p[(p.rule == RULE) & (p.learner == PRIMARY)]
        for i, (a, b) in enumerate(specs[sc]):
            q = p[(p.exp == a) & (p.b == b)].sort_values("level")
            ax.errorbar(q.level + (i - 1.5) * 0.6, q.mean_difference_A_minus_B,
                        yerr=[q.mean_difference_A_minus_B - q.ci_low, q.ci_high - q.mean_difference_A_minus_B],
                        marker="o", ms=4, capsize=3, color=COLORS[i], label=f"{a}  minus  {b}")
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel(f"{key} = revealed ratings of the new user")
        ax.set_ylabel("paired AUC difference")
        ax.set_title(title)
        ax.set_xticks(sorted(p.level.unique()))
        ax.legend(fontsize=8)
    fig.suptitle(f"Fuzzified profile against the raw and the crisp profile (95% interval, {PRIMARY}, rule {RULE})", y=1.02)
    fig.savefig(out / "fig3_fuzzy_vs_controls.png")
    plt.close(fig)


def fig4(base, out):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for col, (sc, (folder, key, title)) in enumerate(SCEN.items()):
        s = with_learner(load(base, folder, "summary_cls"), "method")
        s = s[s.rule == RULE]
        levels = sorted(s.level.unique())
        for row, (metric, label) in enumerate([("f1_liked", "F1 of the liked class"), ("accuracy", "accuracy")]):
            ax = axes[row, col]
            guesses = [("FULL", "#e45756", (s.exp == "FULL") & (s.learner == PRIMARY)), ("all liked", "#bbbbbb", s.method == "ALL_YES")]
            if metric == "accuracy":
                guesses.insert(1, ("all not liked", "#777777", s.method == "ALL_NO"))
            for name, color, sel in guesses:
                q = s[sel].sort_values("level")
                if q.empty:
                    continue
                ax.errorbar(q.level, q[metric], yerr=[q[metric] - q[f"{metric}_low"], q[f"{metric}_high"] - q[metric]], marker="o", ms=4,
                            capsize=3, color=color, label=name, ls="-" if name == "FULL" else "--")
            ax.set_title(f"{title}  -  {label}")
            if metric == "f1_liked":
                ax.text(0.98, 0.04, "all not liked: F1 = 0", transform=ax.transAxes, ha="right", fontsize=8, color="#777777")
            ax.set_xlabel(f"{key}")
            ax.set_xticks(levels)
            if col == 0 and row == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"F1 and accuracy of the full model against the trivial guesses (threshold learned on training users, {PRIMARY}, rule {RULE})", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "fig4_f1_accuracy_baselines.png")
    plt.close(fig)


def fig5(base, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (sc, (folder, key, title)) in zip(axes, SCEN.items()):
        s = with_learner(load(base, folder, "summary_auc"), "method")
        s = s[(s.rule == RULE) & (s.level == s.level.min()) & s.learner.notna()]
        exps = [e for e in MAIN[sc] if e in set(s[s.learner == "rf"].exp)]
        learners = ["logreg", "hgb_c", "rf"]
        w = 0.8 / 3
        for j, ln in enumerate(learners):
            q = s[s.learner == ln].set_index("exp").reindex(exps)
            ax.bar(np.arange(len(exps)) + (j - 1) * w, q.mean_auc, w, yerr=[q.mean_auc - q.ci_low, q.ci_high - q.mean_auc],
                   capsize=2, color=COLORS[j], label=ln)
        ax.set_xticks(range(len(exps)))
        ax.set_xticklabels(exps, rotation=20)
        lo = float(s.mean_auc.min())
        ax.set_ylim(max(0.5, lo - 0.05), None)
        ax.set_ylabel("mean per-user AUC")
        ax.set_title(f"{title}  ({key}={int(s.level.min())})")
        ax.set_ylim(top=float(s.mean_auc.max()) + 0.03)
        ax.legend(fontsize=8, ncol=3, loc="upper center")
    fig.suptitle(f"Three learners on the same experiments (rule {RULE}; random forest only on the short list)", y=1.02)
    fig.savefig(out / "fig5_learners.png")
    plt.close(fig)


def main():
    quick = "--quick" in sys.argv
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    base = OUT_DIR / f"14_final{'_quick' if quick else ''}{tag}"
    out = ensure_dir(base / "figures")
    for fn in (fig1, fig2, fig3, fig4, fig5):
        fn(base, out)
        print("written:", fn.__name__)
    print(f"figures in: {out}")


if __name__ == "__main__":
    main()
