"""
STEP 13 - ablation tables (no model is trained, the paired results of earlier steps are only collected).

For every component the table shows: the difference in mean per-user AUC when the component is added,
its 95% bootstrap interval over users (paired, same users) and the verdict. Removing a component is the
same pair read backwards (without LLM = D0n, D2n, D3n, D4n; with LLM = D0, D2, D3, D4).
Output: outputs/13_ablation/ablation.csv and ablation_report.txt
"""
import os
import sys

import pandas as pd

OUT = os.path.join("outputs", "13_ablation")
os.makedirs(OUT, exist_ok=True)

# (scenario, folder, learner, axis column, levels, [(component, A, B)])  A = with component, B = without
SPECS = [
    ("A new user", "06_report_tpu200", "hgb", "N", [0, 5, 20, 50], [
        ("history of N revealed ratings", "E1_{L}", "G0h_{L}"),
        ("raw profile on top of film popularity", "E0b_{L}", "G0_{L}"),
        ("profile on top of history", "E1P_{L}", "E1_{L}"),
        ("LLM fame columns (GPT-4o) on top of history", "E2_{L}", "E1_{L}"),
    ]),
    ("B new film", "07_new_film_models_sonnet", "hgb", "M", [0, 5], [
        ("LLM fame (Sonnet) on top of track record", "C3_{L}", "C2_{L}"),
        ("LLM fame (Sonnet) on top of track record + affinity", "C4_{L}", "C6_{L}"),
        ("affinity on top of track record", "C6_{L}", "C2_{L}"),
        ("LLM fame (Sonnet) when M ratings of the film are known", "F3_{L}", "F3n_{L}"),
    ]),
    ("C both new, Sonnet", "08_both_new_models_sonnet", "hgb_c", "N", [0, 5, 20], [
        ("LLM fame (Sonnet) on top of track record", "D0_{L}", "D0n_{L}"),
        ("LLM fame (Sonnet) on top of track record + profile", "D2_{L}", "D2n_{L}"),
        ("LLM fame (Sonnet) with revealed ratings", "D3_{L}", "D3n_{L}"),
        ("LLM fame (Sonnet) with revealed ratings + profile", "D4_{L}", "D4n_{L}"),
        ("raw profile, without LLM", "D2n_{L}", "D0n_{L}"),
        ("raw profile, with LLM", "D2_{L}", "D0_{L}"),
        ("revealed ratings, with LLM", "D3_{L}", "D0_{L}"),
    ]),
    ("C both new, GPT-4o + fuzzy", "08_both_new_models_fuzzy", "hgb_c", "N", [0, 5, 20], [
        ("LLM fame (GPT-4o) on top of track record + profile", "D2_{L}", "D2n_{L}"),
        ("fuzzy sets F1 instead of raw profile, without LLM", "ZF1n_{L}", "D2n_{L}"),
        ("fuzzy c-means F2 instead of raw profile, without LLM", "ZF2n_{L}", "D2n_{L}"),
        ("fuzzy rules F3 instead of raw profile, without LLM", "ZF3n_{L}", "D2n_{L}"),
        ("crisp sets C1 instead of raw profile, without LLM", "ZC1n_{L}", "D2n_{L}"),
        ("fuzzy sets F1 instead of raw profile, with LLM", "ZF1_{L}", "D2_{L}"),
        ("fuzzy sets F1 instead of crisp C1, without LLM", "ZF1n_{L}", "ZC1n_{L}"),
        ("fuzzy sets F1 instead of crisp C1, with LLM", "ZF1_{L}", "ZC1_{L}"),
    ]),
]
# four states of scenario C (fuzzy x LLM), same run so the LLM source is the same (GPT-4o)
FOUR = [("none (raw profile, no LLM)", "D2n"), ("fuzzy only (F1)", "ZF1n"),
        ("LLM only", "D2"), ("fuzzy + LLM", "ZF1")]


def load(folder, kind):
    return pd.read_csv(os.path.join("outputs", folder, f"final_{kind}.csv"))


rows, lines = [], []
for scen, folder, learner, ax, levels, comps in SPECS:
    try:
        pairs, summ = load(folder, "pairs"), load(folder, "summary")
    except FileNotFoundError:
        lines.append(f"{scen}: folder {folder} not found, skipped")
        continue
    for rule in ("gt7", "ge7"):
        lines.append(f"\n=== {scen} | learner {learner} | rule {rule} | difference in mean per-user AUC, paired 95% interval ===")
        head = f"{'component (with minus without)':62s}" + "".join(f"{ax}={v:<22d}" for v in levels)
        lines.append(head)
        for name, a, b in comps:
            a, b = a.format(L=learner), b.format(L=learner)
            cells = []
            for v in levels:
                q = pairs[(pairs.rule == rule) & (pairs[ax] == v) & (pairs.A == a) & (pairs.B == b)]
                if q.empty:
                    cells.append("n/a".ljust(24))
                    continue
                r = q.iloc[0]
                cells.append(f"{r.mean_difference_A_minus_B:+.4f} [{r.ci_low:+.4f},{r.ci_high:+.4f}] {r.verdict[0]}".ljust(24))
                rows.append(dict(scenario=scen, learner=learner, rule=rule, level=v, component=name, A=a, B=b,
                                 difference=r.mean_difference_A_minus_B, ci_low=r.ci_low, ci_high=r.ci_high,
                                 verdict=r.verdict))
            lines.append(f"{name:62s}" + "".join(cells))
        if scen.startswith("C both new, GPT-4o"):
            lines.append(f"-- four states, mean AUC [95% interval], {learner}")
            for label, m in FOUR:
                cells = []
                for v in levels:
                    q = summ[(summ.rule == rule) & (summ[ax] == v) & (summ.method == f"{m}_{learner}")]
                    cells.append("n/a".ljust(24) if q.empty else
                                 f"{q.iloc[0].mean_auc:.4f} [{q.iloc[0].ci_low:.3f},{q.iloc[0].ci_high:.3f}]".ljust(24))
                lines.append(f"{label:62s}" + "".join(cells))
lines.append("\nverdict letter: b = better, w = worse, n = no clear difference (interval contains 0)")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "ablation.csv"), index=False)
text = "\n".join(lines)
open(os.path.join(OUT, "ablation_report.txt"), "w", encoding="utf8").write(text)
sys.stdout.buffer.write((text + "\n").encode("utf8"))
