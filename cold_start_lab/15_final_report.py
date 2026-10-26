"""Step 15: tables of the final protocol (reads the partial files of step 14; no model is trained).

For every scenario (A new user, B new film, C both new), label rule, learner and level (N or M):
  summary_auc.csv   mean per-user AUC with a 95% bootstrap interval over users
  pairs_auc.csv     paired differences (with component minus without) with interval and verdict
  summary_cls.csv   accuracy, F1 of the liked class, weighted F1, pooled over all test rows, bootstrap over users,
                    next to the two trivial guesses ALL_NO (nobody liked) and ALL_YES (everybody liked)
  pairs_cls.csv     the same paired differences for the three classification metrics
  ablation.csv      all pairs of the three scenarios in one table (add-one and leave-one-block-out)
  final_report.txt  the main tables for the primary learner
Output: outputs/14_final[_quick]/  (one folder per scenario and one combined folder `all`)

Run  : python 15_final_report.py [--quick] [--tag _ge7]
"""
import importlib
import sys

import numpy as np
import pandas as pd

from config import OUT_DIR, RANDOM_STATE, ensure_dir
from labutils import summarize

PRIMARY = "hgb_c"
N_BOOT = 2000
SCEN = {"A_new_user": ("A", "N"), "B_new_film": ("B", "M"), "C_both_new": ("C", "N")}
MET = ["accuracy", "f1_liked", "f1_weighted"]


def verdict(lo, hi):
    """Bounds are rounded to 4 decimals first, so a difference of 0.000001 is not called better or worse."""
    lo, hi = round(float(lo), 4), round(float(hi), 4)
    return "better" if lo > 0 else ("worse" if hi < 0 else "no clear difference")


def metrics(c):
    """c = (..., 4) counts tp, fp, fn, tn -> dict of arrays."""
    tp, fp, fn, tn = (c[..., i].astype(float) for i in range(4))
    n = np.maximum(tp + fp + fn + tn, 1)
    pos, neg = tp + fn, fp + tn
    f1p = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    f1n = np.where(2 * tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1), 0.0)
    return {"accuracy": (tp + tn) / n, "f1_liked": f1p, "f1_weighted": (pos * f1p + neg * f1n) / n}


def cls_summary(df, keys, pairs):
    """Pooled metrics with a bootstrap interval over users, and paired differences (same bootstrap draws)."""
    rng = np.random.default_rng(RANDOM_STATE)
    summ, diffs = [], []
    for key_vals, sub in df.groupby(keys):
        key_vals = key_vals if isinstance(key_vals, tuple) else (key_vals,)
        key = dict(zip(keys, key_vals))
        users = np.sort(sub["user_id"].unique())
        pos_of = {u: i for i, u in enumerate(users.tolist())}
        C = {}
        for m, g in sub.groupby("method"):
            arr = np.zeros((len(users), 4))
            idx = g["user_id"].map(pos_of).to_numpy()
            np.add.at(arr, idx, g[["tp", "fp", "fn", "tn"]].to_numpy(dtype=float))
            C[m] = arr
        if not C or len(users) < 5:
            continue
        ref = next(iter(C.values()))
        P, Ng = ref[:, 0] + ref[:, 2], ref[:, 1] + ref[:, 3]
        C["ALL_NO"] = np.column_stack([np.zeros_like(P), np.zeros_like(P), P, Ng])
        C["ALL_YES"] = np.column_stack([P, Ng, np.zeros_like(P), np.zeros_like(P)])
        boot = rng.integers(0, len(users), size=(N_BOOT, len(users)))
        bm = {m: metrics(c[boot].sum(axis=1)) for m, c in C.items()}
        for m, c in C.items():
            pt = metrics(c.sum(axis=0))
            row = {**key, "method": m, "users": len(users)}
            for k in MET:
                lo, hi = np.percentile(bm[m][k], [2.5, 97.5])
                row.update({k: float(pt[k]), f"{k}_low": float(lo), f"{k}_high": float(hi)})
            summ.append(row)
        for a, b in pairs:
            if a in C and b in C:
                row = {**key, "A": a, "B": b, "users": len(users)}
                pa, pb = metrics(C[a].sum(axis=0)), metrics(C[b].sum(axis=0))
                for k in MET:
                    d = bm[a][k] - bm[b][k]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    row.update({f"{k}_diff": float(pa[k] - pb[k]), f"{k}_low": float(lo), f"{k}_high": float(hi),
                                f"{k}_verdict": verdict(lo, hi)})
                diffs.append(row)
    return pd.DataFrame(summ), pd.DataFrame(diffs)


def component_of(a, b):
    """Readable name of what the pair a-minus-b tests."""
    if b.startswith("FULL-"):
        return "leave-one-out", f"remove {b[5:]} from the full model"
    if b == "B0_const":
        return "baseline", f"{a} against nothing"
    return "add-one", f"{a} against {b}"


def main():
    quick = "--quick" in sys.argv
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    base = OUT_DIR / f"14_final{'_quick' if quick else ''}{tag}"
    proto = importlib.import_module("14_final_protocol")
    all_pairs_auc, all_pairs_cls, lines = [], [], []
    for folder, (sc, level_key) in SCEN.items():
        d = base / folder
        if not (d / "partial_auc.csv").exists():
            print(f"{folder}: no results yet, skipped")
            continue
        auc = pd.read_csv(d / "partial_auc.csv")
        cls = pd.read_csv(d / "partial_cls.csv")
        pairs_spec = {"A": proto.A_PAIRS, "B": proto.B_PAIRS, "C": proto.C_PAIRS}[sc]
        learners = sorted({m.split("|")[1] for m in auc["method"].unique() if "|" in m})
        full_pairs = [(f"{a}|{k}", f"{b}|{k}" if b != "B0_const" else b) for k in learners for a, b in pairs_spec]
        methods = set(auc["method"].unique())
        full_pairs = [(a, b) for a, b in full_pairs if a in methods and b in methods]
        s_auc, p_auc = summarize(auc, ["rule", "level"], full_pairs)
        p_auc["verdict"] = [verdict(lo, hi) for lo, hi in zip(p_auc["ci_low"], p_auc["ci_high"])]
        cls_pairs = [(a, b) for a, b in full_pairs if b != "B0_const"]
        s_cls, p_cls = cls_summary(cls, ["rule", "level"], cls_pairs)
        for name, tab in (("summary_auc", s_auc), ("pairs_auc", p_auc), ("summary_cls", s_cls), ("pairs_cls", p_cls)):
            tab.to_csv(d / f"{name}.csv", index=False)
        for tab, store in ((p_auc, all_pairs_auc), (p_cls, all_pairs_cls)):
            t = tab.copy()
            t.insert(0, "scenario", sc)
            t["learner"] = t["A"].str.split("|").str[1]
            t["a"], t["b"] = t["A"].str.split("|").str[0], t["B"].str.split("|").str[0]
            t[["kind", "component"]] = [component_of(a, b) for a, b in zip(t["a"], t["b"])]
            store.append(t)

        # ---- text tables for the primary learner
        lines += [f"{'=' * 100}", f"SCENARIO {sc}: {folder}   primary learner: {PRIMARY} (declared before the final run)",
                  f"{level_key} = " + ("revealed ratings of the new user" if level_key == "N" else "known ratings of the new film"), ""]
        for rule in sorted(s_auc["rule"].unique()):
            levels = sorted(s_auc["level"].unique())
            lines.append(f"--- label rule {rule}: mean per-user AUC [95% interval]")
            lines.append("  " + "experiment".ljust(26) + "".join(f"{level_key}={v}".rjust(24) for v in levels))
            exps = [m for m in s_auc["method"].unique() if m.endswith(f"|{PRIMARY}") or "|" not in m]
            for m in exps:
                cells = []
                for v in levels:
                    q = s_auc[(s_auc["rule"] == rule) & (s_auc["level"] == v) & (s_auc["method"] == m)]
                    cells.append((f"{q['mean_auc'].iloc[0]:.4f} [{q['ci_low'].iloc[0]:.3f},{q['ci_high'].iloc[0]:.3f}]"
                                  if len(q) else "-").rjust(24))
                lines.append("  " + m.split("|")[0].ljust(26) + "".join(cells))
            lines += ["", f"--- label rule {rule}: F1 of the liked class / accuracy (pooled, [95% interval over users])"]
            lines.append("  " + "experiment".ljust(26) + "".join(f"{level_key}={v}".rjust(34) for v in levels))
            for m in [x for x in s_cls["method"].unique() if x.endswith(f"|{PRIMARY}") or x.startswith("ALL_")]:
                cells = []
                for v in levels:
                    q = s_cls[(s_cls["rule"] == rule) & (s_cls["level"] == v) & (s_cls["method"] == m)]
                    cells.append((f"F1 {q['f1_liked'].iloc[0]:.3f}  acc {q['accuracy'].iloc[0]:.3f}  wF1 {q['f1_weighted'].iloc[0]:.3f}"
                                  if len(q) else "-").rjust(34))
                lines.append("  " + m.split("|")[0].ljust(26) + "".join(cells))
            lines += ["", f"--- label rule {rule}: paired AUC differences, {PRIMARY} (with minus without)"]
            sub = p_auc[(p_auc["rule"] == rule) & p_auc["A"].str.endswith(f"|{PRIMARY}")]
            for (a, b), g in sub.groupby(["A", "B"], sort=False):
                kind, text = component_of(a.split("|")[0], b.split("|")[0])
                cells = "   ".join(f"{level_key}={int(x['level'])}: {x['mean_difference_A_minus_B']:+.4f} "
                                   f"[{x['ci_low']:+.4f},{x['ci_high']:+.4f}] {x['verdict'][:6]}"
                                   for _, x in g.sort_values("level").iterrows())
                lines.append(f"  [{kind}] {text}: {cells}")
            lines.append("")
    if all_pairs_auc:
        out = ensure_dir(base / "all")
        pa, pc = pd.concat(all_pairs_auc), pd.concat(all_pairs_cls)
        pa.to_csv(out / "ablation_auc.csv", index=False)
        pc.to_csv(out / "ablation_cls.csv", index=False)
        (out / "final_report.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        print(f"\nfiles written to: {out}")


if __name__ == "__main__":
    main()
