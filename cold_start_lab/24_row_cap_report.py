"""Step 24: the row cap learning curve (reads the partial files of step 14; no model is trained).

Levels of the per-user training cap: 100, 200, 400, all (no cap) and all_uw (no cap, every user the same total weight).
Every level has the same folds, seeds, users and test rows; only the training rows change, so all comparisons are paired by user.

Judgement rules, fixed BEFORE any result of this experiment was seen (AUC, cap "all" = the full level):
  real effect          in logreg AND hgb_c: the 95% interval at the full level excludes 0 and |effect| > NOISE_FLOOR, with the same sign
  lack of data         in a learner: the effect is clear at cap 100 (interval excludes 0, |effect| > NOISE_FLOOR), its size falls at every
                       step 100 > 200 > 400 > all without changing sign, and at the full level the interval includes 0 or |effect| <= NOISE_FLOOR
  not established      everything else
A/H in scenarios B and C always enter together in step 14, so they are one block there ("A+H").

Output: outputs/row_cap_curve/  gains.csv  block_effects.csv  block_changes.csv  verdicts.csv  rows.csv  checks.txt  tables_fa.md
        outputs/row_cap_curve_report.md  (findings_fa.md, written by hand after the results, goes at the top of the report)

Run  : python 24_row_cap_report.py
"""
import importlib
import re

import numpy as np
import pandas as pd

from config import OUT_DIR, RANDOM_STATE, ensure_dir

fr = importlib.import_module("15_final_report")

N_BOOT = 2000
NOISE_FLOOR = 0.0006
N_FOLDS_ALL = 15
LEARNERS = ["logreg", "hgb_c", "rf"]
CURVE = ["100", "200", "400", "all"]
CAPS = CURVE + ["all_uw"]
DIRS = {"100": ("14_final_filmq_tpu100", "14_final_tpu100"), "200": ("14_final_filmq", "14_final"),
        "400": ("14_final_filmq_tpu400", "14_final_tpu400"), "all": ("14_final_filmq_tpuall", "14_final_tpuall"),
        "all_uw": ("14_final_filmq_tpuall_uw", "14_final_tpuall_uw")}
LOGS = {"100": "_tpu100", "400": "_tpu400", "all": "_tpuall", "all_uw": "_tpuall_uw"}
CAP_PAIRS = [("100", "200"), ("400", "200"), ("all", "200"), ("all", "400"), ("all_uw", "all"), ("all_uw", "200")]
CHANGE_PAIRS = [("100", "200"), ("400", "200"), ("all", "200"), ("all_uw", "all")]
SCEN = {"A": ("A_new_user", "N"), "B": ("B_new_film", "M"), "C": ("C_both_new", "N")}
HEADLINE = "FULLKQ"       # declared before the runs: the model with every block except the public vote
MEAN_LEVEL = -1           # pseudo level: mean over the levels of N or M

# block, with, without. add-one chains first, then leave-one-out inside the full models
BLOCKS = {
    "A": [("add", "P", "P", "B0_const"), ("add", "H", "P+H", "P"), ("add", "L", "P+H+L", "P+H"), ("add", "A", "P+H+L+A", "P+H+L"),
          ("add", "K", "P+H+K", "P+H"), ("add", "Q", "P+H+K+Q", "P+H+K"), ("add", "PUB", "P+H+K+PUB", "P+H+K"),
          ("add", "PRraw", "P+H+PRraw", "P+H"), ("add", "PRcrisp", "P+H+PRcrisp", "P+H"), ("add", "PRfuzzy", "P+H+PRfuzzy", "P+H"),
          ("versus", "fuzzy-raw", "P+H+PRfuzzy", "P+H+PRraw"), ("versus", "fuzzy-crisp", "P+H+PRfuzzy", "P+H+PRcrisp"),
          ("loo", "P", "FULL", "FULL-P"), ("loo", "H", "FULL", "FULL-H"), ("loo", "L", "FULL", "FULL-L"), ("loo", "A", "FULL", "FULL-A"),
          ("loo", "PRfuzzy", "FULL", "FULL-PR"), ("loo", "K", "FULLK", "FULL"), ("loo", "Q", "FULLKQ", "FULLK"), ("loo", "PUB", "FULLKP", "FULLK")],
    "B": [("add", "T", "T", "B0_const"), ("add", "L", "T+L", "T"), ("add", "A+H", "T+A+H", "T"), ("add", "K", "T+K", "T"),
          ("add", "Q", "T+K+Q", "T+K"), ("add", "PUB", "T+K+PUB", "T+K"),
          ("loo", "FH", "FULL", "FULL-FH"), ("loo", "T", "FULL", "FULL-T"), ("loo", "L", "FULL", "FULL-L"), ("loo", "A+H", "FULL", "FULL-AH"),
          ("loo", "K", "FULLK", "FULL"), ("loo", "Q", "FULLKQ", "FULLK"), ("loo", "PUB", "FULLKP", "FULLK")],
    "C": [("add", "T", "T", "B0_const"), ("add", "L", "T+L", "T"), ("add", "A+H", "T+A+H", "T"), ("add", "K", "T+K", "T"),
          ("add", "Q", "T+K+Q", "T+K"), ("add", "PUB", "T+K+PUB", "T+K"),
          ("add", "PRraw", "T+L+PRraw", "T+L"), ("add", "PRcrisp", "T+L+PRcrisp", "T+L"), ("add", "PRfuzzy", "T+L+PRfuzzy", "T+L"),
          ("versus", "fuzzy-raw", "T+L+PRfuzzy", "T+L+PRraw"), ("versus", "fuzzy-crisp", "T+L+PRfuzzy", "T+L+PRcrisp"),
          ("loo", "T", "FULL", "FULL-T"), ("loo", "L", "FULL", "FULL-L"), ("loo", "PRfuzzy", "FULL", "FULL-PR"), ("loo", "A+H", "FULL", "FULL-AH"),
          ("loo", "K", "FULLK", "FULL"), ("loo", "Q", "FULLKQ", "FULLK"), ("loo", "PUB", "FULLKP", "FULLK")],
}

METRICS = {"auc": "AUC", "f1_liked": "F1", "accuracy": "accuracy"}


# ----------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------
def read_dir(name, folder):
    d = OUT_DIR / name / folder
    if not (d / "partial_auc.csv").exists():
        return None
    auc, cls = pd.read_csv(d / "partial_auc.csv"), pd.read_csv(d / "partial_cls.csv")
    folds = len(auc[["repeat", "fold"]].drop_duplicates())
    if folds < N_FOLDS_ALL:
        print(f"{name}/{folder}: only {folds} of {N_FOLDS_ALL} folds, left out")
        return None
    return (auc.groupby(["level", "user_id", "method"])["auc"].mean().unstack("method"),
            cls.groupby(["level", "method", "user_id"])[["tp", "fp", "fn", "tn"]].sum())


class Level:
    """Everything of one cap and one level of N or M: per-user AUC table and per-user confusion counts."""

    def __init__(self, wide, counts):
        self.wide, self.counts = wide, counts


def load_cap(folder, cap, checks):
    q, p = (read_dir(n, folder) for n in DIRS[cap])
    if q is None and p is None:
        return None
    if q is not None and p is not None:
        both = [m for m in q[0].columns if m in p[0].columns and "|" in m]
        for k in LEARNERS:
            cols = [m for m in both if m.endswith(f"|{k}")]
            if cols:
                dif = (q[0][cols] - p[0][cols]).abs()
                checks.append(f"{folder} cap {cap} {k}: {len(cols)} models in both lists, largest per-user difference {np.nanmax(dif.to_numpy()):.6f}, "
                              f"mean {np.nanmean(dif.to_numpy()):.8f}")
    wide = q[0] if q is not None else p[0]
    cnt = q[1] if q is not None else p[1]
    if q is not None and p is not None:
        wide = wide.join(p[0][[m for m in p[0].columns if m not in wide.columns]])
        known = set(q[1].index.get_level_values("method"))
        cnt = pd.concat([cnt, p[1][~p[1].index.get_level_values("method").isin(known)]])
    out = {}
    users = np.sort(wide.index.get_level_values("user_id").unique())
    for lv in wide.index.get_level_values("level").unique():
        w = wide.xs(lv, level="level").reindex(users).dropna(axis=1, how="all")
        c = {}
        for m, g in cnt.xs(lv, level="level").groupby(level="method"):
            c[m] = g.droplevel("method").reindex(users).fillna(0).to_numpy(dtype=float)
        out[int(lv)] = Level(w, c)
    keep = set.intersection(*[set(v.wide.columns) for v in out.values()])
    out[MEAN_LEVEL] = Level(sum(v.wide[sorted(keep)] for v in out.values()) / len(out),
                            {m: sum(v.counts[m] for v in out.values()) for m in keep if all(m in v.counts for v in out.values())})
    return out, users


# ----------------------------------------------------------------------------------------------
# paired statistics: the same bootstrap draws of users for every cap and model of a scenario
# ----------------------------------------------------------------------------------------------
class Boot:
    def __init__(self, n_users, seed):
        idx = np.random.default_rng(seed).integers(0, n_users, size=(N_BOOT, n_users))
        self.W = np.zeros((N_BOOT, n_users))
        np.add.at(self.W, (np.arange(N_BOOT)[:, None], idx), 1.0)

    def mean(self, d):
        """Bootstrap of the mean of a per-user vector (NaN skipped)."""
        ok = ~np.isnan(d)
        return (self.W @ np.where(ok, d, 0.0)) / np.maximum(self.W @ ok.astype(float), 1)

    def cls(self, c):
        return fr.metrics(self.W @ c)


def interval(point, boots):
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"": float(point), "_low": float(lo), "_high": float(hi)}


def linear(boot, terms):
    """terms = [(sign, level_object, method)]: returns {metric: point, low, high} for the signed sum of the models.
    AUC is a mean of per-user differences; F1 and accuracy are pooled metrics (as in step 15), so their difference is taken after pooling."""
    row = {}
    if all(m in lv.wide.columns or m == "B0_const" for _, lv, m in terms):
        d = sum(s * (lv.wide[m].to_numpy() if m != "B0_const" else np.full(len(lv.wide), 0.5)) for s, lv, m in terms)
        row.update({f"auc{k}": v for k, v in interval(np.nanmean(d), boot.mean(d)).items()})
    if all(m in lv.counts for _, lv, m in terms):
        for met in ("f1_liked", "accuracy"):
            point = sum(s * fr.metrics(lv.counts[m].sum(axis=0))[met] for s, lv, m in terms)
            bs = sum(s * boot.cls(lv.counts[m])[met] for s, lv, m in terms)
            row.update({f"{met}{k}": v for k, v in interval(point, bs).items()})
    return row


def clear(lo, hi):
    return fr.verdict(lo, hi) != "no clear difference"


def learner_state(curve):
    """curve = {cap: (effect, low, high)} of one learner -> (real_sign, lack_of_data)."""
    e, lo, hi = curve["all"]
    real = int(np.sign(e)) if clear(lo, hi) and abs(e) > NOISE_FLOOR else 0
    vals = [curve[c][0] for c in CURVE]
    s = np.sign(vals[0])
    size = [s * v for v in vals]
    first_clear = clear(*curve["100"][1:]) and abs(vals[0]) > NOISE_FLOOR
    falling = all(a > b for a, b in zip(size, size[1:])) and size[-2] > 0
    gone = not clear(lo, hi) or abs(e) <= NOISE_FLOOR
    return real, bool(first_clear and falling and gone)


def judge(states):
    """states = {learner: (real_sign, lack)} for logreg and hgb_c."""
    (r1, l1), (r2, l2) = states["logreg"], states["hgb_c"]
    if r1 == r2 and r1 != 0:
        return "real positive" if r1 > 0 else "real negative"
    if l1 and l2:
        return "lack of data"
    if l1 or l2:
        return "lack of data in one learner"
    return "not established"


# ----------------------------------------------------------------------------------------------
# Persian text of the tables
# ----------------------------------------------------------------------------------------------
FA_DIGITS = str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫")
BLOCK_FA = {"P": "محبوبیت فیلم (P)", "H": "تاریخچه‌ی کاربر (H)", "L": "شهرت مدل زبانی (L)", "A": "هم‌خوانی با فیلم‌های دیده‌شده (A)",
            "A+H": "هم‌خوانی و تاریخچه (A+H)", "T": "سابقه‌ی کارگردان و بازیگر (T)", "K": "محتوای فیلم (K)", "Q": "امتیاز فیلم از سونت (Q)",
            "PUB": "رأی عمومی (PUB)", "FH": "تاریخچه‌ی فیلم (FH)", "PRraw": "پروفایل خام (PRraw)", "PRcrisp": "پروفایل قطعی (PRcrisp)",
            "PRfuzzy": "پروفایل فازی (PRfuzzy)", "fuzzy-raw": "فازی در برابر خام (fuzzy-raw)", "fuzzy-crisp": "فازی در برابر قطعی (fuzzy-crisp)"}
KIND_FA = {"add": "افزودن", "versus": "مقایسه", "loo": "حذف از مدل کامل"}
CAP_FA = {"100": "سقف ۱۰۰", "200": "سقف ۲۰۰", "400": "سقف ۴۰۰", "all": "بدون سقف", "all_uw": "بدون سقف با وزن برابر"}
VERDICT_FA = {"real positive": "اثر واقعی مثبت", "real negative": "اثر واقعی منفی", "lack of data": "ناشی از کمبود داده",
              "lack of data in one learner": "کمبود داده فقط در یک یادگیرنده", "not established": "ثابت‌نشده", "incomplete": "ناقص"}
LEARNER_FA = {"logreg": "رگرسیون لجستیک (logreg)", "hgb_c": "تقویت گرادیان (hgb_c)", "rf": "جنگل تصادفی (rf)"}
METRIC_FA = {"auc": "مساحت زیر منحنی (AUC)", "f1_liked": "اف‌وان (F1)", "accuracy": "دقت (accuracy)"}
SCEN_FA = {"A": "سناریوی کاربر جدید (A)", "B": "سناریوی فیلم جدید (B)", "C": "سناریوی هر دو جدید (C)"}


def fa(x, nd=4):
    return f"{abs(x):.{nd}f}".translate(FA_DIGITS)


def fa_signed(x, nd=4):
    return ("منفی " if round(x, nd) < 0 else "") + fa(x, nd)


def fa_cell(row, met, nd=4):
    """«افزایش ۰٫۰۰۱۲ (از ۰٫۰۰۰۵ تا ۰٫۰۰۱۹)»: the cell always starts with a Persian word."""
    x = row.get(met, np.nan)
    if pd.isna(x):
        return "اجرا نشده"
    word = "افزایش" if round(x, nd) > 0 else ("کاهش" if round(x, nd) < 0 else "بدون تغییر")
    return f"{word} {fa(x, nd)} (از {fa_signed(row[met + '_low'], nd)} تا {fa_signed(row[met + '_high'], nd)})"


def level_fa(key, lv):
    return "میانگین سطوح" if lv == MEAN_LEVEL else f"سطح {key}={str(lv).translate(FA_DIGITS)}"


def md_table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *["| " + " | ".join(r) + " |" for r in rows], ""]


def wrap_rtl(lines):
    return ['<div dir="rtl" style="text-align: right">', "", *lines, "", "</div>", ""]


# ----------------------------------------------------------------------------------------------
# rows actually used, from the logs of step 14
# ----------------------------------------------------------------------------------------------
def rows_used():
    out = []
    for sc, (folder, key) in SCEN.items():
        pat = re.compile(rf"\[{folder}\] repeat \d+ fold \d+ rule gt7 {key}=(\d+): (\d+) training rows")
        sources = {"200": [OUT_DIR / f"filmq_run_{sc}.log", OUT_DIR / "filmq_run.log"]}
        text_by_cap = {}
        for cap, files in sources.items():
            text_by_cap[cap] = "\n".join(f.read_text(encoding="utf-8", errors="ignore") for f in files if f.exists())
        log = OUT_DIR / f"rowcap_run_{sc}.log"
        if log.exists():
            parts = re.split(r"=== scenario \w cap \d+ weighted \w+ tag (\S+) .*\n", log.read_text(encoding="utf-8", errors="ignore"))
            for tag, body in zip(parts[1::2], parts[2::2]):
                for cap, suffix in LOGS.items():
                    if tag == f"_filmq{suffix}":
                        text_by_cap[cap] = body
        for cap, text in text_by_cap.items():
            found = [(int(a), int(b)) for a, b in pat.findall(text)]
            for lv in sorted({a for a, _ in found}):
                vals = [b for a, b in found if a == lv]
                out.append({"scenario": sc, "cap": cap, "level": lv, "mean_training_rows": float(np.mean(vals)), "fits": len(vals)})
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------
def main():
    out = ensure_dir(OUT_DIR / "row_cap_curve")
    gains, effects, changes, verdicts, checks = [], [], [], [], []
    for si, (sc, (folder, key)) in enumerate(SCEN.items()):
        data, users = {}, None
        for cap in CAPS:
            got = load_cap(folder, cap, checks)
            if got is None:
                checks.append(f"{folder} cap {cap}: no complete results")
                continue
            data[cap], u = got
            if users is None:
                users = u
            elif not np.array_equal(users, u):
                raise SystemExit(f"{folder} cap {cap}: the test users differ from the other levels")
        checks.append(f"{folder}: {len(users)} test users in every level, levels found {list(data)}")
        boot = Boot(len(users), [RANDOM_STATE, si, 24])
        levels = sorted(next(iter(data.values())))
        for lv in levels:
            at = {cap: data[cap][lv] for cap in data}
            # ---- gains of every model between levels of the cap
            methods = set.intersection(*[set(v.wide.columns) for v in at.values()]) if at else set()
            for m in sorted(x for x in methods if "|" in x):
                exp, k = m.split("|")
                for ca, cb in CAP_PAIRS:
                    if ca in at and cb in at:
                        gains.append({"scenario": sc, "level": lv, "model": exp, "learner": k, "cap_a": ca, "cap_b": cb,
                                      **linear(boot, [(1, at[ca], m), (-1, at[cb], m)])})
            # ---- effect of every block at every cap, and how the effect changes between caps
            for kind, block, a, b in BLOCKS[sc]:
                for k in LEARNERS:
                    ma, mb = f"{a}|{k}", (b if b == "B0_const" else f"{b}|{k}")
                    base = {"scenario": sc, "level": lv, "kind": kind, "block": block, "with": a, "without": b, "learner": k}
                    curve = {}
                    for cap, x in at.items():
                        if ma not in x.wide.columns or (mb != "B0_const" and mb not in x.wide.columns):
                            continue
                        row = linear(boot, [(1, x, ma), (-1, x, mb)])
                        effects.append({**base, "cap": cap, **row})
                        curve[cap] = row
                    for ca, cb in CHANGE_PAIRS:
                        if ca in curve and cb in curve:
                            changes.append({**base, "cap_a": ca, "cap_b": cb,
                                            **linear(boot, [(1, at[ca], ma), (-1, at[ca], mb), (-1, at[cb], ma), (1, at[cb], mb)])})
        eff = pd.DataFrame([e for e in effects if e["scenario"] == sc and e["level"] != MEAN_LEVEL])
        for (lv, kind, block, a, b), g in eff.groupby(["level", "kind", "block", "with", "without"], sort=False):
            states, row = {}, {"scenario": sc, "level": lv, "kind": kind, "block": block, "with": a, "without": b}
            for k in ("logreg", "hgb_c"):
                gk = g[g["learner"] == k].set_index("cap")
                if not all(c in gk.index for c in CURVE):
                    continue
                curve = {c: (gk.loc[c, "auc"], gk.loc[c, "auc_low"], gk.loc[c, "auc_high"]) for c in CURVE}
                states[k] = learner_state(curve)
                row.update({f"{k}_{c}": curve[c][0] for c in CURVE})
                row.update({f"{k}_all_low": curve["all"][1], f"{k}_all_high": curve["all"][2]})
            row["verdict"] = judge(states) if len(states) == 2 else "incomplete"
            verdicts.append(row)
        print(f"{folder}: done", flush=True)

    gains, effects, changes, verdicts = (pd.DataFrame(x) for x in (gains, effects, changes, verdicts))
    rows = rows_used()
    for name, tab in (("gains", gains), ("block_effects", effects), ("block_changes", changes), ("verdicts", verdicts), ("rows", rows)):
        tab.to_csv(out / f"{name}.csv", index=False)
    (out / "checks.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
    print("\n".join(checks))
    write_tables(out, gains, effects, changes, verdicts, rows)


def write_tables(out, gains, effects, changes, verdicts, rows):
    # ---- main report: tables that are small enough to read
    L = []
    L += ["## سطرهای آموزشی واقعی", "", "مقادیر زیر میانگین تعداد سطرهای آموزش در هر تقسیم (fold) است.", ""]
    head = ["سناریو و سطح", *[CAP_FA[c] for c in CURVE]]
    tr = []
    for sc, (_, key) in SCEN.items():
        sub = rows[rows["scenario"] == sc]
        for lv in sorted(sub["level"].unique()):
            cells = []
            for c in CURVE:
                q = sub[(sub["cap"] == c) & (sub["level"] == lv)]
                cells.append(f"حدود {int(round(q['mean_training_rows'].iloc[0])):,}".translate(FA_DIGITS).replace(",", "٬") if len(q) else "نامعلوم")
            tr.append([f"{SCEN_FA[sc]}، {level_fa(key, lv)}", *cells])
    L += md_table(head, tr)

    L += [f"## سود مدل اصلی ({HEADLINE}) از سقف ۲۰۰ تا بی‌سقف", "",
          "مقادیر زیر تفاوت جفت‌شده روی همان کاربران است و داخل پرانتز بازه‌ی ۹۵٪ بوت‌استرپ (bootstrap) آمده.", ""]
    for k in ("hgb_c", "logreg", "rf"):
        L += [f"### یادگیرنده‌ی {LEARNER_FA[k]}", ""]
        tr = []
        for sc, (_, key) in SCEN.items():
            sub = gains[(gains["scenario"] == sc) & (gains["model"] == HEADLINE) & (gains["learner"] == k) & (gains["cap_a"] == "all") & (gains["cap_b"] == "200")]
            for _, x in sub.sort_values("level").iterrows():
                tr.append([f"{SCEN_FA[sc]}، {level_fa(key, x['level'])}", *[fa_cell(x, m) for m in METRICS]])
        L += md_table(["سناریو و سطح", *METRIC_FA.values()], tr)

    L += ["## منحنی یادگیری مدل اصلی", "", "مقادیر زیر تفاوت هر سقف با سقف ۲۰۰ در مساحت زیر منحنی (AUC) برای یادگیرنده‌ی تقویت گرادیان (hgb_c) است.", ""]
    tr = []
    for sc, (_, key) in SCEN.items():
        sub = gains[(gains["scenario"] == sc) & (gains["model"] == HEADLINE) & (gains["learner"] == "hgb_c") & (gains["cap_b"] == "200")]
        for lv in sorted(sub["level"].unique()):
            cells = []
            for c in ("100", "400", "all", "all_uw"):
                q = sub[(sub["level"] == lv) & (sub["cap_a"] == c)]
                cells.append(fa_cell(q.iloc[0], "auc") if len(q) else "اجرا نشده")
            tr.append([f"{SCEN_FA[sc]}، {level_fa(key, lv)}", *cells])
    L += md_table(["سناریو و سطح", CAP_FA["100"], CAP_FA["400"], CAP_FA["all"], CAP_FA["all_uw"]], tr)

    L += ["## اثر وزن برابر کاربران", "", "مقادیر زیر تفاوت «بدون سقف با وزن برابر» و «بدون سقف بدون وزن» برای مدل اصلی است.", ""]
    tr = []
    for sc, (_, key) in SCEN.items():
        for k in ("hgb_c", "logreg", "rf"):
            sub = gains[(gains["scenario"] == sc) & (gains["model"] == HEADLINE) & (gains["learner"] == k) & (gains["cap_a"] == "all_uw") & (gains["cap_b"] == "all")]
            for _, x in sub.sort_values("level").iterrows():
                tr.append([f"{SCEN_FA[sc]}، {level_fa(key, x['level'])}، {LEARNER_FA[k]}", *[fa_cell(x, m) for m in METRICS]])
    L += md_table(["سناریو، سطح و یادگیرنده", *METRIC_FA.values()], tr)

    L += ["## حکم هر بلوک طبق قاعده‌های از پیش ثابت‌شده", "",
          "هر خانه حکم آن سطح است و عدد کنارش اثر بی‌سقف در یادگیرنده‌ی تقویت گرادیان (hgb_c) و رگرسیون لجستیک (logreg) است.", ""]
    for sc, (_, key) in SCEN.items():
        sub = verdicts[verdicts["scenario"] == sc]
        lvs = sorted(sub["level"].unique())
        L += [f"### {SCEN_FA[sc]}", ""]
        tr = []
        for (kind, block), g in sub.groupby(["kind", "block"], sort=False):
            cells = []
            for lv in lvs:
                q = g[g["level"] == lv]
                if len(q) == 0:
                    cells.append("اجرا نشده")
                    continue
                x = q.iloc[0]
                nums = f"{fa_signed(x['hgb_c_all'])} و {fa_signed(x['logreg_all'])}" if "hgb_c_all" in x and pd.notna(x.get("hgb_c_all")) else ""
                cells.append(f"{VERDICT_FA[x['verdict']]}، {nums}" if nums else VERDICT_FA[x["verdict"]])
            tr.append([f"{KIND_FA[kind]} {BLOCK_FA[block]}", *cells])
        L += md_table(["بلوک", *[level_fa(key, v) for v in lvs]], tr)

    L += ["## اثرهایی که با برداشتن سقف تغییر کردند", "",
          "سطرهای زیر فقط مواردی است که بازه‌ی تفاوت «اثر بی‌سقف منهای اثر در سقف ۲۰۰» صفر را در بر نمی‌گیرد و اندازه‌ی این تغییر از کف نوفه (۰٫۰۰۰۶) بیشتر است.", ""]
    ch = changes[(changes["cap_a"] == "all") & (changes["cap_b"] == "200") & (changes["level"] != MEAN_LEVEL) & changes["learner"].isin(["logreg", "hgb_c"])]
    ch = ch[[clear(a, b) and abs(c) > NOISE_FLOOR for a, b, c in zip(ch["auc_low"], ch["auc_high"], ch["auc"])]]
    tr = [[f"{SCEN_FA[x['scenario']]}، {level_fa(SCEN[x['scenario']][1], x['level'])}", f"{KIND_FA[x['kind']]} {BLOCK_FA[x['block']]}", LEARNER_FA[x["learner"]], fa_cell(x, "auc")]
          for _, x in ch.iterrows()]
    L += md_table(["سناریو و سطح", "بلوک", "یادگیرنده", "تغییر اثر"], tr) if tr else ["هیچ موردی این شرط را نداشت.", ""]

    findings = out / "findings_fa.md"
    head_lines = findings.read_text(encoding="utf-8").splitlines() if findings.exists() else ["# گزارش منحنی یادگیری سقف سطرها", ""]
    (OUT_DIR / "row_cap_curve_report.md").write_text("\n".join(wrap_rtl(head_lines + [""] + L)), encoding="utf-8")

    # ---- appendix: every block, level, cap, learner and metric
    A = ["# پیوست: جدول کامل اثر بلوک‌ها", ""]
    for sc, (_, key) in SCEN.items():
        A += [f"## {SCEN_FA[sc]}", ""]
        sub = effects[effects["scenario"] == sc]
        for lv in sorted(sub["level"].unique()):
            for k in LEARNERS:
                for met, name in METRIC_FA.items():
                    g = sub[(sub["level"] == lv) & (sub["learner"] == k)]
                    if g.empty or met not in g.columns or g[met].notna().sum() == 0:
                        continue
                    A += [f"### {level_fa(key, lv)}، {LEARNER_FA[k]}، {name}", ""]
                    tr = []
                    for (kind, block), b in g.groupby(["kind", "block"], sort=False):
                        cells = []
                        for c in CAPS:
                            q = b[b["cap"] == c]
                            cells.append(fa_cell(q.iloc[0], met) if len(q) else "اجرا نشده")
                        tr.append([f"{KIND_FA[kind]} {BLOCK_FA[block]}", *cells])
                    A += md_table(["بلوک", *[CAP_FA[c] for c in CAPS]], tr)
    (out / "tables_fa.md").write_text("\n".join(wrap_rtl(A)), encoding="utf-8")
    print(f"files in {out}\nreport: {OUT_DIR / 'row_cap_curve_report.md'}")


if __name__ == "__main__":
    main()
