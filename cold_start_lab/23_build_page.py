"""
STEP 23 - builds the meeting page (outputs/23_page/meeting_page.html) from the result files.
Needs: outputs/22_page/page_data.json (step 22) and outputs/21_sim/sim_data.json (step 21).
Style: the CSS of the page published on 2026-10-20 is reused so the page keeps its look; new components are added.
"""
import json
import re

import pandas as pd

from config import OUT_DIR, ensure_dir

OLD = r"C:\Users\Mo\.claude\projects\C--Users-Mo-Desktop-Universiteit-QIAU-MA-Thesis\50d0e207-74cb-4f0a-b98a-feb470b33f83\tool-results\artifact-6a47868e-1792485860-6c8a.html"
OUT = ensure_dir(OUT_DIR / "23_page")
D = json.loads((OUT_DIR / "22_page" / "page_data.json").read_text(encoding="utf-8"))
SIM = (OUT_DIR / "21_sim" / "sim_data.json").read_text(encoding="utf-8")
FA = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def f(x, nd=3):
    return f"{x:.{nd}f}".replace(".", "٫").replace("-", "−").translate(FA)


def sg(x, nd=4):
    return ("+" if x >= 0 else "−") + f(abs(x), nd)


def fi(n):
    return str(n).translate(FA)


VERD = {"better": ("بهتر", "ok"), "worse": ("بدتر", "bad"), "no clear difference": ("روشن نیست", "na")}


def eff(r, nd=4):
    if not r:
        return "<td class='na'>—</td>"
    v = VERD[r[3]]
    return f"<td class='{v[1]}'><span class='num'>{sg(r[0], nd)}</span><br><small class='num'>[{sg(r[1], nd)} ، {sg(r[2], nd)}]</small><br><b>{v[0]}</b></td>"


def tbl(head, rows, cls=""):
    cls = cls  # 'wide' keeps a minimum width and scrolls on narrow screens
    return f"<div class='tblwrap'><table class='{cls}'><thead><tr>{''.join(f'<th>{h}</th>' for h in head)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


# ------------------------------------------------------------------ static tables
cm = [("پیشنهاد جدید: امتیاز مستقیم هر فیلم با Sonnet", "next", "جلسهٔ بعد", "آزمونی که می‌تواند ادعای نوآوری LLM را تأیید یا رد کند. حدود ۳ دلار؛ اجرا نشده."),
      ("Ablation study", "ok", "انجام شد", "محبوبیت در کاربر جدید و محتوای TMDB در فیلم جدید مهم‌ترین بلوک‌اند (بخش ۳)."),
      ("FUZZY LLM 4 STEP", "ok", "انجام شد، برداشت ما: فازی × LLM", "LLM حدود ۰٫۰۰۷ تا ۰٫۰۰۹ AUC اضافه می‌کند؛ فازی ناچیز و ناپایدار است (بخش ۳)."),
      ("Benchmark with CSP 2023", "ok", "انجام شد", "روش مقالهٔ CSP 2023 از «فقط محبوبیت فیلم» بهتر نیست (بخش ۵)."),
      ("Why these algorithms (مزایا و توجیه)", "part", "نیمه‌کاره", "جدول و متن توجیه آماده است؛ منبع علمی ادعاهای کلی مانده (بخش ۸)."),
      ("New user", "ok", "انجام شد", "فقط با محبوبیت فیلم AUC حدود ۰٫۸۱۲ می‌گیریم (بخش ۴)."),
      ("Sparsity", "ok", "انجام شد", "با کم شدن نمره‌ها، محتوای TMDB ارزشمندتر می‌شود؛ LLM نه (بخش ۴)."),
      ("Dataset with no rate / test new item", "ok", "انجام شد", "فیلم بدون نمره (M=۰) و با ۵ نمره؛ M=۲۰ و ۵۰ اجرا نشده (بخش ۴)."),
      ("User with no rating", "ok", "انجام شد", "کاربر بدون نمره (N=۰) تا ۴۰۰ نمره؛ ۴۰۰ نمره فقط حدود ۰٫۰۱۴ اضافه می‌کند (بخش ۴)."),
      ("Effect of each on F1 and accuracy", "ok", "انجام شد", "دقت و F1 کنار دو حدس ساده؛ اثر بلوک‌ها روی F1 در بخش ۶."),
      ("Figures", "ok", "انجام شد", "چهار نمودار در همین صفحه."),
      ("Innovation: LLM", "part", "ادعا فروتنانه شد", "LLM بدون TMDB اثر دارد و با TMDB تقریباً زائد است (بخش ۹)."),
      ("DELETE LLM", "ok", "انجام شد، برداشت ما: حذف در ablation", "حذف LLM در A و B اثری ندارد و در C حدود ۰٫۰۰۲ می‌گیرد (بخش ۳)."),
      ("Recommendation: get user (Top-10)", "ok", "انجام شد", "شبیه‌ساز Top-10؛ لیست ما از «فقط محبوبیت» بهتر نیست (بخش ۷).")]
ST = {"ok": "pill ok", "part": "pill part", "next": "pill nx"}
T_CMT = tbl(["کامنت استاد", "وضعیت", "یافتهٔ اصلی"],
            [f"<tr><td class='en'>{a}</td><td><span class='{ST[b]}'>{c}</span></td><td>{d}</td></tr>" for a, b, c, d in cm])

t = D["tmdb"]
rows = []
for s, nm in (("A", "A: کاربر جدید"), ("B", "B: فیلم جدید"), ("C", "C: هر دو جدید")):
    pk = t["pairs"][s]
    rows.append(f"<tr><th>{nm}</th><td class='num'>{f(t[s]['base'], 4)}</td><td class='num'>{f(t[s]['FULL'], 4)}</td><td class='num'><b>{f(t[s]['FULLK'], 4)}</b></td>"
                f"{eff(pk['K'])}<td class='num ceil'>{f(t[s]['FULLKP'], 4)}</td></tr>")
T_TMDB = tbl(["سناریو (سطح صفر)", "خط پایه", "مدل بدون TMDB", "مدل با محتوای TMDB", "اثر محتوای TMDB", "با امتیاز عمومی TMDB (سقف)"], rows, "wide")

lab = {"P": "محبوبیت فیلم (P)", "T": "سابقهٔ کارگردان و بازیگر (T)", "L": "شهرت LLM (L)", "PR": "پروفایل فازی (PR)", "K": "محتوای TMDB (K)", "H": "تاریخچهٔ کاربر (H)", "A": "شباهت (A)"}
ab = D["abl"]
rows = []
for b in ("P", "T", "L", "PR", "K"):
    cells = "".join(eff(ab[s].get(b)) if b in ab[s] else "<td class='na'>—</td>" for s in "ABC")
    rows.append(f"<tr><th>{lab[b]}</th>{cells}</tr>")
T_ABL = tbl(["بلوکِ حذف‌شده از مدل نهایی", "A: کاربر جدید", "B: فیلم جدید", "C: هر دو جدید"], rows)

fx = D["fxl"]
lbl = {"00": "۰۰ بدون فازی، بدون LLM", "01": "۰۱ بدون فازی، با LLM", "10": "۱۰ با فازی، بدون LLM", "11": "۱۱ با فازی، با LLM"}
rows = [f"<tr><th>{lbl[k]}</th><td class='num'>{f(fx['mean'][k]['0'], 4)}</td><td class='num'>{f(fx['mean'][k]['20'], 4)}</td></tr>" for k in ("00", "01", "10", "11")]
T_FXL = tbl(["حالت (سناریوی C)", "AUC در N=۰", "AUC در N=۲۰"], rows)
efl = [("اثر فازی بدون LLM (۱۰ − ۰۰)", "fuzzy_noLLM"), ("اثر فازی با LLM (۱۱ − ۰۱)", "fuzzy_LLM"), ("اثر LLM بدون فازی (۰۱ − ۰۰)", "llm_noFuzzy"), ("اثر LLM با فازی (۱۱ − ۱۰)", "llm_fuzzy")]
rows = [f"<tr><th>{n}</th>{eff(fx['eff'][k]['0'])}{eff(fx['eff'][k]['20'])}</tr>" for n, k in efl]
T_FXE = tbl(["اثر", "N=۰", "N=۲۰"], rows)


fxm_lbl = {"00": ("بدون فازی", "بدون LLM"), "01": ("بدون فازی", "با LLM"), "10": ("با فازی", "بدون LLM"), "11": ("با فازی", "با LLM")}


def fx_cell(k):
    m = fx["mean"][k]
    return (f"<div class='mx'><span class='mxk'>{fi(k)}</span><b class='num'>{f(m['0'], 4)}</b><small>N=۰</small>"
            f"<span class='num mxs'>{f(m['20'], 4)} در N=۲۰</span></div>")


def fx_eff(title, key):
    r0, r20 = fx["eff"][key]["0"], fx["eff"][key]["20"]
    def one(r):
        v = VERD[r[3]]
        return f"<div class='ef {v[1]}'><b class='num'>{sg(r[0])}</b><span class='num'>[{sg(r[1])} ، {sg(r[2])}]</span><em>{v[0]}</em></div>"
    return f"<div class='card efc'><h4>{title}</h4><div class='efr'><span>N=۰</span>{one(r0)}</div><div class='efr'><span>N=۲۰</span>{one(r20)}</div></div>"


T_FXM = ("<div class='mxwrap'><div class='mxh'></div><div class='mxh'>بدون LLM</div><div class='mxh'>با LLM</div>"
         f"<div class='mxh r'>بدون فازی<br><small>پروفایل خام</small></div>{fx_cell('00')}{fx_cell('01')}"
         f"<div class='mxh r'>با فازی<br><small>پروفایل فازی F1</small></div>{fx_cell('10')}{fx_cell('11')}</div>"
         "<p class='mini'>عدد بزرگ: میانگین AUC در N=۰ (کد ۰۰ تا ۱۱: رقم اول فازی، رقم دوم LLM). عدد کوچک: همان در N=۲۰.</p>"
         "<div class='efgrid'>" + fx_eff("اثر فازی بدون LLM (۱۰ − ۰۰)", "fuzzy_noLLM") + fx_eff("اثر فازی با LLM (۱۱ − ۰۱)", "fuzzy_LLM")
         + fx_eff("اثر LLM بدون فازی (۰۱ − ۰۰)", "llm_noFuzzy") + fx_eff("اثر LLM با فازی (۱۱ − ۱۰)", "llm_fuzzy") + "</div>")

dl = D["delL"]
rows = []
for s, nm in (("A", "A: کاربر جدید"), ("B", "B: فیلم جدید"), ("C", "C: هر دو جدید")):
    lvA = "0" if s != "B" else "0"
    rows.append(f"<tr><th>{nm}</th>{eff(dl['FULLK'][s]['0'])}{eff(dl['FULL'][s]['0'])}</tr>")
T_DEL = tbl(["سناریو (سطح صفر)", "مدل نهایی با TMDB: اثر حذف LLM", "مدل پایه بدون TMDB: اثر حذف LLM"], rows)

sp = D["sparse"]
keep = sp["keep"]
rows = [f"<tr><th>محتوای TMDB (K)، {nm}</th><td class='num'>{sg(sp['effK'][s][0])}</td><td class='num'>{sg(sp['effK'][s][-1])}</td></tr>"
        for s, nm in (("A", "A (روی محبوبیت)"), ("B", "B"), ("C", "C"))]
rows += [f"<tr><th>شهرت LLM (L)، {nm}</th><td class='num'>{sg(sp['effL'][s][0])}</td><td class='num'>{sg(sp['effL'][s][-1])}</td></tr>" for s, nm in (("B", "B"), ("C", "C"))]
T_SPE = tbl(["اثر روی AUC (اختلاف جفت‌شده)", "۱۰۰٪ نمره‌ها", "۵٪ نمره‌ها"], rows)

bm = D["bench"]
rows = []
for k, nm in (("POP_mean", "فقط محبوبیت فیلم (میانگین نمره)"), ("CSP23_content", "CSP 2023، فقط بخش محتوا"), ("CSP23_total", "CSP 2023، روش کامل مقاله"), ("HGB_FULL", "مدل ما (hgb_c)")):
    a = bm[k]["auc"]
    rows.append(f"<tr><th>{nm}</th><td class='num'>{f(a[0], 4)} <small>[{f(a[1], 3)} ، {f(a[2], 3)}]</small></td><td class='num'>{f(bm[k]['top5'], 2)}</td><td class='num'>{fi(bm[k]['imp'])}٪</td></tr>".replace(".", "٫") if False else
                f"<tr><th>{nm}</th><td class='num'>{f(a[0], 4)} <small>[{f(a[1], 3)} ، {f(a[2], 3)}]</small></td><td class='num'>{f(bm[k]['top5'], 2)}</td><td class='num'>{f(bm[k]['imp'], 1)}٪</td></tr>")
T_BEN = tbl(["روش (کاربر جدید، N=۰)", "AUC [بازهٔ ۹۵٪]", "میانگین نمرهٔ ۵ پیشنهاد برتر", "بهبود نسبت به میانگین کاربر"], rows)

cl = D["cls"]
rows = []
for s, nm in (("A", "A: کاربر جدید"), ("B", "B: فیلم جدید"), ("C", "C: هر دو جدید")):
    c = cl[s]
    rows.append(f"<tr><th>{nm}</th><td class='num'><b>{f(c['model']['acc'])}</b></td><td class='num'><b>{f(c['model']['f1'])}</b></td>"
                f"<td class='num'>{f(c['no']['acc'])}</td><td class='num'>{f(c['no']['f1'])}</td><td class='num'>{f(c['yes']['acc'])}</td><td class='num'>{f(c['yes']['f1'])}</td></tr>")
T_CLS = tbl(["سناریو (سطح صفر)", "مدل نهایی: دقت", "مدل نهایی: F1", "«همه نپسندیده‌اند»: دقت", "F1", "«همه پسندیده‌اند»: دقت", "F1"], rows, "wide")

pc = {s: pd.read_csv(OUT_DIR / "14_final_ablk" / sc_ / "pairs_cls.csv") for s, sc_ in (("A", "A_new_user"), ("B", "B_new_film"), ("C", "C_both_new"))}


def cls_cell(s, blk):
    d_ = pc[s]
    q = d_[(d_.rule == "gt7") & (d_.level == 0) & (d_.A == "FULLK|hgb_c") & (d_.B == f"FULLK-{blk}|hgb_c")]
    if q.empty:
        return "<td class='na'>—</td>"
    r = q.iloc[0]
    v = VERD[r.accuracy_verdict]
    return f"<td class='{v[1]}'><span class='num'>دقت {sg(r.accuracy_diff, 3)}</span><br><span class='num'>F1 {sg(r.f1_liked_diff, 3)}</span></td>"


blkn = [("P", "محبوبیت فیلم (P)"), ("T", "سابقهٔ کارگردان و بازیگر (T)"), ("AH", "تاریخچه و شباهت کاربر"), ("L", "شهرت LLM (L)"), ("PR", "پروفایل فازی (PR)"), ("K", "محتوای TMDB (K)")]
T_CLSB = tbl(["بلوکِ حذف‌شده", "A: کاربر جدید", "B: فیلم جدید", "C: هر دو جدید"],
             [f"<tr><th>{n}</th>{''.join(cls_cell(s, b) for s in 'ABC')}</tr>" for b, n in blkn])

tk = D["topk"]
rows = []
for k, nm in (("RANDOM", "تصادفی"), ("POP", "فقط محبوبیت فیلم"), ("FULLK_noL", "مدل نهایی بدون LLM"), ("FULLK", "مدل نهایی (با LLM و TMDB)")):
    r0, r20 = tk["0"][k], tk["20"][k]
    rows.append(f"<tr><th>{nm}</th><td class='num'>{f(r0['ndcg'][0])} <small>[{f(r0['ndcg'][1])} ، {f(r0['ndcg'][2])}]</small></td><td class='num'>{f(r0['p10'])}</td>"
                f"<td class='num'>{f(r20['ndcg'][0])}</td><td class='num'>{f(r20['p10'])}</td></tr>")
T_TOP = tbl(["روش", "NDCG@10 (N=۰)", "P@10 (N=۰)", "NDCG@10 (N=۲۰)", "P@10 (N=۲۰)"], rows)

ln = D["learners"]
rows = [f"<tr><th>{nm}</th><td class='num'>{f(ln[s]['logreg'], 4)}</td><td class='num'><b>{f(ln[s]['hgb_c'], 4)}</b></td><td class='num'>{f(ln[s]['rf'], 4)}</td>"
        f"<td class='num'>{f(ln[s]['hgb_BK'], 4)}</td><td class='num'>{f(ln[s]['mlp_BK'], 4)}</td></tr>"
        for s, nm in (("A", "A: کاربر جدید"), ("B", "B: فیلم جدید"), ("C", "C: هر دو جدید"))]
T_LRN = tbl(["سناریو (سطح صفر)", "رگرسیون لجستیک", "hgb_c (اصلی)", "جنگل تصادفی", "hgb_c با TMDB", "شبکهٔ عصبی MLP با TMDB"], rows, "wide")

# ------------------------------------------------------------------ CSS from the old page + new components
old = open(OLD, encoding="utf-8").read()
css = re.findall(r"<style>(.*?)</style>", old, re.S)
CSS = css[1] if len(css) > 1 else css[0]
CSS += """
h2{font-size:24px}
h2 small{font-size:14px;color:var(--muted);font-weight:400}
.navchips{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px}
.navchips a{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:3px 12px;border-radius:99px;text-decoration:none;font-size:13.5px}
.navchips a:hover{border-color:var(--teal);color:var(--teal)}
.pill{display:inline-block;font-size:12.5px;font-weight:600;padding:1px 10px;border-radius:99px;white-space:nowrap}
.pill.ok{background:var(--teal-soft);color:var(--teal)}.pill.part{background:var(--amber-soft);color:var(--amber)}.pill.nx{background:var(--violet-soft);color:var(--violet)}
td.ok{background:color-mix(in srgb,var(--teal) 13%,transparent)} td.bad{background:color-mix(in srgb,var(--coral) 15%,transparent)}
td.na{color:var(--muted)} td small{color:var(--muted)} td b{font-weight:600}
td.ceil{color:var(--amber)} td.en{direction:ltr;text-align:right;unicode-bidi:isolate}
table td,table th{font-size:14px}
.note{border-radius:12px;padding:12px 16px;background:var(--amber-soft);color:var(--fg);font-size:14.5px;margin-top:10px}
.note b{color:var(--amber)}
.ask{border-radius:12px;padding:12px 16px;background:var(--violet-soft);font-size:14.5px;margin-top:10px}
.ask b{color:var(--violet)}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.fig{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:12px 14px;min-width:0}
.fig h3{margin:0 0 4px;font-size:16px}.fig p{margin:0 0 6px;font-size:13.5px;color:var(--muted)}
.fig svg{display:block;width:100%;height:auto;direction:ltr}
.sv text{fill:var(--muted);font-size:11px;font-family:var(--font)}
.sv .ttl{fill:var(--fg);font-size:12.5px;font-weight:600}
.sv .ax{stroke:var(--line);stroke-width:1} .sv .grid{stroke:var(--line);stroke-width:.6;stroke-dasharray:3 3}
.sv .base{stroke:var(--muted);stroke-width:1.2;stroke-dasharray:5 4;fill:none}
.leg{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12.5px;color:var(--muted);margin-top:4px}
.leg i{display:inline-block;width:18px;height:3px;border-radius:2px;margin-left:6px;vertical-align:middle}
.c1{--c:var(--teal)}.c2{--c:var(--violet)}.c3{--c:var(--amber)}.c4{--c:var(--coral)}.c5{--c:var(--fg)}
.sv .l{stroke:var(--c);stroke-width:2;fill:none} .sv .p{fill:var(--c)} .sv .bar{fill:var(--c)} .sv .wh{stroke:var(--fg);stroke-width:1.4;fill:none}
.leg i{background:var(--c)}
.stabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.stabs button{border:1px solid var(--line);background:var(--card);color:var(--fg);padding:6px 16px;border-radius:99px;cursor:pointer;font-size:14px}
.stabs button[aria-selected="true"]{background:var(--teal);color:#fff;border-color:var(--teal)}
:root[data-theme="dark"] .stabs button[aria-selected="true"]{color:#0b1a1c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .stabs button[aria-selected="true"]{color:#0b1a1c}}
.ctrl{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:end;margin-bottom:12px}
.ctrl label{display:flex;flex-direction:column;gap:2px;font-size:13px;color:var(--muted);min-width:0}
.ctrl select{min-width:150px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:8px 0 12px}
.kpi{background:var(--bg);border-radius:12px;padding:8px 12px}.kpi b{display:block;font-size:20px}.kpi span{font-size:12.5px;color:var(--muted)}
ol.top li{grid-template-columns:auto 1fr auto auto auto}
ol.top .t1{font-weight:600;overflow-wrap:anywhere}
ol.top .why{display:block}
.hit{font-weight:800}.hit.y{color:var(--teal)}.hit.n{color:var(--coral)}
ol.top li.new{outline:2px solid var(--violet)}
.range{width:100%;direction:ltr}
.stepline{display:flex;flex-wrap:wrap;gap:4px 8px;font-size:12.5px;color:var(--muted);margin:6px 0}
.stepline span.on{color:var(--teal);font-weight:700}
.flowrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:10px}
.flowrow .step{border-radius:12px;min-width:0}.flowrow b{display:block;color:var(--teal);margin-bottom:2px}
footer ul{margin:6px 0 0;padding-right:20px}

.tblwrap table{min-width:0}
.tblwrap table.wide{min-width:620px}
.tblwrap td,.tblwrap th{overflow-wrap:anywhere}
.symbols{margin-top:14px;padding:12px 16px}
.symbols .sg{display:flex;flex-wrap:wrap;gap:6px 10px;margin-top:8px}
.symbols .sg span{display:inline-flex;align-items:center;gap:4px;background:var(--bg);border-radius:99px;padding:2px 10px;font-size:13px}
.symbols .sg i{font-style:normal;font-weight:800;color:var(--teal);direction:ltr;unicode-bidi:isolate}
.mxwrap{display:grid;grid-template-columns:minmax(80px,.8fr) 1fr 1fr;gap:8px;align-items:stretch}
.mxh{font-weight:600;font-size:14px;align-self:center;text-align:center}.mxh.r{text-align:right}.mxh small{color:var(--muted);font-weight:400}
.mx{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:10px 12px;text-align:center;position:relative;min-width:0}
.mx b{display:block;font-size:22px}.mx small{color:var(--muted)}.mxs{display:block;font-size:12.5px;color:var(--muted)}
.mxk{position:absolute;top:6px;right:10px;font-size:12px;color:var(--violet);font-weight:700}
.efgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px;margin-top:12px}
.efc{padding:10px 14px}.efc h4{margin:0 0 6px;font-size:14px}
.efr{display:grid;grid-template-columns:44px 1fr;gap:6px;align-items:center;margin-bottom:4px;font-size:13px;color:var(--muted)}
.ef{display:flex;flex-wrap:wrap;gap:2px 8px;align-items:baseline;border-radius:8px;padding:3px 8px;background:var(--bg);color:var(--fg)}
.ef b{font-size:15px}.ef span{font-size:12px;color:var(--muted)}.ef em{font-style:normal;font-size:12px;font-weight:600}
.ef.ok{background:color-mix(in srgb,var(--teal) 14%,transparent)}.ef.bad{background:color-mix(in srgb,var(--coral) 16%,transparent)}
@media (max-width:520px){.mxwrap{grid-template-columns:70px 1fr 1fr;gap:6px}.mx b{font-size:18px}}
.anim{padding:14px 16px;min-width:0}
.anhead{display:flex;flex-wrap:wrap;gap:6px 12px;align-items:center;justify-content:space-between;margin-bottom:8px}
.anhead b{font-size:16px}
.anctl{display:flex;gap:6px}.anctl button{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:99px;padding:3px 12px;cursor:pointer;font-size:13px}
.anstage{position:relative;overflow:hidden;border-radius:12px;background:var(--bg)}
.anrow{position:absolute;right:0;left:0;top:0;height:40px;display:grid;grid-template-columns:28px 1fr auto auto;gap:8px;align-items:center;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:0 10px;transition:transform 1.6s cubic-bezier(.4,0,.2,1),opacity 1.1s;overflow:hidden;min-width:0}
.anrow.new{border-color:var(--violet);box-shadow:0 0 0 2px var(--violet-soft)}
.anrow .rk{font-weight:800;color:var(--teal);text-align:center}
.anrow .ti{min-width:0;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-size:14px}.anrow .ti small{color:var(--muted)}
.anrow .scb{width:70px;height:8px;border-radius:4px;background:var(--track);overflow:hidden;direction:ltr}.anrow .scb i{display:block;height:100%;background:var(--teal);transition:width 1.2s}
.anrow .res{min-width:56px;text-align:left;font-weight:700;font-size:13px}
.anrow .res.y{color:var(--teal)}.anrow .res.n{color:var(--coral)}
.ancap{margin:8px 0 0;font-size:14px}.ankpi{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;color:var(--muted);margin-top:6px}
.ankpi b{color:var(--fg)}
.gauge{position:relative;height:34px;background:var(--track);border-radius:10px;overflow:hidden;direction:ltr;margin:10px 0}
.gfill{position:absolute;left:0;top:0;bottom:0;width:50%;background:var(--teal);border-radius:10px;transition:width 1.6s cubic-bezier(.4,0,.2,1),background .6s}
.gfill.drop{background:var(--coral)}.gfill.gain{background:var(--amber)}.gfill.flat{background:var(--violet)}
.gref{position:absolute;top:0;bottom:0;width:2px;background:var(--fg);opacity:.7}
.gval{position:absolute;left:10px;top:50%;transform:translateY(-50%);font-weight:800;color:#fff;font-size:15px;mix-blend-mode:normal;text-shadow:0 1px 2px rgba(0,0,0,.35)}
.achips{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0}
.achip{border:2px solid var(--teal);border-radius:12px;padding:4px 12px;font-size:13.5px;background:var(--card);transition:transform .9s,opacity .9s,border-color .6s,background .6s}
.achip.off{opacity:.35;transform:translateY(8px) scale(.94);border-style:dashed;border-color:var(--coral);text-decoration:line-through}
.achip.act{background:var(--coral-soft)}
.arank{display:grid;gap:5px;margin-top:8px}
.arank div{display:grid;grid-template-columns:150px 1fr 70px;gap:8px;align-items:center;font-size:13px}
.arank .tr{height:12px;background:var(--track);border-radius:6px;overflow:hidden;direction:ltr}.arank .tr i{display:block;height:100%;background:var(--teal);border-radius:6px;width:0;transition:width 1.6s}
.arank .tr i.neg{background:var(--coral)}
@media (max-width:520px){.arank div{grid-template-columns:100px 1fr 58px}.anrow .scb{display:none}}

.anbox{background:var(--bg);border-radius:14px;padding:10px}
.ancapbox{background:var(--card);border:1px solid var(--violet);border-radius:12px;padding:10px 14px;margin-bottom:10px;font-size:15.5px;line-height:1.9;min-height:96px}
.ancapbox b{display:block;color:var(--violet);font-size:13.5px;margin-bottom:2px}
.ticker{height:5px;background:var(--track);border-radius:3px;margin-top:10px;overflow:hidden;direction:ltr}
.ticker i{display:block;height:100%;width:0;background:var(--violet)}
.anbox .anstage{background:transparent}
@media (prefers-reduced-motion:reduce){.anrow,.gfill,.achip,.arank .tr i{transition:none}}

#pop .s{display:block}
#pop .pl{display:block;margin-top:6px;padding-top:6px;border-top:1px dashed var(--line);color:var(--muted);font-size:13.5px}
.hq{margin:0 3px}
"""

# ------------------------------------------------------------------ the page
BODY = r"""
<div class="wrap">
<header>
  <div class="eyebrow">محمد فراهانی · پایان‌نامهٔ کارشناسی ارشد · دانشگاه آزاد قزوین · جلسهٔ دوم با استاد راهنما</div>
  <h1>کاهش مشکل شروع سرد در توصیه‌گر فیلم با فازی‌سازی اطلاعات شبکهٔ اجتماعی و مدل‌های زبان</h1>
  <p class="lead">این صفحه فقط تغییرات و نتایج نهایی از جلسهٔ قبل را نشان می‌دهد. همهٔ اعداد از یادگیرندهٔ اصلی (hgb_c)، قاعدهٔ «پسند یعنی نمره بیشتر از ۷» و میانگین AUC هر کاربر با بازهٔ اطمینان ۹۵٪ آمده‌اند. هر جدول فقط بهترین مدل را کنار یک حدس ساده می‌گذارد.</p>
  <nav class="navchips" aria-label="فهرست بخش‌ها">
    <a href="#s0">کامنت‌ها</a><a href="#s1">مفاهیم</a><a href="#s2">محتوای TMDB</a><a href="#s3">ablation، فازی و LLM</a><a href="#s4">اگر نمره کم باشد</a>
    <a href="#s5">محک CSP 2023</a><a href="#s6">F1 و دقت</a><a href="#s7">Top-10 و شبیه‌ساز</a><a href="#s8">الگوریتم‌ها</a><a href="#s9">نوآوری LLM و جلسهٔ بعد</a>
  </nav>
  <div class="card symbols"><b>نمادهایی که در صفحه می‌بینید</b>
    <div class="sg">
      <span><i>سناریو A</i> کاربر جدید<button type="button" class="hq" data-h="sA" aria-label="توضیح سناریو A">؟</button></span>
      <span><i>سناریو B</i> فیلم جدید<button type="button" class="hq" data-h="sB" aria-label="توضیح سناریو B">؟</button></span>
      <span><i>سناریو C</i> هر دو جدید<button type="button" class="hq" data-h="sC" aria-label="توضیح سناریو C">؟</button></span>
      <span><i>P</i> محبوبیت فیلم<button type="button" class="hq" data-h="bP" aria-label="توضیح P">؟</button></span>
      <span><i>H</i> تاریخچهٔ کاربر<button type="button" class="hq" data-h="bH" aria-label="توضیح H">؟</button></span>
      <span><i>T</i> سابقهٔ کارگردان و بازیگر<button type="button" class="hq" data-h="bT" aria-label="توضیح T">؟</button></span>
      <span><i>L</i> شهرت LLM<button type="button" class="hq" data-h="bL" aria-label="توضیح L">؟</button></span>
      <span><i>A</i> شباهت به نمره‌های کاربر<button type="button" class="hq" data-h="bA" aria-label="توضیح A">؟</button></span>
      <span><i>PR</i> پروفایل فازی توییتر<button type="button" class="hq" data-h="bPR" aria-label="توضیح PR">؟</button></span>
      <span><i>FH</i> نمرهٔ معلوم خود فیلم<button type="button" class="hq" data-h="bFH" aria-label="توضیح FH">؟</button></span>
      <span><i>K</i> محتوای TMDB<button type="button" class="hq" data-h="bK" aria-label="توضیح K">؟</button></span>
      <span><i>PUB</i> امتیاز عمومی TMDB (سقف)<button type="button" class="hq" data-h="pub" aria-label="توضیح PUB">؟</button></span>
      <span><i>N</i> نمره‌های آشکارشدهٔ کاربر<button type="button" class="hq" data-h="nrev" aria-label="توضیح N">؟</button></span>
      <span><i>M</i> نمره‌های معلوم فیلم<button type="button" class="hq" data-h="mrev" aria-label="توضیح M">؟</button></span>
    </div>
    <p class="mini" style="margin:6px 0 0">حرف A دو معنی دارد: «سناریو A» (کاربر جدید) و بلوک «A» (شباهت). در نمودارها و جدول‌ها نام کامل نوشته شده است.</p>
  </div>
</header>

<section id="s0">
  <h2>۰. کامنت‌های استاد: چه شد، چه ماند</h2>
  <p class="sub">ردیف اول پیشنهاد جدیدی است که برای جلسهٔ بعد می‌ماند. سه ردیف «برداشت ما» نیاز به تأیید استاد دارند.</p>
  <!--T_CMT-->
</section>

<section id="s1">
  <h2>۱. مفاهیم پایه <small>یادآوری کوتاه</small></h2>
  <div class="grid3">
    <div class="card"><span class="tag t-coral">A: کاربر جدید</span><h3>فیلم‌ها شناخته‌اند</h3><p>کاربر هیچ نمره‌ای نداده (N=۰) یا N نمره آشکار شده است. فیلم‌ها از قبل نمره دارند.</p></div>
    <div class="card"><span class="tag t-amber">B: فیلم جدید</span><h3>کاربران شناخته‌اند</h3><p>فیلم هیچ نمره‌ای ندارد (M=۰) یا M نمرهٔ معلوم دارد. فیلم‌های آزمون از آموزش کنار گذاشته می‌شوند.</p></div>
    <div class="card"><span class="tag t-violet">C: هر دو جدید</span><h3>سخت‌ترین حالت</h3><p>هم کاربر هم فیلم جدیدند؛ فقط پروفایل کاربر و محتوای فیلم در دست است.</p></div>
  </div>
  <div class="grid3" style="margin-top:14px">
    <div class="card"><h3>AUC هر کاربر</h3><p>از دو فیلم، یکی مورد علاقه و یکی نه: با چه احتمالی فیلم مورد علاقه را بالاتر می‌گذاریم؟ ۰٫۵ یعنی حدس تصادفی و ۱ یعنی کامل. برای هر کاربر جدا حساب می‌شود و میانگین گرفته می‌شود.</p></div>
    <div class="card"><h3>بلوک‌های اطلاعات</h3>
      <dl class="p en-dt"><dt>P<button type="button" class="hq" data-h="bP" aria-label="توضیح بلوک P">؟</button></dt><dd>محبوبیت فیلم از کاربران آموزش</dd><dt>H<button type="button" class="hq" data-h="bH" aria-label="توضیح بلوک H">؟</button></dt><dd>تاریخچهٔ کاربر</dd><dt>T<button type="button" class="hq" data-h="bT" aria-label="توضیح بلوک T">؟</button></dt><dd>سابقهٔ کارگردان و بازیگر</dd><dt>L<button type="button" class="hq" data-h="bL" aria-label="توضیح بلوک L">؟</button></dt><dd>شهرت از LLM (Sonnet 5.5)</dd><dt>A<button type="button" class="hq" data-h="bA" aria-label="توضیح بلوک A">؟</button></dt><dd>شباهت به نمره‌های کاربر</dd><dt>PR<button type="button" class="hq" data-h="bPR" aria-label="توضیح بلوک PR">؟</button></dt><dd>پروفایل فازی توییتر</dd><dt>FH<button type="button" class="hq" data-h="bFH" aria-label="توضیح بلوک FH">؟</button></dt><dd>نمره‌های معلوم خود فیلم</dd><dt>K<button type="button" class="hq" data-h="bK" aria-label="توضیح بلوک K">؟</button></dt><dd>محتوای TMDB</dd></dl></div>
    <div class="card"><h3>فازی در برابر خام و قطعی</h3><p>پروفایل را سه‌گونه به مدل می‌دهیم: خام، قطعی (هر کاربر فقط در یک دسته) و فازی (درجهٔ عضویت در چند دسته). اثر فازی فقط با این دو کنترل سنجیده می‌شود.</p>
      <label class="mini" for="fzs">تعداد دنبال‌کننده: <b class="num" id="fzv">800</b></label>
      <input type="range" id="fzs" class="range" min="0" max="4000" step="10" value="800">
      <div id="fzout" class="mini"></div></div>
  </div>
</section>

<section id="s2">
  <h2>۲. محتوای TMDB <small>مهم‌ترین تغییر نسبت به جلسهٔ قبل</small></h2>
  <p class="sub">برای هر فیلم از TMDB ژانر، سال، مدت، کلیدواژه، خلاصهٔ داستان، کشور و بودجه را گرفتیم و به‌عنوان یک بلوک (K) به مدل دادیم. امتیاز عمومی TMDB (PUB) هم بررسی شد، ولی چون خودش یک نظر عمومی است، فقط به‌عنوان «سقف» جدا گزارش می‌شود و در مدل رسمی نیست.</p>
  <!--T_TMDB-->
  <div class="call"><strong>نتیجه:</strong> برای فیلم جدید محتوای TMDB حدود ۰٫۰۵ AUC اضافه می‌کند و بزرگ‌ترین بهبود کل پروژه است. برای کاربر جدید تقریباً اثری ندارد، چون محبوبیت فیلم همه‌چیز را می‌گیرد. امتیاز عمومی TMDB حدود ۰٫۰۶ دیگر اضافه می‌کند، ولی چون خودش «نظر دیگران» است، آن را سقف می‌نامیم.</div>
</section>

<section id="s3">
  <h2>۳. ablation، فازی × LLM و حذف LLM</h2>
  <p class="sub">ablation یعنی از مدل نهایی (با TMDB) هر بلوک را یکی‌یکی برمی‌داریم و می‌بینیم AUC چقدر افت می‌کند. عدد مثبت یعنی بلوک کمک می‌کند. سطح صفر: N=۰ یا M=۰.</p>
  <!--T_ABL-->
  <div class="fig" style="margin-top:12px"><h3>نمودار ۱: ضرر حذف هر بلوک</h3><p>نوار یعنی افت AUC؛ خط باریک بازهٔ ۹۵٪ است. خانهٔ خالی یعنی بلوک در آن سناریو نیست.</p><div class="two" id="fig1"></div></div>
  <h3 style="margin:22px 0 4px">ablation در حرکت <small class="mini">خودکار؛ لازم نیست چیزی انتخاب کنید</small></h3>
  <div class="card anim" id="anAbl" aria-label="انیمیشن ablation"></div>
  <div class="call"><strong>خلاصه:</strong> در کاربر جدید فقط محبوبیت (P) مهم است. در فیلم و هر دو جدید، محتوای TMDB (K) مهم‌ترین است. در سناریوی B با ۵ نمرهٔ معلوم از خود فیلم (FH)، آن بلوک با حدود ۰٫۰۴۸ دومین بلوک مهم است. پروفایل فازی در C حتی کمی ضرر دارد.</div>

  <h3 style="margin:24px 0 4px">فازی × LLM: چهار حالت <span class="en">(FUZZY LLM 4 STEP)</span></h3>
  <p class="sub">سناریوی C، چهار ترکیب. «بدون فازی» یعنی پروفایل خام استاندارد. همهٔ ترکیب‌ها یک سابقهٔ سازندگان دارند.</p>
  <!--T_FXM-->
  <div class="call"><strong>خلاصه:</strong> LLM در هر دو حالت و هر دو سطح حدود ۰٫۰۰۷ تا ۰٫۰۰۹ AUC اضافه می‌کند. اثر فازی بین ۰٫۰۰۰۳− و ۰٫۰۰۰۴+ نوسان می‌کند و علامتش عوض می‌شود؛ از چهار حالت، تفاوت اصلی از ستون LLM می‌آید.</div>

  <h3 style="margin:24px 0 4px">حذف LLM <span class="en">(DELETE LLM)</span></h3>
  <!--T_DEL-->
  <div class="call"><strong>خلاصه:</strong> حذف LLM از مدل نهایی در A و B اثری ندارد و در C حدود ۰٫۰۰۲ تا ۰٫۰۰۳ AUC می‌گیرد. بدون TMDB اثر LLM بیشتر است (در C تا ۰٫۰۰۹). یعنی وقتی TMDB هست، LLM تا حد زیادی زائد است.</div>
  <div class="ask"><b>نیاز به تأیید استاد:</b> «چهار حالت» را فازی × LLM و «DELETE LLM» را حذف LLM در ablation فهمیده‌ایم. اگر منظور چیز دیگری بود (مثلاً کاربر × فیلم، یا حذف LLM از کل پایان‌نامه)، بفرمایید.</div>
</section>

<section id="s4">
  <h2>۴. اگر نمره کم باشد <small>Sparsity، کاربر بدون نمره، فیلم بدون نمره</small></h2>
  <p class="sub">دو کامنت یک سؤال‌اند و در یک بخش آمده‌اند. نمودار ۲: از نمره‌های آموزشی ۱۰۰٪ تا ۵٪ را نگه داشتیم؛ کاربران و ردیف‌های آزمون دست‌نخورده‌اند. نمودار ۳: کاربر یا فیلم از «هیچ نمره» شروع می‌کند و نمره‌های معلوم کم‌کم زیاد می‌شوند.</p>
  <div class="fig"><h3>نمودار ۲: AUC در برابر درصد نمره‌های آموزشی (N=۰ یا M=۰)</h3><div class="two" id="fig2"></div></div>
  <div style="height:12px"></div>
  <!--T_SPE-->
  <div class="fig" style="margin-top:12px"><h3>نمودار ۳: AUC در برابر تعداد نمره‌های معلوم</h3><p>A و C: N نمرهٔ کاربر (تا ۴۰۰). B: M نمرهٔ خود فیلم (۰ و ۵). خط‌چین خط پایه است.</p><div class="two" id="fig3"></div></div>
  <div class="call"><strong>خلاصه:</strong> با کم شدن نمره‌ها، اجزایی که از نمره می‌آیند (محبوبیت، سابقهٔ سازندگان) بدتر می‌شوند، و محتوای TMDB ارزش بیشتری پیدا می‌کند (در B و C از حدود ۰٫۰۵ به ۰٫۰۶). اثر LLM ثابت می‌ماند و در تنکی بیشتر نمی‌شود. برای کاربر بدون نمره، حتی ۴۰۰ نمره فقط حدود ۰٫۰۱۴ اضافه می‌کند؛ برای فیلم، ۵ نمرهٔ معلوم خیلی کمک می‌کند (۰٫۷۳ به ۰٫۷۸).</div>
  <p class="mini">فیلم با ۲۰ و ۵۰ نمرهٔ معلوم (M=۲۰ و ۵۰) اجرا نشده است.</p>
</section>

<section id="s5">
  <h2>۵. محک با CSP 2023 <small>Benchmark</small></h2>
  <p class="sub">دو روش دو مقاله را از روی متن بازسازی و در پروتکل خودمان اجرا کردیم (همان کاربران جداشده و همان ردیف‌های آزمون): مقالهٔ دیتاست CSP 2023 (محتوا + کاربران هم‌شبیه، پیشنهاد ۵ فیلم برتر) و مقالهٔ ۲۰۲۰ (مدل جدا برای هر فیلم با جنگل تصادفی).</p>
  <!--T_BEN-->
  <div class="ask" style="margin-top:10px"><b>جمع‌بندی صادقانه:</b> نمی‌توان گفت مدل ما در پیشنهاد به کاربر جدید برتر است. در AUC کمی بهتر است (۰٫۰۰۱۶+ نسبت به «فقط محبوبیت» و ۰٫۰۰۳۳+ نسبت به روش کامل CSP 2023؛ هر دو معنادار ولی بسیار کوچک). در معیارهای بالای لیست کمی بدتر است: میانگین نمرهٔ ۵ فیلم برتر ۸٫۶۶ در برابر ۸٫۷۸ و بهبود نسبت به میانگین کاربر ۲۶٫۳٪ در برابر ۲۸٫۰٪. دلیلش این است که AUC ترتیب همهٔ فیلم‌های کاربر را می‌سنجد، ولی دو ستون آخر فقط بالای لیست را؛ «فقط محبوبیت» دقیقاً محبوب‌ترین فیلم‌ها را بالا می‌گذارد. <b>برتری واقعی مدل ما در فیلم جدید است</b> (سناریوهای B و C)، جایی که روش‌های این دو مقاله به نمرهٔ خودِ فیلم نیاز دارند و برای فیلم بدون نمره کار نمی‌کنند.</div>
  <div class="fig" style="margin-top:12px"><h3>نمودار ۴: AUC روش‌ها</h3><div id="fig4"></div></div>
  <div class="call"><strong>نتیجه:</strong> روش مقالهٔ CSP 2023 از «فقط محبوبیت فیلم» حدود ۰٫۰۰۱۸ AUC کمتر است. «بهبود ۱۶٫۵٪ نسبت به میانگین کاربر» که مقاله گزارش می‌کند، در دادهٔ ما با خود محبوبیت فیلم هم می‌آید (حدود ۲۸٪). مدل جدا برای هر فیلم (مقالهٔ ۲۰۲۰) روی ۱۵ فیلم پرنمره از حدس «همه پسندیده‌اند» بهتر نیست: دقت ۰٫۸۵۹ در برابر ۰٫۸۶۵ با قاعدهٔ «نمره ≥ ۷» خود مقاله.</div>
  <div class="note"><b>قیدها:</b> بازسازی ما ساده‌شده است (برش دودویی ویژگی‌ها روی میانهٔ کاربران آموزش، بدون مرحلهٔ تصویر کردن با ویژگی‌های فیلم). «دقت ۶۰٪» و MRR مقاله تعریف روشنی ندارند و عیناً بازتولید نشدند. در دادهٔ ما هیچ فیلمی بیش از ۴۸۰ نمره ندارد، پس ۱۵ فیلم پرنمره جایگزین شد. مدل ما در این جدول از محبوبیت فقط حدود ۰٫۰۰۱۶ AUC بهتر است.</div>
</section>

<section id="s6">
  <h2>۶. F1 و دقت</h2>
  <p class="sub">مدل نهایی در برابر دو حدس ساده، سطح صفر. دقت و F1 از همهٔ ردیف‌های آزمون با هم حساب شده‌اند و آستانهٔ تصمیم فقط از کاربران آموزش یاد گرفته شده است.</p>
  <!--T_CLS-->
  <div class="fig" style="margin-top:12px"><h3>نمودار ۵: دقت و F1 در برابر دو حدس ساده</h3><p>مدل نهایی (سطح صفر) در کنار «همه نپسندیده‌اند» و «همه پسندیده‌اند». دقت و F1 از همهٔ ردیف‌های آزمون با هم حساب شده‌اند.</p><div class="two" id="fig5"></div></div>
  <div class="note"><b>هشدار:</b> آستانه برای بیشینه‌کردن F1 انتخاب می‌شود، پس دقت بعضی مدل‌ها از «همه نپسندیده‌اند» کمتر است (در سناریوی C دقت مدل ۰٫۶۰۲ و حدس ساده ۰٫۶۴۲ است). F1 در هر سه سناریو از هر دو حدس ساده بالاتر است.</div>
  <h3 style="margin:22px 0 4px">اثر هر بلوک روی دقت و F1</h3>
  <p class="sub">افتِ دقت و F1 کلاس «پسندیده» وقتی بلوک از مدل نهایی حذف می‌شود (سطح صفر). عدد مثبت یعنی بلوک کمک می‌کند.</p>
  <!--T_CLSB-->
  <div class="call"><strong>خلاصه:</strong> در A محبوبیت (P) مهم‌ترین است، در C محتوای TMDB، و در B تاریخچه و شباهت کاربر و بعد TMDB. LLM روی دقت و F1 تقریباً هیچ اثری ندارد؛ پروفایل فازی کمی ضرر دارد (دقت در A حدود ۰٫۰۰۶ و در C حدود ۰٫۰۱۲ کمتر). تاریخچهٔ کاربر در B برای دقت و F1 کلی مهم است ولی در AUC هر کاربر اثری ندارد، چون به مدل می‌گوید «این کاربر اغلب فیلم‌ها را دوست دارد» و این ترتیب فیلم‌های هر کاربر را عوض نمی‌کند.</div>
</section>

<section id="s7">
  <h2>۷. پیشنهاد ۱۰ فیلم برای یک کاربر <small>Recommendation: get user</small></h2>
  <p class="sub">برای هر کاربر آزمون، فیلم‌هایی که امتیاز داده را با نمرهٔ مدل مرتب می‌کنیم و ۱۰ فیلم اول را پیشنهاد می‌دهیم. چون هر فیلم نمرهٔ واقعی دارد، لیست قابل سنجش است. NDCG@10 هر چه به ۱ نزدیک‌تر باشد، ترتیب لیست به بهترین ترتیب ممکن نزدیک‌تر است.</p>
  <!--T_TOP-->
  <div class="call"><strong>یافتهٔ صادقانه:</strong> برای کاربر جدید، لیست مدل نهایی (NDCG ۰٫۸۸۳) از «فقط محبوب‌ترین فیلم‌ها» (۰٫۸۸۸) بهتر نیست؛ با ۲۰ نمرهٔ آشکار نیز تفاوت روشنی ندارد. هر دو از لیست تصادفی (۰٫۷۰۰) خیلی بهترند. در سناریوی کاربر جدید حذف LLM هیچ تغییری در لیست نمی‌دهد.</div>

  <h3 style="margin:24px 0 4px">Top-10 در حرکت <small class="mini">خودکار؛ از ۱۰ کاربر آزمون، یکی‌یکی، به ترتیب شماره</small></h3>
  <div class="card anim" id="anTop" aria-label="انیمیشن ساخت لیست Top-10"></div>

  <h3 style="margin:24px 0 4px">شبیه‌ساز تعاملی <small class="mini">از نتایج واقعی یک فولد؛ ۱۰ کاربر آزمون و ۸ فیلم جدید با بذر ثابت انتخاب شده‌اند</small></h3>
  <div class="card simcard">
    <div class="stabs" role="tablist" id="stabs">
      <button role="tab" aria-selected="true" data-t="t1">Top-10 برای یک کاربر</button>
      <button role="tab" aria-selected="false" data-t="t2">تأثیر هر بلوک در لیست</button>
      <button role="tab" aria-selected="false" data-t="t3">کارکرد گام‌به‌گام یک فیلم</button>
      <button role="tab" aria-selected="false" data-t="t4">فیلم جدید: چه کسانی دوستش دارند؟</button>
    </div>
    <div id="t1"></div><div id="t2" hidden></div><div id="t3" hidden></div><div id="t4" hidden></div>
  </div>
</section>

<section id="s8">
  <h2>۸. الگوریتم‌ها و دلیل انتخابشان</h2>
  <p class="sub">AUC مدل کامل در سطح صفر. «با TMDB» دو ستون آخر از اجرای جداگانه‌ای است که شبکهٔ عصبی را هم دارد.</p>
  <!--T_LRN-->
  <ul class="lim">
    <li><b>لجستیک:</b> خط پایهٔ ساده و قابل‌تفسیر.</li>
    <li><b>hgb_c:</b> تقویت گرادیانی منظم (۳۰۰ درخت کوچک، سرعت یادگیری ۰٫۰۳)؛ برای دادهٔ جدولی و تعامل ویژگی‌ها؛ پیش از اجرای نهایی به‌عنوان یادگیرندهٔ اصلی اعلام شد.</li>
    <li><b>جنگل تصادفی:</b> الگوریتم مقاله‌ها؛ برای مقایسهٔ منصفانه با آن‌ها.</li>
  </ul>
  <div class="call"><strong>نتیجه:</strong> سه یادگیرنده بسیار نزدیک‌اند (در سطح صفر حدود ۰٫۰۰۴، و در سطح‌های دیگر حداکثر حدود ۰٫۰۰۶ AUC)، پس نتایج به انتخاب الگوریتم وابسته نیست. شبکهٔ عصبی کوچک در A تقریباً هم‌سطح است (حدود ۰٫۰۰۴ کمتر) و در B و C حدود ۰٫۰۳ از hgb_c بدتر است. فیلترینگ همکارانه و فاکتورگیری ماتریسی برای کاربر یا فیلم بدون نمره کار نمی‌کنند، چون به نمرهٔ قبلی نیاز دارند.</div>
  <div class="note"><b>مانده:</b> منبع علمی ادعاهای کلی دربارهٔ الگوریتم‌ها (مثل برتری تقویت گرادیانی روی دادهٔ جدولی) هنوز پیدا نشده و بدون منبع در پایان‌نامه نمی‌آید.</div>
  <p class="mini" style="margin-top:10px">کنار گذاشته‌شده، با اثر ناچیز: سلیقهٔ ژانری کاربر و شباهت فیلم‌ها (خام، قطعی، فازی) حداکثر حدود ۰٫۰۰۵ AUC اضافه می‌کنند و فازی با خام تفاوت روشنی ندارد؛ بردار متنی خلاصهٔ داستان اثری ندارد یا کمی منفی است.</p>
</section>

<section id="s9">
  <h2>۹. نوآوری LLM و پیشنهاد جلسهٔ بعد</h2>
  <div class="card best">
    <h3>آنچه با داده‌ها می‌توان ادعا کرد</h3>
    <p>ستون شهرت سازندگان از LLM، وقتی هیچ فراداده‌ای (مثل TMDB) در دست نیست، AUC مدل ساده را برای فیلم و هر دو جدید حدود ۰٫۰۰۷ تا ۰٫۰۱۰ بالا می‌برد (اعتبارسنجی با چهار ارزیاب انسانی و تکرارپذیری ۹۹٫۲٪ در بله/نه). وقتی TMDB هست، سهم اضافهٔ آن به صفر تا حدود ۰٫۰۰۳ می‌رسد. پس LLM عامل اصلی بهبود مدل نهایی نیست.</p>
  </div>
  <div class="ask" style="margin-top:12px"><b>پیشنهاد برای جلسهٔ بعد:</b> تا اینجا از LLM فقط «شهرت کارگردان و بازیگران» را پرسیده‌ایم. آزمون بعدی این است که از Sonnet برای هر فیلم دو نمره بگیریم: «چقدر شناخته‌شده است» و «چقدر مردم و منتقدان دوستش دارند»، فقط با عنوان، سال و کارگردان (بدون هیچ داده‌ای از کاربران یا نمره‌های CSP). حدود ۳ دلار هزینه دارد. اگر چیزی بیشتر از TMDB بدهد، ادعای نوآوری LLM تأیید می‌شود؛ اگر نه، نتیجهٔ منفی صادقانه گزارش می‌شود. این نمره دانش عمومی دربارهٔ فیلم است و باید جدا از «بدون اطلاع بیرونی» معرفی شود.</div>
  <h3 style="margin:18px 0 6px">کارهای مانده برای جلسهٔ بعد</h3>
  <ul class="lim">
    <li>آزمون امتیاز مستقیم هر فیلم با Sonnet (بالا).</li>
    <li>منبع علمی برای توجیه الگوریتم‌ها.</li>
    <li>تأیید استاد دربارهٔ برداشت‌ها: «چهار حالت»، «DELETE LLM» و اینکه امتیاز عمومی TMDB فقط سقف بماند.</li>
    <li>اختیاری: فیلم با ۲۰ و ۵۰ نمرهٔ معلوم، و قاعدهٔ «نمره ≥ ۷» (ge7) در پیوست.</li>
  </ul>
</section>

<footer>
  <b>محدودیت‌ها:</b>
  <ul>
    <li>همه‌جا قاعدهٔ «نمرهٔ بیشتر از ۷» و یادگیرندهٔ hgb_c؛ هر عدد میانگین ۱۵ اجرا (۵ بخش × ۳ تکرار) است.</li>
    <li>شبیه‌ساز از یک فولد ساخته شده و کاربران و فیلم‌هایش با بذر ثابت انتخاب شده‌اند، نه با نگاه به نتیجه.</li>
    <li>تفاوت‌های کمتر از حدود ۰٫۰۰۱ AUC در حد نوسان‌اند و نباید تفسیر شوند. تعداد مقایسه‌ها زیاد است و بازه‌ها برای چندمقایسه‌ای تعدیل نشده‌اند.</li>
  </ul>
</footer>
</div>
"""

JS = r"""
const D = __DATA__;
const S = __SIM__;
const fa = (x, nd = 3) => Number(x).toLocaleString('fa-IR', {minimumFractionDigits: nd, maximumFractionDigits: nd});
const sgn = (x, nd = 4) => (x >= 0 ? '+' : '−') + fa(Math.abs(x), nd);
const NS = 'http://www.w3.org/2000/svg';
const el = (t, a = {}, txt) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); if (txt !== undefined) e.textContent = txt; return e; };
const $ = id => document.getElementById(id);

/* ---------- fuzzy slider ---------- */
(function () {
  const q = [57, 347, 1965];
  const lo = v => v <= q[0] ? 1 : v >= q[1] ? 0 : (q[1] - v) / (q[1] - q[0]);
  const hi = v => v <= q[1] ? 0 : v >= q[2] ? 1 : (v - q[1]) / (q[2] - q[1]);
  const md = v => v <= q[0] || v >= q[2] ? 0 : v <= q[1] ? (v - q[0]) / (q[1] - q[0]) : (q[2] - v) / (q[2] - q[1]);
  const s = $('fzs'), out = $('fzout'), vv = $('fzv');
  function draw() {
    const v = +s.value; vv.textContent = v.toLocaleString('fa-IR');
    const m = [['کم', lo(v)], ['متوسط', md(v)], ['زیاد', hi(v)]];
    const crisp = m.reduce((a, b) => b[1] > a[1] ? b : a)[0];
    out.innerHTML = 'قطعی: فقط «' + crisp + '» (۱۰۰٪). فازی: ' + m.map(x => x[0] + ' ' + Math.round(x[1] * 100).toLocaleString('fa-IR') + '٪').join('، ') + '<br>قله‌ها روی صدک‌های ۱۰ و ۵۰ و ۹۰ کاربران آموزش (۵۷، ۳۴۷، ۱٬۹۶۵).';
  }
  s.addEventListener('input', draw); draw();
})();

/* ---------- charts ---------- */
function legend(items) { return '<div class="leg">' + items.map(i => '<span class="' + i[1] + '"><i></i>' + i[0] + '</span>').join('') + '</div>'; }
function nice(lo, hi) { const span = hi - lo || 0.01; const step = Math.pow(10, Math.floor(Math.log10(span / 4))); const m = [1, 2, 5, 10].find(k => span / (k * step) <= 5) * step; return [Math.floor(lo / m) * m, Math.ceil(hi / m) * m, m]; }
function lineChart(host, o) {
  const W = 330, H = 220, L = 48, R = 10, T = 26, B = 40;
  const all = o.series.flatMap(s => s.ys).concat(o.base ? [o.base] : []).filter(v => v != null);
  let [y0, y1, st] = nice(Math.min(...all), Math.max(...all));
  const svg = el('svg', {viewBox: '0 0 ' + W + ' ' + H, class: 'sv', role: 'img', 'aria-label': o.title});
  svg.appendChild(el('text', {x: W / 2, y: 14, 'text-anchor': 'middle', class: 'ttl'}, o.title));
  const X = i => L + (o.xs.length === 1 ? 0 : i * (W - L - R) / (o.xs.length - 1)), Y = v => T + (y1 - v) * (H - T - B) / (y1 - y0);
  for (let v = y0; v <= y1 + 1e-9; v += st) {
    svg.appendChild(el('line', {x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: 'grid'}));
    svg.appendChild(el('text', {x: L - 5, y: Y(v) + 3.5, 'text-anchor': 'end'}, fa(v, 2)));
  }
  svg.appendChild(el('line', {x1: L, x2: W - R, y1: H - B, y2: H - B, class: 'ax'}));
  o.xs.forEach((x, i) => svg.appendChild(el('text', {x: X(i), y: H - B + 14, 'text-anchor': 'middle'}, x)));
  svg.appendChild(el('text', {x: (L + W - R) / 2, y: H - 6, 'text-anchor': 'middle'}, o.xlabel));
  if (o.base) svg.appendChild(el('line', {x1: L, x2: W - R, y1: Y(o.base), y2: Y(o.base), class: 'base'}));
  o.series.forEach(s => {
    const g = el('g', {class: s.c});
    g.appendChild(el('polyline', {points: s.ys.map((v, i) => v == null ? null : X(i) + ',' + Y(v)).filter(Boolean).join(' '), class: 'l'}));
    s.ys.forEach((v, i) => { if (v != null) g.appendChild(el('circle', {cx: X(i), cy: Y(v), r: 3, class: 'p'})); });
    svg.appendChild(g);
  });
  const d = document.createElement('div'); d.appendChild(svg);
  d.insertAdjacentHTML('beforeend', legend(o.series.map(s => [s.name, s.c]).concat(o.base ? [[o.baseName, 'c5']] : [])));
  host.appendChild(d);
}
function barChart(host, o) {
  const W = 330, rowH = 26, T = 24, L = 118, R = 14, H = T + o.rows.length * rowH + 22;
  const vals = o.rows.flatMap(r => [r.lo, r.hi, r.v, o.zero]).filter(v => v != null);
  let [x0, x1] = [Math.min(...vals), Math.max(...vals)];
  const pad = (x1 - x0) * 0.08; x0 -= pad; x1 += pad;
  const svg = el('svg', {viewBox: '0 0 ' + W + ' ' + H, class: 'sv', role: 'img', 'aria-label': o.title});
  const X = v => L + (v - x0) * (W - L - R) / (x1 - x0);
  svg.appendChild(el('text', {x: W / 2, y: 14, 'text-anchor': 'middle', class: 'ttl'}, o.title));
  o.rows.forEach((r, i) => {
    const y = T + i * rowH, c = r.c || 'c1';
    svg.appendChild(el('text', {x: L - 6, y: y + rowH / 2 + 3.5, 'text-anchor': 'end'}, r.label));
    if (r.v == null) { svg.appendChild(el('text', {x: L + 6, y: y + rowH / 2 + 3.5}, 'نیست')); return; }
    const g = el('g', {class: c});
    g.appendChild(el('rect', {x: Math.min(X(o.zero), X(r.v)), y: y + 5, width: Math.max(1, Math.abs(X(r.v) - X(o.zero))), height: rowH - 10, class: 'bar', rx: 2}));
    if (r.lo != null) { const cy = y + rowH / 2; g.appendChild(el('line', {x1: X(r.lo), x2: X(r.hi), y1: cy, y2: cy, class: 'wh'})); }
    svg.appendChild(g);
    svg.appendChild(el('text', {x: Math.max(X(r.v), X(o.zero)) + 4, y: y + rowH / 2 + 3.5}, o.fmt(r.v)));
  });
  svg.appendChild(el('line', {x1: X(o.zero), x2: X(o.zero), y1: T - 2, y2: H - 20, class: 'ax'}));
  const ticks = o.ticks || [x0 + pad, x1 - pad];
  ticks.forEach(t => svg.appendChild(el('text', {x: X(t), y: H - 6, 'text-anchor': 'middle'}, o.fmt(t))));
  host.appendChild(svg);
}
(function () {
  /* figure 1: ablation */
  const host = $('fig1'), names = {P: 'محبوبیت (P)', T: 'سابقهٔ سازندگان (T)', L: 'شهرت LLM (L)', PR: 'پروفایل فازی (PR)', K: 'محتوای TMDB (K)'};
  [['A', 'A: کاربر جدید', ['P', 'L', 'PR', 'K']], ['B', 'B: فیلم جدید', ['T', 'L', 'K']], ['C', 'C: هر دو جدید', ['T', 'L', 'PR', 'K']]].forEach(([s, t, bl]) => {
    const d = document.createElement('div'); host.appendChild(d);
    barChart(d, {title: t, zero: 0, fmt: v => sgn(v, 3),
      rows: bl.map(b => { const r = D.abl[s][b]; return {label: names[b], v: r ? r[0] : null, lo: r ? r[1] : null, hi: r ? r[2] : null, c: r && r[0] < 0 ? 'c4' : (b === 'K' ? 'c1' : b === 'L' ? 'c3' : 'c2')}; })});
  });
  /* figure 2: sparsity */
  const h2 = $('fig2'), xs = D.sparse.keep.map(k => k.toLocaleString('fa-IR') + '٪');
  [['A', 'A: کاربر جدید', [['P', 'محبوبیت (P)', 'c2'], ['P+K', 'محبوبیت + TMDB', 'c1']]],
   ['B', 'B: فیلم جدید', [['T', 'سابقهٔ سازندگان (T)', 'c2'], ['T+L', 'سابقه + LLM', 'c3'], ['FULL', 'مدل بدون TMDB', 'c4'], ['FULLK', 'مدل نهایی', 'c1']]],
   ['C', 'C: هر دو جدید', [['T', 'سابقهٔ سازندگان (T)', 'c2'], ['T+L', 'سابقه + LLM', 'c3'], ['FULL', 'مدل بدون TMDB', 'c4'], ['FULLK', 'مدل نهایی', 'c1']]]].forEach(([s, t, ser]) => {
    const d = document.createElement('div'); h2.appendChild(d);
    lineChart(d, {title: t, xs, xlabel: 'درصد نمره‌های آموزشی', series: ser.map(([m, n, c]) => ({name: n, c, ys: D.sparse.auc[s][m]}))});
  });
  /* figure 3: known ratings */
  const h3 = $('fig3'), c = D.curve, nx = c.N.map(n => n.toLocaleString('fa-IR'));
  let d = document.createElement('div'); h3.appendChild(d);
  lineChart(d, {title: 'A: کاربر جدید', xs: nx, xlabel: 'N = نمره‌های آشکارشدهٔ کاربر', series: [{name: 'مدل (با TMDB)', c: 'c1', ys: c.A_BK}], base: c.A_pop, baseName: 'فقط محبوبیت'});
  d = document.createElement('div'); h3.appendChild(d);
  lineChart(d, {title: 'B: فیلم جدید', xs: ['۰', '۵'], xlabel: 'M = نمره‌های معلوم فیلم', series: [{name: 'سابقهٔ سازندگان (T)', c: 'c2', ys: c.B_T}, {name: 'مدل بدون TMDB', c: 'c4', ys: c.B_FULL}, {name: 'مدل نهایی', c: 'c1', ys: c.B_FULLK}]});
  d = document.createElement('div'); h3.appendChild(d);
  lineChart(d, {title: 'C: هر دو جدید', xs: nx, xlabel: 'N = نمره‌های آشکارشدهٔ کاربر', series: [{name: 'مدل (با TMDB)', c: 'c1', ys: c.C_BK}], base: c.C_base, baseName: 'سابقهٔ سازندگان'});
  /* figure 5: accuracy and F1 against the two trivial guesses */
  (function () {
    const host = $('fig5'), cl = D.cls;
    const series = [['مدل نهایی', 'c1', 'model'], ['همه نپسندیده‌اند', 'c5', 'no'], ['همه پسندیده‌اند', 'c3', 'yes']];
    [['acc', 'دقت (accuracy)'], ['f1', 'F1 کلاس «پسندیده»']].forEach(([key, title]) => {
      const W = 330, H = 230, L = 36, R = 10, T = 26, B = 40, gw = (W - L - R) / 3, bw = gw / 4.2;
      const svg = el('svg', {viewBox: '0 0 ' + W + ' ' + H, class: 'sv', role: 'img', 'aria-label': title});
      svg.appendChild(el('text', {x: W / 2, y: 14, 'text-anchor': 'middle', class: 'ttl'}, title));
      const Y = v => T + (1 - v) * (H - T - B);
      [0, 0.25, 0.5, 0.75, 1].forEach(v => { svg.appendChild(el('line', {x1: L, x2: W - R, y1: Y(v), y2: Y(v), class: 'grid'})); svg.appendChild(el('text', {x: L - 5, y: Y(v) + 3.5, 'text-anchor': 'end'}, fa(v, 2))); });
      ['A', 'B', 'C'].forEach((s, gi) => {
        series.forEach(([n, c, k], si) => {
          const v = cl[s][k][key], x = L + gi * gw + gw * 0.12 + si * (bw + 3);
          const g = el('g', {class: c}); g.appendChild(el('rect', {x, y: Y(v), width: bw, height: Math.max(1, Y(0) - Y(v)), class: 'bar', rx: 2})); svg.appendChild(g);
          svg.appendChild(el('text', {x: x + bw / 2, y: Y(v) - 3, 'text-anchor': 'middle', style: 'font-size:9.5px'}, fa(v, 2)));
        });
        svg.appendChild(el('text', {x: L + gi * gw + gw / 2, y: H - B + 15, 'text-anchor': 'middle'}, s));
      });
      svg.appendChild(el('line', {x1: L, x2: W - R, y1: Y(0), y2: Y(0), class: 'ax'}));
      const d = document.createElement('div'); d.appendChild(svg); d.insertAdjacentHTML('beforeend', legend(series.map(x => [x[0], x[1]]))); host.appendChild(d);
    });
  })();
  /* figure 4: benchmark */
  const b = D.bench, rows = [['POP_mean', 'فقط محبوبیت', 'c2'], ['CSP23_content', 'CSP23 محتوا', 'c3'], ['CSP23_total', 'CSP23 کامل', 'c4'], ['HGB_FULL', 'مدل ما', 'c1']];
  barChart($('fig4'), {title: 'AUC کاربر جدید (محور از ۰٫۷۹)', zero: 0.79, ticks: [0.79, 0.80, 0.81, 0.82], fmt: v => fa(v, 3),
    rows: rows.map(([k, n, cc]) => ({label: n, v: b[k].auc[0], lo: b[k].auc[1], hi: b[k].auc[2], c: cc}))});
})();

/* ---------- simulator ---------- */
const ML = {RANDOM: 'تصادفی', POP: 'فقط محبوبیت فیلم', P: 'محبوبیت (مدل P)', 'P+H': '+ تاریخچهٔ کاربر', 'P+H+A': '+ شباهت', 'P+H+A+L': '+ LLM', '+PR': '+ پروفایل فازی',
  FULLK: 'مدل نهایی (با LLM و TMDB)', FULLK_noL: 'مدل نهایی بدون LLM', CSP23: 'روش مقالهٔ CSP 2023'};
const MB = {RANDOM: 'تصادفی', T: 'فقط سابقهٔ سازندگان', 'T+L': '+ LLM', K: 'فقط محتوای TMDB', 'T+K': 'سابقه + TMDB', FULLK: 'مدل نهایی', FULLK_noL: 'مدل نهایی بدون LLM'};
const nfa = n => Number(n).toLocaleString('fa-IR');
const fm = i => S.films[i];
const top10 = (u, N, m) => u.rows.map((r, i) => [r, i]).sort((a, b) => (b[0].s[N][m] - a[0].s[N][m]) || (a[1] - b[1])).slice(0, 10).map(x => x[0]);
const filmLine = r => { const f = fm(r.f); return f.t + ' <span class="why">' + (f.d || '') + (f.y ? ' · ' + f.y.toString().replace(/\d/g, d => '۰۱۲۳۴۵۶۷۸۹'[d]) : '') + (f.g.length ? ' · ' + f.g.join('، ') : '') + '</span>'; };
const hitMark = r => r.r > 7 ? '<span class="hit y" title="کاربر پسندیده">✔</span>' : '<span class="hit n" title="کاربر نپسندیده">✘</span>';
function listHtml(rows, mark, prev) {
  return '<ol class="top">' + rows.map(r => {
    const isNew = prev && !prev.some(p => p.f === r.f);
    return '<li' + (isNew ? ' class="new"' : '') + '><span class="t1">' + filmLine(r) + '</span><span class="num mini">' + fa(r.cur, 2) + '</span><span class="num pp">' + nfa(r.r) + '</span>' + hitMark(r) + '</li>';
  }).join('') + '</ol><p class="mini">ستون‌ها: فیلم، نمرهٔ مدل، نمرهٔ واقعی کاربر (از ۱۰)، پسندید (نمره بیشتر از ۷).</p>';
}
const tabs = $('stabs');
tabs.addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; tabs.querySelectorAll('button').forEach(x => x.setAttribute('aria-selected', x === b)); ['t1', 't2', 't3', 't4'].forEach(t => $(t).hidden = t !== b.dataset.t); });
const userOpts = S.A.users.map(u => '<option value="' + (u.id - 1) + '">کاربر آزمون ' + nfa(u.id) + ' (' + nfa(u.n) + ' فیلم نمره‌دار)</option>').join('');
const NOPT = '<option value="0">N = ۰ (هیچ نمره‌ای از کاربر)</option><option value="20">N = ۲۰ نمرهٔ آشکار</option>';

/* tab 1 */
(function () {
  const h = $('t1');
  h.innerHTML = '<div class="ctrl"><label>کاربر<select id="u1">' + userOpts + '</select></label><label>تعداد نمرهٔ آشکار<select id="n1">' + NOPT + '</select></label><label>مدل<select id="m1">' +
    ['FULLK', 'FULLK_noL', 'POP', 'CSP23', 'RANDOM'].map(m => '<option value="' + m + '">' + ML[m] + '</option>').join('') + '</select></label></div><div id="o1"></div>';
  function draw() {
    const u = S.A.users[+$('u1').value], N = $('n1').value, m = $('m1').value;
    const rows = top10(u, N, m).map(r => Object.assign({}, r, {cur: r.s[N][m]}));
    const hits = rows.filter(r => r.r > 7).length, all = u.rows.filter(r => r.r > 7).length / u.rows.length;
    $('o1').innerHTML = '<div class="kpis"><div class="kpi"><b>' + nfa(hits) + ' از ۱۰</b><span>فیلم پسندیده‌شده در لیست</span></div><div class="kpi"><b class="num">' + fa(u.ndcg[N][m]) + '</b><span>NDCG@10 این مدل</span></div><div class="kpi"><b class="num">' + fa(u.ndcg[N].POP) + '</b><span>NDCG@10 فقط محبوبیت</span></div><div class="kpi"><b class="num">' + fa(u.ndcg[N].RANDOM) + '</b><span>NDCG@10 تصادفی</span></div><div class="kpi"><b>' + nfa(Math.round(all * 100)) + '٪</b><span>سهم فیلم‌های پسندیدهٔ این کاربر</span></div></div>' + listHtml(rows);
  }
  ['u1', 'n1', 'm1'].forEach(i => $(i).addEventListener('change', draw)); draw();
})();

/* tab 2 */
(function () {
  const steps = ['POP', 'P', 'P+H', 'P+H+A', 'P+H+A+L', '+PR', 'FULLK'], names = ['فقط محبوبیت', 'مدل P', '+ تاریخچه', '+ شباهت', '+ LLM', '+ پروفایل فازی', '+ TMDB'];
  const h = $('t2');
  h.innerHTML = '<p class="mini">بلوک‌ها را یکی‌یکی روشن کنید و ببینید لیست چقدر عوض می‌شود. کادر بنفش یعنی این فیلم در مرحلهٔ قبل در لیست نبود.</p><div class="ctrl"><label>کاربر<select id="u2">' + userOpts + '</select></label><label>تعداد نمرهٔ آشکار<select id="n2">' + NOPT +
    '</select></label></div><label class="mini" for="s2">مرحله: <b id="s2l"></b></label><input type="range" id="s2" class="range" min="0" max="6" step="1" value="0"><div class="stepline" id="s2s"></div><label class="chk" id="noLw" hidden><input type="checkbox" id="noL"> حذف LLM در مرحلهٔ آخر</label><div id="o2"></div>';
  function modelAt(i) { return i === 6 && $('noL').checked ? 'FULLK_noL' : steps[i]; }
  function draw() {
    const u = S.A.users[+$('u2').value], N = $('n2').value, i = +$('s2').value;
    $('noLw').hidden = i !== 6;
    $('s2s').innerHTML = names.map((n, k) => '<span class="' + (k <= i ? 'on' : '') + '">' + n + '</span>').join(' ← ');
    $('s2l').textContent = names[i];
    const cur = top10(u, N, modelAt(i)), prev = i > 0 ? top10(u, N, modelAt(i - 1)) : null, pop = top10(u, N, 'POP');
    const rows = cur.map(r => Object.assign({}, r, {cur: r.s[N][modelAt(i)]}));
    const same = cur.filter(r => pop.includes(r)).length, chg = prev ? cur.filter(r => !prev.includes(r)).length : 0;
    $('o2').innerHTML = '<div class="kpis"><div class="kpi"><b>' + nfa(same) + ' از ۱۰</b><span>مشترک با لیست «فقط محبوبیت»</span></div><div class="kpi"><b>' + nfa(chg) + '</b><span>فیلم جدید نسبت به مرحلهٔ قبل</span></div><div class="kpi"><b class="num">' + fa(u.ndcg[N][modelAt(i)]) + '</b><span>NDCG@10</span></div><div class="kpi"><b>' + nfa(cur.filter(r => r.r > 7).length) + ' از ۱۰</b><span>پسندیده‌شده</span></div></div>' + listHtml(rows, null, prev);
  }
  ['u2', 'n2', 's2', 'noL'].forEach(i => $(i).addEventListener('input', draw)); draw();
})();

/* tab 3 */
(function () {
  const h = $('t3');
  h.innerHTML = '<p class="mini">یک فیلم را از ورودی‌هایش تا جایگاه نهایی در لیست دنبال کنید (N = ۰، مدل نهایی). اعداد همه واقعی‌اند.</p><div class="ctrl"><label>کاربر<select id="u3">' + userOpts + '</select></label><label>فیلمی که دنبال می‌کنیم<select id="r3"><option value="0">رتبهٔ ۱ در لیست</option><option value="2">رتبهٔ ۳</option><option value="9">رتبهٔ ۱۰</option><option value="49">رتبهٔ ۵۰</option><option value="-1">آخرین رتبه</option></select></label></div><div id="o3"></div>';
  function draw() {
    const u = S.A.users[+$('u3').value], k = +$('r3').value;
    const ord = u.rows.map((r, i) => [r, i]).sort((a, b) => (b[0].s['0'].FULLK - a[0].s['0'].FULLK) || (a[1] - b[1])).map(x => x[0]);
    const idx = k < 0 ? ord.length - 1 : Math.min(k, ord.length - 1), r = ord[idx], f = fm(r.f);
    const cnt = Math.round(Math.exp(r.P[2]) - 1);
    $('o3').innerHTML = '<div class="flowrow">' +
      '<div class="step"><b>۱. ورودی‌های فیلم</b>' + f.t + '<br><span class="mini">' + (f.d || '') + (f.g.length ? ' · ' + f.g.join('، ') : '') + '</span><br>' +
      'محبوبیت (P): ' + nfa(Math.round(r.P[0] * 100)) + '٪ از کاربران آموزش پسندیده‌اند، میانگین نمره ' + fa(r.P[1], 1) + ' از ' + nfa(cnt) + ' نمره.<br>شهرت LLM (L): کارگردان ' + fa(r.L[0], 1) + '، بازیگران ' + fa(r.L[1], 1) + ' از ۱۰.</div>' +
      '<div class="step"><b>۲. ورودی‌های کاربر</b>فقط پروفایل توییتری (فازی‌شده)، چون N=۰ و هیچ نمره‌ای از او نیست. پروفایل برای همهٔ فیلم‌های این کاربر یکی است و رتبهٔ فیلم‌ها را خودش عوض نمی‌کند.</div>' +
      '<div class="step"><b>۳. مدل hgb_c</b>۳۰۰ درخت کوچک (هر کدام ۴ برگ) پشت سر هم؛ هر درخت خطای قبلی‌ها را کمی اصلاح می‌کند و جمعشان به احتمال «پسند» تبدیل می‌شود.</div>' +
      '<div class="step"><b>۴. نمرهٔ مدل</b><span class="big num">' + fa(r.s['0'].FULLK, 2) + '</span><br>محبوبیت فقط همین فیلم: ' + fa(r.s['0'].POP, 2) + '</div>' +
      '<div class="step"><b>۵. رتبه</b>رتبهٔ ' + nfa(idx + 1) + ' از ' + nfa(ord.length) + ' فیلم این کاربر.</div>' +
      '<div class="step"><b>۶. واقعیت</b>نمرهٔ واقعی کاربر: ' + nfa(r.r) + ' از ۱۰ ' + hitMark(r) + '</div></div>' +
      '<p class="mini">برای این فیلم حذف LLM نمره را تقریباً عوض نمی‌کند: ' + fa(r.s['0'].FULLK_noL, 2) + ' در برابر ' + fa(r.s['0'].FULLK, 2) + '.</p>';
  }
  ['u3', 'r3'].forEach(i => $(i).addEventListener('change', draw)); draw();
})();

/* tab 4 */
(function () {
  const h = $('t4');
  h.innerHTML = '<p class="mini">یک فیلم جدید (که هیچ نمره‌ای از آن در آموزش نبود) انتخاب کنید. مدل کاربرانی را که فکر می‌کند دوستش دارند بالاتر می‌گذارد. نمرهٔ واقعی هر کاربر را می‌دانیم.</p><div class="ctrl"><label>فیلم جدید<select id="f4">' +
    S.B.films.map((f, i) => '<option value="' + i + '">' + fm(f.f).t + ' (' + nfa(f.n) + ' کاربر)</option>').join('') + '</select></label><label>مدل<select id="m4">' + ['FULLK', 'FULLK_noL', 'T', 'K', 'RANDOM'].map(m => '<option value="' + m + '">' + MB[m] + '</option>').join('') + '</select></label></div><div id="o4"></div>';
  function auc(us, m) { let w = 0, t = 0; us.forEach(a => { if (!a.y) return; us.forEach(b => { if (b.y) return; t++; w += a.s[m] > b.s[m] ? 1 : a.s[m] === b.s[m] ? 0.5 : 0; }); }); return t ? w / t : null; }
  function draw() {
    const fl = S.B.films[+$('f4').value], m = $('m4').value, f = fm(fl.f);
    const us = fl.users.slice().sort((a, b) => b.s[m] - a.s[m] || a.u - b.u), top = us.slice(0, 10);
    const liked = us.filter(x => x.y).length, hits = top.filter(x => x.y).length;
    $('o4').innerHTML = '<div class="kpis"><div class="kpi"><b>' + nfa(hits) + ' از ۱۰</b><span>کاربر بالای لیست که واقعاً پسندید</span></div><div class="kpi"><b>' + nfa(Math.round(liked / us.length * 100)) + '٪</b><span>سهم کل کاربرانی که پسندیدند</span></div><div class="kpi"><b class="num">' + (auc(us, m) == null ? '—' : fa(auc(us, m))) + '</b><span>AUC این فیلم (بین کاربران)</span></div><div class="kpi"><b class="num">' + fa(auc(us, 'RANDOM')) + '</b><span>AUC تصادفی</span></div></div>' +
      '<p>' + f.t + ' <span class="mini">' + (f.d || '') + (f.g.length ? ' · ' + f.g.join('، ') : '') + '</span></p><ol class="top">' + top.map(x => '<li><span class="t1">کاربر ' + nfa(x.u) + '</span><span class="num mini">' + fa(x.s[m], 2) + '</span><span></span>' + (x.y ? '<span class="hit y">✔</span>' : '<span class="hit n">✘</span>') + '</li>').join('') + '</ol><p class="mini">ده کاربر اول بر اساس نمرهٔ مدل؛ ✔ یعنی کاربر واقعاً فیلم را پسندیده است.</p>';
  }
  ['f4', 'm4'].forEach(i => $(i).addEventListener('change', draw)); draw();
})();


/* ---------- animation: how the Top-10 list is built, block by block (one step = 6 seconds, the explanation is inside the animation) ---------- */
function ticker(bar, ms) { bar.style.transition = 'none'; bar.style.width = '0'; void bar.offsetWidth; bar.style.transition = 'width ' + ms + 'ms linear'; bar.style.width = '100%'; }
function freeze(bar) { const w = getComputedStyle(bar).width; bar.style.transition = 'none'; bar.style.width = w; }
(function () {
  const host = $('anTop'); if (!host) return;
  const STEP = 6000;
  const steps = ['POP', 'P', 'P+H', 'P+H+A', 'P+H+A+L', '+PR', 'FULLK'];
  const names = ['فقط محبوبیت فیلم', 'مدل P', '+ تاریخچهٔ کاربر (H)', '+ شباهت (A)', '+ شهرت LLM (L)', '+ پروفایل فازی (PR)', '+ محتوای TMDB (K)'];
  const caps = ['خط پایه. از میان فیلم‌هایی که این کاربر نمره داده، ده فیلمِ محبوب‌تر (بیشترین سهم پسندیده در کاربران آموزش) را برمی‌داریم. هیچ چیزی دربارهٔ خودِ کاربر به‌کار نمی‌رود.',
    'مدل P. همان محبوبیت (سهم پسندیده، میانگین نمره و تعداد نمره) را به یادگیرندهٔ hgb_c می‌دهیم. لیست تقریباً همان است، چون مدل چیزی جز محبوبیت ندارد.',
    '+ تاریخچهٔ کاربر (H). این کاربر هیچ نمره‌ای آشکار نکرده (N=۰)، پس تاریخچه‌ای برای خلاصه کردن نیست و لیست عوض نمی‌شود.',
    '+ شباهت (A). شباهت هم از نمره‌های آشکارشدهٔ کاربر ساخته می‌شود، پس برای N=۰ اثری ندارد.',
    '+ شهرت LLM (L). امتیاز شهرت کارگردان و بازیگران وارد مدل می‌شود؛ ممکن است ترتیب چند فیلم کمی عوض شود.',
    '+ پروفایل فازی توییتر (PR). این پروفایل برای همهٔ فیلم‌های یک کاربر یکی است، پس فقط از راه ترکیب با ویژگی‌های فیلم اثر می‌گذارد.',
    '+ محتوای TMDB (K). ژانر، سال و کلیدواژهٔ فیلم‌ها وارد می‌شود. این مدل نهایی و لیست نهایی است.'];
  const rowH = 46, nU = S.A.users.length;
  host.innerHTML = '<div class="anhead"><b id="atT"></b><div class="anctl"><button type="button" id="atP">توقف</button><button type="button" id="atS">قدم بعدی</button><button type="button" id="atU">کاربر بعدی</button></div></div>' +
    '<div class="stepline" id="atL"></div><div class="anbox"><div class="ancapbox"><b id="atCh"></b><span id="atC"></span></div><div class="anstage" id="atG"></div><div class="ankpi" id="atK"></div><div class="ticker"><i id="atB"></i></div></div>';
  const stage = $('atG'); stage.style.height = (10 * rowH + 8) + 'px';
  const els = new Map();
  let u = 0, tick = 0, vis = true, timer = null, playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  const mk = r => { const e = document.createElement('div'); e.className = 'anrow'; const f = fm(r.f);
    e.innerHTML = '<span class="rk"></span><span class="ti">' + f.t + ' <small>' + (f.d || '') + '</small></span><span class="scb"><i></i></span><span class="res"></span>'; return e; };
  function schedule() { clearTimeout(timer); if (playing && vis) { ticker($('atB'), STEP); timer = setTimeout(next, STEP); } }
  function draw() {
    const user = S.A.users[u];
    $('atT').textContent = 'کاربر آزمون ' + nfa(user.id) + ' از ' + nfa(nU) + ' (' + nfa(user.n) + ' فیلم نمره‌دار، بدون هیچ نمرهٔ آشکار)';
    const s = Math.min(tick, 6), reveal = tick >= 7, model = steps[s];
    const cur = top10(user, '0', model), prev = s > 0 ? top10(user, '0', steps[s - 1]) : null, pop = top10(user, '0', 'POP');
    $('atL').innerHTML = names.map((n, k) => '<span class="' + (k <= s ? 'on' : '') + '">' + n + '</span>').join(' ← ');
    const keep = new Set(cur.map(r => r.f));
    els.forEach((e, f) => { if (!keep.has(f)) { e.style.opacity = 0; els.delete(f); setTimeout(() => e.remove(), 1700); } });
    const mx = Math.max(...cur.map(r => r.s['0'][model]));
    cur.forEach((r, i) => {
      let e = els.get(r.f);
      if (!e) { e = mk(r); els.set(r.f, e); stage.appendChild(e); e.style.opacity = 0; e.style.transform = 'translateY(' + (i * rowH + 4) + 'px)'; void e.offsetWidth; }
      e.style.opacity = 1; e.style.transform = 'translateY(' + (i * rowH + 4) + 'px)';
      e.classList.toggle('new', !!prev && !prev.some(p => p.f === r.f));
      e.querySelector('.rk').textContent = nfa(i + 1);
      e.querySelector('.scb i').style.width = Math.round(r.s['0'][model] / mx * 100) + '%';
      const rs = e.querySelector('.res');
      if (reveal) { rs.className = 'res ' + (r.r > 7 ? 'y' : 'n'); rs.textContent = (r.r > 7 ? '✔ ' : '✘ ') + nfa(r.r); } else { rs.className = 'res'; rs.textContent = ''; }
    });
    const chg = prev ? cur.filter(r => !prev.some(p => p.f === r.f)).length : 0, same = cur.filter(r => pop.some(p => p.f === r.f)).length;
    if (!reveal) {
      $('atCh').textContent = 'مرحلهٔ ' + nfa(s + 1) + ' از ۷';
      $('atC').textContent = caps[s];
      $('atK').innerHTML = '<span>فیلم جدید نسبت به مرحلهٔ قبل: <b>' + nfa(chg) + '</b> (کادر بنفش)</span><span>مشترک با لیست «فقط محبوبیت»: <b>' + nfa(same) + ' از ۱۰</b></span>';
    } else {
      const h = cur.filter(r => r.r > 7).length;
      $('atCh').textContent = 'پایان: نمرهٔ واقعی کاربر';
      $('atC').textContent = 'نمرهٔ واقعی این کاربر آشکار شد. ✔ یعنی نمره بیشتر از ۷ (پسندیده) و ✘ یعنی نپسندیده. مدل هیچ‌وقت این نمره‌ها را ندیده بود.';
      $('atK').innerHTML = '<span>پسندیده‌شده: <b>' + nfa(h) + ' از ۱۰</b></span><span>NDCG@10 مدل نهایی: <b>' + fa(user.ndcg['0'].FULLK) + '</b></span><span>فقط محبوبیت: <b>' + fa(user.ndcg['0'].POP) + '</b></span><span>تصادفی: <b>' + fa(user.ndcg['0'].RANDOM) + '</b></span>';
    }
    schedule();
  }
  function next() { tick++; if (tick > 7) { tick = 0; u = (u + 1) % nU; stage.innerHTML = ''; els.clear(); } draw(); }
  $('atP').addEventListener('click', () => { playing = !playing; $('atP').textContent = playing ? 'توقف' : 'ادامه'; if (playing) schedule(); else { clearTimeout(timer); freeze($('atB')); } });
  $('atS').addEventListener('click', next);
  $('atU').addEventListener('click', () => { tick = 0; u = (u + 1) % nU; stage.innerHTML = ''; els.clear(); draw(); });
  if (!playing) $('atP').textContent = 'ادامه';
  if ('IntersectionObserver' in window) new IntersectionObserver(en => { vis = en[0].isIntersecting; if (vis) schedule(); else clearTimeout(timer); }, {threshold: 0.25}).observe(host);
  draw();
})();

/* ---------- animation: the ablation runs by itself (one step = 6 seconds, explanation inside) ---------- */
(function () {
  const host = $('anAbl'); if (!host) return;
  const STEP = 6000;
  const sc = [['A', 'سناریو A: کاربر جدید (N=۰)', ['P', 'L', 'PR', 'K']], ['B', 'سناریو B: فیلم جدید (M=۰)', ['T', 'L', 'K']], ['C', 'سناریو C: هر دو جدید (N=۰)', ['T', 'L', 'PR', 'K']]];
  const nm = {P: 'محبوبیت (P)', T: 'سابقهٔ سازندگان (T)', L: 'شهرت LLM (L)', PR: 'پروفایل فازی (PR)', K: 'محتوای TMDB (K)'};
  host.innerHTML = '<div class="anhead"><b id="abT"></b><div class="anctl"><button type="button" id="abP">توقف</button><button type="button" id="abS">قدم بعدی</button></div></div>' +
    '<div class="anbox"><div class="ancapbox"><b id="abCh"></b><span id="abX"></span></div><div class="gauge"><div class="gfill" id="abF"></div><div class="gref" id="abR"></div><span class="gval num" id="abV"></span></div><div class="mini" id="abN"></div><div class="achips" id="abC"></div><div class="arank" id="abK"></div><div class="ticker"><i id="abB"></i></div></div>';
  const ev = [];
  sc.forEach(([s, t, bl], si) => { ev.push({si, k: 'full'}); bl.forEach(b => ev.push({si, k: 'rm', b})); ev.push({si, k: 'sum'}); });
  let i = 0, vis = true, timer = null, playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  function schedule() { clearTimeout(timer); if (playing && vis) { ticker($('abB'), STEP); timer = setTimeout(next, STEP); } }
  function draw() {
    const e = ev[i], [s, t, bl] = sc[e.si], full = D.fullk[s]['0'];
    const effs = bl.map(b => D.abl[s][b][0]), lo = full - Math.max(0.02, Math.max(...effs.map(Math.abs))) - 0.006, hi = full + 0.004;
    const W = v => Math.max(2, Math.min(100, (v - lo) / (hi - lo) * 100));
    $('abT').textContent = t; $('abN').textContent = 'خط سیاه: AUC مدل نهایی (' + fa(full, 4) + '). نوار: AUC مدل بعد از برداشتن بلوک.';
    $('abR').style.left = W(full) + '%';
    $('abC').innerHTML = bl.map(b => '<span class="achip' + (e.k === 'rm' && e.b === b ? ' off act' : '') + '">' + nm[b] + '</span>').join('');
    const F = $('abF'), V = $('abV'), K = $('abK'); K.innerHTML = '';
    if (e.k === 'rm') {
      const r = D.abl[s][e.b], w = full - r[0]; F.style.width = W(w) + '%'; F.className = 'gfill ' + (r[3] === 'better' ? 'drop' : r[3] === 'worse' ? 'gain' : 'flat'); V.textContent = fa(w, 4);
      $('abCh').textContent = 'برداشتن «' + nm[e.b] + '»';
      $('abX').innerHTML = 'AUC بدون این بلوک: <b>' + fa(w, 4) + '</b>. ' + (r[3] === 'better' ? 'افت <b>' + fa(r[0], 4) + '</b> (بازهٔ ۹۵٪ از ' + fa(r[1], 4) + ' تا ' + fa(r[2], 4) + ') و بازه صفر را شامل نمی‌شود، پس این بلوک واقعاً مهم است.' : r[3] === 'worse' ? 'مدل بدون این بلوک بهتر می‌شود (' + fa(-r[0], 4) + ' بیشتر)، یعنی این بلوک در این سناریو کمی ضرر دارد.' : 'تغییر ناچیز است (' + sgn(-r[0], 4) + ') و بازه صفر را شامل می‌شود، پس این بلوک چیزی به مدل اضافه نمی‌کند.');
    } else if (e.k === 'sum') {
      F.style.width = W(full) + '%'; F.className = 'gfill'; V.textContent = fa(full, 4);
      const sorted = bl.slice().sort((a, b) => D.abl[s][b][0] - D.abl[s][a][0]), mx = Math.max(...effs.map(Math.abs));
      $('abCh').textContent = 'جمع‌بندی ' + t;
      $('abX').textContent = 'بلوک‌ها از مهم‌ترین به کم‌اهمیت‌ترین. نوار یعنی افت AUC هنگام برداشتن آن بلوک؛ نوار قرمز یعنی با برداشتنش مدل بهتر می‌شود.';
      K.innerHTML = sorted.map(b => '<div><span>' + nm[b] + '</span><span class="tr"><i class="' + (D.abl[s][b][0] < 0 ? 'neg' : '') + '" style="width:0" data-w="' + Math.round(Math.abs(D.abl[s][b][0]) / mx * 100) + '"></i></span><span class="num">' + sgn(D.abl[s][b][0], 4) + '</span></div>').join('');
      requestAnimationFrame(() => requestAnimationFrame(() => K.querySelectorAll('i').forEach(x => { x.style.width = x.dataset.w + '%'; })));
    } else {
      F.style.width = W(full) + '%'; F.className = 'gfill'; V.textContent = fa(full, 4);
      $('abCh').textContent = 'شروع: مدل نهایی کامل';
      $('abX').textContent = 'همهٔ بلوک‌ها در مدل هستند و AUC میانگین کاربران آزمون ' + fa(full, 4) + ' است. حالا هر بار فقط یک بلوک را برمی‌داریم و دوباره می‌سنجیم؛ هر چه افت بزرگ‌تر، بلوک مهم‌تر است.';
    }
    schedule();
  }
  function next() { i = (i + 1) % ev.length; draw(); }
  $('abP').addEventListener('click', () => { playing = !playing; $('abP').textContent = playing ? 'توقف' : 'ادامه'; if (playing) schedule(); else { clearTimeout(timer); freeze($('abB')); } });
  $('abS').addEventListener('click', next);
  if (!playing) $('abP').textContent = 'ادامه';
  if ('IntersectionObserver' in window) new IntersectionObserver(en => { vis = en[0].isIntersecting; if (vis) schedule(); else clearTimeout(timer); }, {threshold: 0.25}).observe(host);
  draw();
})();

/* ---------- glossary: a small question mark after every concept; hover, focus or tap shows a short and a longer explanation ---------- */
const G = {
 auc: ["AUC (سطح زیر منحنی ROC)", "احتمال اینکه مدل فیلم موردعلاقهٔ یک کاربر را بالاتر از فیلم غیرموردعلاقه‌اش بگذارد.", "برای هر کاربر آزمون همهٔ جفت‌های (فیلم پسندیده، فیلم نپسندیده) را مقایسه می‌کنیم و می‌شماریم در چند جفت مدل فیلم پسندیده را بالاتر گذاشته است. ۰٫۵ یعنی حدس تصادفی و ۱ یعنی ترتیب کامل. چون برای هر کاربر جدا حساب می‌شود، فقط ترتیب فیلم‌های همان کاربر را می‌سنجد و بعد میانگین کاربران گرفته می‌شود."],
 ablation: ["ablation (حذف جزء)", "از مدل کامل یک جزء را برمی‌داریم و می‌بینیم چقدر بد می‌شود.", "اگر برداشتن یک بلوک اطلاعات (مثلاً LLM) AUC را تقریباً عوض نکند، آن بلوک به این مدل چیزی اضافه نکرده است. اگر افت بزرگ باشد، بلوک مهم است. این روش سهم هر جزء را نسبت به بقیهٔ مدل می‌سنجد، نه به‌تنهایی."],
 ci: ["بازهٔ اطمینان ۹۵٪", "بازه‌ای که نشان می‌دهد به عدد چقدر اطمینان داریم.", "کاربران را ۲۰۰۰ بار با جایگذاری دوباره نمونه‌گیری می‌کنیم (بوت‌استرپ) و میانگین را هر بار حساب می‌کنیم؛ ۹۵٪ میانیِ آن‌ها بازه است. اگر بازهٔ یک تفاوت صفر را شامل نشود، تفاوت را واقعی می‌دانیم، وگرنه می‌نویسیم «روشن نیست»."],
 paired: ["اختلاف جفت‌شده", "تفاضل دو مدل روی همان کاربران.", "برای هر کاربر AUC دو مدل را از هم کم می‌کنیم و میانگین این تفاضل‌ها را با بازه‌اش گزارش می‌کنیم. چون هر دو مدل روی همان کاربران آزمون‌اند، تفاوت‌های کوچک دقیق‌تر دیده می‌شوند تا وقتی دو میانگین جدا مقایسه شوند."],
 hgb: ["hgb_c (تقویت گرادیانی منظم)", "چند صد درخت کوچک که هر کدام خطای قبلی‌ها را کم می‌کنند.", "مدل از ۳۰۰ درخت با فقط ۴ برگ ساخته شده، با سرعت یادگیری کم و حداقل ۴۰۰ ردیف در هر برگ؛ این تنظیمات ملایم جلوی بیش‌برازش را می‌گیرند. برای دادهٔ جدولی و تعامل ویژگی‌ها مناسب است و یادگیرندهٔ اصلی ماست."],
 lr: ["رگرسیون لجستیک", "برای هر ویژگی یک وزن می‌گذارد و جمع وزن‌دار را به احتمال تبدیل می‌کند.", "ساده‌ترین مدل استاندارد و قابل‌تفسیر است و خط پایهٔ ماست. فقط رابطهٔ خطی را می‌گیرد و ترکیب دو ویژگی را خودش پیدا نمی‌کند."],
 rf: ["جنگل تصادفی", "تعداد زیادی درخت تصمیم مستقل که با رأی‌گیری تصمیم می‌گیرند.", "الگوریتمی که مقاله‌های پایه به‌کار برده‌اند، پس برای مقایسهٔ منصفانه نگهش داشته‌ایم. در اجرای ما به‌صورت سبک (۶۰ درخت) اجرا شد."],
 mlp: ["MLP (شبکهٔ عصبی کوچک)", "شبکه‌ای با دو لایهٔ پنهان.", "۶۴ و ۳۲ نرون، با ورودی استانداردشده و توقف زودهنگام. روی دادهٔ جدولی ما در B و C حدود ۰٫۰۳ AUC بدتر از hgb_c بود؛ این با تجربهٔ عمومی دربارهٔ دادهٔ جدولی می‌خواند."],
 ndcg: ["NDCG@10", "کیفیت ترتیب ۱۰ فیلم پیشنهادی، بین ۰ و ۱.", "فیلم‌های بالاتر لیست وزن بیشتری دارند و نمرهٔ واقعی کاربر ارزش هر فیلم است. ۱ یعنی لیست دقیقاً به بهترین ترتیب ممکن از فیلم‌های همان کاربر است. لیست تصادفی در دادهٔ ما حدود ۰٫۷ می‌گیرد، نه صفر."],
 p10: ["P@10", "چند درصد از ۱۰ فیلم پیشنهادی را کاربر واقعاً پسندیده است.", "«پسندیده» یعنی نمرهٔ کاربر بیشتر از ۷. چون حدود یک‌سوم فیلم‌ها پسندیده‌اند، لیست تصادفی حدود ۰٫۳۷ می‌گیرد."],
 mrr: ["MRR", "میانگین وارون رتبهٔ اولین پاسخ درست در لیست.", "مقاله‌های پایه این معیار را گزارش کرده‌اند ولی تعریف دقیقش در متنشان روشن نیست، پس عیناً بازتولید نشد."],
 f1: ["F1 کلاس «پسندیده»", "میانگین هماهنگ دقتِ مثبت (precision) و بازیابی (recall) برای فیلم‌های پسندیده.", "دقتِ مثبت یعنی از فیلم‌هایی که «پسندیده» پیش‌بینی شد چند تا واقعاً پسندیده بود، و بازیابی یعنی از فیلم‌های پسندیده چند تا پیدا شد. F1 هر دو را با هم می‌سنجد. حدس «همه پسندیده‌اند» در دادهٔ ما F1 حدود ۰٫۵۲ دارد."],
 acc: ["دقت (accuracy)", "چند درصد پیش‌بینی‌های «پسندیده یا نپسندیده» درست بود.", "چون حدود ۶۴٪ ردیف‌ها «نپسندیده» است، حدس «همه نپسندیده‌اند» به‌تنهایی دقت حدود ۰٫۶۴ می‌گیرد؛ پس دقت بدون مقایسه با آن گمراه‌کننده است."],
 thr: ["آستانهٔ تصمیم", "نمره‌ای که بالاتر از آن می‌گوییم «پسندیده».", "آستانه را فقط از کاربران آموزش یاد می‌گیریم (جایی که F1 بیشینه می‌شود) و روی کاربران آزمون اعمال می‌کنیم. چون برای F1 تنظیم می‌شود، گاهی دقت را فدا می‌کند."],
 trivial: ["حدس ساده", "مدلی که چیزی یاد نگرفته: همه را «نپسندیده» یا همه را «پسندیده» می‌گوید.", "هر مدل واقعی باید از این حدس‌ها بهتر باشد. برای رتبه‌بندی، حدس ساده «فقط محبوبیت فیلم» است که AUC حدود ۰٫۸۱ دارد و شکستنش سخت است."],
 tmdb: ["TMDB", "پایگاه دادهٔ عمومی فیلم‌ها.", "از آن ژانر، سال، مدت، کلیدواژه، خلاصهٔ داستان، کشور و بودجهٔ هر فیلم را گرفته‌ایم (برای ۱۱۰۱ از ۱۱۰۵ فیلم جور شد). هیچ نمره‌ای از کاربران CSP در آن نیست."],
 pub: ["امتیاز عمومی TMDB (PUB)", "میانگین نمرهٔ عموم مردم به فیلم در TMDB.", "چون خودش نظر دیگران دربارهٔ فیلم است، آن را «سقف» می‌نامیم و در مدل رسمی نمی‌گذاریم. حدود ۰٫۰۶ AUC به فیلم جدید اضافه می‌کند."],
 ceil: ["سقف", "حد بالایی که با اطلاعات بیرونی می‌شود رسید.", "جزو مدل رسمی نیست؛ فقط نشان می‌دهد اگر نظر عمومی هم در دست باشد، نتیجه چقدر بهتر می‌شود."],
 fuzzy: ["فازی‌سازی", "به‌جای «زیاد یا کم»، درجهٔ عضویت در چند دسته می‌دهد.", "مثلاً ۸۰۰ دنبال‌کننده ۷۲٪ «متوسط» و ۲۸٪ «زیاد» است. قله‌ها روی صدک‌های ۱۰ و ۵۰ و ۹۰ کاربران آموزش‌اند. برای اینکه بفهمیم خودِ فازی بودن ارزش دارد، باید با پروفایل خام و قطعی مقایسه شود."],
 crisp: ["قطعی (crisp)", "هر کاربر فقط در یک دسته است.", "همان دسته‌های فازی، ولی بالاترین درجه صددرصد و بقیه صفر. کنترل اصلی فازی است."],
 raw: ["خام (raw)", "خودِ عدد بدون دسته‌بندی.", "۱۲ ستون پروفایل فقط هم‌مقیاس شده‌اند. اگر فازی از خام بهتر نباشد، یعنی دسته‌بندی فازی چیزی به خود عدد اضافه نکرده است."],
 nrev: ["N: نمره‌های آشکارشدهٔ کاربر", "تعداد نمره‌هایی که از خود کاربر جدید می‌دانیم.", "N=۰ یعنی هیچ نمره‌ای از او نیست و فقط پروفایل داریم. نمره‌های آشکارشده تصادفی انتخاب می‌شوند و فیلم‌های آزمون هیچ‌وقت جزو آن‌ها نیستند."],
 mrev: ["M: نمره‌های معلوم فیلم", "تعداد نمره‌هایی که از خودِ فیلم جدید می‌دانیم.", "M=۰ یعنی هیچ کاربری هنوز این فیلم را نمره نداده است. M=۵ یعنی پنج نمرهٔ کاربران آموزش به آن فیلم را می‌دانیم."],
 sparsity: ["تنکی (Sparsity)", "کم بودن نمره‌ها در داده.", "به‌صورت تصادفی فقط ۵۰٪ تا ۵٪ از نمره‌های آموزش را نگه می‌داریم و AUC را روی همان کاربران آزمون می‌سنجیم. نشان می‌دهد هر بلوک وقتی داده کم است چقدر ارزش دارد."],
 csp: ["CSP 2023", "دیتاست و مقالهٔ مرجع پروژه.", "دیتاستی که پروفایل توییتری کاربران را با نمره‌هایشان به فیلم‌ها وصل می‌کند. مقالهٔ آن روشی (محتوا و کاربران هم‌شبیه) برای پیشنهاد به کاربر جدید دارد که اینجا به‌عنوان محک اجرا شده است."],
 top10: ["Top-10", "ده فیلم برتر پیشنهادی برای یک کاربر.", "فیلم‌هایی را که کاربر رتبه داده با نمرهٔ مدل مرتب می‌کنیم و ده تای اول را پیشنهاد می‌دهیم. چون نمرهٔ واقعی هر فیلم را داریم، لیست قابل سنجش است."],
 rule: ["قاعدهٔ «پسند»", "فیلمی پسندیده است که نمرهٔ کاربر به آن بیشتر از ۷ باشد.", "این قاعدهٔ اصلی ماست (gt7) و حدود ۳۵٪ نمره‌ها پسند حساب می‌شوند. قاعدهٔ «۷ و بیشتر» (ge7) در آزمایش‌های اصلی نیست."],
 ge7: ["ge7", "قاعدهٔ «نمره ۷ یا بیشتر».", "قاعدهٔ مقالهٔ ۲۰۲۰. فقط برای مقایسه با آن مقاله به‌کار رفت."],
 llm: ["LLM (مدل زبانی بزرگ)", "مدلی که از متن‌های فراوان یاد گرفته و دانش عمومی دارد.", "در این پروژه از آن می‌خواهیم برای هر کارگردان و بازیگر امتیاز شهرت ۱ تا ۱۰ بدهد. این امتیاز یک ستون ورودی (بلوک L) می‌شود و خودِ پیش‌بینی را انجام نمی‌دهد."],
 sonnet: ["Sonnet 5.5", "مدل زبانی‌ای که امتیاز شهرت را داد.", "برای ۱۳٬۰۵۲ نفر (کارگردان و بازیگر) امتیاز گرفتیم و با چهار ارزیاب انسانی اعتبارسنجی کردیم؛ تکرار دو اجرا ۹۹٫۲٪ هم‌خوانی بله یا نه داشت."],
 fold: ["فولد", "یکی از ۵ بخش کاربران که برای آزمون کنار گذاشته می‌شود.", "کاربران آزمون هیچ‌وقت در آموزش نیستند. کل کار ۳ بار با تقسیم‌های مختلف تکرار می‌شود، پس هر عدد میانگین ۱۵ اجراست."],
 seed: ["بذر ثابت", "عدد ثابتی که انتخاب تصادفی را تکرارپذیر می‌کند.", "کاربران و فیلم‌های شبیه‌ساز با بذر انتخاب شده‌اند، نه با نگاه به نتیجه، تا نمونه گل‌چین‌شده نباشد."],
 usersplit: ["کاربران آموزش و آزمون", "کاربر آزمون کاربری است که مدل هرگز ندیده است.", "مدل فقط با کاربران آموزش یاد می‌گیرد. اطلاعاتی از کاربران آزمون (حتی آمار فیلم) نباید به آموزش نشت کند، وگرنه نتیجه خوش‌بینانه می‌شود."],
 textvec: ["بردار متنی", "خلاصهٔ داستان فیلم که به چند عدد تبدیل شده.", "هم با TF-IDF و هم با یک مدل زبانی کوچک (MiniLM) امتحان شد. اثری نداشت یا کمی منفی بود، پس در مدل نیست."],
 genretaste: ["سلیقهٔ ژانری", "ساخت سلیقهٔ هر کاربر از نمره‌هایش به هر ژانر.", "به سه شکل (خام، قطعی، فازی) امتحان شد و اثرش حداکثر حدود ۰٫۰۰۵ AUC بود؛ فازی با خام تفاوت روشنی نداشت."],
 cf: ["فیلترینگ همکارانه (CF)", "پیشنهاد بر اساس نمره‌های کاربران مشابه.", "به نمره‌های قبلی همان کاربر یا فیلم نیاز دارد. برای کاربر یا فیلم بدون نمره چیزی برای کار کردن ندارد، به همین دلیل در این پروژه به‌کار نرفت."],
 mf: ["فاکتورگیری ماتریسی", "جدول نمره‌ها را به دو ماتریس کوچک تجزیه می‌کند.", "برای هر کاربر و هر فیلم یک بردار پنهان یاد می‌گیرد. کاربر یا فیلم بدون نمره بردار ندارد، پس برای شروع سرد به‌تنهایی کار نمی‌کند."],
 sA: ["سناریو A: کاربر جدید", "کاربر هیچ نمره‌ای ندارد، ولی فیلم‌ها از قبل نمره دارند.", "این سناریو را با N=۰ (هیچ نمره‌ای از کاربر) یا N نمرهٔ آشکارشده می‌سنجیم. فیلم‌ها شناخته‌اند، پس محبوبیتشان از کاربران آموزش حساب می‌شود."],
 sB: ["سناریو B: فیلم جدید", "فیلم هیچ نمره‌ای ندارد، ولی کاربران شناخته‌اند.", "فیلم‌های آزمون از آموزش کنار گذاشته می‌شوند. با M=۰ هیچ نمره‌ای از خود فیلم نیست و با M=۵ پنج نمره از کاربران آموزش می‌دانیم."],
 sC: ["سناریو C: هر دو جدید", "هم کاربر و هم فیلم جدیدند.", "سخت‌ترین حالت: فقط پروفایل توییتری کاربر و اطلاعات فیلم (سازندگان، LLM، TMDB) در دست است."],
 bP: ["محبوبیت فیلم (P)", "از نمره‌های کاربران آموزش حساب می‌شود.", "سه عدد برای هر فیلم: سهم کاربران آموزش که پسندیده‌اند، میانگین نمره و لگاریتم تعداد نمره‌ها. به‌تنهایی AUC حدود ۰٫۸۱ برای کاربر جدید می‌دهد."],
 bH: ["تاریخچهٔ کاربر (H)", "خلاصهٔ نمره‌های آشکارشدهٔ خود کاربر.", "مثلاً سهم فیلم‌هایی که کاربر پسندیده و اینکه چقدر سخت‌گیر است. وقتی N=۰ است، چیزی برای خلاصه کردن نیست."],
 bT: ["سابقهٔ کارگردان و بازیگر (T)", "نمرهٔ فیلم‌های دیگر همین کارگردان و بازیگران.", "از کاربران آموزش حساب می‌شود و برای فیلم جدید جایگزین محبوبیت است، چون خودِ فیلم نمره‌ای ندارد."],
 bL: ["شهرت LLM (L)", "امتیاز ۱ تا ۱۰ شهرت کارگردان و بازیگران از Sonnet.", "دو عدد برای هر فیلم (شهرت کارگردان و شهرت بازیگران). فقط دانش عمومی LLM دربارهٔ سازندگان است، نه نمرهٔ خودِ فیلم."],
 bA: ["شباهت (A)", "نمرهٔ کاربر به فیلم‌های هم‌کارگردان یا هم‌بازیگر.", "بر اساس نمره‌های آشکارشدهٔ کاربر حساب می‌شود، پس وقتی N=۰ است خالی است."],
 bPR: ["پروفایل فازی توییتر (PR)", "۱۲ ستون فعالیت توییتری کاربر که فازی شده‌اند.", "شامل سال ساخت حساب، تعداد دوست و دنبال‌کننده، ساعت و روز فعالیت و غیره. فقط آمار فعالیت است و هیچ محتوای سلیقه‌ای ندارد."],
 bFH: ["نمرهٔ معلوم خود فیلم (FH)", "M نمرهٔ کاربران آموزش به همان فیلم جدید.", "فقط وقتی M بزرگ‌تر از صفر باشد وجود دارد. در B با M=۵ دومین بلوک مهم است."],
 bK: ["محتوای TMDB (K)", "ژانر، سال، مدت، کلیدواژه و غیره از TMDB.", "از نمره‌ها ساخته نمی‌شود، پس وقتی نمره کم است افت نمی‌کند. برای فیلم جدید بزرگ‌ترین بهبود پروژه را می‌دهد."]
};
const TERMS = [["NDCG@10", "ndcg"], ["P@10", "p10"], ["MRR", "mrr"], ["AUC", "auc"], ["Ablation", "ablation"], ["ablation", "ablation"], ["بازهٔ اطمینان", "ci"], ["بازهٔ ۹۵٪", "ci"], ["اختلاف جفت‌شده", "paired"],
  ["hgb_c", "hgb"], ["رگرسیون لجستیک", "lr"], ["لجستیک", "lr"], ["جنگل تصادفی", "rf"], ["MLP", "mlp"], ["شبکهٔ عصبی", "mlp"], ["F1", "f1"], ["accuracy", "acc"], ["آستانه", "thr"], ["حدس‌های ساده", "trivial"], ["حدس ساده", "trivial"],
  ["امتیاز عمومی TMDB", "pub"], ["PUB", "pub"], ["سقف", "ceil"], ["TMDB", "tmdb"], ["پروفایل فازی", "fuzzy"], ["فازی‌سازی", "fuzzy"], ["قطعی", "crisp"], ["خام", "raw"], ["تنکی", "sparsity"], ["Sparsity", "sparsity"],
  ["CSP 2023", "csp"], ["Top-10", "top10"], ["بیشتر از ۷", "rule"], ["ge7", "ge7"], ["LLM", "llm"], ["Sonnet", "sonnet"], ["فولد", "fold"], ["بذر ثابت", "seed"], ["کاربران آموزش", "usersplit"], ["کاربر آزمون", "usersplit"],
  ["کاربران آزمون", "usersplit"], ["بردار متنی", "textvec"], ["سلیقهٔ ژانری", "genretaste"], ["فیلترینگ همکارانه", "cf"], ["فاکتورگیری ماتریسی", "mf"], ["N=", "nrev"], ["M=", "mrev"],
  ["سابقهٔ سازندگان (T)", "bT"], ["A: کاربر جدید", "sA"], ["B: فیلم جدید", "sB"], ["C: هر دو جدید", "sC"], ["محبوبیت فیلم (P)", "bP"], ["تاریخچهٔ کاربر (H)", "bH"], ["سابقهٔ کارگردان و بازیگر (T)", "bT"], ["شهرت LLM (L)", "bL"], ["شباهت (A)", "bA"], ["پروفایل فازی (PR)", "bPR"], ["نمرهٔ معلوم خود فیلم (FH)", "bFH"], ["محتوای TMDB (K)", "bK"]];
(function () {
  const skipTags = new Set(['SCRIPT', 'STYLE', 'svg', 'SELECT', 'OPTION', 'BUTTON', 'TEXTAREA', 'INPUT', 'NAV', 'TITLE']);
  document.querySelectorAll('.wrap > header, .wrap > section, .wrap > footer').forEach(sec => {
    const used = new Set(), nodes = [];
    const w = document.createTreeWalker(sec, NodeFilter.SHOW_TEXT);
    while (w.nextNode()) {
      const n = w.currentNode, pe = n.parentElement;
      if (!pe || !n.nodeValue.trim() || skipTags.has(pe.nodeName) || pe.closest('svg, dl.p, #fzout, .hq, .simcard, nav, .symbols, .anim')) continue;
      nodes.push(n);
    }
    nodes.forEach(node => {
      let cur = node;
      for (;;) {
        const txt = cur.nodeValue; let best = null;
        for (const [pat, key] of TERMS) {
          if (used.has(key)) continue;
          const i = txt.indexOf(pat);
          if (i < 0) continue;
          if (!best || i < best.i || (i === best.i && pat.length > best.pat.length)) best = {i, pat, key};
        }
        if (!best) return;
        used.add(best.key);
        const rest = cur.splitText(best.i + best.pat.length);
        const b = document.createElement('button'); b.type = 'button'; b.className = 'hq'; b.dataset.h = best.key; b.setAttribute('aria-label', 'توضیح: ' + G[best.key][0]); b.textContent = '؟';
        cur.parentNode.insertBefore(b, rest);
        cur = rest;
      }
    });
  });
  const pop = $('pop'); let curB = null, touch = false;
  function show(b) {
    const g = G[b.dataset.h]; pop.innerHTML = '<b class="t">' + g[0] + '</b><span class="s">' + g[1] + '</span><span class="pl">' + g[2] + '</span>'; pop.hidden = false;
    const r = b.getBoundingClientRect(), pw = pop.offsetWidth, ph = pop.offsetHeight;
    const x = Math.min(Math.max(8, r.left + r.width / 2 - pw / 2), innerWidth - pw - 8); let y = r.bottom + 8;
    if (y + ph > innerHeight - 8) y = Math.max(8, r.top - ph - 8);
    pop.style.left = x + 'px'; pop.style.top = y + 'px'; curB = b;
  }
  function hide() { pop.hidden = true; curB = null; }
  const hq = e => e.target.closest ? e.target.closest('.hq') : null;
  document.addEventListener('pointerdown', e => { touch = e.pointerType === 'touch'; }, true);
  document.addEventListener('mouseover', e => { if (touch) return; const b = hq(e); if (b) show(b); else if (curB && !e.target.closest('#pop')) hide(); });
  document.addEventListener('focusin', e => { const b = hq(e); if (b) show(b); });
  document.addEventListener('focusout', e => { if (hq(e)) hide(); });
  document.addEventListener('click', e => { const b = hq(e); if (b) { curB === b ? hide() : show(b); } else if (!e.target.closest('#pop')) hide(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hide(); });
})();
"""

page = ("<title>سرد-شروع و فازی‌سازی</title>\n<meta name=\"claude:viewport-fit-sides\" content=\"cover\">\n"
        "<link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;800&display=swap\">\n"
        "<style>" + CSS + "</style>\n" + BODY + "\n<div id=\"pop\" role=\"tooltip\" hidden></div>\n<script>\n" + JS.replace("__DATA__", json.dumps(D, ensure_ascii=False)).replace("__SIM__", SIM) + "\n</script>\n")
for key, val in (("T_CMT", T_CMT), ("T_TMDB", T_TMDB), ("T_ABL", T_ABL), ("T_FXM", T_FXM), ("T_DEL", T_DEL), ("T_SPE", T_SPE),
                 ("T_BEN", T_BEN), ("T_CLS", T_CLS), ("T_CLSB", T_CLSB), ("T_TOP", T_TOP), ("T_LRN", T_LRN)):
    page = page.replace(f"<!--{key}-->", val)
(OUT / "meeting_page.html").write_text(page, encoding="utf-8")
print("written", OUT / "meeting_page.html", len(page) // 1024, "KB")
