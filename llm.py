import pandas as pd
import os
import json
import re
import time
import logging
from openai import OpenAI
from tqdm import tqdm

# ╔══════════════════════════════════════════════════════════════╗
# ║                  ۱. تنظیمات اولیه                           ║
# ╚══════════════════════════════════════════════════════════════╝

API_KEY    = "YOUR_API_KEY_HERE"              # ← توکن API خود را بگذارید
BASE_URL   = ""      # ← آدرس سرویس (اوپن‌ای‌آی یا واسط)
MODEL_NAME = ""                    # ← نام مدل

# مسیر فایل‌ها (ویندوزی)
INPUT_CSV       = r"I:\farhani codes\movie\movies.csv"
OUTPUT_CSV      = r"I:\farhani codes\movie\movies_with_llm_person_fame_scores.csv"
CHECKPOINT_FILE = r"I:\farhani codes\movie\person_fame_scores_checkpoint.json"

BATCH_SIZE   = 10    # ← بچ‌های ۱۰ تایی (طبق درخواست شما)
MAX_RETRIES  = 3     # تعداد تلاش مجدد برای هر بچ
SLEEP_BETWEEN_BATCHES = 0.5  # مکث بین بچ‌ها (ثانیه)

# ستون‌هایی که شامل نام افراد هستند
PERSON_COLUMNS = ["directors", "actors"]

# مقادیر نامعتبر که باید فیلتر شوند
INVALID_NAMES = {
    "NO_CREDITED_ACTORS", "Varios", "NaN", "", "N/A",
    "Unknown", "None", "No Credit", "Uncredited",
}

# تنظیمات لاگ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("FameScorer")


# ╔══════════════════════════════════════════════════════════════╗
# ║            ۲. توابع کمکی — پاکسازی و استخراج                ║
# ╚══════════════════════════════════════════════════════════════╝

def clean_name(name: str) -> str:
    """پاکسازی یک نام از کاراکترهای اضافی"""
    name = name.strip().strip("|").strip()
    # حذف فاصله‌های تکراری
    name = re.sub(r"\s+", " ", name)
    return name


def extract_names_from_cell(cell_value) -> list[str]:
    """
    استخراج لیست نام‌ها از یک سلول.
    مقادیر با | جدا شده‌اند.
    """
    if pd.isna(cell_value) or str(cell_value).strip() == "":
        return []

    raw_names = str(cell_value).split("|")
    cleaned = []
    for name in raw_names:
        name = clean_name(name)
        if name and name not in INVALID_NAMES:
            cleaned.append(name)
    return cleaned


def extract_unique_names(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """
    استخراج تمام نام‌های یکتا از ستون‌های مشخص‌شده.
    """
    unique_names = set()
    for col in columns:
        if col not in df.columns:
            log.warning(f"ستون '{col}' در فایل وجود ندارد. رد می‌شود.")
            continue
        for cell in df[col].dropna():
            names = extract_names_from_cell(cell)
            unique_names.update(names)

    result = sorted(unique_names)
    log.info(f"تعداد کل نام‌های یکتا استخراج‌شده: {len(result)}")
    return result


# ╔══════════════════════════════════════════════════════════════╗
# ║            ۳. ارتباط با LLM و دریافت نمرات                  ║
# ╚══════════════════════════════════════════════════════════════╝

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def build_prompt(names: list[str]) -> str:
    """ساخت پرامپت دقیق انگلیسی برای مدل"""

    names_formatted = "\n".join(f"{i+1}. {name}" for i, name in enumerate(names))

    prompt = f"""You are an expert film historian and pop-culture analyst specializing in global cinema fame and recognition.

## Task
Rate the global fame and mainstream recognition of each person listed below on a scale from 1.0 to 10.0.

## Scoring Rubric
- 9.0 – 10.0 : Global megastar / legendary auteur universally recognized (e.g., Steven Spielberg, Leonardo DiCaprio, Meryl Streep, Alfred Hitchcock).
- 7.0 – 8.9  : Very famous actor/director with multiple major mainstream hits and wide international recognition.
- 5.0 – 6.9  : Well-known working professional; recognizable to film enthusiasts and general audiences in specific regions.
- 3.0 – 4.9  : Niche or character actor / lesser-known director; recognized mainly by dedicated fans or within specific genres.
- 1.0 – 2.9  : Obscure, local, or virtually unknown figure outside very specialized circles.

## Important Rules
1. Consider GLOBAL fame, not just one country.
2. Directors and actors are judged by the same 1-10 scale.
3. If a person is primarily known in a non-English-speaking film industry, still rate their international recognition.
4. Return ONLY a valid JSON object. No markdown, no explanation, no extra text.
5. Keys must be the EXACT names as provided. Values must be numeric (float).

## People to Rate
{names_formatted}

## Output Format (strict JSON)
{{
  "Name One": 7.5,
  "Name Two": 3.0
}}"""

    return prompt


def call_llm_for_scores(names: list[str]) -> dict:
    """
    ارسال یک بچ از نام‌ها به LLM و دریافت نمرات.
    با تلاش مجدد و مدیریت خطا.
    """
    prompt = build_prompt(names)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise JSON-output assistant. "
                            "You respond ONLY with valid JSON objects. "
                            "No markdown, no code fences, no commentary."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2000,
            )

            raw_text = response.choices[0].message.content.strip()

            # حذف احتمالی ```json ... ```
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
            raw_text = raw_text.strip()

            parsed = json.loads(raw_text)

            # اعتبارسنجی: مطمئن شویم همه نام‌ها وجود دارند
            missing = [n for n in names if n not in parsed]
            if missing:
                log.warning(
                    f"تلاش {attempt}: {len(missing)} نام در پاسخ LLM نبود. "
                    f"نام‌های گم‌شده: {missing[:5]}..."
                )
                # برای نام‌های گم‌شده نمره پیش‌فرض ۳ می‌گذاریم
                for m in missing:
                    parsed[m] = 3.0

            # تبدیل مقادیر به float
            validated = {}
            for name, score in parsed.items():
                try:
                    validated[name] = round(float(score), 2)
                except (ValueError, TypeError):
                    validated[name] = 3.0

            return validated

        except json.JSONDecodeError:
            log.warning(
                f"تلاش {attempt}/{MAX_RETRIES}: خطای JSON. "
                f"پاسخ خام: {raw_text[:200]}..."
            )
            time.sleep(2)

        except Exception as e:
            log.error(f"تلاش {attempt}/{MAX_RETRIES}: خطای API → {e}")
            time.sleep(4)

    # اگر همه تلاش‌ها شکست خورد، نمره پیش‌فرض
    log.error("تمام تلاش‌ها برای این بچ ناموفق بود. نمره پیش‌فرض ۳.۰ اختصاص می‌یابد.")
    return {name: 3.0 for name in names}


# ╔══════════════════════════════════════════════════════════════╗
# ║         ۴. پردازش بچ‌به‌بچ + چک‌پوینت                       ║
# ╚══════════════════════════════════════════════════════════════╝

def load_checkpoint() -> dict:
    """بارگذاری نمرات ذخیره‌شده قبلی"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"چک‌پوینت بارگذاری شد: {len(data)} نام قبلاً نمره گرفته.")
        return data
    return {}


def save_checkpoint(scores: dict):
    """ذخیره نمرات تا این لحظه"""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)


def score_all_persons(all_names: list[str]) -> dict:
    """
    نمره‌دهی به تمام افراد در بچ‌های ۱۰ تایی.
    از چک‌پوینت استفاده می‌کند تا کار تکراری انجام نشود.
    """
    scores = load_checkpoint()

    # فقط نام‌هایی که هنوز نمره نگرفته‌اند
    remaining = [n for n in all_names if n not in scores]

    if not remaining:
        log.info("همه نام‌ها قبلاً نمره گرفته‌اند. ✓")
        return scores

    # ساخت بچ‌های ۱۰ تایی
    batches = [
        remaining[i : i + BATCH_SIZE]
        for i in range(0, len(remaining), BATCH_SIZE)
    ]

    log.info(
        f"تعداد بچ‌ها: {len(batches)} | "
        f"اندازه بچ: {BATCH_SIZE} | "
        f"نام‌های باقی‌مانده: {len(remaining)}"
    )

    for batch_idx, batch in enumerate(
        tqdm(batches, desc="نمره‌دهی با LLM", unit="بچ"), 1
    ):
        batch_scores = call_llm_for_scores(batch)
        scores.update(batch_scores)

        # ذخیره چک‌پوینت بعد از هر بچ
        save_checkpoint(scores)

        log.debug(
            f"بچ {batch_idx}/{len(batches)} انجام شد. "
            f"نمونه: {list(batch_scores.items())[:2]}"
        )

        # مکث برای رعایت محدودیت نرخ درخواست
        time.sleep(SLEEP_BETWEEN_BATCHES)

    log.info(f"نمره‌دهی کامل شد. مجموع افراد نمره‌داده‌شده: {len(scores)}")
    return scores


# ╔══════════════════════════════════════════════════════════════╗
# ║      ۵. محاسبه میانگین نمرات برای هر فیلم                   ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_average_score(
    cell_value,
    scores_dict: dict,
) -> float | None:
    """
    میانگین نمره شهرت افراد یک سلول (مثلاً لیست بازیگران یک فیلم).
    اگر هیچ نمره‌ای موجود نبود، None برمی‌گرداند.
    """
    names = extract_names_from_cell(cell_value)
    if not names:
        return None

    valid_scores = []
    for name in names:
        if name in scores_dict:
            valid_scores.append(scores_dict[name])

    if not valid_scores:
        return None

    return round(sum(valid_scores) / len(valid_scores), 4)


# ╔══════════════════════════════════════════════════════════════╗
# ║                      ۶. تابع اصلی (main)                     ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    log.info("=" * 60)
    log.info("  شروع فرایند نمره‌دهی شهرت")
    log.info("=" * 60)

    # ── مرحله ۱: خواندن فایل ──────────────────────────────────
    if not os.path.exists(INPUT_CSV):
        log.error(f"فایل ورودی پیدا نشد: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    log.info(f"فایل خوانده شد. تعداد فیلم‌ها: {len(df)} | ستون‌ها: {list(df.columns)}")

    # ── مرحله ۲: استخراج نام‌های یکتا ────────────────────────
    all_unique_names = extract_unique_names(df, PERSON_COLUMNS)

    if not all_unique_names:
        log.error("هیچ نامی استخراج نشد. خروجی گرفته می‌شود.")
        return

    # ── مرحله ۳: نمره‌دهی با LLM ─────────────────────────────
    person_scores = score_all_persons(all_unique_names)

    # ── مرحله ۴: محاسبه میانگین و افزودن ستون‌های جدید ──────
    log.info("محاسبه میانگین نمرات برای هر فیلم...")

    df["director_fame_score_llm"] = df["directors"].apply(
        lambda x: compute_average_score(x, person_scores)
    )

    df["actors_fame_score_llm"] = df["actors"].apply(
        lambda x: compute_average_score(x, person_scores)
    )

    # ── مرحله ۵: ذخیره خروجی نهایی ───────────────────────────
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"خروجی نهایی ذخیره شد: {OUTPUT_CSV}")

    # ── مرحله ۶: پاکسازی چک‌پوینت ────────────────────────────
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log.info("فایل چک‌پوینت حذف شد (کار کامل شد). ✓")

    # ── خلاصه آماری ──────────────────────────────────────────
    log.info("─" * 50)
    log.info("  خلاصه نتایج")
    log.info("─" * 50)
    log.info(f"  تعداد فیلم‌ها:                  {len(df)}")
    log.info(f"  تعداد افراد نمره‌داده‌شده:        {len(person_scores)}")
    log.info(
        f"  میانگین نمره کارگردانان:       "
        f"{df['director_fame_score_llm'].mean():.4f}"
    )
    log.info(
        f"  میانگین نمره بازیگران:         "
        f"{df['actors_fame_score_llm'].mean():.4f}"
    )
    log.info(
        f"  فیلم‌های بدون نمره کارگردان:    "
        f"{df['director_fame_score_llm'].isna().sum()}"
    )
    log.info(
        f"  فیلم‌های بدون نمره بازیگر:      "
        f"{df['actors_fame_score_llm'].isna().sum()}"
    )
    log.info("─" * 50)
    log.info("  فرایند با موفقیت به پایان رسید. ✅")
    log.info("=" * 60)


# ╔══════════════════════════════════════════════════════════════╗
# ║                         اجرا                                ║
# ╚══════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    main()