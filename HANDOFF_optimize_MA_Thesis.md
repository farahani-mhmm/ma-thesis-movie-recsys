# Handoff: optimize MA Thesis (updated 2026-11-02)

Read this first in a new chat. Reply to the user in Persian, RTL (every line/bullet starts with a Persian word; product names in Persian script, Latin only in parentheses). Supervisor = «بهروز». Ask questions as multiple choice. Report all effects honestly, claims must match evidence. Memory files load automatically; this file holds what memory does not.

## Project
- Thesis: cold start in movie recommendation with fuzzified Twitter-profile info + LLMs. Lab: `cold_start_lab/` (git branch `my-practice`, many files uncommitted). Raw data read-only in `CSP Dataset/`. LLM scripts in `pipeline_run/`.
- 3 scenarios: A new user, B new film, C both new. Label gt7 (rate > 7). Per-user AUC (primary) + F1/accuracy. Learners logreg / hgb_c (primary) / rf. User-grouped 5 folds x 3 repeats, N = 0/5/20/50 revealed ratings (B: M = 0/5 known film ratings).
- Step 14 `14_final_protocol.py`: local, no API, ~1 h. Reads the LLM fame CSV (default `pipeline_run/llm_claude_outputs/full_sonnet55/movies_with_sonnet_fame.csv`; `--llm-csv` to change, `--out-tag` for a separate output folder, `--tmdb` for K/PUB, `--taste` for E/GR/GC/GF/SIM). Step 15 = report.
- Blocks: P popularity(3), H history(4), L LLM fame dir + top-3 actors(2), A affinity(6, uses fame), T track record(4), FH known ratings of the new film, PR profile 12 cols (raw/crisp/fuzzy F1); optional K TMDB content(24), PUB TMDB public vote(4, a "ceiling"), E text vector, GR/GC/GF genre taste, SIM.
- Model output = probability 0-1 of rating > 7 per (user, film); top-10 = sort. Step 14 reports AUC/F1; top-k is `20_topk.py`.

## Key results (hgb_c)
- A: P alone AUC ~0.812; H +0.001..0.006 (grows with N); profile (any form) ~+0.0003; fuzzy vs raw/crisp no clear difference; L, A ~0.
- B: FULL 0.682 (M=0) / 0.770 (M=5); FH most important. C: FULL ~0.667; T and L main; profile +0.002 vs T, vanishes with L.
- TMDB content K: B 0.682 -> 0.731, C 0.667 -> 0.721 (biggest gain). PUB -> ~0.78-0.79 (ceiling). Text embeddings E slightly hurt. Step 9 fuzzy F1/F2/F3 + kNN all within ~0.0007 of raw.
- 12 profile cols cannot predict user taste (CV R2 <= 0); no subgroup where profile helps (memory `project-profile-no-taste-signal`). Limit = information in the profile, not the fuzzy method.
- Sonnet fame: 1105 films, 13,052 people (batches of 10). Final scores use run 1 only; 406 requests of run 2 also exist (usable to check repeatability); 32% people "unknown".

## Plan: one session per independent experiment
Rule: work that needs another step's result stays in the same session; independent work gets its own session.
Sidebar: file each new session into the group "MA Thesis" (ColdStart work goes in "ColdStart Practice"). Names in English, numbered: 02 Exp Test Q, 03 Exp RAG, 04 Playground, 05 Exp Gemini.

### Session 1: Test Q (film-level LLM scores) - do first
0. Git first (memory `project-git-workflow-experiments`): `git status`, check the files to be committed contain no API keys, raw CSP data or big outputs, show the list to the user, and after their OK make one clean commit on `my-practice`. Then make branch `exp/q-film-scores`. No worktrees, since outputs and data are not in git.
1. Buy Liara credit (~600-700k toman covers it), then `python llm_film_scores.py` (dry run, compare its estimate with ~$1.6) and `--yes`. The script exists (2026-10-27, never run): Sonnet, title + year + first director only, popularity + quality 1-10, 3 runs, median, 10 films/request, supports Liara via ANTHROPIC_BASE_URL. Output `pipeline_run/llm_claude_outputs/film_scores/film_scores.csv` (folder empty now).
2. Add block Q (film_pop, film_quality) to step 14 (scenarios B and C, also A for completeness).
3. Run step 14 with `--tmdb` and a new `--out-tag` (old results stay untouched).
4. Step 15; compare T+K vs T+K+Q, Q vs PUB. Report Q as "public knowledge about the film", separate from no-outside-info.

**Outcome (DONE 2026-11-03, branch `exp/q-film-scores`, commits 7a9bbd9 + dd21455):**
- Scores: Liara is OpenAI-compatible (model `anthropic/claude-sonnet-5.5`); `llm_film_scores.py` now detects Liara, runs an endpoint check (saved `film_scores/endpoint_check.json`; reply model field matched, id is Liara's own, no Anthropic token count since no Anthropic key) and stops if the first request fails. 333/333 requests ok, 1104 of 1105 films scored, 272,793 in / 84,971 out tokens (about $1.40). 3 runs agree: popularity identical for 904 films, quality for 986. Spearman: Sonnet quality vs TMDB vote average 0.82, Sonnet popularity vs log vote count 0.65.
- Step 14: `--filmq --out-tag _filmq` (block Q = pop, quality, unknown flag; implies --tmdb). Output `cold_start_lab/outputs/14_final_filmq/` (report `outputs/filmq_report.txt`). Run split: Windows B all + A, C repeats 0-1; Mac repeat 2 of A and C (same package versions, `requirements-lock.txt`). Reproducibility check against the old all-Windows `_tmdb` run on repeat 2: hgb_c identical, logreg mean per-user diff < 0.0001, rf mean 0.0004-0.002 (thread nondeterminism); Windows-vs-Windows repeat 0 identical.
- B and C (new film): T+K+Q vs T+K **+0.048 to +0.057 AUC, better** at every level and learner (T 0.657, T+K 0.720, T+K+Q 0.769, T+K+PUB 0.782). Q fills about 80% of the K -> PUB gap. Q alone 0.749 vs PUB 0.774-0.785 (worse by 0.022-0.036). With PUB present Q adds only +0.001 to +0.003 (mostly "better", tiny). B: FULLKQ vs FULLK +0.047 at M=0, +0.015 at M=5 (less needed once a few ratings are known).
- A (new user, films already rated by others): Q adds nothing (all differences within +-0.0006, "no clear").
- F1 (liked) and accuracy agree: T+K+Q vs T+K in B F1 +0.031..+0.037, accuracy +0.040..+0.066; in C F1 +0.034..+0.041, accuracy +0.041..+0.070 (all "better"). Q vs PUB worse in all scenarios (F1 -0.010..-0.026, accuracy -0.028..-0.068). Q beyond PUB: F1 -0.000..+0.005, accuracy mixed (better/no clear/worse by learner). A: F1 and accuracy mixed around zero.
- Thesis wording: Q is public knowledge about the film recalled by the LLM (like PUB, not "no outside information"); it is a cheap substitute for TMDB public votes, most useful when the film has no ratings at all.

### Session 2: Test RAG (content traits from film text) - after Session 1
- Give Sonnet each film's TMDB overview + keywords + tagline (already in `cold_start_lab/outputs/16_tmdb/movies_tmdb.csv`); ask darkness, pace, complexity, violence, humor, arthouse 1-10. Never include vote average / vote count / popularity in the prompt (would leak PUB).
- Compare 3 arms: memory only (title/year) vs memory + text (RAG) vs no LLM (E text vector). Add block with raw / crisp / fuzzy versions (the fuzzy idea moves to the film side). Est. Sonnet ~1.4M toman, 1-2 days work. Optional: 2-4 human raters on 30-50 films (like step 10).

### Session 3: Playground (demos)
- Private repo `farahani-mhmm/thesis-playground`, local `MA Thesis/thesis-playground` (ignored by the outer repo). EVERYTHING inside it in English (memory `project-thesis-playground`).
- Done: demo 1 block explorer (Streamlit `app.py`, data from `build_data.py`). Run: `..\cold_start_lab\.venv\Scripts\python.exe -m streamlit run app.py`.
- Next: demo RAG vs LLM memory (10 films side by side), demo fuzzy vs crisp vs raw (slider + membership plot). Optional `.devcontainer` for Codespaces.

### Session 4: Gemini comparison (optional)
- `pipeline_run/llm_gemini.py` (copy of llm_claude.py, same prompt/batches, resumable, retries 429/5xx/network). Pilot of 40 directors unfinished: gemini-3.8-flash gave 503 "high demand", 3.7-flash hit a network drop (fixed, not re-run). Partial results in `llm_gemini_outputs/pilot40` and `pilot40_g37`. Then compare with Sonnet and the 4 human raters (step 10). Never mix models inside one run. Free-tier daily limit unknown; limits are per project, do not rotate keys.

## Tools written (2026-10-31 to 11-02)
- `pipeline_run/anthropic_test.py`, `gemini_test.py`, `key_check.py` (identify a key's provider, list models; `--provider aimlapi`).
- `cold_start_lab/tg_run.py`: run any command, send start/progress/alerts/end to Telegram (env TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID). Bot not created yet.
- Installed in `cold_start_lab/.venv`: anthropic, google-genai, pypdf, streamlit.

## API access facts
- Anthropic key works but has no credit. Liara: Sonnet 5.5, effective ~359k toman per $ (261k + 25% fee + 10% tax), minimum top-up ~416k (calculator artifact https://claude.ai/artifact/ArUGFgK3zW7Mnha1qeS5bF). Opus 5.5 = 2x Sonnet price; Sonnet is enough and keeps the thesis consistent.
- Gemini key (format `AQ.`, 53 chars) works; it was exposed in terminal output, so the user should delete it and make a new one.
- GitHub Models retired 2026-07-30; Copilot is not an API; Azure for Students $100 might allow GPT (unverified).
- AvalAI (avalai.ir): Iranian gateway, 20k toman gift (too small), OpenAI/Anthropic/GenAI compatible, has a research grant ("AvalAward"); check that `claude-sonnet-5-5` is offered before buying.
- Env vars are per terminal window (`$env:X = "..."`); `setx` needs a new window (Claude app terminal: app restart).

## Later ideas
Active learning (which films to ask a new user), offline calibration check of probabilities, small user study (ratings 1-10), Telegram bot as cold-start front end (needs بهروز's OK).
