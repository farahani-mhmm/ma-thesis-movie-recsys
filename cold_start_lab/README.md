# Cold-start lab

The experiments of the thesis "Mitigation of the cold start problem in movie recommender systems using
fuzzification of users' social network information and LLMs".
Inputs are only READ from `../CSP Dataset/`. Everything the scripts write goes to `outputs/`.
Run every script from this folder (`cd cold_start_lab`).

## Question
When a user, a film, or both are new, which information helps a model rank films for a test user, and does it
beat simple baselines?

| Scenario | New | Known |
|---|---|---|
| A, new user | the user | the films (popularity from other users) |
| B, new item | the film | the user (ratings of other films) |
| C, both new | user and film | nothing about either, only content and profile |

## Rules
* Split by USER (test users never appear in training), 5 folds x 3 repeats, fixed seeds.
* N = 0, 1, 3, 5, 10, 20, 50 revealed ratings of the new user; the test pool is the user's ratings at positions >= 50
  (the same for every N). In scenarios B and C the test films are also held out of training.
* Both like rules are always reported: `rate > 7` (gt7) and `rate >= 7` (ge7).
* Main metric: per-user AUC (mean over users) with a 95% bootstrap interval over users. Paired differences are
  reported against the matching control.
* Everything learned from data (like ratios, similarities, scalers, fuzzy sets, clusters) is fitted on training
  users only; training rows get their film statistics out-of-fold over users.
* The primary learner of each step is declared before the results are seen (`hgb`, later `hgb_b` or `hgb_c`);
  logistic regression is the check.

## Steps
| step | file | scenario | what it does |
|---|---|---|---|
| 1 | `01_prepare_data.py` | | clean data from `../CSP Dataset/` |
| 2 | `02_make_splits.py` | | user folds and the random reveal order |
| 3 | `03_baselines.py` | A | majority guess and popularity baselines |
| 4 | `04_profile_models.py` | A | raw profile only (per-film models like the old paper), films-only control |
| 5 | `05_history_models.py` | A | popularity + history of N revealed ratings, + profile, + LLM columns, + affinity (E1 to E4 in the code); `--with-uu` adds user-user collaborative filtering |
| 6 | `06_report.py` | A | confidence intervals, paired differences and curves for steps 4 and 5 |
| 7 | `07_new_film_models.py` | B | content only (track record, LLM fame) and M known ratings of the film |
| 8 | `08_both_new_models.py` | C | both new; options for stronger learners, products, fuzzified profiles |
| 9 | `09_fuzzification.py` | A | fuzzification of the profile (F1, F2, F3) against raw, log/circular and crisp controls; global and kNN use |
| 10 | `10_llm_columns.py` | | checks the LLM fame scores against 4 human raters, builds better film columns (no API call) |
| 11 | `11_export_demo.py` | | collects the headline numbers into `outputs/11_demo/results.json` |
| 12 | `12_export_topk.py` | | exports a small Top-10 recommender (logistic regression) for the demo page |

| 13 | `13_ablation.py` | | collects the paired differences of the earlier steps (old exploratory runs) into one table |
| 14 | `14_final_protocol.py` | A, B, C | **the final protocol**: one script, three research questions, block-based names (P, H, T, L, A, FH, PR), add-one and leave-one-out experiments, AUC plus F1 and accuracy, Sonnet columns, fuzzifier F1 against raw and crisp controls, learners logreg / hgb_c / rf |
| 15 | `15_final_report.py` | A, B, C | tables of step 14: AUC and F1 with intervals, paired differences, one combined ablation table |

Final run (about one hour of computer time, no API cost):

```powershell
..\.venv\Scripts\python.exe 14_final_protocol.py
..\.venv\Scripts\python.exe 15_final_report.py
```

Appendix with the second label rule: add `--rules ge7 --out-tag _ge7` to step 14 and `--tag _ge7` to step 15. A stopped run continues where it stopped (the finished folds are skipped).

`run_all.py` runs steps 4, 5 and 6 in a row. `run_extra.py` runs the user-user experiment and step 8.

## Options (not every step has every option)
| option | meaning |
|---|---|
| `--quick` | smoke test: 1 repeat, 1 fold, a few minutes; writes to folders ending in `_quick` |
| `--rules gt7,ge7`, `--n-list 0,5,20`, `--repeats`, `--folds` | restrict what is run |
| `--train-per-user 200` | training rows per user (the large sample is used for the reported results) |
| `--out-tag _name` | suffix for the output folder, so earlier results are never overwritten |
| `--learners logreg,hgb,hgb_b,hgb_c,logreg_c` | learners (step 8, 9). `hgb_c` is the strongest regularised tree model |
| `--interact` | step 8: explicit profile x film product columns |
| `--fuzzy` | step 8: fuzzified profile representations (R2, C1, F1, F2, F3) |
| `--llm` (gpt4o, sonnet, sonnet_max or sonnet_mean) | which LLM fame columns the film features use (default `gpt4o`; `sonnet` = director mean + mean of the first 3 billed actors) |
| `--force` | recompute even if the outputs already exist |

Example, scenario C with the Sonnet columns and the fuzzified profile:

```powershell
..\.venv\Scripts\python.exe 08_both_new_models.py --fuzzy --llm sonnet --learners hgb_b,hgb_c,logreg --out-tag _fuzzy_sonnet --train-per-user 200
```

## Names
| name | inputs of the model |
|---|---|
| B0 / B1_pop_* | no model: majority guess / film popularity from training users |
| E0a / E0b / G0 | step 4: per-film raw profile / film statistics + profile / film statistics only |
| G0h, E1, E1P, E2, E3, E4 | step 5: control, + history, + profile, + GPT-4o fame, + affinity, + all |
| C1 to C6, F1 to F3 | step 7: new-film experiments (content, affinity, known ratings of the film) |
| D0n to D6 | step 8: both new (n = without LLM columns; 0 = track record, 2 = + profile, 3/4 = + revealed ratings) |
| Z\<rep\> | step 8 with `--fuzzy`: profile representation R2, C1, F1, F2 or F3 |
| R1, R2, C1, F1, F2, F3 | step 9: raw, log/circular, crisp, fuzzy sets, fuzzy c-means, fuzzy rules |

"Method" = which inputs. "Learner" = which algorithm.

The thesis summary page numbers the experiments E1 to E10 differently from these code names; see the summary page.

## LLM fame columns
The scores come from `../pipeline_run/` (`llm_liara.py` or `llm_claude.py`, API key in an environment variable,
a dry run is the default and `--yes` spends money). `finalize_partial.py` builds the final person and film files from a
partial run without calling the API. Step 10 and the `--llm sonnet` option read
`../pipeline_run/llm_claude_outputs/full_sonnet55/movies_with_sonnet_fame.csv`.

## Install (once)
From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
