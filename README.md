# Mitigating the cold start problem in movie recommender systems using fuzzification of users' social network information and LLMs

MA thesis, Islamic Azad University, Qazvin. Author: Mohammad Farahani.

This repository holds the code, the small result files and the LLM fame scores of the thesis experiments.

## What the project does

A recommender cannot rely on past ratings when the user, the film, or both are new (the *cold start problem*).
The project measures, under three cold start scenarios, what helps a model rank films for a test user:

| Scenario | What is new |
|---|---|
| new user | the user has few or no ratings (only a Twitter profile) |
| new item | the film has no ratings (only director and cast) |
| both new | the user and the film are new |

It compares film popularity, revealed ratings, the raw Twitter profile, **fuzzified** profiles (three fuzzifiers and crisp / raw controls), and **LLM fame scores** of directors and actors (Claude Sonnet 5.5, with GPT-4o for comparison).

Metric: per-user AUC (Area Under the ROC Curve) with a 95% bootstrap interval over test users, for two like rules (rating > 7 and rating >= 7). Test users are never in the training data.

## Layout

| Folder | Content |
|---|---|
| `cold_start_lab/` | the experiments (numbered steps `01_` to `12_`), see `cold_start_lab/README.md` |
| `pipeline_run/` | the LLM scoring scripts (`llm_liara.py`, `llm_claude.py`) and the earlier GPT-4o pipeline |
| `CSP Dataset/` | the CSP dataset (MIT licence, copied from the original repository) |
| `Director fame rating/` | ratings of 40 directors by 4 human raters, used to check the LLM scores |

## How to run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
cd cold_start_lab
..\.venv\Scripts\python.exe 01_prepare_data.py
..\.venv\Scripts\python.exe 02_make_splits.py
..\.venv\Scripts\python.exe 03_baselines.py
```

Each later step writes its tables to `cold_start_lab/outputs/`. Most scripts accept `--quick` for a short smoke test. The LLM scoring scripts need an API key set as an environment variable (never write keys in the code).

## Data and licences

- Code: MIT (see `LICENSE`).
- CSP dataset: MIT, taken from <https://github.com/lynchblue/movie-rating-dataset>. Please cite:
  Herce-Zelaya, J., Porcel, C., Tejeda-Lorente, A., Bernabé-Moreno, J., Herrera-Viedma, E. (2023).
  *Introducing CSP Dataset: A Dataset Optimized for the Study of the Cold Start Problem in Recommender Systems.* Information, 14(1), 19.
- Large intermediate files and the Python environment are not included (see `.gitignore`).
- The CSP data was collected from public web sources (film ratings and Twitter profiles). Check the terms of those sources before reusing the data.

## Status

Work in progress. The results shown in the thesis are the ones in `cold_start_lab/outputs/*/final_report.txt`.
