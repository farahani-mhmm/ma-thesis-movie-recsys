"""Step 1: read the CSP files, check them, clean them, add the two "liked" labels, save clean copies.

Needs only pandas and numpy (the thesis .venv already has them). No API calls, nothing is installed.

Outputs (in outputs/01_data/):
    ratings_clean.csv   user_id, movie_id, rate, date, liked_gt7, liked_ge7
    users_clean.csv     user_id + the 12 profile columns (twitterName is dropped)
    movies_clean.csv    the film table, unchanged apart from renaming id -> movie_id
    data_report.json    every count and check, machine readable
    data_report.txt     the same, for people
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

from config import (CSV_ENCODING, DATA_DIR, LIKE_RULES, MIN_RATINGS_FOR_EVAL, N_GRID, OUT_DIR,
                    PROFILE_COLS, ensure_dir)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quantiles(series):
    q = series.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).round(1)
    return {"min": float(q.iloc[0]), "p05": float(q.iloc[1]), "p25": float(q.iloc[2]), "median": float(q.iloc[3]),
            "p75": float(q.iloc[4]), "p95": float(q.iloc[5]), "max": float(q.iloc[6])}


def main():
    files = {name: DATA_DIR / name for name in ("ratings.csv", "movies.csv", "user_profiles.csv")}
    for name, path in files.items():
        if not path.exists():
            sys.exit(f"Missing input file: {path}")

    report = {"inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()}}

    ratings = pd.read_csv(files["ratings.csv"], encoding=CSV_ENCODING)
    movies = pd.read_csv(files["movies.csv"], encoding=CSV_ENCODING)
    users = pd.read_csv(files["user_profiles.csv"], encoding=CSV_ENCODING)

    # --- column checks -------------------------------------------------------------------------
    need = {"ratings": {"id", "rate", "movie_id", "date"}, "movies": {"id"}, "users": {"id", *PROFILE_COLS}}
    for label, df in (("ratings", ratings), ("movies", movies), ("users", users)):
        missing = need[label] - set(df.columns)
        if missing:
            sys.exit(f"{label} is missing columns: {sorted(missing)}")

    ratings = ratings.rename(columns={"id": "user_id"})
    users = users.rename(columns={"id": "user_id"})
    movies = movies.rename(columns={"id": "movie_id"})

    report["raw_rows"] = {"ratings": int(len(ratings)), "movies": int(len(movies)), "users": int(len(users))}

    # --- ratings: types, duplicates ------------------------------------------------------------
    ratings["date"] = pd.to_datetime(ratings["date"], errors="coerce")
    report["ratings_bad_dates"] = int(ratings["date"].isna().sum())
    if not ratings["rate"].between(1, 10).all():
        sys.exit("Some ratings are outside 1..10")
    ratings["rate"] = ratings["rate"].astype("int64")

    dup_pairs = int(ratings.duplicated(["user_id", "movie_id"], keep=False).sum())
    ratings = (ratings.sort_values(["user_id", "movie_id", "date"], kind="mergesort")
               .drop_duplicates(["user_id", "movie_id"], keep="last"))
    report["ratings_duplicate_user_movie_rows"] = dup_pairs
    report["rows_after_dedup"] = int(len(ratings))

    # --- referential integrity -----------------------------------------------------------------
    unknown_movie_rows = int((~ratings["movie_id"].isin(movies["movie_id"])).sum())
    report["ratings_with_unknown_movie"] = unknown_movie_rows
    if unknown_movie_rows:
        ratings = ratings[ratings["movie_id"].isin(movies["movie_id"])]

    # --- users: only users that have a profile (all experiments use the same users) ------------
    if users["user_id"].duplicated().any():
        sys.exit("Duplicate user ids in user_profiles.csv")
    users_null = int(users[PROFILE_COLS].isna().sum().sum())
    report["profile_null_cells"] = users_null
    has_profile = ratings["user_id"].isin(users["user_id"])
    report["users_in_ratings_without_profile"] = int(ratings.loc[~has_profile, "user_id"].nunique())
    report["rows_dropped_no_profile"] = int((~has_profile).sum())
    ratings = ratings[has_profile].copy()
    users = users[users["user_id"].isin(ratings["user_id"])].copy()
    report["profile_users_without_ratings"] = int((~pd.read_csv(files["user_profiles.csv"], encoding=CSV_ENCODING)["id"]
                                                  .isin(ratings["user_id"])).sum())

    # --- labels --------------------------------------------------------------------------------
    for key, (op, thr) in LIKE_RULES.items():
        flag = ratings["rate"] > thr if op == ">" else ratings["rate"] >= thr
        ratings[f"liked_{key}"] = flag.astype("int8")

    # --- summaries -----------------------------------------------------------------------------
    n_rows = int(len(ratings))
    report["clean"] = {
        "rows": n_rows,
        "users": int(ratings["user_id"].nunique()),
        "films": int(ratings["movie_id"].nunique()),
        "density": float(n_rows / (ratings["user_id"].nunique() * ratings["movie_id"].nunique())),
        "mean_rate": float(ratings["rate"].mean()),
        "like_rate": {key: float(ratings[f"liked_{key}"].mean()) for key in LIKE_RULES},
    }
    per_user = ratings.groupby("user_id").size()
    per_film = ratings.groupby("movie_id").size()
    report["ratings_per_user"] = quantiles(per_user)
    report["ratings_per_film"] = quantiles(per_film)
    report["users_with_fewer_than_min_ratings"] = {
        "min_required": int(MIN_RATINGS_FOR_EVAL),
        "count": int((per_user < MIN_RATINGS_FOR_EVAL).sum()),
    }
    # AUC per user is undefined when a user likes everything or nothing.
    report["users_without_both_classes"] = {
        key: int(((ratings.groupby("user_id")[f"liked_{key}"].mean() == 0)
                  | (ratings.groupby("user_id")[f"liked_{key}"].mean() == 1)).sum())
        for key in LIKE_RULES
    }
    report["n_grid"] = N_GRID

    # --- save ----------------------------------------------------------------------------------
    out = ensure_dir(OUT_DIR / "01_data")
    ratings_out = ratings[["user_id", "movie_id", "rate", "date"] + [f"liked_{k}" for k in LIKE_RULES]]
    ratings_out.sort_values(["user_id", "movie_id"]).to_csv(out / "ratings_clean.csv", index=False, encoding=CSV_ENCODING)
    users[["user_id"] + PROFILE_COLS].sort_values("user_id").to_csv(out / "users_clean.csv", index=False, encoding=CSV_ENCODING)
    movies.sort_values("movie_id").to_csv(out / "movies_clean.csv", index=False, encoding=CSV_ENCODING)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return str(o)

    (out / "data_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=default), encoding="utf-8")

    c = report["clean"]
    lines = [
        "STEP 1 - DATA REPORT",
        f"raw rows: ratings={report['raw_rows']['ratings']}, films={report['raw_rows']['movies']}, profiles={report['raw_rows']['users']}",
        f"duplicate (user, film) rows: {report['ratings_duplicate_user_movie_rows']}",
        f"ratings dropped because the user has no profile: {report['rows_dropped_no_profile']} "
        f"({report['users_in_ratings_without_profile']} users)",
        f"profile users without ratings: {report['profile_users_without_ratings']}",
        f"profile null cells: {report['profile_null_cells']}",
        "",
        f"CLEAN DATA: {c['rows']} ratings, {c['users']} users, {c['films']} films, density {c['density']:.3f}",
        f"mean rating {c['mean_rate']:.2f}",
        f"liked rate: rate > 7 = {c['like_rate']['gt7']:.3f} | rate >= 7 = {c['like_rate']['ge7']:.3f}",
        f"ratings per user: {report['ratings_per_user']}",
        f"ratings per film: {report['ratings_per_film']}",
        f"users with fewer than {MIN_RATINGS_FOR_EVAL} ratings (too few for the largest N): "
        f"{report['users_with_fewer_than_min_ratings']['count']}",
        f"users who like all or none (per-user AUC undefined): {report['users_without_both_classes']}",
        "",
        f"files written to: {out}",
    ]
    text = "\n".join(lines)
    (out / "data_report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
