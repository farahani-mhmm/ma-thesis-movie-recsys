"""Film attributes used for content-based features (director/actor overlap and the two GPT-4o fame columns)."""
import numpy as np
import pandas as pd

INVALID_NAMES = {"NO_CREDITED_ACTORS", "Varios", "NaN", "", "N/A", "Unknown", "None", "No Credit", "Uncredited"}
MAX_ACTORS = 8       # the first cast members carry the signal (same limit as the old pipeline)


def tokens(cell, limit=None):
    if pd.isna(cell):
        return []
    out = [t.strip() for t in str(cell).split("|")]
    out = [t for t in out if t and t not in INVALID_NAMES]
    return out[:limit] if limit else out


def share_matrix(token_lists):
    """M x M float matrix: 1 when two different films share at least one token (director or actor)."""
    vocab, rows, cols = {}, [], []
    for i, toks in enumerate(token_lists):
        for t in set(toks):
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
    A = np.zeros((len(token_lists), max(len(vocab), 1)), dtype=np.float32)
    if rows:
        A[rows, cols] = 1.0
    S = (A @ A.T) > 0
    np.fill_diagonal(S, False)
    return S.astype(np.float32)


class Films:
    pass


def build_films(movies, film_ids):
    mv = movies.drop_duplicates("movie_id").set_index("movie_id").loc[film_ids]
    F = Films()
    fd = pd.to_numeric(mv["director_fame_score_llm"], errors="coerce")
    fa = pd.to_numeric(mv["actors_fame_score_llm"], errors="coerce")
    F.mean_fame_dir = float(fd.mean())
    F.mean_fame_act = float(fa.mean())
    F.fame_dir = fd.fillna(F.mean_fame_dir).to_numpy(dtype=np.float32)
    F.fame_act = fa.fillna(F.mean_fame_act).to_numpy(dtype=np.float32)
    F.dir_share = share_matrix([tokens(x) for x in mv["directors"]])
    F.act_share = share_matrix([tokens(x, MAX_ACTORS) for x in mv["actors"]])
    return F
