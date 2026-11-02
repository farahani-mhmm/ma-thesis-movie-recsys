"""Blocks that describe how well a candidate film fits the TASTE of a user, built from the user's known ratings and the
content of the films (TMDB genres and text). Used by step 14 with --taste.

    E    text vector of the film: TF-IDF of overview + keywords + tagline, reduced to 20 SVD dimensions (unsupervised,
         fitted on the text of all films, no rating is used)
    GR   genre taste, RAW numbers      mean / max / min of (user's like rate in the film's genres - user's overall like rate)
                                       and the mean log of how many known films the user has in these genres
    GC   genre taste, CRISP version    each genre is -1 / 0 / +1 with hard cut-offs, plus the share of genres with enough evidence
    GF   genre taste, FUZZY version    each genre has membership degrees "likes" / "dislikes" and "enough evidence" (linear shoulders),
                                       averaged over the film's genres.  Same information as GR and GC, only the representation differs.
    SIM  similar films                 the user's known ratings of the 10 films most similar to the candidate (cosine of genres + text),
                                       weighted average of (liked - user's like rate), and the mean similarity of these 10

When the candidate film itself is among the user's known ratings (training rows of scenario B), its own rating is taken
out of every count (leave-one-out), so a training row never sees its own label.
"""
import numpy as np
import pandas as pd

GENRE_SMOOTH = 5.0
EVID_CRISP = 5.0
EVID_FULL = 10.0
CUT_CRISP = 0.10
FUZ_LO, FUZ_HI = 0.05, 0.20       # |deviation| <= 0.05: membership 0, >= 0.20: membership 1
TOPK = 10


def text_vectors(tmdb, film_ids, dims=20, seed=7):
    """E block: TF-IDF + truncated SVD of overview, keywords and tagline. Rows follow film_ids; L2-normalised."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    t = tmdb.drop_duplicates("csp_id").set_index("csp_id").reindex(film_ids)
    docs = (t["overview"].fillna("") + " " + t["keywords"].fillna("").str.replace("|", " ", regex=False) + " "
            + t["tagline"].fillna("")).tolist()
    X = TfidfVectorizer(stop_words="english", min_df=3, max_df=0.5, sublinear_tf=True).fit_transform(docs)
    Z = TruncatedSVD(n_components=dims, random_state=seed).fit_transform(X)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    Z[(t["overview"].fillna("") == "").to_numpy()] = 0.0
    return Z.astype(np.float32)


def deep_vectors(emb, ids, film_ids, dims=20, seed=7):
    """E block from precomputed sentence embeddings (step 18): PCA to `dims` numbers, L2-normalised, rows follow film_ids.
    Same 20-number size as the TF-IDF vector so the two are comparable; films without text keep a zero vector."""
    from sklearn.decomposition import PCA
    pos = pd.Series(np.arange(len(ids)), index=ids)
    X = emb[pos.reindex(film_ids).to_numpy()]
    Z = PCA(n_components=dims, random_state=seed).fit_transform(X)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
    Z[~X.any(axis=1)] = 0.0
    return Z.astype(np.float32)


def similarity_matrix(genres, E):
    """Cosine similarity of [genre vector, text vector] between all films; the diagonal is set to 0."""
    G = genres / np.maximum(np.linalg.norm(genres, axis=1, keepdims=True), 1e-9)
    V = np.hstack([G, E])
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    S = (V @ V.T).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    return S


def taste_blocks(genres, S, cand, ruf, lu, h_u, g_rate, smooth_h):
    """GR, GC, GF, SIM for the candidate films `cand` of one user.

    genres: M x G multi-hot; S: M x M similarity; ruf: float mask of the user's known films; lu: liked values on known films
    (0 elsewhere); h_u: the user's smoothed like rate (including the candidate when it is a known film)."""
    n_c = len(cand)
    rev = np.flatnonzero(ruf > 0)
    Gc = genres[cand]                                          # C x G
    cnt = np.maximum(Gc.sum(axis=1), 1.0)
    own = (ruf[cand] > 0).astype(np.float32)                   # candidate is itself a known film (training rows of B)
    lu_c = lu[cand]
    # user's like rate without the candidate
    n_all = float(ruf.sum())
    h_c = (float(lu.sum()) - own * lu_c + smooth_h * g_rate) / (n_all - own + smooth_h)
    if len(rev):
        Gr = genres[rev]
        n_g = Gr.sum(axis=0)[None, :] - Gc * own[:, None]                    # C x G counts (candidate removed)
        l_g = (Gr * lu[rev][:, None]).sum(axis=0)[None, :] - Gc * (own * lu_c)[:, None]
    else:
        n_g = np.zeros_like(Gc)
        l_g = np.zeros_like(Gc)
    tau = (l_g + GENRE_SMOOTH * h_c[:, None]) / (n_g + GENRE_SMOOTH)
    d = tau - h_c[:, None]                                     # deviation of the genre like rate from the user's own rate
    m = Gc > 0
    # ---- raw
    mean_d = (Gc * d).sum(axis=1) / cnt
    max_d = np.where(m, d, -np.inf).max(axis=1)
    min_d = np.where(m, d, np.inf).min(axis=1)
    max_d[~np.isfinite(max_d)] = 0.0
    min_d[~np.isfinite(min_d)] = 0.0
    mean_logn = (Gc * np.log1p(n_g)).sum(axis=1) / cnt
    GR = np.column_stack([mean_d, max_d, min_d, mean_logn])
    # ---- crisp
    ev = n_g >= EVID_CRISP
    s = np.where(ev & (d > CUT_CRISP), 1.0, np.where(ev & (d < -CUT_CRISP), -1.0, 0.0))
    GC = np.column_stack([(Gc * s).sum(axis=1) / cnt, (Gc * ev).sum(axis=1) / cnt])
    # ---- fuzzy
    e = np.minimum(1.0, n_g / EVID_FULL)
    mu_like = np.clip((d - FUZ_LO) / (FUZ_HI - FUZ_LO), 0.0, 1.0)
    mu_dis = np.clip((-d - FUZ_LO) / (FUZ_HI - FUZ_LO), 0.0, 1.0)
    GF = np.column_stack([(Gc * e * mu_like).sum(axis=1) / cnt, (Gc * e * mu_dis).sum(axis=1) / cnt, (Gc * e).sum(axis=1) / cnt])
    # ---- similar films
    if len(rev):
        Sm = S[np.ix_(cand, rev)]                              # C x R
        pos = np.searchsorted(rev, cand)
        hit = (pos < len(rev)) & (rev[np.minimum(pos, len(rev) - 1)] == cand)
        Sm[np.flatnonzero(hit), pos[hit]] = -1.0               # the candidate is not its own neighbour
        k = min(TOPK, Sm.shape[1])
        idx = np.argpartition(-Sm, k - 1, axis=1)[:, :k]
        w = np.take_along_axis(Sm, idx, axis=1)
        w = np.clip(w, 0.0, None)
        resid = lu[rev][idx] - h_c[:, None]
        sim_score = (w * resid).sum(axis=1) / (w.sum(axis=1) + 1e-3)
        SIM = np.column_stack([sim_score, w.mean(axis=1)])
    else:
        SIM = np.zeros((n_c, 2))
    return {k: v.astype(np.float32) for k, v in (("GR", GR), ("GC", GC), ("GF", GF), ("SIM", SIM))}
