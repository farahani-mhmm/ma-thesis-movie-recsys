"""
STEP 18 - deep text vectors of the films (sentence embeddings), computed once and saved.

Text of a film = overview + keywords + tagline from TMDB (same text as the TF-IDF vector E).
The pretrained model (all-MiniLM-L6-v2, 384 numbers per film) never sees a rating: it is a fixed text encoder.

Needs torch and sentence-transformers, so run it with the separate environment:
    ..\\.venv_dl\\Scripts\\python.exe 18_deep_text_embed.py
Output: outputs/16_tmdb/film_text_emb.npy (rows follow film_text_emb_ids.csv)
"""
import numpy as np
import pandas as pd

from config import OUT_DIR

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
src = OUT_DIR / "16_tmdb" / "movies_tmdb.csv"
t = pd.read_csv(src).drop_duplicates("csp_id")
docs = (t["overview"].fillna("") + ". " + t["keywords"].fillna("").str.replace("|", ", ", regex=False) + ". "
        + t["tagline"].fillna("")).tolist()
empty = (t["overview"].fillna("") == "").to_numpy()

from sentence_transformers import SentenceTransformer

enc = SentenceTransformer(MODEL, device="cpu")
Z = enc.encode(docs, batch_size=64, show_progress_bar=False, normalize_embeddings=True).astype(np.float32)
Z[empty] = 0.0                                    # films without any text get a zero vector, like in the TF-IDF block
np.save(OUT_DIR / "16_tmdb" / "film_text_emb.npy", Z)
t[["csp_id"]].to_csv(OUT_DIR / "16_tmdb" / "film_text_emb_ids.csv", index=False)
print("films:", len(t), "| without text:", int(empty.sum()), "| shape:", Z.shape, "| model:", MODEL)
