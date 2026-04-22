"""
Sentence-transformer embeddings for free-text clinical fields.

Encodes MIMIC-IV chief complaints and eICU admission diagnoses using
all-mpnet-base-v2, then merges the resulting 768-dim vectors into each dataset.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM   = 768
BATCH_SIZE  = 64
DEVICE      = "mps"  # set to "cuda" or "cpu" if needed


def encode(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, show_progress_bar=True, device=DEVICE, batch_size=BATCH_SIZE)


def embed_mimic(df: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    complaints = df["chiefcomplaint"].fillna("unknown").str.upper().tolist()
    emb = encode(complaints, model)
    emb_df = pd.DataFrame(emb, columns=[f"cc_{i}" for i in range(EMBED_DIM)],
                          index=df.index)
    return pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)


def embed_eicu(df: pd.DataFrame, model: SentenceTransformer) -> pd.DataFrame:
    diagnoses = df["admission_dx"].fillna("unknown").str.upper().tolist()
    emb = encode(diagnoses, model)
    emb_df = pd.DataFrame(emb, columns=[f"cc_{i}" for i in range(EMBED_DIM)],
                          index=df.index)
    return pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)


if __name__ == "__main__":
    from prepare_features import load_mimic, load_eicu

    print(f"Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    mimic = load_mimic()
    mimic_emb = embed_mimic(mimic, model)
    print(f"MIMIC with embeddings: {mimic_emb.shape}")

    eicu = load_eicu()
    eicu_emb = embed_eicu(eicu, model)
    print(f"eICU  with embeddings: {eicu_emb.shape}")
