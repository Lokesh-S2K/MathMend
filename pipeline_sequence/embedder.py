"""
math_retrieval.modules.embedder
--------------------------------

Phase 2 — Reasoning & text embedding utilities

Responsibilities:
- Load a sentence-transformer / Math-aware transformer model for reasoning/text embeddings.
- Provide efficient batched encoding, optional GPU use, and L2/cosine normalization.
- Provide small "text builder" utilities to create the input text for the model from problem + fingerprint.
- Save/load embeddings to disk (numpy / jsonl), and support incremental caching (avoid re-embedding).
- Small CLI demo function to show end-to-end behavior.

Notes:
- For structure-based retrieval we prefer the deterministic structure_vector (features module).
  Use this embedder for: (a) reasoning embeddings stored alongside structure vectors for explanations,
  or (b) a fallback transformer-based embedding during prototyping.
"""

from typing import List, Iterable, Optional, Dict, Any, Tuple
import os
import json
import logging
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from tqdm import tqdm

# Hugging Face / sentence-transformers
from sentence_transformers import SentenceTransformer

# local config
try:
    from pipeline_sequence.config import MODEL_NAME, REASONING_MODEL_NAME, VECTOR_DIMENSION, EMBEDDINGS_JSONL, OUTPUT_DIR
except Exception:
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    REASONING_MODEL_NAME = MODEL_NAME
    VECTOR_DIMENSION = 384
    OUTPUT_DIR = Path("output")
    EMBEDDINGS_JSONL = OUTPUT_DIR / "embeddings.jsonl"

# logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------
# Model loader (singleton-like)
# ---------------------------

@lru_cache(maxsize=2)
def load_model(model_name: Optional[str] = None, use_auth_token: Optional[str] = None) -> SentenceTransformer:
    """
    Load and cache the SentenceTransformer model.
    - model_name: HF model (default from config)
    - use_auth_token: optional token for private models
    """
    mn = model_name or REASONING_MODEL_NAME or MODEL_NAME
    logger.info("Loading embedding model: %s", mn)
    # The SentenceTransformer constructor will choose GPU if available (PyTorch)
    model = SentenceTransformer(mn, device=None)  # device inference is automatic
    return model


# ---------------------------
# Input formatting helpers
# ---------------------------

def build_text_for_embedding(problem_text: str, fingerprint: Optional[str] = None, max_len: int = 512) -> str:
    """
    Build a single input string for the transformer embedding model.
    Keep it deterministic and compact:
      "Problem: {problem_text} Equations: {fingerprint}"
    We optionally clip if too long.
    """
    pt = problem_text.strip() if isinstance(problem_text, str) else ""
    eq = fingerprint.strip() if isinstance(fingerprint, str) else ""
    if eq:
        combined = f"Problem: {pt} Equations: {eq}"
    else:
        combined = f"Problem: {pt}"
    # optional simple trimming
    if len(combined) > max_len:
        return combined[: max_len - 3] + "..."
    return combined


# ---------------------------
# Embedding utilities
# ---------------------------

def encode_texts(texts: Iterable[str],
                 model_name: Optional[str] = None,
                 batch_size: int = 64,
                 normalize: bool = True,
                 show_progress: bool = True) -> np.ndarray:
    """
    Encode a list/iterable of strings into a numpy array of shape (N, D).
    - batch_size controls batching for GPU/CPU efficiency.
    - normalize: if True, L2-normalize each vector (useful for cosine search).
    - Returns float32 numpy array.
    """
    model = load_model(model_name)
    encs = []
    it = texts
    if show_progress:
        it = tqdm(texts, desc="Embedding texts", unit="rows")
    # accumulate in batches
    buffer = []
    for s in it:
        buffer.append(s)
        if len(buffer) >= batch_size:
            arr = model.encode(buffer, convert_to_numpy=True, show_progress_bar=False)
            encs.append(arr)
            buffer = []
    # final batch
    if buffer:
        arr = model.encode(buffer, convert_to_numpy=True, show_progress_bar=False)
        encs.append(arr)
    if not encs:
        return np.zeros((0, VECTOR_DIMENSION), dtype=np.float32)
    out = np.vstack(encs).astype(np.float32)
    if normalize and out.shape[0] > 0:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out = out / norms
    return out


def encode_dataframe(df: pd.DataFrame,
                     problem_col: str = "clean_text",
                     fingerprint_col: str = "symbolic_fingerprint",
                     output_col: str = "reasoning_vector",
                     model_name: Optional[str] = None,
                     batch_size: int = 64,
                     normalize: bool = True,
                     text_builder=build_text_for_embedding) -> pd.DataFrame:
 
    df = df.copy()
    texts = []
    for _, row in df.iterrows():
        pt = row.get(problem_col, "") or ""
        fp = row.get(fingerprint_col, None)
        texts.append(text_builder(pt, fp))
    vectors = encode_texts(texts, model_name=model_name, batch_size=batch_size, normalize=normalize)
    df[output_col] = list(vectors)
    return df



def save_embeddings_jsonl(records: List[Dict[str, Any]], out_path: Optional[str] = None):

    out = out_path or str(EMBEDDINGS_JSONL)
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf8") as fh:
        for rec in records:
            # ensure embedding is serializable
            emb = rec.get("embedding")
            if isinstance(emb, (list, tuple)):
                emb_list = emb
            elif hasattr(emb, "tolist"):
                emb_list = emb.tolist()
            else:
                emb_list = []
            rec_copy = dict(rec)
            rec_copy["embedding"] = emb_list
            fh.write(json.dumps(rec_copy) + "\n")
    logger.info("Saved %d embeddings to %s", len(records), out)


def load_embeddings_jsonl(path: str) -> List[Dict[str, Any]]:
    recs = []
    p = Path(path)
    if not p.exists():
        return recs
    with p.open("r", encoding="utf8") as fh:
        for line in fh:
            j = json.loads(line)
            if "embedding" in j and isinstance(j["embedding"], list):
                j["embedding"] = np.array(j["embedding"], dtype=np.float32)
            recs.append(j)
    return recs


def build_embedding_records(df: pd.DataFrame,
                            id_col: str = "id",
                            vector_col: str = "reasoning_vector",
                            fingerprint_col: str = "symbolic_fingerprint",
                            metadata_cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:

    metadata_cols = metadata_cols or []
    recs = []
    for _, r in df.iterrows():
        eid = r.get(id_col)
        emb = r.get(vector_col)
        if emb is None:
            continue
        meta = {c: r.get(c) for c in metadata_cols}
        recs.append({
            "id": eid,
            "embedding": emb,
            "fingerprint": r.get(fingerprint_col),
            "metadata": meta
        })
    return recs


def embed_with_cache(df: pd.DataFrame,
                     id_col: str = "id",
                     problem_col: str = "clean_text",
                     fingerprint_col: str = "symbolic_fingerprint",
                     cache_path: Optional[str] = None,
                     model_name: Optional[str] = None,
                     batch_size: int = 64,
                     normalize: bool = True) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Embed DataFrame rows with caching support:
      - If cache_path exists, load existing embeddings and only embed missing ids.
      - Returns: (df_with_vectors, list_of_new_records_saved)
    """
    cache_path = cache_path or str(EMBEDDINGS_JSONL)
    cache_records = load_embeddings_jsonl(cache_path) if Path(cache_path).exists() else []
    existing_map = {r["id"]: r for r in cache_records if "id" in r}

    # Prepare list of texts to embed for missing rows
    to_embed_rows = []
    to_embed_ids = []
    for _, r in df.iterrows():
        rid = r.get(id_col)
        if rid in existing_map:
            continue
        to_embed_rows.append(build_text_for_embedding(r.get(problem_col, ""), r.get(fingerprint_col)))
        to_embed_ids.append(rid)

    new_records = []
    if to_embed_rows:
        vectors = encode_texts(to_embed_rows, model_name=model_name, batch_size=batch_size, normalize=normalize)
        for rid, vec in zip(to_embed_ids, vectors):
            rec = {"id": rid, "embedding": vec, "fingerprint": None}  # fingerprint can be filled later
            new_records.append(rec)
        # append new records to file (append mode)
        all_records = cache_records + new_records
        # write back
        save_embeddings_jsonl(all_records, cache_path)
    else:
        logger.info("All rows already present in cache at %s", cache_path)

    # Build df with reasoning_vector column populated (from cache + new)
    id_to_vec = {r["id"]: np.array(r["embedding"], dtype=np.float32) for r in (cache_records + new_records) if "id" in r}
    df = df.copy()
    vectors = []
    for _, r in df.iterrows():
        vec = id_to_vec.get(r.get(id_col))
        vectors.append(vec)
    df["reasoning_vector"] = vectors
    return df, new_records


# ---------------------------
# Demo / CLI
# ---------------------------

def demo_encode_sample():
    """
    Quick demo that loads small sample DataFrame and encodes.
    Intended to be run interactively during development.
    """
    sample = [
        {"id": 1, "clean_text": "A is 5 years older than B. A + B = 25.", "symbolic_fingerprint": "v1 - v2 - 5; v1 + v2 - 25"},
        {"id": 2, "clean_text": "Boat goes 30 km downstream in 2 hours and back in 3 hours.", "symbolic_fingerprint": "2*v1 + 2*v2 - 30; 3*v1 - 3*v2 - 30"},
    ]
    df = pd.DataFrame(sample)
    df_enc = encode_dataframe(df, problem_col="clean_text", fingerprint_col="symbolic_fingerprint", output_col="reasoning_vector")
    print("Encoded vectors shape:", np.vstack(df_enc["reasoning_vector"].to_list()).shape)
    recs = build_embedding_records(df_enc, id_col="id", vector_col="reasoning_vector", fingerprint_col="symbolic_fingerprint", metadata_cols=["clean_text"])
    save_embeddings_jsonl(recs, str(OUTPUT_DIR / "demo_embeddings.jsonl"))
    print("Wrote demo embeddings to output/demo_embeddings.jsonl")


if __name__ == "__main__":
    demo_encode_sample()
