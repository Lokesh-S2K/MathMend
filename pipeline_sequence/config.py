"""
math_retrieval.config
---------------------

Centralized configuration for the Hybrid Embedding project (MVP).

This file contains:
- paths and filenames
- model & vector configuration
- DB placeholders (DO NOT commit secrets)
- FAISS/index parameters
- logging defaults

Usage:
    from math_retrieval.config import *

Fill DB_PARAMS from environment variables or a secure vault before using database features.
"""

from pathlib import Path
import os
import logging
from typing import Dict, Any

# ---------------------------
# Project paths & directories
# ---------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent  # if module under math_retrieval/
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
CACHE_DIR = ROOT_DIR / "cache"

# create directories if they do not exist (safe to call at import time)
for d in (DATA_DIR, OUTPUT_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Input / output filenames (relative to DATA_DIR / OUTPUT_DIR)
RAW_DATA_CSV = DATA_DIR / "dataset_raw.csv"
CLEANED_DATA_CSV = DATA_DIR / "dataset_cleaned.csv"
GOLD_ANNOTATIONS_CSV = DATA_DIR / "gold_annotations.csv"

EMBEDDINGS_JSONL = OUTPUT_DIR / "embeddings.jsonl"   # optional per-row embeddings + metadata
STRUCT_FEATURES_NPY = OUTPUT_DIR / "structure_features.npy"
FAISS_INDEX_PATH = OUTPUT_DIR / "faiss.index"
FAISS_ID_MAP = OUTPUT_DIR / "faiss_id_map.json"      # mapping from index to problem id + metadata

# ---------------------------
# Model configuration
# ---------------------------
# Default model: lightweight sentence-transformer (MVP). Optionally swap to MathBERT/other.
# NOTE: for research/prod you can replace with a Math-aware model name (e.g., "bert-math" if available).
MODEL_NAME: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Dimension for the default model. If you switch models, update this accordingly.
VECTOR_DIMENSION: int = int(os.getenv("VECTOR_DIM", 384))

# When using reasoning embeddings separate from structure, you may choose different dim/model.
REASONING_MODEL_NAME: str = os.getenv("REASONING_MODEL", MODEL_NAME)

# ---------------------------
# FAISS / Indexing defaults
# ---------------------------
FAISS_USE_GPU: bool = bool(int(os.getenv("FAISS_USE_GPU", "0")))  # 0/1
# If you need large-scale indexing in production, consider IVF+PQ or HNSW. For MVP:
# - For <= 50k vectors, IndexFlatL2 (exact) is fine.
# - For > 50k consider IndexHNSWFlat or IndexIVFFlat.
FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "FlatL2")  # options: FlatL2, HNSW, IVF
FAISS_HNSW_M: int = int(os.getenv("FAISS_HNSW_M", "32"))
FAISS_NLIST: int = int(os.getenv("FAISS_NLIST", "100"))  # for IVF

# ---------------------------
# Canonicalization / parsing
# ---------------------------
# SymPy parsing transformations: used consistently throughout code.
SYMPY_TRANSFORMATIONS = ("standard_transformations", "implicit_multiplication_application")

# How many WL iterations to compute (for WL histogram baseline)
WL_ITERATIONS: int = int(os.getenv("WL_ITERS", "2"))

# Max number of variables to pad coefficient vectors to (for fixed length features)
MAX_VARIABLES: int = int(os.getenv("MAX_VARS", "6"))

# Feature vector sizes (baseline WL+coeff)
WL_FEATURE_SIZE: int = int(os.getenv("WL_FEATURE_SIZE", "64"))   # target fixed size for WL histogram (hash to this many bins)
COEFF_VECTOR_SIZE: int = MAX_VARIABLES * 1  # per-equation coefficient vector length (flattening rules applied later)
STRUCTURE_VECTOR_SIZE: int = int(os.getenv("STRUCTURE_VECTOR_SIZE", "256"))  # final concatenated/padded size

# ---------------------------
# DB / Storage (placeholders)
# ---------------------------
# DO NOT hardcode secrets in code - use environment variables or a secure vault.
DB_PARAMS: Dict[str, Any] = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "mathdb"),
    "user": os.getenv("DB_USER", "youruser"),
    "password": os.getenv("DB_PASSWORD", "changeme"),
    # optionally add sslmode, connect_timeout, etc.
}

# Postgres table names (example)
PG_TABLE_PROBLEMS = os.getenv("PG_TABLE_PROBLEMS", "problems")
PG_TABLE_EMBEDDINGS = os.getenv("PG_TABLE_EMBEDDINGS", "problem_embeddings")

# ---------------------------
# Logging defaults
# ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"

def configure_logging(level: int = None):
    """Configure root logger with project defaults (call once on program start)."""
    lvl = level or getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format=LOG_FORMAT)

# quick configure on import during development (comment out in production as desired)
configure_logging()

# ---------------------------
# Small helper utils
# ---------------------------

def model_info() -> dict:
    """Return a small summary of model & index config (useful for printing in logs)."""
    return {
        "model_name": MODEL_NAME,
        "vector_dim": VECTOR_DIMENSION,
        "faiss_index_type": FAISS_INDEX_TYPE,
        "wl_iterations": WL_ITERATIONS,
        "max_vars": MAX_VARIABLES,
        "structure_vector_size": STRUCTURE_VECTOR_SIZE,
    }

# End of config.py
