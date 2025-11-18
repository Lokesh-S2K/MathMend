"""
math_retrieval.modules.features
--------------------------------

Phase 2 — Feature Engineering for Structure Embeddings

Responsibilities:
- Build operator/operand graphs from canonicalized SymPy equation strings.
- Compute Weisfeiler-Lehman (WL) subtree counts via iterative relabeling.
- Convert WL counts into a fixed-length histogram vector (hash-binned).
- Build flattened & padded coefficient vectors for linear systems.
- Concatenate features into a fixed-length "structure vector".
- Provide batch utilities for DataFrame processing and saving results.

Notes:
- This is a deterministic, interpretable baseline. Later you can replace/augment
  these features with a learned GNN embedding or a projection/ML model.
"""

from typing import Dict, List, Tuple, Any, Optional
from collections import Counter, defaultdict
import hashlib
import math
import json
import logging

import numpy as np
import pandas as pd
import networkx as nx
from sympy import Symbol
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# local config (assumes package import)
try:
    from pipeline_sequence.config import WL_ITERATIONS, WL_FEATURE_SIZE, MAX_VARIABLES, STRUCTURE_VECTOR_SIZE
except Exception:
    # safe defaults if config import fails
    WL_ITERATIONS = 2
    WL_FEATURE_SIZE = 64
    MAX_VARIABLES = 6
    STRUCTURE_VECTOR_SIZE = 256

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# sympy parse transformations (used only if needed)
_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)


# ---------------------------
# Helper functions
# ---------------------------

def _hash_to_bin(label: str, bins: int) -> int:
    """Stable hash of a label string mapped to [0, bins-1]."""
    h = hashlib.sha256(label.encode("utf8")).hexdigest()
    # take last 8 hex chars for an int, mod bins
    val = int(h[-8:], 16)
    return val % bins


def _safe_float(x: Any) -> float:
    """Convert sympy Number or other numeric to float safely."""
    try:
        return float(x)
    except Exception:
        try:
            # attempt numeric evaluation
            return float(x.evalf())
        except Exception:
            return 0.0


# ---------------------------
# Expression -> Graph
# ---------------------------

def expr_to_graph(expr_str: str) -> Optional[nx.Graph]:
    """
    Build a small operator/operand graph for a canonical expression string.
    Expect expr_str as e.g. "2*v1 + 3*v2 - 30" (a SymPy-style str).
    Graph nodes are labeled with 'type' in {'op', 'var', 'const'} and 'tok' the token string.

    Returns: networkx.Graph or None if parse fails.
    """
    if not isinstance(expr_str, str) or expr_str.strip() == "":
        return None

    try:
        expr = parse_expr(expr_str, transformations=_TRANSFORMATIONS)
    except Exception as e:
        logger.debug("expr_to_graph parse_expr failed for '%s': %s", expr_str, e)
        return None

    G = nx.Graph()
    node_id = 0

    # recursive traversal to produce nodes
    def add_node(tok_type: str, tok_str: str) -> int:
        nonlocal node_id
        nid = f"n{node_id}"
        G.add_node(nid, type=tok_type, tok=tok_str)
        node_id += 1
        return nid

    def traverse(sym):
        """
        Traverse SymPy expression, create nodes and edges.
        For atomic symbols (Symbol), create 'var' node.
        For numbers, create 'const' node.
        For operations (Add, Mul, Pow, etc.), create 'op' node and connect to children.
        """
        from sympy import Add, Mul, Pow, Integer, Rational

        if sym.is_Symbol:
            nid = add_node("var", str(sym))
            return nid
        if sym.is_Number:
            nid = add_node("const", str(sym))
            return nid

        # for named ops
        if sym.func == Add:
            op_node = add_node("op", "+")
            for arg in sym.args:
                child = traverse(arg)
                G.add_edge(op_node, child)
            return op_node

        if sym.func == Mul:
            op_node = add_node("op", "*")
            for arg in sym.args:
                child = traverse(arg)
                G.add_edge(op_node, child)
            return op_node

        if sym.func == Pow:
            op_node = add_node("op", "^")
            base = traverse(sym.args[0])
            exp = traverse(sym.args[1])
            G.add_edge(op_node, base)
            G.add_edge(op_node, exp)
            return op_node

        # fallback for other functions (sin, cos, etc.) - treat as op with name
        op_node = add_node("op", str(sym.func))
        for arg in sym.args:
            child = traverse(arg)
            G.add_edge(op_node, child)
        return op_node

    try:
        root = traverse(expr)
    except Exception as e:
        logger.debug("expr_to_graph traversal failed for '%s': %s", expr_str, e)
        return None

    return G


# ---------------------------
# WL Subtree counts (lightweight)
# ---------------------------

def weisfeiler_lehman_subtree_counts(G: nx.Graph, iterations: int = WL_ITERATIONS) -> Counter:
    """
    Compute WL subtree label counts for a graph G.
    Implementation idea:
      - Initial label = node type + tok (e.g., 'var:v1' or 'op:+').
      - For each iteration, for each node, compute new label = hash(current_label + sorted(neighbor_labels))
      - Tally counts of labels observed across all iterations (multiset).
    Returns: Counter {label_str: count}
    """
    if G is None or len(G) == 0:
        return Counter()

    # initial labels
    labels = {}
    for n, data in G.nodes(data=True):
        lab = f"{data.get('type','unk')}:{data.get('tok','')}"
        labels[n] = lab

    counts = Counter()
    # seed counts: count initial labels
    counts.update(labels.values())

    for it in range(iterations):
        new_labels = {}
        for n in G.nodes():
            neigh_labels = sorted([labels[nb] for nb in G.neighbors(n)])
            # create a composite label and hash it to keep strings short
            composite = labels[n] + "|" + "|".join(neigh_labels)
            # short stable label: use sha1 hex digest truncated
            h = hashlib.sha1(composite.encode("utf8")).hexdigest()[:16]
            new_label = f"wl{it}:{h}"
            new_labels[n] = new_label
        labels = new_labels
        counts.update(labels.values())

    return counts


def wl_counts_to_histogram(counts: Counter, bins: int = WL_FEATURE_SIZE) -> np.ndarray:
    """
    Convert a Counter of WL labels into a fixed-length hashed histogram.
    Uses stable label hashing to bin counts into 'bins' buckets.
    Returns numpy array shape (bins,) of float counts (optionally normalized later).
    """
    hist = np.zeros(bins, dtype=float)
    if not counts:
        return hist
    for label, cnt in counts.items():
        idx = _hash_to_bin(label, bins)
        hist[idx] += cnt
    # normalize by L2 norm to stabilize magnitudes if desired
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist = hist / norm
    return hist


# ---------------------------
# Coefficient vector utilities
# ---------------------------

def coeff_matrix_to_flat_vector(coeff_matrix: Optional[List[List[float]]], max_vars: int = MAX_VARIABLES) -> np.ndarray:
    """
    Given coeff_matrix (list of lists) where each row corresponds to [a1, a2, ..., const],
    flatten and pad/truncate to a stable sized vector.

    Strategy:
      - If coeff_matrix is None: return zero vector of length (max_vars + 1) * max_equations
      - To keep fixed size regardless of number of equations, we:
          * Limit number of equations considered to `max_eqs` (here we pick 3 as default)
          * For each equation, take the first max_vars coefficients (padding or truncating)
          * Append constant term
      - If total dimension less than target_dim, zero-pad to target.
    """
    # target per-equation width = max_vars + 1 (constants)
    per_eq = max_vars + 1
    # choose how many equations to consider; here pick up to 3 equations for compactness
    max_eqs = 3
    target_dim = per_eq * max_eqs

    vec = np.zeros(target_dim, dtype=float)
    if not coeff_matrix:
        return vec

    # iterate through first max_eqs equations
    for i, row in enumerate(coeff_matrix[:max_eqs]):
        # row is list of floats length maybe max_vars+1 or other
        truncated = [float(x) if x is not None else 0.0 for x in (row[:max_vars] + [row[-1]] if len(row) >= max_vars + 1 else (row + [0.0] * (max_vars + 1 - len(row))))]
        # ensure length = per_eq
        truncated = truncated[:per_eq] + [0.0] * max(0, per_eq - len(truncated))
        start = i * per_eq
        vec[start:start + per_eq] = truncated[:per_eq]
    # normalize (L2)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


# ---------------------------
# Simple numeric invariants
# ---------------------------

def compute_simple_invariants(parsed_obj: Dict[str, Any]) -> np.ndarray:
    """
    Compute a small vector of interpretable numeric invariants:
      - number of variables (normalized)
      - number of equations (normalized)
      - indicator all_linear (0/1)
      - symbolic diversity score (if present in parsed_obj or fingerprint)
    Returns small numpy array length 4.
    """
    if not isinstance(parsed_obj, dict):
        return np.zeros(4, dtype=float)
    var_count = len(parsed_obj.get("variables", []) or [])
    eq_count = len(parsed_obj.get("canonical_equations_renamed", []) or [])
    all_linear = 1.0 if parsed_obj.get("is_all_linear", False) else 0.0
    # symbolic diversity: approximate using fingerprint string length
    fp = parsed_obj.get("fingerprint") or ""
    diversity = min(len(fp) / 100.0, 1.0) if fp else 0.0

    # normalize counts by heuristics
    var_norm = min(var_count / float(MAX_VARIABLES), 1.0)
    eq_norm = min(eq_count / 3.0, 1.0)  # we often limited to 3 eqs
    return np.array([var_norm, eq_norm, all_linear, diversity], dtype=float)


# ---------------------------
# Build structure vector
# ---------------------------

def build_structure_vector_from_parsed(parsed_obj: Dict[str, Any],
                                       wl_bins: int = WL_FEATURE_SIZE,
                                       wl_iterations: int = WL_ITERATIONS,
                                       max_vars: int = MAX_VARIABLES,
                                       target_dim: int = STRUCTURE_VECTOR_SIZE) -> np.ndarray:
    """
    Build the final deterministic structure vector for a parsed canonicalizer object.

    Steps:
      - Extract coefficient matrix -> flattened normalized vec (dim = per_eq * max_eqs)
      - For each canonical equation, build expr graph and run WL counts -> histogram
      - Aggregate equation-level histograms by averaging
      - Compute simple numeric invariants
      - Concatenate [coeff_vec, wl_hist, invariants] -> pad/truncate to target_dim

    Returns:
      - numpy array shape (target_dim,)
    """
    # coeff vector
    coeff_vec = coeff_matrix_to_flat_vector(parsed_obj.get("coeff_matrix"), max_vars=max_vars)

    # WL histograms for each canonical equation (if any)
    eqs = parsed_obj.get("canonical_equations_renamed", []) or []
    if eqs:
        per_eq_hists = []
        for eq in eqs:
            G = expr_to_graph(eq)
            counts = weisfeiler_lehman_subtree_counts(G, iterations=wl_iterations)
            hist = wl_counts_to_histogram(counts, bins=wl_bins)
            per_eq_hists.append(hist)
        # average across equations for fixed-length representation
        wl_hist = np.mean(per_eq_hists, axis=0) if per_eq_hists else np.zeros(wl_bins, dtype=float)
    else:
        wl_hist = np.zeros(wl_bins, dtype=float)

    invariants = compute_simple_invariants(parsed_obj)  # length 4

    # concatenate: coeff_vec + wl_hist + invariants
    raw = np.concatenate([coeff_vec, wl_hist, invariants], axis=0).astype(float)

    # pad or truncate to target_dim
    if raw.shape[0] >= target_dim:
        vec = raw[:target_dim]
    else:
        pad = np.zeros(target_dim - raw.shape[0], dtype=float)
        vec = np.concatenate([raw, pad], axis=0)

    # final normalization (L2)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec


# ---------------------------
# DataFrame-level batch processing
# ---------------------------

def compute_structure_features_df(df: pd.DataFrame,
                                  parsed_col: str = "parsed_equations",
                                  output_col: str = "structure_vector",
                                  wl_bins: int = WL_FEATURE_SIZE,
                                  wl_iterations: int = WL_ITERATIONS,
                                  max_vars: int = MAX_VARIABLES,
                                  target_dim: int = STRUCTURE_VECTOR_SIZE) -> pd.DataFrame:
    """
    Compute structure vectors for every row in df and add as new column `output_col`.
    Returns new DataFrame copy.
    """
    df = df.copy()
    vectors = []
    for idx, row in df.iterrows():
        parsed = row.get(parsed_col, {})
        if not isinstance(parsed, dict):
            parsed = {}
        vec = build_structure_vector_from_parsed(parsed,
                                                 wl_bins=wl_bins,
                                                 wl_iterations=wl_iterations,
                                                 max_vars=max_vars,
                                                 target_dim=target_dim)
        vectors.append(vec)
    df[output_col] = vectors
    return df


def save_structure_features(df: pd.DataFrame, vector_col: str = "structure_vector", out_path: str = None):
    """
    Persist structure vectors as numpy array and store mapping to problem ids (if present).
    """
    vecs = np.vstack(df[vector_col].to_list()) if len(df) > 0 else np.zeros((0, STRUCTURE_VECTOR_SIZE))
    if out_path:
        np.save(out_path, vecs)
    return vecs


# ---------------------------
# CLI / Demo
# ---------------------------

def demo_with_samples():
    """Quick demo using synthetic parsed objects to show behavior."""
    samples = [
        {
            "canonical_equations_renamed": ["v1 - 2*v2 + 30", "v2 - v1 - 20"],
            "variables": ["x", "y"],
            "var_map": {"x": "v1", "y": "v2"},
            "fingerprint": "v1 - 2*v2 + 30; v2 - v1 - 20",
            "coeff_matrix": [[1, -2, 30], [-1, 1, -20]],
            "is_all_linear": True,
        },
        {
            "canonical_equations_renamed": ["2*v1 + 3*v2 - 60"],
            "variables": ["a", "b"],
            "var_map": {"a": "v1", "b": "v2"},
            "fingerprint": "2*v1 + 3*v2 - 60",
            "coeff_matrix": [[2, 3, -60]],
            "is_all_linear": True,
        },
        # nonlinear / no-coeff sample
        {
            "canonical_equations_renamed": ["v1*v2 - 50"],
            "variables": ["l", "w"],
            "var_map": {"l": "v1", "w": "v2"},
            "fingerprint": "v1*v2 - 50",
            "coeff_matrix": None,
            "is_all_linear": False,
        }
    ]
    df = pd.DataFrame([{"parsed_equations": s} for s in samples])
    print(df)
    out = compute_structure_features_df(df, parsed_col="parsed_equations", output_col="structure_vector")
    for i, row in out.iterrows():
        print(row)
        print(f"Sample {i}: vector norm {np.linalg.norm(row['structure_vector']):.4f}, dim {len(row['structure_vector'])}")


if __name__ == "__main__":
    demo_with_samples()
