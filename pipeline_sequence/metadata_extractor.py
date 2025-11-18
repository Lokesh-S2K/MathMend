"""
math_retrieval.modules.metadata_extractor
-----------------------------------------

Phase 1 — Metadata Extraction

Purpose:
---------
Extract interpretable metadata features from cleaned and/or canonicalized
math word problems. These features later help in analytics dashboards,
filtering, and baseline retrieval.

Functionalities:
----------------
- Detect the problem domain (algebra, geometry, physics, arithmetic, etc.)
- Detect the dominant operation type (+, -, ×, ÷)
- Infer difficulty (simple heuristic)
- Count equations, variables, and coefficients (from canonicalizer output)
- Compute symbolic diversity score for structure analytics
- Attach all metadata as a structured dictionary to each record

Input:
------
- DataFrame with columns like:
    * clean_text (from cleaning stage)
    * parsed_equations (from canonicalizer)
    * symbolic_fingerprint (optional)
    
Output:
-------
- Adds columns:
    * metadata_dict
    * problem_type
    * operation_type
    * difficulty_level
    * variable_count
    * equation_count
    * coefficient_count

This module is lightweight and does not depend on ML models; it uses regex
and keyword dictionaries for efficiency and interpretability.
"""

import re
import math
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

# setup logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# -----------------------------------
# Keyword Dictionaries
# -----------------------------------

PROBLEM_TYPE_KEYWORDS = {
    "algebra": [
        "variable", "equation", "unknown", "x", "y", "solve", "system of equations"
    ],
    "geometry": [
        "triangle", "rectangle", "circle", "area", "perimeter", "radius",
        "diameter", "angle", "base", "height"
    ],
    "physics": [
        "speed", "distance", "time", "velocity", "acceleration", "force",
        "mass", "energy", "work", "power"
    ],
    "arithmetic": [
        "add", "sum", "difference", "product", "quotient", "average",
        "ratio", "percentage", "increase", "decrease"
    ],
    "statistics": [
        "mean", "median", "mode", "probability", "distribution", "data",
        "range", "variance", "standard deviation"
    ],
    "finance": [
        "simple interest", "compound interest", "principal", "rate",
        "profit", "loss", "discount", "investment"
    ]
}

OPERATION_KEYWORDS = {
    "addition": ["add", "plus", "sum", "increase"],
    "subtraction": ["subtract", "minus", "difference", "decrease", "less"],
    "multiplication": ["multiply", "times", "product"],
    "division": ["divide", "quotient", "per"],
    "mixed": ["and", "combined", "together"]
}


# -----------------------------------
# Utility Functions
# -----------------------------------

def detect_problem_type(text: str) -> Optional[str]:
    """Return the best-matching problem type based on keyword presence."""
    if not isinstance(text, str):
        return None
    text_l = text.lower()
    scores = {}
    for domain, kws in PROBLEM_TYPE_KEYWORDS.items():
        count = sum(1 for k in kws if k in text_l)
        scores[domain] = count
    if not any(scores.values()):
        return "unknown"
    # choose highest count domain
    return max(scores, key=scores.get)


def detect_operation_type(text: str) -> Optional[str]:
    """Detect dominant operation keyword."""
    if not isinstance(text, str):
        return None
    text_l = text.lower()
    counts = {op: sum(1 for k in kws if k in text_l) for op, kws in OPERATION_KEYWORDS.items()}
    if not any(counts.values()):
        return "unknown"
    return max(counts, key=counts.get)


def estimate_difficulty(eq_count: int, var_count: int) -> str:
    """
    Simple difficulty heuristic:
    - Easy: 1 eqn, ≤ 2 vars
    - Medium: 2–3 eqns, ≤ 3 vars
    - Hard: >3 eqns or ≥ 4 vars
    """
    if eq_count <= 1 and var_count <= 2:
        return "Easy"
    elif eq_count <= 3 and var_count <= 3:
        return "Medium"
    return "Hard"


def analyze_structure(parsed_obj: Dict[str, Any]) -> Tuple[int, int, int]:
    """
    Extract equation/variable/coefficient counts from canonicalizer output dict.
    Returns (eq_count, var_count, coeff_count)
    """
    if not isinstance(parsed_obj, dict):
        return (0, 0, 0)

    eq_count = len(parsed_obj.get("canonical_equations_renamed", []) or [])
    var_count = len(parsed_obj.get("variables", []) or [])

    coeff_matrix = parsed_obj.get("coeff_matrix")
    if coeff_matrix and isinstance(coeff_matrix, list):
        # flatten coefficient matrix to count nonzero entries
        coeff_count = sum(1 for row in coeff_matrix for c in row if abs(c) > 0)
    else:
        coeff_count = 0

    return eq_count, var_count, coeff_count


def symbolic_diversity_score(fingerprint: Optional[str]) -> float:
    """
    Compute a numeric diversity score from symbolic fingerprint.
    Idea: longer and more variable-dense fingerprints indicate more complex problems.
    """
    if not isinstance(fingerprint, str):
        return 0.0
    tokens = re.findall(r"[A-Za-z]+", fingerprint)
    unique_tokens = len(set(tokens))
    length_factor = min(len(fingerprint) / 50.0, 3.0)  # cap contribution
    return round(unique_tokens * 0.5 + length_factor, 2)


# -----------------------------------
# Core Metadata Extraction
# -----------------------------------

def extract_metadata_row(row: pd.Series) -> Dict[str, Any]:
    """
    Compute metadata for a single DataFrame row.
    Expects columns: 'clean_text', 'parsed_equations', 'symbolic_fingerprint'
    """
    text = row.get("clean_text", "")
    parsed_obj = row.get("parsed_equations", {})
    fingerprint = row.get("symbolic_fingerprint", None)

    problem_type = detect_problem_type(text)
    operation_type = detect_operation_type(text)
    eq_count, var_count, coeff_count = analyze_structure(parsed_obj)
    difficulty = estimate_difficulty(eq_count, var_count)
    diversity = symbolic_diversity_score(fingerprint)

    return {
        "problem_type": problem_type,
        "operation_type": operation_type,
        "difficulty_level": difficulty,
        "equation_count": eq_count,
        "variable_count": var_count,
        "coefficient_count": coeff_count,
        "symbolic_diversity_score": diversity,
    }


def extract_metadata_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Batch metadata extraction on DataFrame.
    Adds:
      - metadata_dict
      - problem_type
      - operation_type
      - difficulty_level
      - equation_count
      - variable_count
      - coefficient_count
      - symbolic_diversity_score
    """
    meta_records = []
    for _, row in df.iterrows():
        meta = extract_metadata_row(row)
        meta_records.append(meta)

    meta_df = pd.DataFrame(meta_records)
    df = pd.concat([df.reset_index(drop=True), meta_df], axis=1)
    df["metadata_dict"] = meta_records
    return df


# -----------------------------------
# Demo / Standalone Testing
# -----------------------------------

def demo():
    """Demonstration with dummy samples."""
    data = [
        {
            "clean_text": "Find the area of a rectangle with length 10 cm and width 5 cm.",
            "parsed_equations": {"canonical_equations_renamed": ["v1*v2 - 50"], "variables": ["length", "width"], "coeff_matrix": [[1, 1, -50]]},
            "symbolic_fingerprint": "v1*v2 - 50"
        },
        {
            "clean_text": "Two trains start at the same time. The speed of one is 20 km/h more than the other. They meet after 3 hours.",
            "parsed_equations": {"canonical_equations_renamed": ["v1*3 - v2*3 - 60"], "variables": ["speed1", "speed2"], "coeff_matrix": [[3, -3, -60]]},
            "symbolic_fingerprint": "v1*3 - v2*3 - 60"
        }
    ]
    df = pd.DataFrame(data)
    out = extract_metadata_dataframe(df)
    print(out[["clean_text", "problem_type", "operation_type", "difficulty_level", "equation_count", "variable_count", "symbolic_diversity_score"]])


if __name__ == "__main__":
    demo()
