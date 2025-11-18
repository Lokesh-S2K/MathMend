"""
math_retrieval.modules.canonicalizer
-----------------------------------

Phase 1 — Canonicalization & symbolic fingerprinting.

Responsibilities:
- Extract candidate equations from cleaned problem text or reasoning text.
- Parse equations into SymPy expressions with consistent transformations.
- Canonicalize each equation to a deterministic "expanded LHS = 0" form.
- Deterministically rename variables (v1, v2, ...) to remove naming differences.
- Produce a stable symbolic_fingerprint for a system of equations (string).
- Extract coefficient vectors for linear systems (for feature baselines).
- Provide DataFrame-level canonicalization utilities and validation tools.

Notes:
- This module is intentionally defensive: parsing can fail, and we capture
  parsing diagnostics for inspection.
- Uses SymPy parsing with implicit multiplication support and standard transforms.
"""

from typing import List, Tuple, Dict, Any, Optional
import re
import logging
import json

from sympy import sympify, Eq, simplify, expand, Symbol, Poly, Add, Number
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
import pandas as pd

from pathlib import Path

# relative import from config (assumes package layout)
try:
    from pipeline_sequence.config import (
        SYMPY_TRANSFORMATIONS,
        WL_ITERATIONS,
        MAX_VARIABLES,
        STRUCTURE_VECTOR_SIZE,
    )
except Exception:
    # fallback if run as script for testing
    SYMPY_TRANSFORMATIONS = ("standard_transformations", "implicit_multiplication_application")
    WL_ITERATIONS = 2
    MAX_VARIABLES = 6
    STRUCTURE_VECTOR_SIZE = 256

# Setup logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# prepare sympy transformations object
_transformations = standard_transformations + (implicit_multiplication_application,)


# ---------------------------
# Utilities
# ---------------------------

_EQ_SPLIT_RE = re.compile(r";|\n|\band\b|,|\t")  # splitting tokens for equations candidate lists


def _split_equation_candidates(text: str) -> List[str]:
    """
    Split a block of text into likely equation candidate strings.
    This is intentionally permissive — final parsing step verifies validity.
    """
    if not isinstance(text, str) or text.strip() == "":
        return []
    parts = [p.strip() for p in _EQ_SPLIT_RE.split(text) if p.strip()]
    return parts


def extract_equations_from_reasoning(reasoning_text: str) -> List[str]:
    """
    Heuristic to extract 'final equations' from a reasoning block.
    Looks for explicit markers like "Final Equations:" or else falls back to extracting
    lines containing '=' or expressions that look like equations.
    """
    if not isinstance(reasoning_text, str) or not reasoning_text.strip():
        return []

    # Prefer explicit marker "Final Equations:"
    m = re.search(r"Final Equations:?(.*)", reasoning_text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        after = m.group(1)
        return _split_equation_candidates(after)

    # Otherwise, find lines containing '=' or pattern like 'x+y=10'
    lines = reasoning_text.splitlines()
    candidates = []
    for ln in lines:
        if "=" in ln:
            candidates.extend(_split_equation_candidates(ln))
    # if nothing found, fallback to scanning for near-equation tokens (numbers & variables with =)
    if not candidates:
        tokens = re.findall(r"[A-Za-z0-9\+\-\*/\^\(\)\s=]+=[A-Za-z0-9\+\-\*/\^\(\)\s]+", reasoning_text)
        candidates.extend([t.strip() for t in tokens])
    # final cleanup
    return [c for c in candidates if len(c) > 0]


def _normalize_equation_text(eq_text: str) -> str:
    """
    Lightweight normalization applied before SymPy parsing:
      - replace '^' with '**'
      - strip extraneous trailing dots or non-printable chars
    """
    if not isinstance(eq_text, str):
        return ""
    s = eq_text.replace("^", "**")
    s = s.replace("==", "=")  # avoid double equals
    # remove non-printable weird chars
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\u024F]+", "", s)
    s = s.strip()
    # remove trailing periods that commonly appear after equations in prose
    s = re.sub(r"\.$", "", s)
    return s


# ---------------------------
# Core canonicalization
# ---------------------------

def canonicalize_equation(eq_str: str) -> Tuple[Optional[str], Optional[Any], Optional[str]]:
    """
    Parse and canonicalize a single equation string.
    Returns tuple: (canonical_str, sympy_expr (lhs - rhs), error_message)
    canonical_str: stable string representation of canonicalized equation, e.g. "2*v1 + 3*v2 - 30"
    sympy_expr: SymPy expression representing lhs - rhs (should equal 0)
    error_message: None if OK else diagnostic

    Steps:
      - Normalize eq text
      - Ensure it has an '='; if missing, attempt to treat as expression (lhs)
      - Parse left and right with SymPy with transformations
      - Move everything to LHS, expand & simplify
      - Return textual canonical form (string)
    """
    if not isinstance(eq_str, str) or not eq_str.strip():
        return None, None, "empty input"

    eq_text = _normalize_equation_text(eq_str)
    if not eq_text:
        return None, None, "normalized to empty"

    # Try to split on '='; if multiple '=' present, use first split.
    if "=" in eq_text:
        parts = eq_text.split("=", 1)
        lhs_raw, rhs_raw = parts[0].strip(), parts[1].strip()
    else:
        # treat whole expression as lhs = 0
        lhs_raw, rhs_raw = eq_text, "0"

    try:
        lhs = parse_expr(lhs_raw, transformations=_transformations)
        rhs = parse_expr(rhs_raw, transformations=_transformations)
        expr = expand(simplify(lhs - rhs))
        # final canonical string (sympy's srepr/str may include ordering variations; use str(expr) after simplify)
        canonical = str(simplify(expr))
        return canonical, expr, None
    except Exception as e:
        logger.debug("Parsing failure for eq '%s': %s", eq_text, e)
        return None, None, f"parse_error: {e}"


def _collect_variables_from_expr(expr) -> List[str]:
    """
    Determine variable symbols used in a SymPy expression in a deterministic way.
    Returns sorted list of symbol names.
    """
    if expr is None:
        return []
    syms = sorted({str(s) for s in expr.free_symbols})
    return syms


def _deterministic_var_map(variables: List[str]) -> Dict[str, str]:
    """
    Create a deterministic mapping from original variable names -> v1, v2, ...
    Order is alphabetical on variable name to ensure determinism.
    """
    var_map = {}
    for i, var in enumerate(sorted(variables)):
        var_map[var] = f"v{i+1}"
    return var_map


def _apply_var_renaming_to_string(expr_str: str, var_map: Dict[str, str]) -> str:
    """
    Replace whole-variable occurrences in a string using word-boundary aware substitution.
    """
    s = expr_str
    for old, new in var_map.items():
        # use regex word boundary to avoid partial replacements
        s = re.sub(rf"\b{re.escape(old)}\b", new, s)
    return s


def canonicalize_system(equation_strs: List[str]) -> Dict[str, Any]:
    """
    Given a list of raw equation strings (candidates), attempt to canonicalize
    all parseable ones, produce deterministic variable renaming and a fingerprint.

    Returns dict with keys:
      - 'raw_equations': input list
      - 'parsed': list of dicts: {'raw':, 'canonical':, 'expr': sympy expr or None, 'error':}
      - 'variables': list of variable names (original)
      - 'var_map': mapping original -> v1...
      - 'canonical_equations_renamed': list of canonical eq strings after renaming
      - 'fingerprint': final fingerprint string (sorted canonical eqs joined by ';') or None
      - 'is_all_linear': bool (if all parseable eqs are linear)
      - 'coeff_matrix': list of coefficient vectors (if linear) else None
      - 'errors': list of non-empty error messages
    """
    parsed_list = []
    exprs = []
    errors = []

    for raw in equation_strs:
        canonical, expr, err = canonicalize_equation(raw)
        parsed_list.append({"raw": raw, "canonical": canonical, "expr": expr, "error": err})
        if err:
            errors.append({"raw": raw, "error": err})
        else:
            exprs.append(expr)

    variables = sorted(list({v for expr in exprs for v in _collect_variables_from_expr(expr)}))
    var_map = _deterministic_var_map(variables)

    # Build renamed canonical equations
    renamed = []
    for p in parsed_list:
        if p["canonical"] and p["expr"] is not None:
            renamed_str = _apply_var_renaming_to_string(p["canonical"], var_map)
            renamed.append(renamed_str)
        else:
            # keep raw as placeholder or skip; we maintain order by inserting None
            renamed.append(None)

    # Filter out None and sort the canonical equations for stable fingerprint
    valid_renamed = [r for r in renamed if r is not None]
    # Sorting ensures deterministic ordering of multiple eqs forming a system
    valid_renamed_sorted = sorted(valid_renamed)

    fingerprint = "; ".join(valid_renamed_sorted) if valid_renamed_sorted else None

    # Linear detection & coefficient extraction
    coeff_matrix = []
    all_linear = True
    if exprs:
        # attempt to extract coefficient vectors for each equation, using consistent variable order (v1..vn)
        ren_vars = [var_map[v] for v in variables]  # original->v1 order
        # If ren_vars empty, treat as all constant equations
        for r in valid_renamed_sorted:
            try:
                # parse renamed expression (which is in form '...') back to sympy for Poly
                sym = parse_expr(r, transformations=_transformations)
                # construct polynomial in renamed variables
                if ren_vars:
                    poly = Poly(sym, *[Symbol(v) for v in ren_vars])
                else:
                    poly = Poly(sym)
                # degree check — if multivariate or nonlinear, Poly will represent accordingly
                coeffs = []
                # Build coefficient vector in an order: coefficients for each v_i, then constant term
                for v in ren_vars:
                    # coefficient of variable v (sum of monomials where v appears to power 1)
                    try:
                        coeff = poly.coeffs()  # fallback: but we want specific coeff per var
                        # use poly.coeffs for safety is messy; better use poly.coeff_monomial
                        c = poly.coeff_monomial(Symbol(v))
                    except Exception:
                        c = 0
                    coeffs.append(float(c) if isinstance(c, Number) else float(c.evalf()) if c != 0 else 0.0)
                # constant term: poly.coeff_monomial(1)
                const_term = poly.coeff_monomial(1)
                coeffs.append(float(const_term) if isinstance(const_term, Number) else float(const_term.evalf()) if const_term != 0 else 0.0)
                coeff_matrix.append(coeffs)
            except Exception:
                # not polynomial / not linear => mark not linear
                all_linear = False
                coeff_matrix = None
                break
    else:
        all_linear = False
        coeff_matrix = None

    return {
        "raw_equations": equation_strs,
        "parsed": parsed_list,
        "variables": variables,
        "var_map": var_map,
        "canonical_equations_renamed": valid_renamed_sorted,
        "fingerprint": fingerprint,
        "is_all_linear": all_linear,
        "coeff_matrix": coeff_matrix,
        "errors": errors,
    }


# ---------------------------
# DataFrame-level utilities
# ---------------------------

def canonicalize_dataframe(
    df: pd.DataFrame,
    source_col: str = "clean_text",
    reasoning_col: Optional[str] = None,
    equations_col_out: str = "extracted_equations",
    fingerprint_col: str = "symbolic_fingerprint",
    parsed_col: str = "parsed_equations",
) -> pd.DataFrame:
    """
    Apply extraction + canonicalization across a DataFrame.
    - If reasoning_col is provided, extract equations from reasoning first; else try to extract from source_col.
    - Adds columns:
        - equations_col_out: raw extracted equation candidate list
        - parsed_col: JSON string of parsed dict for each row
        - fingerprint_col: final fingerprint string
    """
    df = df.copy()
    extracted_list = []
    parsed_list = []
    fingerprint_list = []

    for idx, row in df.iterrows():
        reasoning_text = None
        if reasoning_col and isinstance(row.get(reasoning_col, None), str) and row.get(reasoning_col).strip():
            reasoning_text = row[reasoning_col]
        else:
            reasoning_text = row.get(source_col, "")

        candidates = extract_equations_from_reasoning(reasoning_text)
        # as fallback, try to find inline equations like 'x + y = 10' inside source text
        if not candidates:
            # attempt to find eq-looking substrings
            candidates = re.findall(r"[A-Za-z0-9\+\-\*/\^\(\)\s=]+=[A-Za-z0-9\+\-\*/\^\(\)\s]+", reasoning_text)
            candidates = [c.strip() for c in candidates if len(c.strip()) > 0]

        parsed = canonicalize_system(candidates)
        extracted_list.append(candidates)
        parsed_list.append(parsed)
        fingerprint_list.append(parsed.get("fingerprint"))

    df[equations_col_out] = extracted_list
    df[parsed_col] = parsed_list
    df[fingerprint_col] = fingerprint_list

    return df


def validate_against_gold(df: pd.DataFrame, gold_df: pd.DataFrame, id_col: str = "id", fingerprint_col: str = "symbolic_fingerprint") -> Dict[str, Any]:
    """
    Validate canonicalization results against a gold annotated dataset.
    gold_df must contain the expected canonical fingerprint (or canonical equations).
    Returns a summary dict:
      - total compared
      - exact_match_count
      - mismatch_samples (list of examples up to N)
    """
    summary = {"total": 0, "exact_match": 0, "mismatches": []}
    # join on id column
    merged = df.merge(gold_df[[id_col, fingerprint_col]], on=id_col, how="inner", suffixes=("", "_gold"))
    total = len(merged)
    exact = 0
    mismatches = []
    for _, r in merged.iterrows():
        pred = r.get(fingerprint_col)
        gold = r.get(f"{fingerprint_col}_gold")
        if pred == gold:
            exact += 1
        else:
            mismatches.append({"id": r[id_col], "pred": pred, "gold": gold})
    summary["total"] = total
    summary["exact_match"] = exact
    summary["mismatch_count"] = len(mismatches)
    summary["mismatches"] = mismatches[:50]  # limit payload
    return summary


# ---------------------------
# CLI / Demo helpers (no I/O heavy behavior, safe)
# ---------------------------

def demo_on_sample_text(sample_text: str):
    """
    Quick demo function to show canonicalization on a single sample reasoning block.
    """
    print("Input text:\n", sample_text)
    candidates = extract_equations_from_reasoning(sample_text)
    print("\nExtracted candidates:")
    for c in candidates:
        print(" -", c)
    parsed = canonicalize_system(candidates)
    print("\nParsed results:")
    print(json.dumps({
        "variables": parsed["variables"],
        "var_map": parsed["var_map"],
        "canonical_equations": parsed["canonical_equations_renamed"],
        "fingerprint": parsed["fingerprint"],
        "is_all_linear": parsed["is_all_linear"],
    }, indent=2, default=str))


# ---------------------------
# End of canonicalizer.py
# ---------------------------

