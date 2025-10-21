# ==========================================================
# File: modules/cleaning.py
# Phase: 1  →  Text Cleaning & Normalization
# Description:
#   Provides functions to clean raw math word problem text by:
#     - Normalizing whitespace and punctuation
#     - Converting math symbols to Pythonic forms
#     - Removing corrupted or stray characters
#     - Ensuring consistent formatting for equation parsing
# ==========================================================

import re
import pandas as pd
from typing import List, Dict

# ----------------------------------------------------------
# Cleaning Configuration
# ----------------------------------------------------------

REPLACEMENTS = {
    '^': '**',          # exponentiation for Python/SymPy
    '×': '*',
    '÷': '/',
    '–': '-',           # en-dash → hyphen
    '−': '-',           # minus sign → hyphen
    '“': '"', '”': '"', # smart quotes → straight quotes
    '‘': "'", '’': "'",
    '√': 'sqrt',        # for SymPy-friendly parsing
}

UNWANTED_SYMBOLS = [
    '�', '…', '✅', '🔥', '→', '⇒', '←', '🙂', '🙂', '✔', '✖', '#', '•'
]


def normalize_text(text: str) -> str:
    """
    Apply normalization rules:
      - Replace math symbols (^ → **, etc.)
      - Normalize whitespace
      - Remove unwanted Unicode characters
      - Fix double spaces, punctuation spacing
    """
    if not isinstance(text, str):
        return ""

    # Apply replacements
    for old, new in REPLACEMENTS.items():
        text = text.replace(old, new)

    # Remove unwanted symbols
    for sym in UNWANTED_SYMBOLS:
        text = text.replace(sym, '')

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Ensure spacing around basic math operators
    text = re.sub(r'([+\-*/=()])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def clean_equation_text(equation: str) -> str:
    """
    Clean only equation-specific strings (not full problem statements).
    Ensures symbols and operators are consistent.
    """
    eq = normalize_text(equation)
    eq = eq.replace(' = =', '=').replace('==', '=')
    eq = eq.replace('= =', '=')
    eq = re.sub(r'\.\.+', '.', eq)
    return eq.strip()


def clean_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    df = df.copy()
    df['clean_text'] = df[text_col].apply(normalize_text)
    return df


def cleaning_report(df: pd.DataFrame, raw_col: str, clean_col: str) -> Dict[str, int]:
    total = len(df)
    changed = (df[raw_col] != df[clean_col]).sum()
    empty = (df[clean_col].str.strip() == '').sum()

    return {
        "total_rows": total,
        "changed_texts": int(changed),
        "empty_after_clean": int(empty),
        "unchanged_texts": int(total - changed)
    }


def demo_cleaning_example():
    sample = "If A has 2^x + 3y = 10 and B has 4x − 5y = 20, find x and y."
    print("Before:", sample)
    print("After :", normalize_text(sample))


