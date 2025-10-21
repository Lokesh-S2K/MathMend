"""
modules/utils.py
----------------
Shared utilities for logging, I/O, and safe concurrent file operations.
Used across all modules in the math problem retrieval pipeline.
"""

import os
import json
import csv
import logging
import time
from datetime import datetime
from contextlib import contextmanager
from filelock import FileLock
from tqdm import tqdm

# -------------------------------------------------------------------------
# Logging utilities
# -------------------------------------------------------------------------

def get_logger(name: str, level=logging.INFO):
    """
    Configure and return a logger with uniform formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

# -------------------------------------------------------------------------
# File and directory utilities
# -------------------------------------------------------------------------

def ensure_dir(path: str):
    """
    Create directory if it doesn’t exist.
    """
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    """
    Returns a human-readable timestamp string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# -------------------------------------------------------------------------
# JSONL / CSV I/O utilities
# -------------------------------------------------------------------------

def read_jsonl(path: str):
    """
    Reads a JSON Lines (.jsonl) file into a list of dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str, data: list):
    """
    Writes a list of dictionaries to JSON Lines format.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_csv(path: str):
    """
    Reads a CSV file into a list of dicts.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv(path: str, rows: list, fieldnames: list):
    """
    Writes rows (list of dicts) to a CSV file.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# -------------------------------------------------------------------------
# File lock utilities for safe parallel access
# -------------------------------------------------------------------------

@contextmanager
def safe_write(path: str, timeout: int = 10):
    """
    Context manager for safe concurrent writing.
    Uses filelock to prevent corruption when multiple processes write.
    """
    lock_path = path + ".lock"
    ensure_dir(os.path.dirname(path))
    with FileLock(lock_path, timeout=timeout):
        yield


def append_jsonl_safely(path: str, entry: dict):
    """
    Append one JSONL record safely (with file lock).
    """
    ensure_dir(os.path.dirname(path))
    with safe_write(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# -------------------------------------------------------------------------
# Progress + timing helpers
# -------------------------------------------------------------------------

@contextmanager
def timer(section: str):
    """
    Context manager to measure elapsed time for any block.
    """
    start = time.time()
    print(f"⏳ Starting: {section}")
    yield
    end = time.time()
    print(f"✅ Completed {section} in {end - start:.2f}s")


def progress_bar(iterable, desc="Processing", total=None):
    """
    Simple tqdm wrapper for progress tracking.
    """
    return tqdm(iterable, desc=desc, total=total, ncols=80)

# -------------------------------------------------------------------------
# Misc utilities
# -------------------------------------------------------------------------

def flatten_dict(d, parent_key='', sep='.'):
    """
    Flatten nested dict into single-level dict with dot notation keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# -------------------------------------------------------------------------
# Example Usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    logger = get_logger("utils_demo")
    logger.info("Testing utilities...")
    test_path = "tmp/test.jsonl"
    test_data = [{"id": 1, "text": "A test entry"}]
    write_jsonl(test_path, test_data)
    read_back = read_jsonl(test_path)
    logger.info(f"Read back: {read_back}")
