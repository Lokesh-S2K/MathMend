"""
evaluator.py
------------
This module evaluates retrieval performance and canonicalization accuracy.

Metrics:
- Precision@K and Recall@K for structure-based retrieval
- Canonicalization validation using symbolic fingerprint equivalence
- Detailed failure analysis logs

Author: Anirudh Krishna M
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score
from typing import Dict, List, Tuple
from sympy import simplify
import json
import logging

logging.basicConfig(
    filename="logs/evaluator.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class Evaluator:
    def __init__(self, faiss_index, embeddings, metadata_df: pd.DataFrame, id_to_problem: Dict[int, str]):
        """
        Args:
            faiss_index: FAISS index object.
            embeddings: np.ndarray of all embeddings used for indexing.
            metadata_df: DataFrame containing problem_id, fingerprint, metadata.
            id_to_problem: Mapping from FAISS index IDs to problem statements.
        """
        self.index = faiss_index
        self.embeddings = embeddings
        self.metadata = metadata_df
        self.id_to_problem = id_to_problem

    # ------------------------- #
    # Canonicalization Accuracy #
    # ------------------------- #

    def validate_canonicalization(self, gold_pairs: List[Tuple[str, str]]) -> float:
        """
        Check canonical equivalence (symbolic fingerprint equality).

        Args:
            gold_pairs: List of (expected_eq, predicted_eq)
        Returns:
            Accuracy score (float)
        """
        correct = 0
        total = len(gold_pairs)
        for gold, pred in gold_pairs:
            try:
                if simplify(gold) == simplify(pred):
                    correct += 1
            except Exception:
                logging.warning(f"Failed to simplify pair: {gold}, {pred}")
        acc = correct / total if total > 0 else 0
        logging.info(f"Canonicalization accuracy: {acc:.4f}")
        return acc

    # ------------------------- #
    # Retrieval Evaluation      #
    # ------------------------- #

    def precision_at_k(self, query_vecs: np.ndarray, query_ids: List[int], k: int = 5) -> float:
        """
        Compute Precision@K for retrieval based on canonical equivalence.
        """
        hits = 0
        total = len(query_ids)
        for i, qid in enumerate(tqdm(query_ids, desc=f"Evaluating Precision@{k}")):
            query_vec = query_vecs[i].reshape(1, -1)
            D, I = self.index.search(query_vec, k)
            retrieved = I[0].tolist()
            target_fp = self.metadata.iloc[qid]['symbolic_fingerprint']
            retrieved_fps = [self.metadata.iloc[idx]['symbolic_fingerprint'] for idx in retrieved]

            if target_fp in retrieved_fps:
                hits += 1

        precision = hits / total if total > 0 else 0
        logging.info(f"Precision@{k}: {precision:.4f}")
        return precision

    def recall_at_k(self, query_vecs: np.ndarray, query_ids: List[int], k: int = 5) -> float:
        """
        Compute Recall@K for retrieval.
        """
        relevant = 0
        retrieved_relevant = 0
        for i, qid in enumerate(tqdm(query_ids, desc=f"Evaluating Recall@{k}")):
            query_vec = query_vecs[i].reshape(1, -1)
            D, I = self.index.search(query_vec, k)
            retrieved = I[0].tolist()
            target_fp = self.metadata.iloc[qid]['symbolic_fingerprint']

            all_same_fp = self.metadata[self.metadata['symbolic_fingerprint'] == target_fp].index.tolist()
            relevant += len(all_same_fp)
            retrieved_relevant += len(set(retrieved) & set(all_same_fp))

        recall = retrieved_relevant / relevant if relevant > 0 else 0
        logging.info(f"Recall@{k}: {recall:.4f}")
        return recall

    # ------------------------- #
    # Failure Case Logging      #
    # ------------------------- #

    def analyze_failures(self, query_vecs: np.ndarray, query_ids: List[int], k: int = 5, output_path="reports/failures.json"):
        """
        Analyze retrieval failures — where top-k results miss canonical equivalents.
        Save as JSON for inspection.
        """
        failures = []
        for i, qid in enumerate(query_ids):
            query_vec = query_vecs[i].reshape(1, -1)
            D, I = self.index.search(query_vec, k)
            retrieved = I[0].tolist()

            target_fp = self.metadata.iloc[qid]['symbolic_fingerprint']
            retrieved_fps = [self.metadata.iloc[idx]['symbolic_fingerprint'] for idx in retrieved]

            if target_fp not in retrieved_fps:
                failures.append({
                    "query_id": int(qid),
                    "query_problem": self.id_to_problem[qid],
                    "target_fp": target_fp,
                    "retrieved_fps": retrieved_fps,
                    "retrieved_problems": [self.id_to_problem[idx] for idx in retrieved]
                })

        with open(output_path, "w") as f:
            json.dump(failures, f, indent=2)
        logging.info(f"Saved failure analysis to {output_path}")

    # ------------------------- #
    # Overall Report Generator  #
    # ------------------------- #

    def full_report(self, query_vecs, query_ids, gold_pairs):
        """
        Run all evaluation metrics and generate report.
        """
        report = {
            "canonicalization_accuracy": self.validate_canonicalization(gold_pairs),
            "precision@5": self.precision_at_k(query_vecs, query_ids, k=5),
            "recall@5": self.recall_at_k(query_vecs, query_ids, k=5)
        }

        with open("reports/evaluation_summary.json", "w") as f:
            json.dump(report, f, indent=2)

        logging.info("Full evaluation report generated.")
        return report


# ------------------------- #
# Example usage (optional)  #
# ------------------------- #

if __name__ == "__main__":
    import faiss

    # Example dummy data setup
    embeddings = np.random.rand(100, 128).astype("float32")
    index = faiss.IndexFlatL2(128)
    index.add(embeddings)

    metadata = pd.DataFrame({
        "problem_id": range(100),
        "symbolic_fingerprint": [f"eq_{i // 10}" for i in range(100)]
    })
    id_to_problem = {i: f"Problem text {i}" for i in range(100)}

    evaluator = Evaluator(index, embeddings, metadata, id_to_problem)

    gold_pairs = [(f"x+{i}", f"x+{i}") for i in range(10)]
    query_vecs = embeddings[:10]
    query_ids = list(range(10))

    report = evaluator.full_report(query_vecs, query_ids, gold_pairs)
    print(json.dumps(report, indent=2))
