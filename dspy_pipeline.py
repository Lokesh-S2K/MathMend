"""
CalcMate DSPy Hybrid Retrieval Demo with Evaluation
---------------------------------------------------
This script demonstrates the DSPy-powered Smart Retrieval pipeline
for mathematically similar problem discovery and evaluation metrics.
"""

from dspy_modules import SmartRetrievalPipeline
import textwrap
import time

class SmartRetrievalDemo:
    """Demo harness for the SmartRetrievalPipeline."""

    def __init__(self):
        self.pipeline = None
        self.index_path = "output/embeddings/faiss_index_20251015_145706.bin"
        self.idmap_path = "output/embeddings/faiss_id_map_20251015_145706.json"

    def load_system(self):
        """Initialize the DSPy Smart Retrieval system."""
        try:
            print("🔧 Initializing DSPy Hybrid Retrieval System...")
            self.pipeline = SmartRetrievalPipeline(self.index_path, self.idmap_path)
            print("✅ System successfully loaded!\n")
            return True
        except Exception as e:
            print(f"❌ Failed to load system: {e}")
            return False

    def find_most_similar_problem(self, query: str):
        """Run retrieval for a given query and print results."""
        print(f"🔍 Query: {query}")
        print("-" * 70)

        start = time.time()
        prediction = self.pipeline(query, top_k=3)
        elapsed = time.time() - start

        if not prediction.results:
            print("⚠️ No similar problems found.\n")
            return

        for i, res in enumerate(prediction.results, 1):
            print(f"[{i}] Similarity Score: {res['similarity']:.4f}")
            print(textwrap.fill(f"Matched Text: {res['text'][:300]}", width=90))
            print()

        print(f"⏱ Retrieval completed in {elapsed:.3f} seconds\n")

        if prediction.equations:
            print("📘 Extracted Equations (symbolic understanding):")
            for eq in prediction.equations:
                print(f"   • {eq}")
        else:
            print("📘 No explicit equations detected — used pattern-based understanding.")
        print()

    def run_evaluation(self, queries, top_k=3):
        """Run retrieval evaluation metrics for a list of queries."""
        print("\n📊 Evaluating Retrieval Quality...")
        metrics = self.pipeline.evaluate_retrieval_quality(queries, top_k=top_k)
        print(f"Precision@{top_k}: {metrics.precision_at_k:.3f}")
        print(f"Mean Cosine Similarity: {metrics.mean_cosine_similarity:.3f}\n")


# --------------------------------------------------------------------
#  DEMO RUNNER
# --------------------------------------------------------------------
def main():
    """Run the CalcMate-style retrieval showcase and evaluation."""

    demo = SmartRetrievalDemo()

    if not demo.load_system():
        return

    print("\n" + "=" * 70)
    print("CALCMATE DEMO: MATHEMATICAL SIMILARITY SEARCH")
    print("Find problems that are STRUCTURALLY similar, not just word matches!")
    print("=" * 70 + "\n")

    # --- Sample test queries ---
    test_queries = [
        "A motorcycle travels 180 kilometers in 3 hours. What is its average speed?",
        "If a pizza costs $18 and you order 3 pizzas, what's the total cost?",
        "A rectangular playground has length 12 meters and width 8 meters. Calculate its area.",
    ]

    print("🚀 RUNNING SMART RETRIEVAL TESTS...\n")
    for i, query in enumerate(test_queries, 1):
        print(f"TEST {i}/{len(test_queries)}:")
        demo.find_most_similar_problem(query)

        if i < len(test_queries):
            print("═" * 70 + "\n")

    # --- Evaluation ---
    demo.run_evaluation(test_queries, top_k=3)

    print("=" * 70)
    print("✅ DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    print("✨ WHAT MAKES CALCMATE UNIQUE:")
    print("   • Finds *mathematically* similar problems, not just textual matches")
    print("   • Understands symbolic and semantic structure of math problems")
    print("   • Uses hybrid embeddings (text + equation)")
    print("   • Paves the way for a neuro-symbolic solver in the next phase")
    print("=" * 70)
    print("Next: Neuro-symbolic solver that learns to solve using retrieved examples!")
    print("=" * 70)


if __name__ == "__main__":
    main()
