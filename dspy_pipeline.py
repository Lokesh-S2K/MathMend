"""
CalcMate DSPy Hybrid Neuro-Symbolic Retrieval Demo with Evaluation + Explanations
---------------------------------------------------------------------------------
This script demonstrates the DSPy-powered Smart Retrieval pipeline
with neuro-symbolic reasoning for mathematically similar problem discovery,
evaluation metrics, and interpretability.
"""

from dspy_modules import SmartRetrievalPipeline, explain_similarity
from pipeline_sequence.embedder import encode_texts
import textwrap
import time
import numpy as np
from typing import List, Optional


class SmartRetrievalDemo:
    """Demo harness for the SmartRetrievalPipeline with enhanced explanations."""

    def __init__(self):
        self.pipeline = None
        self.index_path = "output/embeddings/faiss_index_20251015_145706.bin"
        self.idmap_path = "output/embeddings/faiss_id_map_20251015_145706.json"

    def load_system(self):
        """Initialize the DSPy Smart Retrieval system with neuro-symbolic reasoning."""
        try:
            print("🔧 Initializing DSPy Hybrid Neuro-Symbolic Retrieval System...")
            # Note: LLM client removed from constructor in fixed version
            # DSPy uses dspy.configure(lm=...) instead
            self.pipeline = SmartRetrievalPipeline(
                self.index_path, 
                self.idmap_path
            )
            print("✅ System successfully loaded!\n")
            return True
        except Exception as e:
            print(f"❌ Failed to load system: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _add_explanations_to_results(self, query: str, results: List[dict]) -> List[dict]:
        """
        Add similarity explanations to each retrieved result.
        
        Args:
            query: The original query text
            results: List of retrieval results with similarity scores
            
        Returns:
            Results enhanced with explanation dictionaries
        """
        try:
            # Encode query for explanation
            query_embedding = encode_texts([query], normalize=True)[0]
            
            enhanced_results = []
            for res in results:
                # Get document text and encode it
                doc_text = res.get("text", "")
                if not doc_text or doc_text.startswith("["):
                    # Skip invalid/error entries
                    enhanced_results.append(res)
                    continue
                
                try:
                    doc_embedding = encode_texts([doc_text], normalize=True)[0]
                    
                    # Generate explanation
                    explanation = explain_similarity(
                        query, 
                        doc_text, 
                        query_embedding, 
                        doc_embedding
                    )
                    
                    res['explanation'] = explanation
                except Exception as e:
                    res['explanation'] = {
                        'error': f"Could not generate explanation: {str(e)}"
                    }
                
                enhanced_results.append(res)
            
            return enhanced_results
        except Exception as e:
            print(f"⚠️ Warning: Could not add explanations: {e}")
            return results

    def find_most_similar_problem(self, query: str, explain: bool = True, top_k: int = 3):
        """Run retrieval for a given query, print results, and show explanations."""
        print(f"🔍 Query: {query}")
        print("-" * 70)

        start = time.time()
        try:
            prediction = self.pipeline(query, top_k=top_k)
        except Exception as e:
            print(f"❌ Error during retrieval: {e}")
            import traceback
            traceback.print_exc()
            return
        
        elapsed = time.time() - start

        # Extract retrieved results (handle different attribute names)
        retrieved_results = getattr(prediction, "results", [])
        
        if not retrieved_results:
            print("⚠️ No similar problems found.")
            print(f"   Result type: {getattr(prediction, 'result_type', 'unknown')}")
            if hasattr(prediction, 'note'):
                print(f"   Note: {prediction.note}")
            print()
            return

        # Add explanations if requested
        if explain:
            retrieved_results = self._add_explanations_to_results(query, retrieved_results)

        # Display results
        print(f"📋 Found {len(retrieved_results)} similar problems:\n")
        for i, res in enumerate(retrieved_results, 1):
            similarity = res.get("similarity", 0.0)
            text = res.get("text", "")
            
            # Skip error entries
            if "error" in res and not text.strip():
                continue
            
            print(f"[{i}] Similarity Score: {similarity:.4f}")
            
            # Truncate long text
            display_text = text[:300] + "..." if len(text) > 300 else text
            print(textwrap.fill(f"Matched Text: {display_text}", width=90))
            
            # Show metadata if available
            if "metadata" in res:
                meta = res["metadata"]
                if "problem_type" in meta:
                    print(f"    Problem Type: {meta['problem_type']}")
            
            print()

            # Show explanation if available
            if explain and "explanation" in res:
                exp = res["explanation"]
                if "error" not in exp:
                    print("🧠 Similarity Analysis:")
                    print(f"   • Embedding Similarity: {exp.get('embedding_cosine_similarity', 'N/A')}")
                    print(f"   • Keyword Overlap: {exp.get('keyword_overlap_score', 'N/A')}")
                    print(f"   • Jaccard Similarity: {exp.get('jaccard_similarity', 'N/A')}")
                    
                    common = exp.get('common_tokens', [])
                    if common and len(common) > 0:
                        print(f"   • Common Terms ({exp.get('common_token_count', len(common))}): {', '.join(common[:8])}")
                    print()

        print(f"⏱ Retrieval + neuro-symbolic reasoning completed in {elapsed:.3f} seconds\n")

        # Show extracted equations if available
        equations = (
            getattr(prediction, "equations", None) or 
            getattr(prediction, "canonical_equations", None) or
            getattr(prediction, "extracted_equations", None)
        )
        if equations:
            print("📘 Extracted/Canonical Equations:")
            for eq in equations:
                print(f"   • {eq}")
            print()

        # Show solution if found
        solution = getattr(prediction, "solution", None)
        if solution:
            result_type = getattr(prediction, "result_type", "unknown")
            print(f"🧮 Solution (via {result_type}):")
            for k, v in solution.items():
                print(f"   • {k} = {v}")
            
            # Show residuals for symbolic solutions
            if hasattr(prediction, "residuals"):
                print(f"\n   Residuals (verification):")
                for eq, val in prediction.residuals.items():
                    if isinstance(val, dict):
                        satisfied = val.get('satisfied', False)
                        status = "✓" if satisfied else "✗"
                        print(f"   {status} {eq}: {val.get('value', 'N/A')}")
                    else:
                        print(f"   • {eq}: {val}")
            print()

        # Show LLM reasoning if available
        reasoning = getattr(prediction, "reasoning", None) or getattr(prediction, "steps", None)
        if reasoning:
            print("💡 LLM Reasoning Steps:")
            # Truncate very long reasoning
            display_reasoning = reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
            print(textwrap.fill(display_reasoning, width=90))
            print()

        # Show note for unresolved cases
        if hasattr(prediction, "note"):
            print(f"ℹ️ Note: {prediction.note}\n")

    def run_evaluation(self, queries: List[str], top_k: int = 3):
        """Run retrieval evaluation metrics for a list of queries."""
        print("\n📊 Evaluating Retrieval Quality...")
        print("=" * 70)

        all_precisions = []
        all_similarities = []
        successful_queries = 0
        symbolic_solves = 0
        llm_solves = 0
        unresolved = 0

        for idx, query in enumerate(queries, 1):
            print(f"Evaluating query {idx}/{len(queries)}...", end="\r")
            
            try:
                prediction = self.pipeline(query, top_k=top_k)
                
                # Track solution types
                result_type = getattr(prediction, "result_type", None)
                if result_type == "symbolic":
                    symbolic_solves += 1
                elif result_type == "llm":
                    llm_solves += 1
                elif result_type == "unresolved":
                    unresolved += 1
                
                # Get retrieved results
                retrieved = (
                    getattr(prediction, "results", None) or
                    getattr(prediction, "retrieved", None) or 
                    []
                )

                if not retrieved:
                    continue

                successful_queries += 1

                # Calculate mean similarity
                sim_scores = [
                    res.get("similarity", 0.0) 
                    for res in retrieved 
                    if "error" not in res
                ]
                if sim_scores:
                    mean_sim = sum(sim_scores) / len(sim_scores)
                    all_similarities.append(mean_sim)

                # Calculate precision based on problem type matching
                hits = 0
                for res in retrieved:
                    if "error" in res:
                        continue
                    
                    res_type = res.get("metadata", {}).get("problem_type", "")
                    res_text = res.get("text", "").lower()
                    
                    # Naive relevance: check if query keywords appear in result
                    query_words = set(query.lower().split())
                    res_words = set(res_text.split())
                    overlap = len(query_words.intersection(res_words))
                    
                    # Consider relevant if >30% word overlap or problem type matches
                    if overlap / len(query_words) > 0.3 or (res_type and res_type.lower() in query.lower()):
                        hits += 1
                
                precision = hits / top_k if top_k > 0 else 0
                all_precisions.append(precision)
                
            except Exception as e:
                print(f"\n⚠️ Warning: Query failed: {e}")
                continue

        print(" " * 50, end="\r")  # Clear progress line
        print(f"\n📈 Evaluation Results (on {successful_queries} successful queries):")
        print("-" * 70)
        
        if all_precisions:
            print(f"  Precision@{top_k}: {np.mean(all_precisions):.3f} ± {np.std(all_precisions):.3f}")
        else:
            print(f"  Precision@{top_k}: N/A (no valid results)")
        
        if all_similarities:
            print(f"  Mean Similarity Score: {np.mean(all_similarities):.3f} ± {np.std(all_similarities):.3f}")
        else:
            print(f"  Mean Similarity Score: N/A")
        
        print(f"\n🔧 Solution Statistics:")
        print(f"  Symbolic Solver: {symbolic_solves} ({symbolic_solves/len(queries)*100:.1f}%)")
        print(f"  LLM Fallback: {llm_solves} ({llm_solves/len(queries)*100:.1f}%)")
        print(f"  Unresolved: {unresolved} ({unresolved/len(queries)*100:.1f}%)")
        print()

    def interactive_mode(self):
        """Run interactive query mode."""
        print("\n" + "=" * 70)
        print("🎯 INTERACTIVE MODE")
        print("=" * 70)
        print("Enter your math problem queries (or 'quit' to exit):\n")
        
        while True:
            try:
                query = input("Query > ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                print()
                self.find_most_similar_problem(query, explain=True, top_k=3)
                print("=" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Error: {e}\n")


# --------------------------------------------------------------------
#  DEMO RUNNER
# --------------------------------------------------------------------
def main():
    demo = SmartRetrievalDemo()

    if not demo.load_system():
        print("\n❌ System initialization failed. Please check:")
        print("   1. Index files exist at specified paths")
        print("   2. All dependencies are installed")
        print("   3. DSPy is properly configured")
        return

    print("\n" + "=" * 70)
    print("CALCMATE DEMO: MATHEMATICAL SIMILARITY SEARCH + NEURO-SYMBOLIC SOLVER")
    print("Find problems that are structurally similar and attempt symbolic/LLM solutions!")
    print("=" * 70 + "\n")

    test_queries = [
        "A motorcycle travels 180 kilometers in 3 hours. What is its average speed?",
        "If a pizza costs $18 and you order 3 pizzas, what's the total cost?",
        "A rectangular playground has length 12 meters and width 8 meters. Calculate its area.",
        "Two numbers sum to 50 and their difference is 10. What are the numbers?",
    ]

    print("🚀 RUNNING SMART RETRIEVAL + NEURO-SYMBOLIC TESTS...\n")
    for i, query in enumerate(test_queries, 1):
        print(f"TEST {i}/{len(test_queries)}:")
        demo.find_most_similar_problem(query, explain=True, top_k=3)
        if i < len(test_queries):
            print("═" * 70 + "\n")

    # Run evaluation
    demo.run_evaluation(test_queries, top_k=3)

    print("=" * 70)
    print("✅ DEMO COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    # Optional: interactive mode
    try:
        response = input("Would you like to try interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            demo.interactive_mode()
    except (KeyboardInterrupt, EOFError):
        print("\n\nGoodbye! 👋")


if __name__ == "__main__":
    main()