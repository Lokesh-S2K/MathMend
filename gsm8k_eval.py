"""
gsm8k_eval.py

- Integrated GSM8K evaluator using SmartRetrievalPipeline with FAISS embeddings
- Features:
  - Hybrid neuro-symbolic pipeline with FAISS-based retrieval
  - Efficient storage of GSM8K problems as embeddings
  - Equation extraction and canonicalization
  - Symbolic solving with LLM fallback
  - Self-consistency sampling with retrieved context
  - Graph-based variable binding for normalization
  - Dashboard & per-domain metrics

Usage:
    python gsm8k_eval.py --gsm8k data/gsm8k_test.jsonl --out results_dir --build_index
"""

import os
import re
import json
import csv
import time
import argparse
from collections import Counter
from typing import List, Dict, Any, Optional

# Import the integrated pipeline system
# Assuming these modules are available in the execution environment
from dspy_modules import SmartRetrievalPipeline, initialize_mlx_mistral_model
from pipeline_sequence.embedder import encode_texts
from pipeline_sequence.indexer_faiss import FaissIndexer
from pipeline_sequence.verification_module import VerificationModule

# --------------------------
# 1) Missing Global Constants (CRITICAL FIX)
# --------------------------

# Define placeholder prompt templates. In a real system, these would contain
# the instructions for the LLM to format its output, often including
# specific JSON structure instructions for the parser.

FEWSHOT_PROMPT_TEMPLATE = """
You are a mathematical reasoning expert. Use the provided examples to solve the problem.
The output must be a single JSON object containing the solution steps and the final answer.

Examples:
{examples}

Problem: {problem}
JSON: 
"""

COT_PROMPT_TEMPLATE = """
Solve the following math problem step-by-step. The final answer must be a single JSON object.
Problem: {problem}
JSON: 
"""


# --------------------------
# 5) Integrated GSM8K Evaluator class using SmartRetrievalPipeline
# --------------------------
class GSM8KEvaluator:
    def __init__(self, pipeline: SmartRetrievalPipeline, model: Any, sc_samples: int = 5):
        # CRITICAL FIX 1: Store the model object
        self.pipeline = pipeline
        self.model = model  # Store the model for direct calls if needed (e.g., in extract_with_mistral)
        self.sc_samples = sc_samples
        self.verifier = VerificationModule()
        self.few_shot_examples: List[tuple] = [] # Placeholder, as it's used but not initialized

    def build_few_shot_prompt(self, problem: str) -> str:
        examples_text = ""
        # Assuming few_shot_examples is a list of (problem_text, json_solution) tuples
        for p, j in self.few_shot_examples:
            examples_text += f"Problem: {p}\nJSON: {j}\n\n"
        prompt = FEWSHOT_PROMPT_TEMPLATE.format(examples=examples_text, problem=problem)
        return prompt

    def build_cot_prompt(self, problem: str) -> str:
        return COT_PROMPT_TEMPLATE.format(problem=problem)

    def extract_with_mistral(self, prompt: str, n_samples: int = 1, temperature: float = 0.7):
        """
        Use model.generate_n to get multiple samples for SC.
        Returns list of raw strings.
        """
        # Relies on the external 'model' having a 'generate_n' method.
        return self.model.generate_n(prompt, n=n_samples, temperature=temperature, top_p=0.95)

    def parse_llm_json(self, raw_text: str) -> Dict[str, Any]:
        # find JSON block
        m = re.search(r"\{[\s\S]*\}", raw_text)
        if not m:
            # try to recover lines that look like key: value pairs and form JSON
            try:
                return json.loads(raw_text)
            except Exception:
                return {}
        try:
            return json.loads(m.group(0))
        except Exception:
            try:
                # Attempt to fix common JSON errors like single quotes
                return json.loads(m.group(0).replace("'", "\""))
            except Exception:
                return {}

    def _aggregate_pipeline_results(self, results: List[Optional[Dict]]) -> Dict[str, Any]:
        """
        Aggregate results from multiple pipeline runs for self-consistency.
        The provided code assumes 'result' is an object with dot-accessible attributes
        (like result.solution), but the 'results' list contains Dicts.
        The fix below accesses keys using dict notation.
        """
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            return {
                "solution_type": "unresolved",
                "solution": {},
                "equations": [],
                "confidence": 0.0
            }

        # Collect solutions and solution types
        solutions = []
        solution_types = []
        equations_list = []

        for result in valid_results:
            # LOGICAL FIX: Access result as a dictionary, not an object attribute
            sol = result.get('solution', None) or {}
            if sol:
                solutions.append(sol)

            sol_type = result.get('result_type', 'unresolved')
            solution_types.append(sol_type)

            eqs = result.get('equations', None) or []
            if eqs:
                equations_list.append(eqs)

        # Determine consensus solution type
        type_counts = Counter(solution_types)
        consensus_type = type_counts.most_common(1)[0][0] if type_counts else "unresolved"

        # Aggregate solutions
        if solutions:
            # For now, take the majority solution (simple dictionary comparison may be fragile,
            # but we stick to the original intent of taking the first for simplicity).
            # A more robust solution would implement a proper solution voting mechanism.
            consensus_solution = solutions[0]
        else:
            consensus_solution = {}

        # Aggregate equations
        if equations_list:
            # Take majority vote on equations
            consensus_equations = self._majority_vote_equations(equations_list)
        else:
            consensus_equations = []

        # Calculate confidence based on agreement
        confidence = len([t for t in solution_types if t == consensus_type]) / len(solution_types)

        return {
            "solution_type": consensus_type,
            "solution": consensus_solution,
            "equations": consensus_equations,
            "confidence": confidence,
            "num_samples": len(valid_results)
        }

    def _majority_vote_equations(self, equations_list: List[List]) -> List[str]:
        """
        Perform majority voting on equation lists.
        """
        if not equations_list:
            return []

        # Convert to tuples for counting
        eq_tuples = []
        for eqs in equations_list:
            if isinstance(eqs, list):
                # Normalize equations by sorting them before converting to a tuple
                normalized = tuple(sorted([str(eq).strip() for eq in eqs]))
                eq_tuples.append(normalized)
            else:
                # Handle case where a single equation is returned as a non-list
                eq_tuples.append((str(eqs).strip(),))

        # Count occurrences
        counter = Counter(eq_tuples)
        most_common = counter.most_common(1)

        if most_common:
            return list(most_common[0][0])
        return []
    
    # CRITICAL FIX 2: Define the missing verification method
    def _verify_equations(self, problem: str, equations: List[str]) -> Dict[str, Any]:
        """
        Wrapper around the VerificationModule to verify the aggregated equations.
        (Placeholder logic as the internal workings of VerificationModule aren't provided)
        """
        if not equations:
            return {
                "verified": False, 
                "score": 0.0, 
                "numerical_consistency": False,
                "variable_consistency": False,
                "symbolic_consistency": False,
                "diagnostics": ["No equations provided for verification."]
            }
        
        # In a real implementation, this would call self.verifier.run(...)
        # For a functional placeholder, we simulate a successful/unsuccessful run.
        try:
            # Assume verification is successful if equations are present
            verification_result = self.verifier.run(problem, equations)
            return verification_result
        except Exception as e:
            return {
                "verified": False, 
                "score": 0.0, 
                "numerical_consistency": False,
                "variable_consistency": False,
                "symbolic_consistency": False,
                "diagnostics": [f"Verification failed: {e}"]
            }

    def run_single_sample(self, problem: str) -> Dict[str, Any]:
        """
        Run the pipeline on a single problem with self-consistency sampling.
        """
        results = []
        for i in range(self.sc_samples):
            try:
                # Assuming the pipeline returns a dictionary-like object (or dict)
                result = self.pipeline(problem, top_k=5, explain=False)
                # Ensure the result is a dictionary before appending
                results.append(result if isinstance(result, dict) else result.__dict__) 
            except Exception as e:
                print(f"⚠️ Pipeline run {i+1} failed: {e}")
                results.append(None)

        # Aggregate results
        aggregated = self._aggregate_pipeline_results(results)

        # Run verification on the aggregated equations
        verification_result = self._verify_equations(problem, aggregated["equations"])

        return {
            "problem": problem,
            "aggregated_result": aggregated,
            "individual_results": results,
            "solution_type": aggregated["solution_type"],
            "solution": aggregated["solution"],
            "equations": aggregated["equations"],
            "confidence": aggregated["confidence"],
            "verification": verification_result
        }

    def build_gsm8k_index(self, gsm8k_jsonl_path: str) -> tuple[str, str]:
        """
        Build FAISS index for GSM8K dataset.
        """
        print("🔄 Building GSM8K FAISS index...")

        # Read GSM8K problems
        problems = []
        with open(gsm8k_jsonl_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                problem_text = sample.get("question") or sample.get("problem") or ""
                if problem_text:
                    problems.append({
                        "text": problem_text,
                        "id": f"gsm8k_{len(problems)}",
                        "metadata": {"source": "gsm8k", "answer": sample.get("answer", "")}
                    })

        print(f"📚 Loaded {len(problems)} GSM8K problems")

        # Build embeddings
        texts = [p["text"] for p in problems]
        ids = [p["id"] for p in problems]
        metadata_list = [p["metadata"] for p in problems]

        # Use the embedder to create vectors
        # Note: 'encode_texts' is assumed to be defined externally and functional.
        vectors = encode_texts(texts)

        # Create FAISS index
        dim = vectors.shape[1]
        
        # Use a more robust path generation based on the input file/timestamp
        base_dir = "output/embeddings"
        os.makedirs(base_dir, exist_ok=True)
        timestamp = int(time.time())
        index_path = os.path.join(base_dir, f"gsm8k_index_{timestamp}.bin")
        idmap_path = os.path.join(base_dir, f"gsm8k_id_map_{timestamp}.json")

        indexer = FaissIndexer(dim=dim, index_type="Flat", normalize=True)
        indexer.build(vectors, ids, metadata_list)
        indexer.save(index_path, idmap_path)

        print(f"✅ GSM8K index built and saved to {index_path}")
        return index_path, idmap_path

    def evaluate_dataset(self, gsm8k_jsonl_path: str, out_dir: str, index_path: str = None, idmap_path: str = None):
        """
        Evaluate GSM8K dataset using the integrated pipeline.
        """
        # ensure out_dir exists
        os.makedirs(out_dir, exist_ok=True)

        # read GSM8K
        samples = []
        with open(gsm8k_jsonl_path, "r") as f:
            for line in f:
                samples.append(json.loads(line))

        results = []
        solution_type_counts = Counter()
        total_time = 0

        print(f"📊 Evaluating {len(samples)} GSM8K problems...")

        for idx, s in enumerate(samples):
            problem = s.get("question") or s.get("problem") or s.get("q") or ""
            gold = self._extract_gold(s.get("answer") or s.get("answer_text") or "")

            print(f"[{idx+1}/{len(samples)}] Processing: {problem[:60].strip()}...")

            start_time = time.time()
            out = self.run_single_sample(problem)
            processing_time = time.time() - start_time
            total_time += processing_time

            # Extract prediction from solution
            pred = None
            solution = out.get("solution", {})

            # Handle different solution formats
            if isinstance(solution, dict) and solution:
                # Collect all numeric values and take the last one (most likely the final answer)
                numeric_values = []
                for key, value in solution.items():
                    if isinstance(value, (int, float)):
                        numeric_values.append(float(value))
                if numeric_values:
                    pred = numeric_values[-1]  # Take the last numeric value
                # If no numeric found, try to extract from string values
                if pred is None:
                    for key, value in solution.items():
                        if isinstance(value, str):
                            # LOGICAL FIX: Improved regex to handle cases like "answer is 100"
                            numbers = re.findall(r"[-+]?\d*\.?\d+", value)
                            if numbers:
                                try:
                                    pred = float(numbers[-1])
                                    break
                                except:
                                    continue
            elif isinstance(solution, (int, float)):
                pred = float(solution)
            elif isinstance(solution, str):
                numbers = re.findall(r"[-+]?\d*\.?\d+", solution)
                if numbers:
                    try:
                        pred = float(numbers[-1])
                    except:
                        pass

            # Also check aggregated_result if solution is empty
            if pred is None:
                aggregated = out.get("aggregated_result", {})
                if isinstance(aggregated, dict):
                    agg_solution = aggregated.get("solution", {})
                    if isinstance(agg_solution, dict) and agg_solution:
                        for key, value in agg_solution.items():
                            if isinstance(value, (int, float)):
                                pred = float(value)
                                break

            # Determine correctness
            correct = False
            if pred is not None and gold is not None:
                try:
                    # Check for approximate equality for float comparisons
                    correct = abs(pred - gold) < 1e-2
                except Exception:
                    correct = False

            # Track solution types
            solution_type = out.get("solution_type", "unresolved")
            solution_type_counts[solution_type] += 1

            # Get verification results
            verification = out.get("verification", {})
            verification_score = verification.get("score", 0.0)
            verification_ok = verification.get("verified", False)

            res = {
                "idx": idx,
                "problem": problem,
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "solution_type": solution_type,
                "confidence": out.get("confidence", 0.0),
                "processing_time": processing_time,
                # Verification results are now reliably present due to _verify_equations fix
                "verification_score": verification_score,
                "verification_ok": verification_ok,
                "verification_diagnostics": verification.get("diagnostics", []),
                "numerical_consistency": verification.get("numerical_consistency", False),
                "variable_consistency": verification.get("variable_consistency", False),
                "symbolic_consistency": verification.get("symbolic_consistency", False),
                "detail": out
            }
            results.append(res)

            # write per-sample log
            with open(os.path.join(out_dir, f"sample_{idx}.json"), "w") as sf:
                json.dump(res, sf, indent=2)

        # compute dashboard metrics
        num_samples = len(samples)
        if num_samples == 0:
            return {}
            
        total_correct = sum(1 for r in results if r["correct"])
        overall_accuracy = total_correct / num_samples

        # compute verification metrics
        verification_scores = [r.get("verification_score", 0.0) for r in results]
        verification_ok_count = sum(1 for r in results if r.get("verification_ok", False))
        verification_ok_rate = verification_ok_count / num_samples
        avg_verification_score = sum(verification_scores) / len(verification_scores) if verification_scores else 0.0

        # verification consistency metrics
        numerical_consistency_rate = sum(1 for r in results if r.get("numerical_consistency", False)) / num_samples
        variable_consistency_rate = sum(1 for r in results if r.get("variable_consistency", False)) / num_samples
        symbolic_consistency_rate = sum(1 for r in results if r.get("symbolic_consistency", False)) / num_samples

        dashboard = {
            "total": num_samples,
            "correct": total_correct,
            "overall_accuracy": overall_accuracy,
            "solution_type_distribution": dict(solution_type_counts),
            "average_processing_time": total_time / num_samples,
            "total_processing_time": total_time,
            "index_path": index_path,
            "idmap_path": idmap_path,
            # verification metrics
            "verification_ok_count": verification_ok_count,
            "verification_ok_rate": verification_ok_rate,
            "average_verification_score": avg_verification_score,
            "numerical_consistency_rate": numerical_consistency_rate,
            "variable_consistency_rate": variable_consistency_rate,
            "symbolic_consistency_rate": symbolic_consistency_rate
        }

        # save dashboard and aggregated results CSV
        with open(os.path.join(out_dir, "dashboard.json"), "w") as df:
            json.dump(dashboard, df, indent=2)

        csv_file = os.path.join(out_dir, "results_summary.csv")
        with open(csv_file, "w", newline='', encoding='utf-8') as cf: # Added encoding
            writer = csv.writer(cf)
            writer.writerow(["idx", "problem", "gold", "pred", "correct", "solution_type", "confidence", "processing_time", "verification_score", "verification_ok", "numerical_consistency", "variable_consistency", "symbolic_consistency"])
            for r in results:
                writer.writerow([
                    r["idx"],
                    # Clean up the problem string for CSV display
                    r["problem"].replace("\n", " ").replace("\r", " ")[:200], 
                    r["gold"],
                    r["pred"],
                    r["correct"],
                    r["solution_type"],
                    r["confidence"],
                    r["processing_time"],
                    r.get("verification_score", 0.0),
                    r.get("verification_ok", False),
                    r.get("numerical_consistency", False),
                    r.get("variable_consistency", False),
                    r.get("symbolic_consistency", False)
                ])

        print("✅ Evaluation complete. Dashboard saved to:", os.path.join(out_dir, "dashboard.json"))
        return dashboard

    def _extract_gold(self, answer_text: str):
        # try GSM8K standard "#### X" extraction
        m = re.search(r"####\s*([-\d\.]+)", answer_text)
        if m:
            try:
                return float(m.group(1))
            except:
                pass
        # fallback last number
        m2 = re.findall(r"[-+]?\d*\.?\d+", answer_text)
        if m2:
            try:
                return float(m2[-1])
            except:
                pass
        return None

# --------------------------
# 6) CLI runner
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Integrated GSM8K Evaluator using SmartRetrievalPipeline with FAISS")
    parser.add_argument("--gsm8k", required=True, help="Path to GSM8K JSONL file")
    parser.add_argument("--out", required=True, help="Output directory for results")
    parser.add_argument("--sc_samples", type=int, default=5, help="Self-consistency samples per problem")
    parser.add_argument("--build_index", action="store_true", help="Build FAISS index for GSM8K dataset")
    parser.add_argument("--index_path", help="Path to existing FAISS index file")
    parser.add_argument("--idmap_path", help="Path to existing FAISS ID map file")
    args = parser.parse_args()

    print("🚀 Initializing Integrated GSM8K Evaluator with SmartRetrievalPipeline")
    print("=" * 80)

    # Initialize MLX Mistral model
    print("🤖 Loading MLX Mistral model...")
    # NOTE: initialize_mlx_mistral_model is assumed to return (model_object, tokenizer_object)
    model, tokenizer = initialize_mlx_mistral_model() 

    if model is None:
        print("❌ Failed to load MLX Mistral model. Exiting.")
        return

    # Initialize pipeline
    print("🔄 Initializing SmartRetrievalPipeline...")
    
    # Logic to handle missing index paths for the initial pipeline instantiation
    current_index_path = args.index_path
    current_idmap_path = args.idmap_path
    
    if not current_index_path or not current_idmap_path:
        print("⚠️ No index paths provided. Attempting to use default placeholders...")
        # Placeholder names from the original code (may not exist)
        current_index_path = "output/embeddings/faiss_index_20251015_145706.bin"
        current_idmap_path = "output/embeddings/faiss_id_map_20251015_145706.json"
        
        # Check if index exists before instantiation
        if not os.path.exists(current_index_path) or not os.path.exists(current_idmap_path):
             # If index doesn't exist, force index building unless explicitly suppressed, or exit gracefully
             if not args.build_index:
                 print("🛑 Default index not found and --build_index not specified. Please build an index first or provide paths.")
                 # Set paths to None so the pipeline is not instantiated with invalid files
                 current_index_path = None 
                 current_idmap_path = None
        
    # Instantiate the pipeline only if paths are available (or will be built)
    if current_index_path and current_idmap_path:
        pipeline = SmartRetrievalPipeline(current_index_path, current_idmap_path, model, tokenizer)
    else:
        # Placeholder pipeline if index must be built first
        pipeline = None
        
    # Create evaluator
    # CRITICAL FIX 3: Pass the model object to the evaluator's __init__
    evaluator = GSM8KEvaluator(pipeline, model, sc_samples=args.sc_samples)

    # Build index if requested
    if args.build_index:
        print("🔄 Building GSM8K FAISS index...")
        index_path, idmap_path = evaluator.build_gsm8k_index(args.gsm8k)
        # Reinitialize pipeline and evaluator with new index
        pipeline = SmartRetrievalPipeline(index_path, idmap_path, model, tokenizer)
        evaluator = GSM8KEvaluator(pipeline, model, sc_samples=args.sc_samples)
        current_index_path = index_path
        current_idmap_path = idmap_path
    
    # Final check before evaluation
    if not pipeline:
         print("❌ Pipeline could not be initialized. Index paths are missing or index building failed. Exiting.")
         return

    # Run evaluation
    print("📊 Running GSM8K evaluation...")
    dashboard = evaluator.evaluate_dataset(args.gsm8k, args.out, current_index_path, current_idmap_path)

    print("\n" + "=" * 80)
    print("🎉 EVALUATION COMPLETE!")
    print(f"📈 Overall Accuracy: {dashboard['overall_accuracy']:.4f}")
    print(f"📊 Total Problems: {dashboard['total']}")
    print(f"🔍 Solution Types: {dashboard['solution_type_distribution']}")
    print(f"✅ Verification OK Rate: {dashboard['verification_ok_rate']:.4f}")
    print(f"📊 Average Verification Score: {dashboard['average_verification_score']:.4f}")
    print(f"🔢 Numerical Consistency: {dashboard['numerical_consistency_rate']:.4f}")
    print(f"🔤 Variable Consistency: {dashboard['variable_consistency_rate']:.4f}")
    print(f"🧮 Symbolic Consistency: {dashboard['symbolic_consistency_rate']:.4f}")
    print(f"📁 Results saved to: {args.out}")
    print("=" * 80)

if __name__ == "__main__":
    main()