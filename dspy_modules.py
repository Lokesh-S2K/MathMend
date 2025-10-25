import dspy
import numpy as np
import os
import faiss
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_community.llms import CTransformers
import requests
from pipeline_sequence.advanced_equation_extractor import extract_equations_advanced
from pipeline_sequence.canonicalizer import canonicalize_system
from pipeline_sequence.features import build_structure_vector_from_parsed
from pipeline_sequence.embedder import encode_texts, build_text_for_embedding
from pipeline_sequence.indexer_faiss import FaissIndexer
import json
import re
from typing import List, Dict, Any, Optional

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sympy import symbols, solve, Eq


# --------------------------
# CTransformers LLM Initialization
# --------------------------
def initialize_ctransformers_model():
    """Initialize CTransformers model for local LLM inference"""
    try:
        print("🔄 Loading CTransformers model...")
        print("📥 This may take a few minutes on first run as the model downloads...")
        
        llm = CTransformers(
            model="TheBloke/Llama-2-7B-Chat-GGML",
            model_file="llama-2-7b-chat.ggmlv3.q4_0.bin",  # Specify model file
            model_type="llama",
            config={
                "max_new_tokens": 256, 
                "temperature": 0.5,
                "context_length": 2048  # Added context length
            }
        )
        print("✅ LLM Model Loaded Successfully!")
        return llm
    except Exception as e:
        print(f"❌ Failed to load CTransformers model: {e}")
        print("💡 The model will be downloaded automatically from Hugging Face")
        print("💡 Make sure you have sufficient disk space (~4GB) and internet connection")
        return None


# --------------------------
# Symbolic Solver
# --------------------------
class SymbolicSolver(dspy.Module):
    """Attempts to solve equations symbolically using SymPy"""
    
    def forward(self, canonical_equations):
        try:
            if not canonical_equations:
                return dspy.Prediction(
                    solution={}, 
                    success=False, 
                    residuals={}, 
                    error_msg="No equations provided"
                )
            
            # Handle different types of canonical equations
            equations_to_solve = []
            for eq in canonical_equations:
                if isinstance(eq, str):
                    # Try to parse string equations
                    try:
                        from sympy import sympify
                        parsed_eq = sympify(eq)
                        equations_to_solve.append(parsed_eq)
                    except:
                        continue
                elif hasattr(eq, 'free_symbols'):
                    # Already a sympy expression
                    equations_to_solve.append(eq)
            
            if not equations_to_solve:
                return dspy.Prediction(
                    solution={}, 
                    success=False, 
                    residuals={}, 
                    error_msg="No valid equations to solve"
                )
            
            syms = sorted(
                {s for e in equations_to_solve for s in e.free_symbols}, 
                key=lambda x: str(x)
            )
            
            if not syms:
                return dspy.Prediction(
                    solution={}, 
                    success=False, 
                    residuals={}, 
                    error_msg="No variables found"
                )
            
            sol = solve(equations_to_solve, syms, dict=True)
            
            if not sol:
                return dspy.Prediction(
                    solution={}, 
                    success=False, 
                    residuals={}, 
                    error_msg="No solution found"
                )
            
            sol0 = sol[0]
            residuals = {}
            for e in equations_to_solve:
                try:
                    # For equations like Eq(x + y, 50), we need to evaluate the difference
                    if hasattr(e, 'lhs') and hasattr(e, 'rhs'):
                        # This is an equation: evaluate lhs - rhs
                        lhs_sub = e.lhs.subs(sol0)
                        rhs_sub = e.rhs.subs(sol0)
                        if hasattr(lhs_sub, 'evalf'):
                            lhs_val = float(lhs_sub.evalf())
                        else:
                            lhs_val = float(lhs_sub)
                        if hasattr(rhs_sub, 'evalf'):
                            rhs_val = float(rhs_sub.evalf())
                        else:
                            rhs_val = float(rhs_sub)
                        val = lhs_val - rhs_val
                    else:
                        # This is an expression: evaluate directly
                        substituted = e.subs(sol0)
                        if hasattr(substituted, 'evalf'):
                            val = float(substituted.evalf())
                        else:
                            val = float(substituted)
                    residuals[str(e)] = val
                except Exception as ex:
                    residuals[str(e)] = f"Error: {str(ex)}"
            
            # Check success - only consider numeric residuals
            numeric_residuals = [v for v in residuals.values() if isinstance(v, (int, float))]
            success = len(numeric_residuals) > 0 and all(abs(v) < 1e-6 for v in numeric_residuals)
            
            return dspy.Prediction(
                solution={str(k): float(v) for k, v in sol0.items()},
                success=success,
                residuals=residuals,
                error_msg=None
            )
        except Exception as e:
            return dspy.Prediction(
                solution={}, 
                success=False, 
                residuals={}, 
                error_msg=f"Solver error: {str(e)}"
            )


# --------------------------
# Verifier Module
# --------------------------
class Verifier(dspy.Module):
    """Verifies if a solution satisfies the canonical equations"""
    
    def __init__(self, tolerance: float = 1e-6):
        super().__init__()
        self.tolerance = tolerance
    
    def forward(self, canonical_equations, candidate_solution, retrieved_solutions=None):
        if not canonical_equations:
            return dspy.Prediction(
                verification={'ok': False, 'residuals': {}, 'error': 'No equations to verify'}
            )
        
        residuals = {}
        for e in canonical_equations:
            try:
                substitutions = {
                    k: candidate_solution.get(str(k), 0) 
                    for k in e.free_symbols
                }
                
                # For equations like Eq(x + y, 50), we need to evaluate the difference
                if hasattr(e, 'lhs') and hasattr(e, 'rhs'):
                    # This is an equation: evaluate lhs - rhs
                    lhs_sub = e.lhs.subs(substitutions)
                    rhs_sub = e.rhs.subs(substitutions)
                    if hasattr(lhs_sub, 'evalf'):
                        lhs_val = float(lhs_sub.evalf())
                    else:
                        lhs_val = float(lhs_sub)
                    if hasattr(rhs_sub, 'evalf'):
                        rhs_val = float(rhs_sub.evalf())
                    else:
                        rhs_val = float(rhs_sub)
                    val = lhs_val - rhs_val
                else:
                    # This is an expression: evaluate directly
                    substituted = e.subs(substitutions)
                    if hasattr(substituted, 'evalf'):
                        val = float(substituted.evalf())
                    else:
                        val = float(substituted)
                
                residuals[str(e)] = {'value': val, 'satisfied': abs(val) < self.tolerance}
            except Exception as ex:
                residuals[str(e)] = {'value': None, 'error': str(ex)}
        
        ok = all(
            r.get('satisfied', False) 
            for r in residuals.values() 
            if isinstance(r, dict)
        )
        
        return dspy.Prediction(
            verification={
                'ok': ok, 
                'residuals': residuals,
                'tolerance': self.tolerance
            }
        )


# --------------------------
# LLM Reasoner with Fixed Prompting
# --------------------------
class LLMReasoner(dspy.Module):
    """Uses LLM to solve problems when symbolic methods fail"""
    
    def __init__(self, llm_model=None):
        super().__init__()
        self.llm_model = llm_model
        if self.llm_model is None:
            self.llm_model = initialize_ctransformers_model()
    
    def build_prompt(self, query: str, canonical_eqs: List, 
                    retrieved_examples: List[Dict], max_shots: int = 3) -> str:
        """Build few-shot prompt using string concatenation"""
        prompt_parts = ["# Mathematical Problem Solving\n"]
        
        # Add examples
        for i, ex in enumerate(retrieved_examples[:max_shots], 1):
            prompt_parts.append(f"\n## Example {i}:")
            prompt_parts.append(f"Problem: {ex.get('text', ex.get('problem_text', 'N/A'))}")
            
            if 'equations' in ex:
                eqs = ex['equations'] if isinstance(ex['equations'], list) else [ex['equations']]
                prompt_parts.append(f"Equations: {'; '.join(str(e) for e in eqs)}")
            
            if 'reasoning_steps' in ex:
                prompt_parts.append(f"Solution Steps:\n{ex['reasoning_steps']}")
            
            if 'solution' in ex:
                prompt_parts.append(f"Final Solution: {ex['solution']}")
        
        # Add current problem
        prompt_parts.append("\n## Current Problem:")
        prompt_parts.append(f"Problem: {query}")
        prompt_parts.append(f"Equations: {'; '.join(str(e) for e in canonical_eqs)}")
        prompt_parts.append("\nProvide step-by-step reasoning and final solution.")
        prompt_parts.append("Format your answer as JSON with 'reasoning' and 'solution' fields.")
        
        return "\n".join(prompt_parts)
    
    def forward(self, query_text: str, canonical_equations: List, 
               retrieved_examples: List[Dict]):
        try:
            prompt = self.build_prompt(query_text, canonical_equations, retrieved_examples)
            
            # Use CTransformers model if available, otherwise fallback to DSPy
            if self.llm_model is not None:
                try:
                    # Use CTransformers model directly
                    response = self.llm_model.invoke(prompt, max_new_tokens=256, temperature=0.5)
                    if isinstance(response, str):
                        response_text = response
                    else:
                        response_text = str(response)
                except Exception as e:
                    print(f"⚠️ CTransformers model failed: {e}")
                    response_text = f"Error with CTransformers model: {str(e)}"
            else:
                # Fallback to DSPy Predict
                try:
                    class ReasoningSignature(dspy.Signature):
                        """Solve math problem with reasoning"""
                        prompt = dspy.InputField()
                        response = dspy.OutputField()
                    
                    predictor = dspy.Predict(ReasoningSignature)
                    result = predictor(prompt=prompt)
                    response_text = result.response
                except Exception as e:
                    response_text = f"Error with DSPy Predict: {str(e)}"
            
            parsed_equations, parsed_solution = self._parse_response(response_text)
            
            return dspy.Prediction(
                llm_steps=response_text, 
                llm_equations=parsed_equations, 
                llm_solution=parsed_solution,
                success=bool(parsed_solution)
            )
        except Exception as e:
            # Fallback: return a simple solution attempt
            return dspy.Prediction(
                llm_steps=f"Error: {str(e)}", 
                llm_equations=[], 
                llm_solution={},
                success=False
            )
    
    def _parse_response(self, response_text: str):
        """Parse LLM response to extract equations and solutions"""
        # Try JSON extraction first
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, flags=re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return data.get("equations", []), data.get("solution", {})
            except json.JSONDecodeError:
                pass
        
        # Fallback: extract variable assignments
        solution = {}
        for match in re.finditer(r'([A-Za-z]\w*)\s*=\s*([-+]?\d*\.?\d+)', response_text):
            var_name, value = match.groups()
            try:
                solution[var_name] = float(value)
            except ValueError:
                continue
        
        # Extract equation-like strings
        equations = re.findall(r'([A-Za-z0-9_+\-*/\s\(\)]+=[^,\n]+)', response_text)
        
        return equations, solution


# --------------------------
# Enhanced Retriever
# --------------------------
class Retriever(dspy.Module):
    """FAISS-based retrieval with error handling"""
    
    def __init__(self, index_path: str, idmap_path: str):
        super().__init__()
        self.index = FaissIndexer.load(index_path, idmap_path)
        
        with open(idmap_path) as f:
            self.db = json.load(f)
        
        self.max_idx = len(self.db) - 1
    
    def forward(self, hybrid_vector, top_k: int = 5):
        try:
            # FaissIndexer.search returns (distances, results_list)
            # where results_list is a list of dicts like {'id': ..., 'metadata': {...}}
            distances, results_list = self.index.search(hybrid_vector, top_k=top_k)
            
            # Ensure distances and results_list are properly formatted
            if hasattr(distances, 'tolist'):
                distances = distances.tolist()
            
            # Handle single query case (not batched)
            if isinstance(distances, list) and len(distances) > 0 and not isinstance(distances[0], (list, tuple)):
                distances = [distances]
                results_list = [results_list]
            
            # Take the first (and only) query result
            if len(distances) > 0:
                distances = distances[0]
                results_list = results_list[0]
            
        except Exception as e:
            return dspy.Prediction(
                results=[], 
                error=f"Search failed: {str(e)}"
            )
        
        retrieved = []
        
        for dist, entry in zip(distances, results_list):
            try:
                # The 'entry' already contains the id and metadata from FaissIndexer
                text = entry.get("metadata", {}).get("text", entry.get("metadata", {}).get("problem_text", "No text available"))
                
                retrieved.append({
                    "similarity": float(dist),
                    "text": text,
                    "id": entry.get("id"),
                    "metadata": entry.get("metadata", {})
                })
            except Exception as e:
                retrieved.append({
                    "similarity": float(dist),
                    "text": f"[Error processing entry: {str(e)}]",
                    "error": str(e)
                })
        
        return dspy.Prediction(results=retrieved)


# --------------------------
# Evaluation Module
# --------------------------
class RetrievalEvaluator(dspy.Module):
    """Evaluate retrieval quality with multiple metrics"""
    
    def forward(self, query_vectors: np.ndarray, target_vectors: np.ndarray, 
               top_k: int = 5, relevance_threshold: float = 0.7):
        """
        Compute retrieval evaluation metrics.
        
        Args:
            query_vectors: NxD query embeddings
            target_vectors: MxD target embeddings
            top_k: Number of results to consider
            relevance_threshold: Similarity threshold for relevance
        """
        if len(query_vectors) == 0 or len(target_vectors) == 0:
            return dspy.Prediction(
                precision_at_k=0.0,
                mean_cosine_similarity=0.0,
                error="Empty input vectors"
            )
        
        # Compute similarities
        cos_sim = cosine_similarity(query_vectors, target_vectors)
        
        precision_scores = []
        mean_cos_scores = []
        mrr_scores = []  # Mean Reciprocal Rank
        
        for i in range(len(query_vectors)):
            # Get top-k indices
            top_indices = np.argsort(cos_sim[i])[::-1][:top_k]
            top_sim_values = cos_sim[i][top_indices]
            
            # Precision@k
            relevant_count = np.sum(top_sim_values >= relevance_threshold)
            precision = relevant_count / top_k
            precision_scores.append(precision)
            
            # Mean similarity
            mean_cos_scores.append(float(np.mean(top_sim_values)))
            
            # MRR: reciprocal rank of first relevant item
            relevant_ranks = np.where(top_sim_values >= relevance_threshold)[0]
            if len(relevant_ranks) > 0:
                mrr_scores.append(1.0 / (relevant_ranks[0] + 1))
            else:
                mrr_scores.append(0.0)
        
        return dspy.Prediction(
            precision_at_k=float(np.mean(precision_scores)),
            mean_cosine_similarity=float(np.mean(mean_cos_scores)),
            mean_reciprocal_rank=float(np.mean(mrr_scores)),
            per_query_precision=precision_scores,
            per_query_similarity=mean_cos_scores
        )


# --------------------------
# Similarity Explanation
# --------------------------
def explain_similarity(query: str, document: str, 
                      query_embedding: np.ndarray, 
                      doc_embedding: np.ndarray) -> Dict[str, Any]:
    """
    Explain why a document is similar to the query using multiple signals.
    """
    explanations = {}
    
    # Embedding cosine similarity
    cos_sim = float(cosine_similarity(
        query_embedding.reshape(1, -1),
        doc_embedding.reshape(1, -1)
    )[0][0])
    explanations['embedding_cosine_similarity'] = round(cos_sim, 4)
    
    # Keyword overlap
    try:
        vectorizer = CountVectorizer(stop_words='english', min_df=1)
        vectorizer.fit([query, document])
        qv = vectorizer.transform([query]).toarray()
        dv = vectorizer.transform([document]).toarray()
        keyword_sim = float(cosine_similarity(qv, dv)[0][0])
        explanations['keyword_overlap_score'] = round(keyword_sim, 4)
    except Exception:
        explanations['keyword_overlap_score'] = 0.0
    
    # Common tokens
    q_tokens = set(query.lower().split())
    d_tokens = set(document.lower().split())
    common = q_tokens.intersection(d_tokens)
    explanations['common_tokens'] = sorted(list(common))[:10]  # Limit to 10
    explanations['common_token_count'] = len(common)
    
    # Jaccard similarity
    jaccard = len(common) / len(q_tokens.union(d_tokens)) if q_tokens or d_tokens else 0.0
    explanations['jaccard_similarity'] = round(jaccard, 4)
    
    return explanations


# --------------------------
# Core Pipeline Modules
# --------------------------
class EquationExtractor(dspy.Module):
    def forward(self, text):
        equations = extract_equations_advanced(text)
        # Ensure we return a list of equation strings, not complex objects
        if equations and isinstance(equations, list):
            # Filter out non-string equations and convert to strings
            equation_strings = []
            for eq in equations:
                if isinstance(eq, str):
                    equation_strings.append(eq)
                elif hasattr(eq, '__str__'):
                    equation_strings.append(str(eq))
            return dspy.Prediction(equations=equation_strings)
        return dspy.Prediction(equations=[])


class Canonicalizer(dspy.Module):
    def forward(self, equations):
        if not equations:
            return dspy.Prediction(canonical=[])
        try:
            canonical_result = canonicalize_system(equations)
            # Extract the actual canonical equations from the result
            if isinstance(canonical_result, dict):
                # Get the parsed expressions (SymPy objects) instead of strings
                parsed_list = canonical_result.get('parsed', [])
                canonical = []
                for parsed_item in parsed_list:
                    if parsed_item.get('expr') is not None:
                        canonical.append(parsed_item['expr'])
                return dspy.Prediction(canonical=canonical)
            else:
                return dspy.Prediction(canonical=canonical_result)
        except Exception as e:
            return dspy.Prediction(canonical=[], error=str(e))


class StructureEncoder(dspy.Module):
    def forward(self, canonical):
        if not canonical:
            return dspy.Prediction(structure_vector=np.zeros(256))
        try:
            vec = build_structure_vector_from_parsed(canonical)
            return dspy.Prediction(structure_vector=vec)
        except Exception as e:
            return dspy.Prediction(structure_vector=np.zeros(256), error=str(e))


class TextEncoder(dspy.Module):
    def forward(self, text):
        try:
            vec = encode_texts([text], normalize=True)[0]
            return dspy.Prediction(text_vector=vec)
        except Exception as e:
            # Return zero vector on error
            return dspy.Prediction(text_vector=np.zeros(384), error=str(e))


class HybridEncoder(dspy.Module):
    def forward(self, structure_vector, text_vector):
        hybrid = np.concatenate([structure_vector, text_vector])
        return dspy.Prediction(hybrid_vector=hybrid)


# --------------------------
# Main Pipeline
# --------------------------
class SmartRetrievalPipeline(dspy.Module):
    """
    DSPy-based hybrid neuro-symbolic retrieval pipeline with:
    - Equation extraction and canonicalization
    - Hybrid embedding (structure + semantics)
    - FAISS vector search
    - Symbolic solver with verification
    - LLM fallback with reasoning
    - Comprehensive error handling
    """
    
    def __init__(self, index_path: str, idmap_path: str, llm_model=None):
        super().__init__()
        
        print("Loading DSPy Hybrid Neuro-Symbolic Retrieval System...")
        print("=" * 60)
        
        # Core modules
        self.eq_extractor = EquationExtractor()
        self.canonicalizer = Canonicalizer()
        self.structure_encoder = StructureEncoder()
        self.text_encoder = TextEncoder()
        self.hybrid_encoder = HybridEncoder()
        self.retriever = Retriever(index_path, idmap_path)
        self.index = self.retriever.index.index
        
        # Metadata
        with open(idmap_path) as f:
            self.problem_db = json.load(f)
        
        # Neuro-symbolic components
        self.sym_solver = SymbolicSolver()
        self.verifier = Verifier()
        self.llm_reasoner = LLMReasoner(llm_model)
        
        print(f"✓ System Loaded Successfully!")
        print(f"  Knowledge Base: {len(self.problem_db)} math problems")
        print(f"  Index dimension: {self.index.d}")
        if self.llm_reasoner.llm_model is not None:
            print(f"  LLM Model: CTransformers (Llama-2-7B-Chat)")
        else:
            print(f"  LLM Model: Not available")
        print()
    
    def forward(self, text: str, top_k: int = 5, explain: bool = False):
        """
        Main forward pass through the neuro-symbolic pipeline.
        """
        # Step 1: Extract equations
        eq_pred = self.eq_extractor(text)
        equations = eq_pred.equations if eq_pred.equations else self._pattern_based_extraction(text)
        
        # Step 2: Canonicalize
        can_pred = self.canonicalizer(equations)
        canonical = can_pred.canonical
        
        # Step 3: Structure encoding
        struct_pred = self.structure_encoder(canonical)
        
        # Step 4: Text encoding
        text_for_embedding = build_text_for_embedding(text, fingerprint=None)
        text_pred = self.text_encoder(text_for_embedding)
        
        # Step 5: Hybrid fusion
        hybrid_pred = self.hybrid_encoder(
            struct_pred.structure_vector, 
            text_pred.text_vector
        )
        
        # Step 6: Retrieval
        retr_pred = self.retriever(hybrid_pred.hybrid_vector, top_k=top_k)
        retrieved_results = retr_pred.results if hasattr(retr_pred, 'results') else []
        
        # Step 7: Attempt symbolic solving
        if canonical:
            sym_out = self.sym_solver(canonical)
            if getattr(sym_out, "success", False):
                ver = self.verifier(canonical, sym_out.solution)
                if ver.verification["ok"]:
                    return dspy.Prediction(
                        result_type="symbolic",
                        solution=sym_out.solution,
                        residuals=sym_out.residuals,
                        results=retrieved_results,
                        equations=canonical
                    )
        
        # Step 8: LLM fallback
        try:
            llm_out = self.llm_reasoner(text, canonical, retrieved_results)
            if getattr(llm_out, "success", False) and llm_out.llm_solution:
                if canonical:
                    ver2 = self.verifier(canonical, llm_out.llm_solution)
                    if ver2.verification["ok"]:
                        return dspy.Prediction(
                            result_type="llm",
                            solution=llm_out.llm_solution,
                            reasoning=llm_out.llm_steps,
                            results=retrieved_results,
                            equations=canonical
                        )
                else:
                    # No equations to verify, trust LLM
                    return dspy.Prediction(
                        result_type="llm",
                        solution=llm_out.llm_solution,
                        reasoning=llm_out.llm_steps,
                        results=retrieved_results
                    )
        except Exception as e:
            print(f"LLM reasoning failed: {e}")
        
        # Step 9: Unresolved - needs human intervention
        return dspy.Prediction(
            result_type="unresolved",
            results=retrieved_results,
            note="Unable to solve automatically - human review required",
            extracted_equations=equations,
            canonical_equations=canonical
        )
    
    def _pattern_based_extraction(self, text: str) -> List[str]:
        """
        Improved pattern-based fallback for equation extraction.
        """
        text_lower = text.lower()
        equations = []
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', text)
        
        if len(numbers) < 2:
            return equations
        
        # Distance/Speed problems
        if any(kw in text_lower for kw in ["speed", "distance", "km", "mph", "km/h"]):
            if "per" in text_lower or "/" in text:
                equations.append(f"speed = distance / time")
        
        # Cost/Price problems
        elif any(kw in text_lower for kw in ["cost", "price", "buy", "total"]):
            if "each" in text_lower:
                equations.append(f"total_cost = quantity * unit_price")
        
        # Percentage problems
        elif "%" in text or "percent" in text_lower:
            equations.append(f"result = base * (percentage / 100)")
        
        # System of equations keywords
        elif "sum" in text_lower and "difference" in text_lower:
            if len(numbers) >= 2:
                equations.extend([
                    f"x + y = {numbers[0]}",
                    f"x - y = {numbers[1]}"
                ])
        
        return equations
    
    def evaluate_retrieval_quality(self, queries: List[str], 
                                   top_k: int = 5, 
                                   sample_size: int = 1000):
        """
        Evaluate retrieval quality efficiently using sampling.
        """
        if not queries:
            print("⚠️ No queries provided")
            return None
        
        print(f"Evaluating on {len(queries)} queries...")
        
        # Get query vectors
        query_vectors = []
        for q in queries:
            # Step 1: Extract equations
            eq_pred = self.eq_extractor(q)
            equations = eq_pred.equations if eq_pred.equations else self._pattern_based_extraction(q)
            
            # Step 2: Canonicalize
            can_pred = self.canonicalizer(equations)
            canonical = can_pred.canonical
            
            # Step 3: Structure encoding
            struct_pred = self.structure_encoder(canonical)
            
            # Step 4: Text encoding
            text_for_embedding = build_text_for_embedding(q, fingerprint=None)
            text_pred = self.text_encoder(text_for_embedding)
            
            # Step 5: Hybrid fusion
            hybrid_pred = self.hybrid_encoder(
                struct_pred.structure_vector, 
                text_pred.text_vector
            )
            query_vectors.append(hybrid_pred.hybrid_vector)
        
        query_vectors = np.vstack(query_vectors)
        
        # Sample target vectors efficiently
        n_total = self.index.ntotal
        sample_size = min(sample_size, n_total)
        sampled_indices = np.random.choice(n_total, sample_size, replace=False)
        
        target_vectors = np.vstack([
            self.index.reconstruct(int(i)) 
            for i in sampled_indices
        ])
        
        evaluator = RetrievalEvaluator()
        metrics = evaluator(query_vectors, target_vectors, top_k=top_k)
        
        print(f"✓ Evaluation complete")
        print(f"  Precision@{top_k}: {metrics.precision_at_k:.4f}")
        print(f"  Mean Cosine Similarity: {metrics.mean_cosine_similarity:.4f}")
        print(f"  Mean Reciprocal Rank: {metrics.mean_reciprocal_rank:.4f}")
        
        return metrics