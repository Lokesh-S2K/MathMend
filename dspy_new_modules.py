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
    
    def _clean_equation_string(self, eq_str):
        """Clean equation string to make it SymPy-compatible"""
        import re
        
        # Remove units and common text
        eq_str = re.sub(r'\b(km/h|km|miles|mph|cm|m|kg|g|lbs|dollars|\$)\b', '', eq_str)
        
        # Fix implicit multiplication (e.g., 2x -> 2*x, xy -> x*y)
        eq_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', eq_str)  # 2x -> 2*x
        eq_str = re.sub(r'([a-zA-Z])(\d+)', r'\1*\2', eq_str)  # x2 -> x*2
        eq_str = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', eq_str)  # xy -> x*y
        
        # Fix common word problems
        eq_str = re.sub(r'\b(length|width|height|area|perimeter|speed|distance|time)\b', 
                       lambda m: m.group(1)[0], eq_str)  # length -> l, width -> w, etc.
        
        # Remove extra spaces and clean up
        eq_str = re.sub(r'\s+', ' ', eq_str).strip()
        
        # Handle specific patterns
        eq_str = eq_str.replace(' x ', ' * ')  # word "x" to multiplication
        eq_str = eq_str.replace('Area =', 'A =')
        eq_str = eq_str.replace('length x width', 'l * w')
        
        return eq_str
    
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
                    # Try to parse string equations - handle both expressions and equations
                    try:
                        from sympy import sympify, Eq, symbols
                        
                        # Clean up the equation string
                        cleaned_eq = self._clean_equation_string(eq)
                        
                        # First try to parse as equation (with =)
                        if '=' in cleaned_eq:
                            # Split by = and create equation
                            parts = cleaned_eq.split('=', 1)
                            if len(parts) == 2:
                                lhs = sympify(parts[0].strip())
                                rhs = sympify(parts[1].strip())
                                equations_to_solve.append(Eq(lhs, rhs))
                            else:
                                # Fallback to direct parsing
                                parsed_eq = sympify(cleaned_eq)
                                equations_to_solve.append(parsed_eq)
                        else:
                            # Parse as expression
                            parsed_eq = sympify(cleaned_eq)
                            equations_to_solve.append(parsed_eq)
                    except Exception as parse_error:
                        print(f"Failed to parse equation '{eq}': {parse_error}")
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
            
            # Extract all symbols from equations
            syms = set()
            for e in equations_to_solve:
                if hasattr(e, 'free_symbols'):
                    syms.update(e.free_symbols)
            
            syms = sorted(syms, key=lambda x: str(x))
            
            if not syms:
                return dspy.Prediction(
                    solution={}, 
                    success=False, 
                    residuals={}, 
                    error_msg="No variables found"
                )
            
            print(f"🔍 Solving equations: {[str(eq) for eq in equations_to_solve]}")
            print(f"🔍 Variables: {[str(s) for s in syms]}")
            
            # Try to solve the system
            sol = solve(equations_to_solve, syms, dict=True)
            
            if not sol:
                return dspy.Prediction(
                    solution={}, 
                    success=False, 
                    residuals={}, 
                    error_msg="No solution found"
                )
            
            sol0 = sol[0]
            print(f"✅ Solution found: {sol0}")
            
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
            
            print(f"🔍 Residuals: {residuals}")
            print(f"🔍 Success: {success}")
            
            return dspy.Prediction(
                solution={str(k): float(v) for k, v in sol0.items()},
                success=success,
                residuals=residuals,
                error_msg=None
            )
        except Exception as e:
            print(f"❌ Solver error: {str(e)}")
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
    
    def _clean_equation_string(self, eq_str):
        """Clean equation string to make it SymPy-compatible"""
        import re
        
        # Remove units and common text
        eq_str = re.sub(r'\b(km/h|km|miles|mph|cm|m|kg|g|lbs|dollars|\$)\b', '', eq_str)
        
        # Fix implicit multiplication (e.g., 2x -> 2*x, xy -> x*y)
        eq_str = re.sub(r'(\d+)([a-zA-Z])', r'\1*\2', eq_str)  # 2x -> 2*x
        eq_str = re.sub(r'([a-zA-Z])(\d+)', r'\1*\2', eq_str)  # x2 -> x*2
        eq_str = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', eq_str)  # xy -> x*y
        
        # Fix common word problems
        eq_str = re.sub(r'\b(length|width|height|area|perimeter|speed|distance|time)\b', 
                       lambda m: m.group(1)[0], eq_str)  # length -> l, width -> w, etc.
        
        # Remove extra spaces and clean up
        eq_str = re.sub(r'\s+', ' ', eq_str).strip()
        
        # Handle specific patterns
        eq_str = eq_str.replace(' x ', ' * ')  # word "x" to multiplication
        eq_str = eq_str.replace('Area =', 'A =')
        eq_str = eq_str.replace('length x width', 'l * w')
        
        return eq_str
    
    def forward(self, canonical_equations, candidate_solution, retrieved_solutions=None):
        if not canonical_equations:
            return dspy.Prediction(
                verification={'ok': False, 'residuals': {}, 'error': 'No equations to verify'}
            )
        
        residuals = {}
        for e in canonical_equations:
            try:
                # Handle string equations by cleaning them first
                if isinstance(e, str):
                    cleaned_eq = self._clean_equation_string(e)
                    from sympy import sympify, Eq
                    if '=' in cleaned_eq:
                        parts = cleaned_eq.split('=', 1)
                        if len(parts) == 2:
                            lhs = sympify(parts[0].strip())
                            rhs = sympify(parts[1].strip())
                            e = Eq(lhs, rhs)
                        else:
                            e = sympify(cleaned_eq)
                    else:
                        e = sympify(cleaned_eq)
                
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
    """Uses LLM to extract equations from problems and similar examples"""
    
    def __init__(self, llm_model=None):
        super().__init__()
        self.llm_model = llm_model if llm_model else initialize_ctransformers_model()
    
    def build_prompt(self, query: str, canonical_eqs: List, retrieved_examples: List[Dict], max_shots: int = 3) -> str:
        prompt_parts = [
        "# Math Problem Analysis\n",
        "Extract mathematical equations from the word problem following these examples.\n"
    ]
    
    # Add similar examples first with clear structure
        for i, ex in enumerate(retrieved_examples[:max_shots], 1):
            prompt_parts.append(f"\nExample {i}:")
            prompt_parts.append(f"Problem: {ex.get('text', '')}")
            if 'equations' in ex:
                eqs = ex['equations'] if isinstance(ex['equations'], list) else [ex['equations']]
                prompt_parts.append("Equations:")
                for eq in eqs:
                    prompt_parts.append(f"- {eq}")
        
        # Add current problem with clear instruction
        prompt_parts.extend([
            f"\nNow solve this problem:",
            f"Problem: {query}",
            "\nExtract the equations in the same format as the examples above.",
            "Return your answer as JSON:",
            '{"equations": ["equation1", "equation2", ...]}',
            "\nResponse:"
        ])
        
        return "\n".join(prompt_parts)

    def forward(self, query_text: str, canonical_equations: List, retrieved_examples: List[Dict]):
        try:
            # Build and send prompt
            prompt = self.build_prompt(query_text, canonical_equations, retrieved_examples)
            
            if self.llm_model:
                # Get LLM response with adjusted parameters
                response = self.llm_model.invoke(
                    prompt,
                    max_new_tokens=256,  # Shorter response focused on equations
                    temperature=0.1,     # More deterministic
                    top_p=0.9,
                    repetition_penalty=1.1
                )
                response_text = str(response)
                print(f"🔍 LLM Raw Response:\n{response_text[:200]}...")  # Debug output
                
                # Parse equations
                equations, success = self._parse_response(response_text)
                
                if not equations and canonical_equations:
                    # Fallback to canonical equations if LLM extraction failed
                    equations = [str(eq) for eq in canonical_equations]
                    success = True
                
                return dspy.Prediction(
                    llm_equations=equations,
                    llm_solution={},
                    llm_steps=response_text,
                    success=bool(equations)  # Success if we got any equations
                )
            else:
                print("⚠️ No LLM model available, using fallback extraction")
                equations = [str(eq) for eq in (canonical_equations or [])]
                return dspy.Prediction(
                    llm_equations=equations,
                    llm_solution={},
                    llm_steps="Used fallback equation extraction",
                    success=bool(equations)
                )
                
        except Exception as e:
            print(f"⚠️ LLM Reasoning failed: {str(e)}")
            return dspy.Prediction(
                llm_equations=[],
                llm_solution={},
                llm_steps=f"Error: {str(e)}",
                success=False
            )

    def _parse_response(self, response_text: str) -> tuple:
        """Enhanced response parsing to extract equations"""
        try:
            # 1. Try to find JSON block first
            json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    if 'equations' in data:
                        equations = data['equations']
                        if isinstance(equations, str):
                            equations = [equations]
                        return equations, True
                except:
                    pass

            # 2. Look for equation patterns
            equations = []
            lines = response_text.split('\n')
            
            for line in lines:
                line = line.strip()
                # Skip non-equation lines
                if not line or not '=' in line:
                    continue
                    
                # Clean up the line
                eq = re.sub(r'^[-*•\s]+', '', line)
                
                # Skip if it contains explanation words
                if any(word in eq.lower() for word in ['therefore', 'thus', 'hence', 'because']):
                    continue
                
                # Clean up equation
                eq = eq.strip(' ."\'')
                if '=' in eq:
                    equations.append(eq)

            return equations, bool(equations)

        except Exception as e:
            print(f"Response parsing error: {e}")
            return [], False

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
        
        # Step 7: Attempt symbolic solving with canonical equations
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
        
        # Step 8: LLM fallback - extract equations and try symbolic solving
        try:
            llm_out = self.llm_reasoner(text, canonical, retrieved_results)
            if getattr(llm_out, "success", False) and llm_out.llm_equations:
                print(f"🔍 LLM extracted equations: {llm_out.llm_equations}")
                
                # Try symbolic solving with LLM extracted equations
                if llm_out.llm_equations:
                    sym_out_llm = self.sym_solver(llm_out.llm_equations)
                    if getattr(sym_out_llm, "success", False):
                        print(f"✅ Symbolic solver succeeded with LLM equations!")
                        return dspy.Prediction(
                            result_type="symbolic_llm",
                            solution=sym_out_llm.solution,
                            residuals=sym_out_llm.residuals,
                            results=retrieved_results,
                            equations=llm_out.llm_equations,
                            reasoning=llm_out.llm_steps
                        )
                
                # If symbolic solving failed, check if LLM provided a solution
                if llm_out.llm_solution:
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
                            results=retrieved_results,
                            equations=llm_out.llm_equations
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