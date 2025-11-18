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
# Symbolic Solver (WORKING VERSION)
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
# Verifier Module (WORKING VERSION)
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
# LLM Reasoner (WORKING VERSION)
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
# Main Pipeline (WORKING VERSION)
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


# --------------------------
# COMPREHENSIVE EVALUATION METRICS
# --------------------------

class ExactMatchEvaluator(dspy.Module):
    """Evaluates exact match between predicted and ground truth solutions"""
    
    def forward(self, predicted_solutions: List[Dict], ground_truth_solutions: List[Dict], tolerance: float = 1e-6):
        if not predicted_solutions or not ground_truth_solutions:
            return dspy.Prediction(exact_match_rate=0.0, exact_matches=0, total_solutions=0)
        
        exact_matches = 0
        total_solutions = len(predicted_solutions)
        
        for pred, gt in zip(predicted_solutions, ground_truth_solutions):
            if self._solutions_match(pred, gt, tolerance):
                exact_matches += 1
        
        exact_match_rate = exact_matches / total_solutions if total_solutions > 0 else 0.0
        
        return dspy.Prediction(
            exact_match_rate=exact_match_rate,
            exact_matches=exact_matches,
            total_solutions=total_solutions
        )
    
    def _solutions_match(self, pred: Dict, gt: Dict, tolerance: float) -> bool:
        """Check if two solutions match within tolerance"""
        if not pred or not gt:
            return False
        
        # Check if all keys match
        if set(pred.keys()) != set(gt.keys()):
            return False
        
        # Check if all values match within tolerance
        for key in pred.keys():
            if key not in gt:
                return False
            if abs(pred[key] - gt[key]) > tolerance:
                return False
        
        return True


class PassAtKEvaluator(dspy.Module):
    """Evaluates Pass@k accuracy for solution generation"""
    
    def forward(self, predicted_solution_lists: List[List[Dict]], ground_truth_solutions: List[Dict], k: int = 1):
        if not predicted_solution_lists or not ground_truth_solutions:
            return dspy.Prediction(pass_at_k_rate=0.0, passes=0, total_problems=0)
        
        passes = 0
        total_problems = len(predicted_solution_lists)
        
        for pred_list, gt in zip(predicted_solution_lists, ground_truth_solutions):
            # Check if any of the top-k solutions pass
            for pred in pred_list[:k]:
                if self._solutions_match(pred, gt):
                    passes += 1
                    break
        
        pass_at_k_rate = passes / total_problems if total_problems > 0 else 0.0
        
        return dspy.Prediction(
            pass_at_k_rate=pass_at_k_rate,
            passes=passes,
            total_problems=total_problems
        )
    
    def _solutions_match(self, pred: Dict, gt: Dict, tolerance: float = 1e-6) -> bool:
        """Check if two solutions match within tolerance"""
        if not pred or not gt:
            return False
        
        # Check if all keys match
        if set(pred.keys()) != set(gt.keys()):
            return False
        
        # Check if all values match within tolerance
        for key in pred.keys():
            if key not in gt:
                return False
            if abs(pred[key] - gt[key]) > tolerance:
                return False
        
        return True


class SymbolicSolvingEvaluator(dspy.Module):
    """Evaluates symbolic solving success rate"""
    
    def forward(self, symbolic_solving_results: List[Dict]):
        if not symbolic_solving_results:
            return dspy.Prediction(symbolic_success_rate=0.0, successful_solves=0, total_attempts=0)
        
        successful_solves = sum(1 for result in symbolic_solving_results if result.get('success', False))
        total_attempts = len(symbolic_solving_results)
        symbolic_success_rate = successful_solves / total_attempts if total_attempts > 0 else 0.0
        
        return dspy.Prediction(
            symbolic_success_rate=symbolic_success_rate,
            successful_solves=successful_solves,
            total_attempts=total_attempts
        )


class LLMSolverAgreementEvaluator(dspy.Module):
    """Evaluates agreement between LLM and symbolic solver solutions"""
    
    def forward(self, llm_solutions: List[Dict], symbolic_solutions: List[Dict], tolerance: float = 1e-6):
        if not llm_solutions or not symbolic_solutions:
            return dspy.Prediction(agreement_rate=0.0, agreements=0, total_comparisons=0)
        
        agreements = 0
        total_comparisons = min(len(llm_solutions), len(symbolic_solutions))
        
        for llm_sol, sym_sol in zip(llm_solutions, symbolic_solutions):
            if self._solutions_match(llm_sol, sym_sol, tolerance):
                agreements += 1
        
        agreement_rate = agreements / total_comparisons if total_comparisons > 0 else 0.0
        
        return dspy.Prediction(
            agreement_rate=agreement_rate,
            agreements=agreements,
            total_comparisons=total_comparisons
        )
    
    def _solutions_match(self, sol1: Dict, sol2: Dict, tolerance: float) -> bool:
        """Check if two solutions match within tolerance"""
        if not sol1 or not sol2:
            return False
        
        # Check if all keys match
        if set(sol1.keys()) != set(sol2.keys()):
            return False
        
        # Check if all values match within tolerance
        for key in sol1.keys():
            if key not in sol2:
                return False
            if abs(sol1[key] - sol2[key]) > tolerance:
                return False
        
        return True


class ReasoningConsistencyEvaluator(dspy.Module):
    """Evaluates consistency of reasoning steps with solutions"""
    
    def forward(self, reasoning_steps: List[str], solutions: List[Dict]):
        if not reasoning_steps or not solutions:
            return dspy.Prediction(consistency_score=0.0, per_case_scores=[])
        
        consistency_scores = []
        
        for reasoning, solution in zip(reasoning_steps, solutions):
            score = self._evaluate_reasoning_consistency(reasoning, solution)
            consistency_scores.append(score)
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        return dspy.Prediction(
            consistency_score=overall_consistency,
            per_case_scores=consistency_scores
        )
    
    def _evaluate_reasoning_consistency(self, reasoning: str, solution: Dict) -> float:
        """Evaluate consistency between reasoning and solution"""
        if not reasoning or not solution:
            return 0.0
        
        # Simple heuristic: check if reasoning mentions solution variables
        reasoning_lower = reasoning.lower()
        consistency_score = 0.0
        
        for var in solution.keys():
            if var.lower() in reasoning_lower:
                consistency_score += 0.3
        
        # Check if reasoning mentions mathematical operations
        math_ops = ['add', 'subtract', 'multiply', 'divide', 'solve', 'equation']
        for op in math_ops:
            if op in reasoning_lower:
                consistency_score += 0.1
        
        return min(1.0, consistency_score)


class RetrievalRecallEvaluator(dspy.Module):
    """Evaluates retrieval recall@k"""
    
    def forward(self, retrieved_results: List[List[Dict]], relevant_docs: List[List[str]], k: int = 5):
        if not retrieved_results or not relevant_docs:
            return dspy.Prediction(recall_at_k=0.0, per_query_recall=[])
        
        per_query_recall = []
        
        for retrieved, relevant in zip(retrieved_results, relevant_docs):
            recall = self._calculate_recall(retrieved, relevant, k)
            per_query_recall.append(recall)
        
        overall_recall = np.mean(per_query_recall) if per_query_recall else 0.0
        
        return dspy.Prediction(
            recall_at_k=overall_recall,
            per_query_recall=per_query_recall
        )
    
    def _calculate_recall(self, retrieved: List[Dict], relevant: List[str], k: int) -> float:
        """Calculate recall@k for a single query"""
        if not retrieved or not relevant:
            return 0.0
        
        # Get top-k retrieved document IDs
        retrieved_ids = [doc.get('id', f'doc_{i}') for i, doc in enumerate(retrieved[:k])]
        
        # Count how many relevant docs are in top-k
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant)
        
        return relevant_retrieved / len(relevant) if relevant else 0.0


class MathematicalEquivalenceEvaluator(dspy.Module):
    """Evaluates mathematical equivalence of expressions"""
    
    def forward(self, predicted_expressions: List[str], ground_truth_expressions: List[str]):
        if not predicted_expressions or not ground_truth_expressions:
            return dspy.Prediction(equivalence_accuracy=0.0, equivalent_count=0, total_expressions=0)
        
        equivalent_count = 0
        total_expressions = min(len(predicted_expressions), len(ground_truth_expressions))
        
        for pred_expr, gt_expr in zip(predicted_expressions, ground_truth_expressions):
            if self._expressions_equivalent(pred_expr, gt_expr):
                equivalent_count += 1
        
        equivalence_accuracy = equivalent_count / total_expressions if total_expressions > 0 else 0.0
        
        return dspy.Prediction(
            equivalence_accuracy=equivalence_accuracy,
            equivalent_count=equivalent_count,
            total_expressions=total_expressions
        )
    
    def _expressions_equivalent(self, expr1: str, expr2: str) -> bool:
        """Check if two mathematical expressions are equivalent"""
        try:
            from sympy import sympify, simplify
            # Parse and simplify both expressions
            parsed1 = simplify(sympify(expr1))
            parsed2 = simplify(sympify(expr2))
            return parsed1 == parsed2
        except:
            # Fallback to string comparison
            return expr1.strip() == expr2.strip()


class FaithfulnessEvaluator(dspy.Module):
    """Evaluates faithfulness of generated text to source text"""
    
    def forward(self, generated_texts: List[str], source_texts: List[str]):
        if not generated_texts or not source_texts:
            return dspy.Prediction(faithfulness_score=0.0, per_text_scores=[])
        
        per_text_scores = []
        
        for generated, source in zip(generated_texts, source_texts):
            score = self._calculate_faithfulness(generated, source)
            per_text_scores.append(score)
        
        overall_faithfulness = np.mean(per_text_scores) if per_text_scores else 0.0
        
        return dspy.Prediction(
            faithfulness_score=overall_faithfulness,
            per_text_scores=per_text_scores
        )
    
    def _calculate_faithfulness(self, generated: str, source: str) -> float:
        """Calculate faithfulness score between generated and source text"""
        if not generated or not source:
            return 0.0
        
        # Simple heuristic: check for common mathematical terms and numbers
        source_words = set(source.lower().split())
        generated_words = set(generated.lower().split())
        
        # Find common mathematical terms
        math_terms = {'equation', 'solve', 'variable', 'number', 'sum', 'difference', 'multiply', 'divide'}
        common_math_terms = source_words.intersection(math_terms).intersection(generated_words)
        
        # Find common numbers
        import re
        source_numbers = set(re.findall(r'\d+', source))
        generated_numbers = set(re.findall(r'\d+', generated))
        common_numbers = source_numbers.intersection(generated_numbers)
        
        # Calculate faithfulness score
        faithfulness_score = 0.0
        if math_terms:
            faithfulness_score += len(common_math_terms) / len(math_terms) * 0.5
        if source_numbers:
            faithfulness_score += len(common_numbers) / len(source_numbers) * 0.5
        
        return min(1.0, faithfulness_score)


class HallucinationRateEvaluator(dspy.Module):
    """Evaluates hallucination rate in generated text"""
    
    def forward(self, generated_texts: List[str], source_texts: List[str]):
        if not generated_texts or not source_texts:
            return dspy.Prediction(hallucination_rate=0.0, hallucination_count=0, total_texts=0)
        
        hallucination_count = 0
        total_texts = len(generated_texts)
        
        for generated, source in zip(generated_texts, source_texts):
            if self._contains_hallucination(generated, source):
                hallucination_count += 1
        
        hallucination_rate = hallucination_count / total_texts if total_texts > 0 else 0.0
        
        return dspy.Prediction(
            hallucination_rate=hallucination_rate,
            hallucination_count=hallucination_count,
            total_texts=total_texts
        )
    
    def _contains_hallucination(self, generated: str, source: str) -> bool:
        """Check if generated text contains hallucinations"""
        if not generated or not source:
            return True
        
        # Simple heuristic: check for made-up numbers or facts not in source
        import re
        
        # Extract numbers from both texts
        source_numbers = set(re.findall(r'\d+', source))
        generated_numbers = set(re.findall(r'\d+', generated))
        
        # Check if generated text contains numbers not in source
        hallucinated_numbers = generated_numbers - source_numbers
        
        # If more than 2 new numbers, consider it hallucination
        return len(hallucinated_numbers) > 2


class ThroughputEvaluator(dspy.Module):
    """Evaluates end-to-end throughput"""
    
    def forward(self, processing_times: List[float], batch_sizes: List[int]):
        if not processing_times or not batch_sizes:
            return dspy.Prediction(throughput=0.0, total_items=0, total_time=0.0, avg_processing_time=0.0)
        
        total_items = sum(batch_sizes)
        total_time = sum(processing_times)
        avg_processing_time = np.mean(processing_times) if processing_times else 0.0
        
        throughput = total_items / total_time if total_time > 0 else 0.0
        
        return dspy.Prediction(
            throughput=throughput,
            total_items=total_items,
            total_time=total_time,
            avg_processing_time=avg_processing_time
        )


class RetrievalPrecisionEvaluator(dspy.Module):
    """Evaluates retrieval precision@k"""
    
    def forward(self, retrieved_results: List[List[Dict]], relevant_docs: List[List[str]], k: int = 5):
        if not retrieved_results or not relevant_docs:
            return dspy.Prediction(precision_at_k=0.0, per_query_precision=[])
        
        per_query_precision = []
        
        for retrieved, relevant in zip(retrieved_results, relevant_docs):
            precision = self._calculate_precision(retrieved, relevant, k)
            per_query_precision.append(precision)
        
        overall_precision = np.mean(per_query_precision) if per_query_precision else 0.0
        
        return dspy.Prediction(
            precision_at_k=overall_precision,
            per_query_precision=per_query_precision
        )
    
    def _calculate_precision(self, retrieved: List[Dict], relevant: List[str], k: int) -> float:
        """Calculate precision@k for a single query"""
        if not retrieved or not relevant:
            return 0.0
        
        # Get top-k retrieved document IDs
        retrieved_ids = [doc.get('id', f'doc_{i}') for i, doc in enumerate(retrieved[:k])]
        
        # Count how many retrieved docs are relevant
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant)
        
        return relevant_retrieved / k if k > 0 else 0.0


class RetrievalRecallKEvaluator(dspy.Module):
    """Evaluates retrieval recall@k (alternative implementation)"""
    
    def forward(self, retrieved_results: List[List[Dict]], relevant_docs: List[List[str]], k: int = 5):
        if not retrieved_results or not relevant_docs:
            return dspy.Prediction(recall_at_k=0.0, per_query_recall=[])
        
        per_query_recall = []
        
        for retrieved, relevant in zip(retrieved_results, relevant_docs):
            recall = self._calculate_recall(retrieved, relevant, k)
            per_query_recall.append(recall)
        
        overall_recall = np.mean(per_query_recall) if per_query_recall else 0.0
        
        return dspy.Prediction(
            recall_at_k=overall_recall,
            per_query_recall=per_query_recall
        )
    
    def _calculate_recall(self, retrieved: List[Dict], relevant: List[str], k: int) -> float:
        """Calculate recall@k for a single query"""
        if not retrieved or not relevant:
            return 0.0
        
        # Get top-k retrieved document IDs
        retrieved_ids = [doc.get('id', f'doc_{i}') for i, doc in enumerate(retrieved[:k])]
        
        # Count how many relevant docs are in top-k
        relevant_retrieved = sum(1 for doc_id in retrieved_ids if doc_id in relevant)
        
        return relevant_retrieved / len(relevant) if relevant else 0.0


class MRREvaluator(dspy.Module):
    """Evaluates Mean Reciprocal Rank (MRR)"""
    
    def forward(self, retrieved_results: List[List[Dict]], relevant_docs: List[List[str]]):
        if not retrieved_results or not relevant_docs:
            return dspy.Prediction(mrr=0.0, per_query_rr=[])
        
        per_query_rr = []
        
        for retrieved, relevant in zip(retrieved_results, relevant_docs):
            rr = self._calculate_reciprocal_rank(retrieved, relevant)
            per_query_rr.append(rr)
        
        overall_mrr = np.mean(per_query_rr) if per_query_rr else 0.0
        
        return dspy.Prediction(
            mrr=overall_mrr,
            per_query_rr=per_query_rr
        )
    
    def _calculate_reciprocal_rank(self, retrieved: List[Dict], relevant: List[str]) -> float:
        """Calculate reciprocal rank for a single query"""
        if not retrieved or not relevant:
            return 0.0
        
        # Find the rank of the first relevant document
        for rank, doc in enumerate(retrieved, 1):
            doc_id = doc.get('id', f'doc_{rank-1}')
            if doc_id in relevant:
                return 1.0 / rank
        
        return 0.0


class NDCGEvaluator(dspy.Module):
    """Evaluates Normalized Discounted Cumulative Gain (NDCG@k)"""
    
    def forward(self, retrieved_results: List[List[Dict]], relevance_scores: List[List[float]], k: int = 5):
        if not retrieved_results or not relevance_scores:
            return dspy.Prediction(ndcg_at_k=0.0, per_query_ndcg=[])
        
        per_query_ndcg = []
        
        for retrieved, scores in zip(retrieved_results, relevance_scores):
            ndcg = self._calculate_ndcg(retrieved, scores, k)
            per_query_ndcg.append(ndcg)
        
        overall_ndcg = np.mean(per_query_ndcg) if per_query_ndcg else 0.0
        
        return dspy.Prediction(
            ndcg_at_k=overall_ndcg,
            per_query_ndcg=per_query_ndcg
        )
    
    def _calculate_ndcg(self, retrieved: List[Dict], relevance_scores: List[float], k: int) -> float:
        """Calculate NDCG@k for a single query"""
        if not retrieved or not relevance_scores:
            return 0.0
        
        # Get top-k relevance scores
        top_k_scores = relevance_scores[:k]
        
        # Calculate DCG
        dcg = sum(score / np.log2(i + 2) for i, score in enumerate(top_k_scores))
        
        # Calculate IDCG (ideal DCG)
        ideal_scores = sorted(relevance_scores, reverse=True)[:k]
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        
        return dcg / idcg if idcg > 0 else 0.0


class CosineSimilarityDistributionEvaluator(dspy.Module):
    """Evaluates cosine similarity score distribution"""
    
    def forward(self, similarity_scores: List[float]):
        if not similarity_scores:
            return dspy.Prediction(
                mean_similarity=0.0,
                std_similarity=0.0,
                min_similarity=0.0,
                max_similarity=0.0,
                median_similarity=0.0,
                q25_similarity=0.0,
                q75_similarity=0.0
            )
        
        similarity_array = np.array(similarity_scores)
        
        return dspy.Prediction(
            mean_similarity=float(np.mean(similarity_array)),
            std_similarity=float(np.std(similarity_array)),
            min_similarity=float(np.min(similarity_array)),
            max_similarity=float(np.max(similarity_array)),
            median_similarity=float(np.median(similarity_array)),
            q25_similarity=float(np.percentile(similarity_array, 25)),
            q75_similarity=float(np.percentile(similarity_array, 75))
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
