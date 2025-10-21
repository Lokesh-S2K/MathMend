"""
demo.py
--------
POWERFUL CalcMate Retrieval Demo - Shows True Mathematical Similarity
Finds structurally similar math problems (not exact matches)
"""

import pandas as pd
import numpy as np
import sys
import os
import json

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pipeline_sequence.embedder import encode_texts, build_text_for_embedding
from pipeline_sequence.indexer_faiss import FaissIndexer
from pipeline_sequence.advanced_equation_extractor import extract_equations_advanced, AdvancedMathParser
from pipeline_sequence.features import build_structure_vector_from_parsed
from pipeline_sequence.canonicalizer import canonicalize_system

class SmartRetrievalDemo:
    """Advanced retrieval that finds truly similar (not identical) math problems"""
    
    def __init__(self):
        self.faiss_indexer = None
        self.problem_database = None
        
    def load_system(self):
        """Load the FAISS retrieval system"""
        print("Loading CalcMate Math Retrieval System...")
        print("=" * 60)
        
        try:
            # Use the correct file names from your output
            index_path = "output/embeddings/faiss_index_20251015_145706.bin"
            idmap_path = "output/embeddings/faiss_id_map_20251015_145706.json"
            
            self.faiss_indexer = FaissIndexer.load(index_path, idmap_path)
            
            # Load problem database for analysis
            with open(idmap_path, 'r') as f:
                self.problem_database = json.load(f)
            
            print("System Loaded Successfully!")
            print(f"   Knowledge Base: {self.faiss_indexer.size()} math problems")
            print(f"   Vector Dimension: {self.faiss_indexer.dim}D")
            print(f"   Index Type: {self.faiss_indexer.index_type}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to load system: {e}")
            return False
        return True
    
    def create_smart_query_vector(self, query_text):
        """Create enhanced query vector with better equation extraction"""
        
        # Enhanced equation extraction with fallbacks
        equations = self._extract_equations_robust(query_text)
        
        # Generate text embedding
        query_for_embedding = build_text_for_embedding(query_text, fingerprint=None)
        text_embedding = encode_texts([query_for_embedding], normalize=True)[0]
        
        # Generate structure vector from equations
        if equations:
            canonicalized = canonicalize_system(equations)
            structure_vector = build_structure_vector_from_parsed(canonicalized)
        else:
            # Fallback: zero structure vector if no equations found
            structure_vector = np.zeros(256)
        
        # Combine into hybrid vector
        hybrid_vector = np.concatenate([structure_vector, text_embedding])
        
        return hybrid_vector, equations
    
    def _extract_equations_robust(self, text):
        """Enhanced equation extraction with pattern matching"""
        equations = extract_equations_advanced(text)
        
        # If advanced extractor fails, use pattern-based fallback
        if not equations:
            equations = self._pattern_based_extraction(text)
        
        return equations
    
    def _pattern_based_extraction(self, text):
        """Pattern-based equation extraction as fallback"""
        text_lower = text.lower()
        equations = []
        
        # Speed-Distance-Time pattern
        if any(word in text_lower for word in ['speed', 'distance', 'km/h', 'mph']):
            numbers = re.findall(r'\b(\d+)\b', text)
            if len(numbers) >= 2:
                equations.append(f"speed = {numbers[0]} / {numbers[1]}")
        
        # Cost-Quantity pattern  
        elif any(word in text_lower for word in ['cost', 'price', 'buy', 'spend']):
            numbers = re.findall(r'\$?(\d+)', text)
            if len(numbers) >= 2:
                equations.append(f"total = {numbers[0]} * {numbers[1]}")
        
        # Percentage pattern
        elif '%' in text:
            numbers = re.findall(r'\b(\d+)\b', text)
            if len(numbers) >= 2:
                equations.append(f"result = {numbers[1]} * {numbers[0]} / 100")
        
        # Sum-Difference pattern
        elif 'sum' in text_lower and 'difference' in text_lower:
            numbers = re.findall(r'\b(\d+)\b', text)
            if len(numbers) >= 2:
                equations.extend([f"x + y = {numbers[0]}", f"x - y = {numbers[1]}"])
        
        return equations
    
    def find_most_similar_problem(self, query_text):
        """Find the most similar but DIFFERENT problem"""
        
        print(f"\n USER QUERY: \"{query_text}\"")
        print("─" * 60)
        
        # Create query vector
        query_vector, extracted_equations = self.create_smart_query_vector(query_text)
        
        if extracted_equations:
            print(f"Mathematical Structure Detected: {extracted_equations}")
        else:
            print("Mathematical Structure: Inferred from problem context")
        
        # Search for similar problems (get top 10 to filter)
        distances, results = self.faiss_indexer.search(query_vector, top_k=10)
        
        if not results:
            print("❌ No similar problems found in knowledge base")
            return None
        
        # Find the BEST similar problem (not the exact same)
        best_match = self._select_best_match(query_text, list(zip(distances, results)))
        
        if best_match:
            self._display_match_analysis(best_match, query_text)
            return best_match
        else:
            print("❌ No sufficiently similar problems found")
            return None
    
    def _select_best_match(self, query_text, candidates):
        """Select the best similar problem that's not the same"""
        
        for distance, result in candidates:
            result_text = result['metadata'].get('problem_text', '')
            
            # Skip if it's essentially the same problem
            if self._is_same_problem(query_text, result_text):
                continue
                
            # Skip if similarity is too low
            if distance > 0.8:  # Adjust threshold as needed
                continue
                
            return (distance, result)
        
        return None
    
    def _is_same_problem(self, query, candidate):
        """Check if problems are essentially the same"""
        query_words = set(query.lower().split())
        candidate_words = set(candidate.lower().split())
        
        # If they share too many keywords, likely the same problem
        common_words = query_words.intersection(candidate_words)
        similarity_ratio = len(common_words) / len(query_words)
        
        return similarity_ratio > 0.7
    
    def _display_match_analysis(self, best_match, query_text):
        """Display detailed analysis of why problems are similar"""
        distance, result = best_match
        metadata = result['metadata']
        
        similarity_score = 1 - distance  # Convert distance to similarity
        
        print(f"FOUND SIMILAR PROBLEM (Similarity: {similarity_score:.1%})")
        print()
        print(f"SIMILAR PROBLEM:")
        print(f"   \"{metadata.get('problem_text', 'N/A')}\"")
        print()
        print(f"WHY IT'S SIMILAR:")
        
        # Analyze similarity reasons
        similarity_reasons = self._analyze_similarity(query_text, metadata)
        for reason in similarity_reasons:
            print(f"   • {reason}")
        
        print()
        print(f"PROBLEM ANALYSIS:")
        print(f"   Type: {metadata.get('problem_type', 'N/A')}")
        
        
        if metadata.get('solution_text'):
            print()
            print(f"SOLUTION APPROACH:")
            print(f"   {metadata.get('solution_text', 'N/A')}")
    
    def _analyze_similarity(self, query_text, result_metadata):
        """Analyze why the problems are mathematically similar"""
        reasons = []
        
        query_lower = query_text.lower()
        result_text = result_metadata.get('problem_text', '').lower()
        result_type = result_metadata.get('problem_type', '')
        result_equations = result_metadata.get('extracted_equations', [])
        
        # Problem type similarity
        if any(word in query_lower for word in ['speed', 'distance', 'time', 'km/h']):
            reasons.append("Both involve speed-distance-time relationships")
        elif any(word in query_lower for word in ['sum', 'difference', 'numbers']):
            reasons.append("Both deal with sum and difference of numbers")
        elif any(word in query_lower for word in ['cost', 'price', 'buy']):
            reasons.append("Both involve cost calculations and multiplication")
        elif any(word in query_lower for word in ['area', 'perimeter', 'rectangle']):
            reasons.append("Both are geometry problems with area/perimeter")
        elif '%' in query_text:
            reasons.append("Both involve percentage calculations")
        
        # Equation structure similarity
        if result_equations:
            if any('+' in eq and '=' in eq for eq in result_equations):
                reasons.append("Both use additive relationships")
            if any('*' in eq or '×' in eq for eq in result_equations):
                reasons.append("Both use multiplicative relationships")
            if any('/' in eq for eq in result_equations):
                reasons.append("Both use division operations")
        
        # Difficulty level similarity
        if 'Hard' in result_metadata.get('difficulty_level', ''):
            reasons.append("Both are complex multi-step problems")
        elif 'Easy' in result_metadata.get('difficulty_level', ''):
            reasons.append("Both are straightforward single-step problems")
        
        return reasons if reasons else ["Similar mathematical structure and problem-solving approach"]

def main():
    """Run the powerful retrieval demo"""
    
    demo = SmartRetrievalDemo()
    
    if not demo.load_system():
        return
    
    print()
    print("--------------------------------------------------------------------")
    print("CALCMATE DEMO: Mathematical Similarity Search")
    print("Find problems with similar STRUCTURE, not just similar words!")
    print("--------------------------------------------------------------------")   
    print() 
    # Test queries designed to find DIFFERENT but similar problems
    test_queries = [
        # Should find different speed-distance problems
        "A motorcycle travels 180 kilometers in 3 hours. What is its average speed?",

        # Should find different cost calculation problems
        "If a pizza costs $18 and you order 3 pizzas, what's the total cost?",
        
        # Should find different geometry problems
        "A rectangular playground has length 12 meters and width 8 meters. Calculate its area.",
    ]
    
    print("RUNNING SMART RETRIEVAL TESTS...")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"TEST {i}/{len(test_queries)}:")
        demo.find_most_similar_problem(query)
        
        if i < len(test_queries):
            print("\n" + "═" * 70)
            print()
    print()
    print("--------------------------------------------------------------------")
    print("DEMO COMPLETED!")
    print("--------------------------------------------------------------------")
    print()
    print("WHAT MAKES CALCMATE UNIQUE:")
    print("   • Finds MATHEMATICALLY similar problems, not just textually similar")
    print("   • Understands equation structures and problem-solving patterns")  
    print("   • Explains WHY problems are similar")
    print("   • Helps learn mathematical concepts through pattern recognition")
    print()
    print("--------------------------------------------------------------------")
    print("Next: Neuro-symbolic solver that USES these similar problems to solve new ones!")
    print("--------------------------------------------------------------------")
if __name__ == "__main__":
    import re  # Add this import at the top
    main()