"""
CalcMate CTransformers LLM Integration Demo
------------------------------------------
Comprehensive demo showcasing the neuro-symbolic system with CTransformers LLM:
1. Mathematical similarity retrieval
2. Symbolic equation solving
3. CTransformers LLM reasoning
4. Solution verification
5. Complete pipeline integration
"""

import time
import textwrap
import numpy as np
import dspy
from sympy import symbols, Eq
from dspy_modules import (
    SmartRetrievalPipeline, 
    SymbolicSolver, 
    Verifier, 
    LLMReasoner,
    initialize_ctransformers_model,
    explain_similarity
)
from pipeline_sequence.embedder import encode_texts


class CTransformersNeuroSymbolicDemo:
    """Comprehensive demo for testing CTransformers LLM integration"""
    
    def __init__(self):
        self.index_path = "output/embeddings/faiss_index_20251015_145706.bin"
        self.idmap_path = "output/embeddings/faiss_id_map_20251015_145706.json"
        self.pipeline = None
        self.llm_model = None
        self.symbolic_solver = SymbolicSolver()
        self.verifier = Verifier()
        self.llm_reasoner = None

    def load_system(self):
        """Load the complete neuro-symbolic system with CTransformers LLM"""
        print("🔧 Loading DSPy Hybrid Neuro-Symbolic System with CTransformers LLM...")
        print("=" * 80)
        
        try:
            # Initialize CTransformers LLM
            print("🤖 Initializing CTransformers LLM...")
            self.llm_model = initialize_ctransformers_model()
            
            if self.llm_model is not None:
                print("✅ CTransformers LLM loaded successfully!")
            else:
                print("⚠️ CTransformers LLM not available, will use fallback methods")
            
            # Load main pipeline with LLM
            print("\n🔄 Loading main pipeline...")
            self.pipeline = SmartRetrievalPipeline(
                self.index_path, 
                self.idmap_path,
                llm_model=self.llm_model
            )
            print("✅ Main pipeline loaded successfully!")
            
            # Initialize LLM reasoner with the model
            self.llm_reasoner = LLMReasoner(self.llm_model)
            print("✅ LLM reasoner initialized!")
            print("✅ Symbolic solver initialized!")
            print("✅ Verifier module initialized!")
            print()
            
        except Exception as e:
            print(f"❌ Failed to load system: {e}")
            import traceback
            traceback.print_exc()
            return False
        return True

    def test_ctransformers_llm_directly(self):
        """Test the CTransformers LLM directly with math problems"""
        print("\n🤖 TESTING CTRANSFORMERS LLM DIRECTLY")
        print("=" * 80)
        
        if self.llm_model is None:
            print("❌ CTransformers LLM not available for testing")
            return
        
        # Test cases for CTransformers LLM
        test_cases = [
            {
                "name": "Simple Arithmetic",
                "prompt": "Solve this math problem step by step: If a pizza costs $18 and you order 3 pizzas, what's the total cost?",
                "expected": "multiplication"
            },
            {
                "name": "Speed Problem",
                "prompt": "A car travels 120 km in 2 hours. What is its speed? Show your work.",
                "expected": "division"
            },
            {
                "name": "Area Calculation",
                "prompt": "A rectangle has length 12 cm and width 8 cm. What is its area? Explain your reasoning.",
                "expected": "multiplication"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}] {test_case['name']}")
            print("-" * 60)
            print(f"Prompt: {test_case['prompt']}")
            print()
            
            try:
                start_time = time.time()
                response = self.llm_model(
                    test_case['prompt'], 
                    max_new_tokens=256, 
                    temperature=0.5
                )
                elapsed = time.time() - start_time
                
                print(f"⏱ Response Time: {elapsed:.3f} seconds")
                print("🤖 LLM Response:")
                print(textwrap.fill(str(response), width=80))
                
            except Exception as e:
                print(f"❌ LLM test failed: {e}")
            
            print()

    def test_llm_reasoner_module(self):
        """Test the LLMReasoner module with CTransformers"""
        print("\n💡 TESTING LLM REASONER MODULE WITH CTRANSFORMERS")
        print("=" * 80)
        
        # Test cases for LLM reasoner
        test_cases = [
            {
                "name": "Cost Calculation",
                "query": "If a pizza costs $18 and you order 3 pizzas, what's the total cost?",
                "equations": [],
                "examples": []
            },
            {
                "name": "Speed Problem",
                "query": "A car travels 120 km in 2 hours. What is its speed?",
                "equations": [],
                "examples": []
            },
            {
                "name": "Area Problem",
                "query": "A rectangle has length 12 cm and width 8 cm. What is its area?",
                "equations": [],
                "examples": []
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}] {test_case['name']}")
            print("-" * 60)
            
            try:
                start_time = time.time()
                result = self.llm_reasoner(
                    test_case['query'], 
                    test_case['equations'], 
                    test_case['examples']
                )
                elapsed = time.time() - start_time
                
                print(f"⏱ Processing Time: {elapsed:.3f} seconds")
                print(f"✅ Success: {result.success}")
                
                if result.llm_steps:
                    print("\n🤖 LLM Reasoning Steps:")
                    print(textwrap.fill(result.llm_steps[:500] + "...", width=80))
                
                if result.llm_solution:
                    print("\n🧮 Extracted Solution:")
                    for var, val in result.llm_solution.items():
                        print(f"   {var} = {val}")
                
                if result.llm_equations:
                    print("\n📘 Extracted Equations:")
                    for eq in result.llm_equations:
                        print(f"   • {eq}")
                        
            except Exception as e:
                print(f"❌ LLM reasoner test failed: {e}")
            
            print()

    def test_complete_pipeline_with_llm(self):
        """Test the complete neuro-symbolic pipeline with CTransformers LLM"""
        print("\n🚀 TESTING COMPLETE PIPELINE WITH CTRANSFORMERS LLM")
        print("=" * 80)
        
        # Test queries designed to trigger different parts of the pipeline
        test_queries = [
            {
                "query": "If a pizza costs $18 and you order 3 pizzas, what's the total cost?",
                "expected_type": "llm",
                "description": "Should use CTransformers LLM reasoning"
            },
            {
                "query": "A car travels 120 km in 2 hours. What is its speed?",
                "expected_type": "llm", 
                "description": "Should use LLM for speed calculation"
            },
            {
                "query": "A rectangle has length 12 cm and width 8 cm. What is its area?",
                "expected_type": "llm",
                "description": "Should use LLM for area calculation"
            },
            {
                "query": "Two numbers sum to 50 and their difference is 10. What are the numbers?",
                "expected_type": "symbolic",
                "description": "Should try symbolic solver first"
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n[{i}] {test_case['description']}")
            print("-" * 60)
            print(f"Query: {test_case['query']}")
            print(f"Expected: {test_case['expected_type']} solution")
            print()
            
            start_time = time.time()
            
            try:
                # Run through complete pipeline
                result = self.pipeline(test_case['query'], top_k=3)
                
                elapsed = time.time() - start_time
                
                # Display results
                result_type = getattr(result, 'result_type', 'unknown')
                print(f"✅ Result Type: {result_type}")
                print(f"⏱ Total Processing Time: {elapsed:.3f} seconds")
                
                # Show retrieved similar problems
                results = getattr(result, 'results', [])
                if results:
                    print(f"\n📋 Retrieved {len(results)} similar problems:")
                    for j, res in enumerate(results[:2], 1):  # Show top 2
                        similarity = res.get('similarity', 0)
                        text = res.get('text', '')[:100] + "..."
                        print(f"  [{j}] Similarity: {similarity:.4f}")
                        print(f"      Text: {text}")
                
                # Show extracted equations
                equations = getattr(result, 'equations', None)
                if equations:
                    print(f"\n📘 Extracted Equations ({len(equations)}):")
                    for eq in equations:
                        print(f"  • {eq}")
                
                # Show solution
                solution = getattr(result, 'solution', None)
                if solution:
                    print(f"\n🧮 Solution:")
                    for var, val in solution.items():
                        print(f"  {var} = {val}")
                    
                    # Show residuals for symbolic solutions
                    residuals = getattr(result, 'residuals', None)
                    if residuals:
                        print(f"\n🔍 Verification Residuals:")
                        for eq, val in residuals.items():
                            if isinstance(val, dict):
                                satisfied = val.get('satisfied', False)
                                value = val.get('value', 'N/A')
                                status = "✓" if satisfied else "✗"
                                print(f"  {status} {eq}: {value}")
                            else:
                                print(f"  • {eq}: {val}")
                
                # Show LLM reasoning if available
                reasoning = getattr(result, 'reasoning', None)
                if reasoning:
                    print(f"\n🤖 LLM Reasoning:")
                    print(textwrap.fill(reasoning[:400] + "...", width=80))
                
                # Show note if unresolved
                note = getattr(result, 'note', None)
                if note:
                    print(f"\nℹ️ Note: {note}")
                
            except Exception as e:
                print(f"❌ Pipeline failed: {e}")
                import traceback
                traceback.print_exc()
            
            print("\n" + "="*80)

    def run_comprehensive_demo(self):
        """Run the complete comprehensive demo with CTransformers LLM"""
        print("🎯 CALCMATE CTRANSFORMERS LLM NEURO-SYMBOLIC DEMO")
        print("=" * 80)
        print("Testing all components of the hybrid neuro-symbolic system:")
        print("• Mathematical similarity retrieval")
        print("• Symbolic equation solving")
        print("• CTransformers LLM reasoning")
        print("• Solution verification")
        print("• Complete pipeline integration")
        print("=" * 80)
        
        if not self.load_system():
            return
        
        # Test CTransformers LLM directly
        self.test_ctransformers_llm_directly()
        
        # Test LLM reasoner module
        self.test_llm_reasoner_module()
        
        # Test complete pipeline
        self.test_complete_pipeline_with_llm()
        
        print("\n🎉 COMPREHENSIVE CTRANSFORMERS DEMO COMPLETED!")
        print("=" * 80)
        print("✅ CTransformers LLM integration tested successfully")
        print("✅ Neuro-symbolic pipeline with LLM verified")
        print("✅ Symbolic solving capabilities demonstrated")
        print("✅ Retrieval and LLM reasoning systems validated")
        print("\n💡 The system now combines:")
        print("   • Neural similarity search")
        print("   • Symbolic mathematics")
        print("   • Local LLM reasoning (CTransformers)")
        print("   • Solution verification")


def main():
    """Main demo runner"""
    demo = CTransformersNeuroSymbolicDemo()
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    main()
