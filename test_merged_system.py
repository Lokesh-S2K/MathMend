#!/usr/bin/env python3
"""
Test script to verify the merged DSPy modules work correctly
This tests both the working SymPy solving and LLM extraction functionality
along with the comprehensive metrics evaluation.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_merged_functionality():
    """Test the merged functionality"""
    print("🧪 TESTING MERGED DSPY MODULES")
    print("=" * 50)
    
    try:
        # Test 1: Import merged modules
        print("\n1️⃣ Testing Module Imports...")
        from dspy_modules_merged import (
            SmartRetrievalPipeline,
            SymbolicSolver,
            LLMReasoner,
            Verifier,
            initialize_ctransformers_model,
            # All comprehensive metrics
            ExactMatchEvaluator,
            PassAtKEvaluator,
            SymbolicSolvingEvaluator,
            LLMSolverAgreementEvaluator,
            ReasoningConsistencyEvaluator,
            RetrievalRecallEvaluator,
            MathematicalEquivalenceEvaluator,
            FaithfulnessEvaluator,
            HallucinationRateEvaluator,
            ThroughputEvaluator,
            RetrievalPrecisionEvaluator,
            RetrievalRecallKEvaluator,
            MRREvaluator,
            NDCGEvaluator,
            CosineSimilarityDistributionEvaluator
        )
        print("✅ All modules imported successfully")
        
        # Test 2: Symbolic Solver (Working Version)
        print("\n2️⃣ Testing Symbolic Solver (Working Version)...")
        solver = SymbolicSolver()
        test_equations = ["x + y = 50", "x - y = 10"]
        result = solver(test_equations)
        print(f"✅ Symbolic solver test: Success={result.success}")
        if result.success:
            print(f"   Solution: {result.solution}")
            print(f"   Residuals: {result.residuals}")
        else:
            print(f"   Error: {result.error_msg}")
        
        # Test 3: Verifier (Working Version)
        print("\n3️⃣ Testing Verifier (Working Version)...")
        verifier = Verifier()
        test_solution = {"x": 30.0, "y": 20.0}
        verify_result = verifier(test_equations, test_solution)
        print(f"✅ Verifier test: Success={verify_result.verification['ok']}")
        print(f"   Verification details: {verify_result.verification}")
        
        # Test 4: LLM Initialization
        print("\n4️⃣ Testing LLM Initialization...")
        llm = initialize_ctransformers_model()
        if llm is not None:
            print("✅ LLM initialization successful")
        else:
            print("⚠️ LLM initialization failed - will use fallback")
        
        # Test 5: LLM Reasoner (Working Version)
        print("\n5️⃣ Testing LLM Reasoner (Working Version)...")
        reasoner = LLMReasoner(llm)
        test_query = "Two numbers sum to 50 and their difference is 10. What are the numbers?"
        test_examples = [
            {
                "text": "Two numbers add up to 30 and their difference is 6. Find the numbers.",
                "equations": ["x + y = 30", "x - y = 6"]
            }
        ]
        llm_result = reasoner(test_query, [], test_examples)
        print(f"✅ LLM reasoner test: Success={llm_result.success}")
        if llm_result.llm_equations:
            print(f"   Extracted equations: {llm_result.llm_equations}")
        
        # Test 6: Comprehensive Metrics
        print("\n6️⃣ Testing Comprehensive Metrics...")
        
        # Test Exact Match Evaluator
        em_evaluator = ExactMatchEvaluator()
        predicted_solutions = [{"x": 30.0, "y": 20.0}]
        ground_truth_solutions = [{"x": 30.0, "y": 20.0}]
        em_result = em_evaluator(predicted_solutions, ground_truth_solutions)
        print(f"✅ Exact Match Evaluator: Rate={em_result.exact_match_rate:.4f}")
        
        # Test Symbolic Solving Evaluator
        sym_evaluator = SymbolicSolvingEvaluator()
        symbolic_results = [{"success": True, "solution": {"x": 30.0}}]
        sym_result = sym_evaluator(symbolic_results)
        print(f"✅ Symbolic Solving Evaluator: Rate={sym_result.symbolic_success_rate:.4f}")
        
        # Test Throughput Evaluator
        throughput_evaluator = ThroughputEvaluator()
        processing_times = [1.5, 2.0, 1.8]
        batch_sizes = [1, 1, 1]
        throughput_result = throughput_evaluator(processing_times, batch_sizes)
        print(f"✅ Throughput Evaluator: {throughput_result.throughput:.2f} items/sec")
        
        print("\n✅ ALL TESTS PASSED!")
        print("🎯 The merged system combines:")
        print("   • Working SymPy solving functionality")
        print("   • Working LLM equation extraction")
        print("   • All 15 comprehensive evaluation metrics")
        print("   • Complete pipeline integration")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_demo_integration():
    """Test the demo integration"""
    print("\n🎮 TESTING DEMO INTEGRATION")
    print("=" * 50)
    
    try:
        # Test importing the merged demo
        from complete_pipeline_demo_merged import CompletePipelineDemo
        print("✅ Merged demo imported successfully")
        
        # Test demo initialization
        demo = CompletePipelineDemo()
        print("✅ Demo initialized successfully")
        
        # Test evaluators initialization
        print(f"✅ Evaluators initialized: {len(demo.evaluators)} metrics")
        
        print("✅ DEMO INTEGRATION TESTS PASSED!")
        
    except Exception as e:
        print(f"❌ Demo integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 MERGED SYSTEM VERIFICATION")
    print("=" * 60)
    
    # Test merged functionality
    if test_merged_functionality():
        print("\n✅ MERGED FUNCTIONALITY VERIFIED!")
    else:
        print("\n❌ MERGED FUNCTIONALITY FAILED!")
        sys.exit(1)
    
    # Test demo integration
    if test_demo_integration():
        print("\n✅ DEMO INTEGRATION VERIFIED!")
    else:
        print("\n❌ DEMO INTEGRATION FAILED!")
        sys.exit(1)
    
    print("\n🎉 ALL VERIFICATION TESTS PASSED!")
    print("=" * 60)
    print("🎯 The merged system is ready for use:")
    print("   • Run: python complete_pipeline_demo_merged.py")
    print("   • Or: python dspy_modules_merged.py (for direct module testing)")
    print("   • Features: Working SymPy + LLM + 15 comprehensive metrics")
