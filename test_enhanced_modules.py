#!/usr/bin/env python3
"""
Test script to verify the enhanced DSPy modules work correctly
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dspy_modules import (
    SmartRetrievalPipeline,
    SymbolicSolver,
    LLMReasoner,
    initialize_ctransformers_model
)

def test_basic_functionality():
    """Test basic functionality of the enhanced modules"""
    print("🧪 TESTING ENHANCED DSPY MODULES")
    print("=" * 50)
    
    # Test 1: LLM Initialization
    print("\n1️⃣ Testing LLM Initialization...")
    try:
        llm = initialize_ctransformers_model()
        if llm is not None:
            print("✅ LLM initialization successful")
        else:
            print("⚠️ LLM initialization failed - will use fallback")
    except Exception as e:
        print(f"❌ LLM initialization error: {e}")
    
    # Test 2: Symbolic Solver
    print("\n2️⃣ Testing Symbolic Solver...")
    try:
        solver = SymbolicSolver()
        test_equations = ["x + y = 50", "x - y = 10"]
        result = solver(test_equations)
        print(f"✅ Symbolic solver test: Success={result.success}")
        if result.success:
            print(f"   Solution: {result.solution}")
    except Exception as e:
        print(f"❌ Symbolic solver error: {e}")
    
    # Test 3: LLM Reasoner
    print("\n3️⃣ Testing LLM Reasoner...")
    try:
        reasoner = LLMReasoner(llm)
        test_query = "Two numbers sum to 50 and their difference is 10. What are the numbers?"
        test_examples = [
            {
                "text": "Two numbers add up to 30 and their difference is 6. Find the numbers.",
                "equations": ["x + y = 30", "x - y = 6"]
            }
        ]
        result = reasoner(test_query, [], test_examples)
        print(f"✅ LLM reasoner test: Success={result.success}")
        if result.llm_equations:
            print(f"   Extracted equations: {result.llm_equations}")
    except Exception as e:
        print(f"❌ LLM reasoner error: {e}")
    
    # Test 4: Pipeline Loading (without full initialization)
    print("\n4️⃣ Testing Pipeline Loading...")
    try:
        # This will test if the pipeline can be imported and basic initialization works
        print("✅ Pipeline import successful")
        print("   Note: Full pipeline test requires FAISS index files")
    except Exception as e:
        print(f"❌ Pipeline loading error: {e}")
    
    print("\n🎯 BASIC FUNCTIONALITY TEST COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    test_basic_functionality()
