# MathMend DSPy Modules Analysis

## Overview

This codebase implements a **DSPy-based hybrid neuro-symbolic retrieval pipeline** for solving mathematical word problems. The system combines:

- **Equation extraction** from natural language text
- **Canonicalization** of mathematical expressions
- **Hybrid embedding** (structural + semantic)
- **FAISS-based vector retrieval** for similar problems
- **Symbolic solving** using SymPy
- **LLM reasoning** as fallback (using CTransformers/Llama-2)
- **Comprehensive evaluation metrics**

The system is designed to automatically solve math word problems by retrieving similar solved examples and applying symbolic computation, with LLM assistance when needed.

## System Architecture

### Core Components

1. **Preprocessing Pipeline**

   - EquationExtractor → Canonicalizer → StructureEncoder → TextEncoder → HybridEncoder

2. **Retrieval System**

   - FAISS-based vector search with hybrid embeddings

3. **Solving Components**

   - SymbolicSolver (SymPy-based)
   - LLMReasoner (CTransformers fallback)
   - Verifier (solution validation)

4. **Evaluation Framework**
   - Multiple evaluators for accuracy, retrieval quality, consistency, etc.

### Data Flow

```
Input Text → Equation Extraction → Canonicalization → Hybrid Embedding → FAISS Retrieval → Symbolic Solving → LLM Fallback → Verification → Output
```

## Working Modules

### 1. CTransformers LLM Initialization (`initialize_ctransformers_model`)

**Status: Working**

- Initializes Llama-2-7B-Chat model via CTransformers
- Handles model download and loading
- Returns LLM instance or None on failure

**Dependencies:** CTransformers, HuggingFace model

### 2. SymbolicSolver

**Status: Working**

- Uses SymPy to solve systems of equations symbolically
- Handles string equation parsing and cleaning
- Computes residuals and success validation
- Robust error handling for malformed equations

**Key Methods:**

- `_clean_equation_string()`: Preprocesses equations for SymPy compatibility
- `forward()`: Main solving logic with comprehensive error handling

### 3. Verifier

**Status: Working**

- Validates solutions against canonical equations
- Computes residuals with configurable tolerance
- Handles both equation objects and string representations

**Key Methods:**

- `_clean_equation_string()`: Same preprocessing as SymbolicSolver
- `forward()`: Verification with detailed residual analysis

### 4. LLMReasoner

**Status: Working**

- Uses LLM to extract equations from problems
- Builds prompts with few-shot examples
- Parses JSON responses for equations
- Fallback to canonical equations if LLM fails

**Key Methods:**

- `build_prompt()`: Constructs reasoning prompts with examples
- `forward()`: LLM inference and response parsing
- `_parse_response()`: Robust JSON and pattern-based parsing

### 5. Retriever

**Status: Working**

- FAISS-based retrieval with error handling
- Loads pre-built index and metadata
- Returns similarity-ranked results

**Dependencies:** FAISS, pre-built index files

### 6. Core Pipeline Modules

#### EquationExtractor

**Status: Working**

- Calls `extract_equations_advanced` from pipeline_sequence
- Returns list of equation strings
- Handles empty results gracefully

#### Canonicalizer

**Status: Working**

- Uses `canonicalize_system` from pipeline_sequence
- Converts equations to SymPy expressions
- Returns parsed canonical forms

#### StructureEncoder

**Status: Working**

- Calls `build_structure_vector_from_parsed`
- Creates structural embeddings from canonical equations

#### TextEncoder

**Status: Working**

- Uses SentenceTransformer for semantic embeddings
- Calls `encode_texts` with normalization

#### HybridEncoder

**Status: Working**

- Concatenates structure and text vectors
- Creates hybrid embedding for retrieval

### 7. SmartRetrievalPipeline (Main Pipeline)

**Status: Working**

- Orchestrates entire neuro-symbolic pipeline
- Implements fallback logic: Symbolic → LLM → Retrieval-only
- Comprehensive error handling and logging

**Key Features:**

- Pattern-based equation extraction fallback
- Multi-stage solving with verification
- Detailed result types and metadata

### 8. Evaluation Metrics

**All Working**

- **ExactMatchEvaluator**: Solution accuracy
- **PassAtKEvaluator**: Top-k accuracy
- **SymbolicSolvingEvaluator**: Symbolic success rates
- **LLMSolverAgreementEvaluator**: LLM-symbolic agreement
- **ReasoningConsistencyEvaluator**: Reasoning quality
- **RetrievalRecallEvaluator**: Retrieval effectiveness
- **MathematicalEquivalenceEvaluator**: Expression equivalence
- **FaithfulnessEvaluator**: Text faithfulness
- **HallucinationRateEvaluator**: Hallucination detection
- **ThroughputEvaluator**: Performance metrics
- **RetrievalPrecisionEvaluator**: Precision metrics
- **MRREvaluator**: Mean Reciprocal Rank
- **NDCGEvaluator**: Normalized DCG
- **CosineSimilarityDistributionEvaluator**: Similarity statistics

### 9. Similarity Explanation (`explain_similarity`)

**Status: Working**

- Analyzes query-document similarity
- Multiple signals: embedding, keywords, tokens, Jaccard
- Returns detailed explanation dictionary

## Dependencies

### Python Packages

- `dspy`: Core framework
- `numpy`, `pandas`, `torch`: Numerical computing
- `faiss`: Vector search
- `sentence-transformers`: Text embeddings
- `transformers`: LLM interface
- `langchain-community`: LLM wrapper
- `sympy`: Symbolic math
- `sklearn`: ML utilities
- `requests`: HTTP client
- `json`, `re`: Standard library

### External Modules

- `pipeline_sequence.advanced_equation_extractor`: Equation extraction
- `pipeline_sequence.canonicalizer`: Equation canonicalization
- `pipeline_sequence.features`: Structure vector building
- `pipeline_sequence.embedder`: Text embedding
- `pipeline_sequence.indexer_faiss`: FAISS indexing

### Data Files

- FAISS index file (`.index`)
- Metadata JSON file (`.json`) with problem database

## How to Work with the System

### For Updates and Modifications

1. **Adding New Modules**

   - Inherit from `dspy.Module`
   - Implement `forward()` method
   - Add to `SmartRetrievalPipeline` if needed

2. **Modifying Existing Components**

   - Locate the relevant class/method
   - Preserve input/output signatures for compatibility
   - Update imports if changing dependencies

3. **Testing Changes**

   - Use evaluation metrics for validation
   - Test with sample math problems
   - Verify end-to-end pipeline functionality

4. **Configuration**
   - LLM parameters in `initialize_ctransformers_model()`
   - Index paths in `Retriever.__init__()`
   - Tolerances in `Verifier` and evaluators

### Key Integration Points

- **Equation Extraction**: `EquationExtractor.forward()`
- **Canonicalization**: `Canonicalizer.forward()`
- **Embedding**: `StructureEncoder`, `TextEncoder`, `HybridEncoder`
- **Retrieval**: `Retriever.forward()`
- **Solving**: `SymbolicSolver.forward()`, `LLMReasoner.forward()`
- **Verification**: `Verifier.forward()`

### Running the System

```python
# Initialize pipeline
pipeline = SmartRetrievalPipeline(index_path="path/to/index", idmap_path="path/to/metadata")

# Solve a problem
result = pipeline("A train travels 60 km in 1 hour. What is its speed?")

# Access results
print(result.solution)  # {'speed': 60.0}
print(result.result_type)  # 'symbolic'
```

## EVOSS Pipeline Integration for Equation Extraction

### Current Implementation

The equation extraction is handled by `EquationExtractor.forward()`, which calls:

```python
equations = extract_equations_advanced(text)
```

This function is imported from `pipeline_sequence.advanced_equation_extractor`.

### Where to Change for EVOSS Integration

1. **Primary Change Location**: `EquationExtractor.forward()` method (around line ~450 in the file)

2. **Integration Steps**:

   - Replace the call to `extract_equations_advanced(text)` with EVOSS pipeline
   - Ensure EVOSS returns a list of equation strings in the same format
   - Update imports to include EVOSS modules
   - Maintain error handling for cases where extraction fails

3. **Code Modification Pattern**:

   ```python
   class EquationExtractor(dspy.Module):
       def forward(self, text):
           # Replace this line:
           # equations = extract_equations_advanced(text)

           # With EVOSS integration:
           equations = evoss_pipeline.extract_equations(text)  # Adjust based on EVOSS API

           # Rest of the method remains the same
           if equations and isinstance(equations, list):
               equation_strings = []
               for eq in equations:
                   if isinstance(eq, str):
                       equation_strings.append(eq)
                   elif hasattr(eq, '__str__'):
                       equation_strings.append(str(eq))
               return dspy.Prediction(equations=equation_strings)
           return dspy.Prediction(equations=[])
   ```

4. **Dependencies to Update**:

   - Remove import: `from pipeline_sequence.advanced_equation_extractor import extract_equations_advanced`
   - Add EVOSS imports at the top of the file

5. **Testing Considerations**:

   - Verify EVOSS output format matches expected input for `Canonicalizer`
   - Test with various math problem types
   - Ensure fallback mechanisms still work if EVOSS fails

6. **Potential Impact**:
   - May affect downstream canonicalization and solving accuracy
   - Could improve equation extraction quality if EVOSS is more robust
   - Requires re-evaluation of the full pipeline performance

### Recommended Integration Approach

1. Create a wrapper function that can switch between extraction methods
2. Add configuration parameter to choose extraction method
3. Gradually migrate and compare performance
4. Update evaluation metrics to track extraction quality

This modular design allows for easy swapping of equation extraction backends while maintaining the rest of the neuro-symbolic pipeline intact.
