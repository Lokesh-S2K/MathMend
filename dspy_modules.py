import dspy
import numpy as np
from pipeline_sequence.advanced_equation_extractor import extract_equations_advanced
from pipeline_sequence.canonicalizer import canonicalize_system
from pipeline_sequence.features import build_structure_vector_from_parsed
from pipeline_sequence.embedder import encode_texts
from pipeline_sequence.indexer_faiss import FaissIndexer
import json
import re
from pipeline_sequence.embedder import build_text_for_embedding

from sklearn.metrics.pairwise import cosine_similarity

class RetrievalEvaluator(dspy.Module):
    """
    Module to evaluate the quality of retrievals:
    - retrieval precision at k
    - mean cosine similarity ranking
    """

    def forward(self, query_vectors, target_vectors, top_k=5):
        """
        Compute evaluation metrics for retrieval embeddings.

        Args:
            query_vectors (np.ndarray): NxD matrix of query embeddings
            target_vectors (np.ndarray): MxD matrix of target embeddings (knowledge base)
            top_k (int): Number of top results to consider for precision@k

        Returns:
            dict: {'precision_at_k': float, 'mean_cosine_similarity': float}
        """

        # Cosine similarity between all queries and target embeddings
        cos_sim = cosine_similarity(query_vectors, target_vectors)

        # Precision@k: assume relevance = max similarity per query is relevant
        precision_scores = []
        mean_cos_scores = []

        for i in range(len(query_vectors)):
            top_indices = cos_sim[i].argsort()[::-1][:top_k]
            top_sim_values = cos_sim[i][top_indices]

            # For demo, consider all top-k as "relevant" if similarity > 0.7
            precision = sum(top_sim_values >= 0.7) / top_k
            precision_scores.append(precision)

            mean_cos_scores.append(top_sim_values.mean())

        return dspy.Prediction(
            precision_at_k=float(np.mean(precision_scores)),
            mean_cosine_similarity=float(np.mean(mean_cos_scores)),
        )



# Input and output schema for DSPy modules
class EquationExtraction(dspy.Signature):
    """Extract mathematical equations from text"""
    text = dspy.InputField(desc="Problem statement text")
    equations = dspy.OutputField(desc="List of extracted equations")

class Canonicalization(dspy.Signature):
    """Canonicalize equation system"""
    equations = dspy.InputField()
    canonical = dspy.OutputField()

class StructureEmbedding(dspy.Signature):
    """Create structural embedding for canonical equations"""
    canonical = dspy.InputField()
    structure_vector = dspy.OutputField()

class TextEmbedding(dspy.Signature):
    """Encode problem text into embedding"""
    text = dspy.InputField()
    text_vector = dspy.OutputField()

class HybridFusion(dspy.Signature):
    """Fuse structure and text embeddings into one"""
    structure_vector = dspy.InputField()
    text_vector = dspy.InputField()
    hybrid_vector = dspy.OutputField()

class RetrievalModule(dspy.Signature):
    """Retrieve top-k similar problems using FAISS"""
    hybrid_vector = dspy.InputField()
    top_k = dspy.InputField(default=5)
    results = dspy.OutputField(desc="List of retrieved problems and similarity scores")


class EquationExtractor(dspy.Module):
    def forward(self, text):
        equations = extract_equations_advanced(text)
        if not equations:
            # fallback
            equations = []
        return dspy.Prediction(equations=equations)


class Canonicalizer(dspy.Module):
    def forward(self, equations):
        if not equations:
            return dspy.Prediction(canonical=[])
        canonical = canonicalize_system(equations)
        return dspy.Prediction(canonical=canonical)


class StructureEncoder(dspy.Module):
    def forward(self, canonical):
        if not canonical:
            return dspy.Prediction(structure_vector=np.zeros(256))
        vec = build_structure_vector_from_parsed(canonical)
        return dspy.Prediction(structure_vector=vec)


class TextEncoder(dspy.Module):
    def forward(self, text):
        vec = encode_texts([text], normalize=True)[0]
        return dspy.Prediction(text_vector=vec)


class HybridEncoder(dspy.Module):
    def forward(self, structure_vector, text_vector):
        hybrid = np.concatenate([structure_vector, text_vector])
        return dspy.Prediction(hybrid_vector=hybrid)


class Retriever(dspy.Module):
    def __init__(self, index_path, idmap_path):
        super().__init__()
        self.index = FaissIndexer.load(index_path, idmap_path)
        with open(idmap_path) as f:
            self.db = json.load(f)

    def forward(self, hybrid_vector, top_k=5):
        distances, results = self.index.search(hybrid_vector, top_k=top_k)
        retrieved = []
        for d, idx in zip(distances, results):
            try:
                entry = self.db[str(idx)]
                text = entry.get("text", "No text field available")
            except Exception:
                text = f"[ID {idx}] (metadata missing)"
            retrieved.append({"similarity": float(d), "text": text})
        return dspy.Prediction(results=retrieved)

    
    
# --- Full DSPy Pipeline ---
class SmartRetrievalPipeline(dspy.Module):
    """
    DSPy-based hybrid retrieval pipeline that combines:
    - Equation understanding (symbolic)
    - Textual embedding (semantic)
    - FAISS retrieval (vector search)
    """

    def __init__(self, index_path, idmap_path):
        super().__init__()

        print("Loading DSPy Hybrid Embedding Retrieval System...")
        print("=" * 60)

        # Core DSPy modules
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

        print(f"System Loaded Successfully!")
        print(f"   Knowledge Base: {len(self.problem_db)} math problems")
        print()

    # --------------------------------------------------------------------
    #  DSPy forward pass — defines the full inference sequence
    # --------------------------------------------------------------------
    def forward(self, text: str, top_k: int = 5):
        """
        DSPy standard forward() → called automatically during inference.
        Handles extraction → canonicalization → embedding → retrieval.
        """

        # Step 1: Equation extraction (with fallback)
        eq_pred = self.eq_extractor(text)
        equations = eq_pred.equations if eq_pred.equations else self._pattern_based_extraction(text)

        # Step 2: Canonicalize and encode structure
        can_pred = self.canonicalizer(equations)
        struct_pred = self.structure_encoder(can_pred.canonical)

        # Step 3: Text embedding
        text_for_embedding = build_text_for_embedding(text, fingerprint=None)
        text_pred = self.text_encoder(text_for_embedding)

        # Step 4: Hybrid fusion
        hybrid_pred = self.hybrid_encoder(struct_pred.structure_vector, text_pred.text_vector)
        hybrid_vec = hybrid_pred.hybrid_vector

        # Step 5: Retrieval
        pred = self.retriever(hybrid_vec, top_k=top_k)

        # Step 6: Return DSPy prediction
        return dspy.Prediction(
            results=pred.results,
            equations=equations,
            hybrid_vector=hybrid_vec
        )

    # --------------------------------------------------------------------
    #  Pattern-based fallback for weak equation detection
    # --------------------------------------------------------------------
    def _pattern_based_extraction(self, text):
        text_lower = text.lower()
        equations = []

        if any(word in text_lower for word in ["speed", "distance", "km/h", "mph"]):
            numbers = re.findall(r"\b(\d+)\b", text)
            if len(numbers) >= 2:
                equations.append(f"speed = {numbers[0]} / {numbers[1]}")
        elif any(word in text_lower for word in ["cost", "price", "buy", "spend"]):
            numbers = re.findall(r"\$?(\d+)", text)
            if len(numbers) >= 2:
                equations.append(f"total = {numbers[0]} * {numbers[1]}")
        elif "%" in text:
            numbers = re.findall(r"\b(\d+)\b", text)
            if len(numbers) >= 2:
                equations.append(f"result = {numbers[1]} * {numbers[0]} / 100")
        elif "sum" in text_lower and "difference" in text_lower:
            numbers = re.findall(r"\b(\d+)\b", text)
            if len(numbers) >= 2:
                equations.extend([f"x + y = {numbers[0]}", f"x - y = {numbers[1]}"])
        return equations
    
    
    # --------------------------------------------------------------------
    #  Evaluation function
    # --------------------------------------------------------------------
    def evaluate_retrieval_quality(self, queries, top_k=5):
        """
        Compute retrieval quality metrics for a list of queries.

        Args:
            queries (list[str]): List of math problem queries
            top_k (int): Number of top results to consider

        Returns:
            dict: Precision@k and mean cosine similarity
        """

        if not queries:
            print("⚠️ No queries provided for evaluation")
            return None

        # Step 1: Compute hybrid embeddings for all queries
        query_vectors = []
        for q in queries:
            hybrid_vec, _ = self.forward(q).hybrid_vector, self._pattern_based_extraction(q)
            query_vectors.append(hybrid_vec)
        query_vectors = np.vstack(query_vectors)

        # Step 2: Use all embeddings from the FAISS index
        # Assuming FAISS index can return vectors for evaluation
        target_vectors = np.array([self.index.reconstruct(i) for i in range(self.index.ntotal)])

        # Step 3: Evaluate with RetrievalEvaluator
        evaluator = RetrievalEvaluator()
        metrics = evaluator(query_vectors, target_vectors, top_k=top_k)
        return metrics
