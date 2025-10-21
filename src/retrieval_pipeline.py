import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer

# We are reusing the exact same functions from the indexing pipeline
# to ensure the query is processed in the same way as the stored data.
from indexing_pipeline import extract_equations, create_symbolic_fingerprint
from config import DB_PARAMS

# --- Configuration (Must match indexing_pipeline.py) ---
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
VECTOR_DIMENSION = 384
# ---------------------------------------------------------

def find_similar_problems(query_problem: str, top_k: int = 5):
    """
    Finds and re-ranks similar problems from the database.
    """
    print("--- Starting Retrieval Pipeline ---")
    
    # 1. Initialize the embedding model
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    # 2. Analyze the user's query
    print("Analyzing query problem...")
    query_equations = extract_equations(query_problem) # Assuming query has reasoning format for now
    query_fingerprint = create_symbolic_fingerprint(query_equations)
    
    text_to_embed = f"Problem: {query_problem} Equations: {query_fingerprint if query_fingerprint else ''}"
    query_embedding = model.encode(text_to_embed).tolist()
    
    print(f"Generated Fingerprint for query: {query_fingerprint}")

    # 3. Candidate Retrieval from PostgreSQL
    conn = None
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        print(f"\nSearching for top {top_k * 4} initial candidates...") # Fetch more to allow for re-ranking

        # Use the <=> operator from pgvector to find the nearest neighbors
        sql_query = """
    SELECT id, problem_text, reasoning_text, symbolic_fingerprint
    FROM math_problems
    ORDER BY embedding <=> %s::vector
    LIMIT %s;
"""
        # We fetch more candidates initially (e.g., 4 * top_k) to give the re-ranker a good selection
        cur.execute(sql_query, (query_embedding, top_k * 4))
        candidates = cur.fetchall()
        
        print(f"Found {len(candidates)} candidates.")

        # 4. Precise Re-ranking
        print("Re-ranking results based on symbolic fingerprints...")
        exact_matches = []
        similar_matches = []

        for candidate in candidates:
            # The columns are id, problem_text, reasoning_text, symbolic_fingerprint
            candidate_id, problem_text, reasoning_text, fingerprint = candidate
            
            # Check for an exact mathematical match
            if query_fingerprint and fingerprint == query_fingerprint:
                exact_matches.append({
                    "id": candidate_id,
                    "problem": problem_text,
                    "reasoning": reasoning_text,
                    "match_type": "Exact Symbolic Match"
                })
            else:
                similar_matches.append({
                    "id": candidate_id,
                    "problem": problem_text,
                    "reasoning": reasoning_text,
                    "match_type": "Similar Semantic Match"
                })
        
        # Combine the lists, with exact matches always first
        final_results = exact_matches + similar_matches
        
        return final_results[:top_k] # Return the top k results after re-ranking

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Database error: {error}")
        return []
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':
    # --- Example Usage ---
    # This example query is taken from your dataset.
    # For a real application, this would be user input.
    # We include a simplified "reasoning" part so the functions can extract the equations.
    example_query = """
    A car travels 180 miles downstream in 3 hours and 120 miles upstream in 3 hours. Write a system of linear equations to represent this information and find the speed of the car in still water and the speed of the current.
    Final Equations: x + y = 60 and x - y = 40
    """

    # Find the top 5 similar problems
    results = find_similar_problems(example_query, top_k=5)

    print("\n--- ✅ Top 5 Similar Problems ---")
    if not results:
        print("No results found.")
    else:
        for i, result in enumerate(results):
            print(f"\n--- Result {i+1} (ID: {result['id']}) ---")
            print(f"Match Type: {result['match_type']}")
            print(f"Problem: {result['problem']}")
            # print(f"Reasoning: {result['reasoning']}") # Optional: uncomment to see full reasoning