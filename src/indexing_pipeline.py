import re
import pandas as pd
import psycopg2
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sympy import sympify, Eq, SympifyError
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

from config import DB_PARAMS

# The specific MathBERT model we'll use
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
VECTOR_DIMENSION = 384 # MathBERT's dimension

def extract_equations(reasoning_text: str) -> list:
    """Extracts equation strings from reasoning text."""
    # Handle cases where reasoning_text might be None or not a string
    if not isinstance(reasoning_text, str):
        return []

    equations_part = re.search(r"Final Equations:(.*)", reasoning_text, re.DOTALL)
    if not equations_part:
        return []
    
    equations_str = equations_part.group(1).strip()
    # A more robust regex to split by 'and' or commas that separate equations
    return [eq.strip() for eq in re.split(r'\s+and\s+|,', equations_str) if eq.strip()]

def create_symbolic_fingerprint(equations: list) -> str:
    """Creates a standardized fingerprint from a list of equation strings."""
    if not equations:
        return None
    
    try:
        # Define transformations to allow for implicit multiplication (e.g., '2x' becomes '2*x')
        transformations = standard_transformations + (implicit_multiplication_application,)
        
        # Standardize variable names to v1, v2, etc.
        all_eq_str = "".join(equations)
        variables = sorted(list(set(re.findall(r'[a-zA-Z]', all_eq_str))))
        var_map = {var: f'v{i+1}' for i, var in enumerate(variables)}
        
        standardized_eqs = []
        for eq_str in equations:
            if '=' not in eq_str:
                continue

            # Replace variables for consistency
            for old, new in var_map.items():
                eq_str = re.sub(r'\b' + old + r'\b', new, eq_str)
            
            # Split into LHS and RHS
            lhs_str, rhs_str = eq_str.split('=', 1)
            
            # Parse with implicit multiplication enabled
            lhs = parse_expr(lhs_str.strip(), transformations=transformations)
            rhs = parse_expr(rhs_str.strip(), transformations=transformations)
            
            # Create the standard form: LHS - RHS = 0
            standard_form = lhs - rhs
            standardized_eqs.append(str(standard_form))
        
        if not standardized_eqs:
            return None

        # Sort the standardized equations to make the order consistent
        standardized_eqs.sort()
        return "; ".join(standardized_eqs)
        
    except (SympifyError, SyntaxError, TypeError, Exception) as e:
        # Catch any parsing error and print a warning, but don't crash
        print(f"Could not parse equations: {equations}. Error: {e}")
        return None

def extract_metadata(problem_text: str) -> dict:
    """Placeholder function to extract metadata."""
    # In a real system, this would use NLP or rules to find the category, etc.
    # For now, we'll use dummy data.
    if "age" in problem_text.lower():
        category = "age_problem"
    elif "apple" in problem_text.lower():
        category = "inventory_problem"
    else:
        category = "finance_problem"
        
    return {
        "problem_category": category,
        "num_variables": 2, # Assuming 2 for these examples
        "equation_type": "linear"
    }

def main():
    """Main function to run the indexing pipeline."""
    print("Starting the indexing pipeline...")
    
    # 1. Load the dataset
    df = pd.read_csv('data/dataset.csv', engine='python', on_bad_lines='warn')
    print(f"Loaded {len(df)} problems from CSV.")
    
    # 2. Initialize the embedding model
    print(f"Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")
    
    # 3. Connect to the database
    conn = psycopg2.connect(**DB_PARAMS)
    cur = conn.cursor()
    
    print("Processing and inserting data into PostgreSQL...")
    # 4. Iterate through the dataset and insert into DB
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        problem = row['problem']
        reasoning = row['reasoning']
        
        # A. Analyze and generate data
        equations = extract_equations(reasoning)
        fingerprint = create_symbolic_fingerprint(equations)
        metadata = extract_metadata(problem)
        
        # B. Create the text to embed
        text_to_embed = f"Problem: {problem} Equations: {fingerprint if fingerprint else ''}"
        
        # C. Generate the embedding
        embedding = model.encode(text_to_embed).tolist()
        
        # D. Insert into the database
        if fingerprint: # Only insert if we have a valid fingerprint
            try:
                cur.execute(
                    """
                    INSERT INTO math_problems 
                    (problem_text, reasoning_text, problem_category, num_variables, equation_type, symbolic_fingerprint, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        problem, 
                        reasoning, 
                        metadata['problem_category'], 
                        metadata['num_variables'], 
                        metadata['equation_type'], 
                        fingerprint, 
                        embedding
                    )
                )
            except psycopg2.errors.UniqueViolation:
                print(f"Skipping duplicate fingerprint: {fingerprint}")
                conn.rollback() # Rollback the failed transaction
                continue
            except Exception as e:
                print(f"An error occurred: {e}")
                conn.rollback()
                continue

    # Commit all transactions and close the connection
    conn.commit()
    cur.close()
    conn.close()
    
    print("Indexing pipeline finished successfully.")

if __name__ == '__main__':
    main()