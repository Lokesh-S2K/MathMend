import psycopg2
from config import DB_PARAMS

# The dimension of the MathBERT embeddings is 768
VECTOR_DIMENSION = 384

def create_table():
    """ Connects to PostgreSQL and creates the math_problems table. """
    conn = None
    try:
        print("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()

        # Drop the table if it already exists to start fresh
        print("Dropping existing table (if any)...")
        cur.execute("DROP TABLE IF EXISTS math_problems;")

        # Create the table
        print("Creating table 'math_problems'...")
        create_table_command = f"""
        CREATE TABLE math_problems (
            id SERIAL PRIMARY KEY,
            problem_text TEXT NOT NULL,
            reasoning_text TEXT,
            problem_category VARCHAR(100),
            num_variables INT,
            equation_type VARCHAR(100),
            symbolic_fingerprint TEXT UNIQUE,
            embedding VECTOR({VECTOR_DIMENSION})
        );
        """
        cur.execute(create_table_command)

        conn.commit()
        print("Table 'math_problems' created successfully.")
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")

if __name__ == '__main__':
    create_table()