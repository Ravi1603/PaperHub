'''import pandas as pd
import psycopg2

def load_data_from_db(file_path='C:\\Users\\rroganna\\Desktop\\paper_hub\\data\\arxiv_papers.csv'):
    # Connection parameters from the docker-compose.yml file
    db_params = {
        'host': 'localhost',
        'database': 'airflow',  # From POSTGRES_DB in docker-compose
        'user': 'airflow',      # From POSTGRES_USER in docker-compose
        'password': 'airflow',  # From POSTGRES_PASSWORD in docker-compose
        'port': '5432'
    }
    
    try:
        print("Connecting to PostgreSQL database...")
        conn = psycopg2.connect(**db_params)
        print("Connection successful!")
        
        # First, check if the papers table exists
        check_table_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'papers'
        );
        """
        with conn.cursor() as cursor:
            cursor.execute(check_table_query)
            table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("The 'papers' table does not exist in the database.")
            print("Available tables:")
            list_tables_query = """
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public';
            """
            tables_df = pd.read_sql(list_tables_query, conn)
            print(tables_df)
            conn.close()
            return None
        
        # If table exists, proceed with the original query
        query = "SELECT * FROM papers"
        print(f"Executing query: {query}")
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Save DataFrame to CSV
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    load_data_from_db()
'''

import pandas as pd

def load_data_from_db():
    """
    Load data from CSV file instead of PostgreSQL database
    """
    try:
        print("Loading data from CSV file...")
        df = pd.read_csv('C:\\Users\\rroganna\\Desktop\\PaperHub\\data\\arxiv_papers.csv')
        print(f"Successfully loaded {len(df)} papers from CSV.")
        return df
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
        return None
