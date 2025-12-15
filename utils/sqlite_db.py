from langchain_community.utilities import SQLDatabase
import sqlite3
import pandas as pd

DB_PATH = "financials.db"

def create_table_from_input(input_path: str, table_name: str, db_path: str = DB_PATH):
    """Reads a input file and creates a table in the SQLite database."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_csv(input_path)
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Successfully created table '{table_name}' from {input_path}")
    except Exception as e:
        print(f"Error creating table '{table_name}': {e}")
    finally:
        conn.close()

def verify_database(db_path: str = DB_PATH):
    """Prints schema info using LangChain wrapper."""
    try:
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        print("\n--- Database Schema ---")
        print(db.get_table_info())
    except Exception as e:
        print(f"Error verifying database: {e}")

if __name__ == "__main__":
    # Example usage with the revenue summary
    create_table_from_input("sec-edgar-filings/revenue_summary.csv", "revenue_summary")
    
    verify_database()