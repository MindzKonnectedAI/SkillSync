# import pandas as pd
# from sqlalchemy import create_engine
# # from langchain_community.utilities import SQLDatabase

# def save_csv_to_sql(output_csv):
#     # Path to the SQLite database file
#     # db_path = "employee.db"
#     db_path = "employee.db"
#     print(f"Database {db_path} does not exist. Creating and loading data.")
#     # Load DataFrame from CSV
#     try:
#         df = pd.read_csv(output_csv, encoding='utf-8')
#     except UnicodeDecodeError:
#         df = pd.read_csv(output_csv, encoding='latin1')
#     # print(df.shape)
#     # print(df.columns.tolist())
#     # Create SQLite engine
#     engine = create_engine(f"sqlite:///{db_path}")
#     # Load DataFrame into SQLite database
#     df.to_sql("employee", engine, if_exists='replace', index=False)
#     # Initialize SQLDatabase object
#     # db = SQLDatabase(engine=engine)
#     print(f"Successfully Created db")

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import os

# Define folder paths
csv_folder = "csv"
db_folder = "db"

# Ensure the csv and db directories exist
os.makedirs(csv_folder, exist_ok=True)
os.makedirs(db_folder, exist_ok=True)

def save_csv_to_sql(csv_path):
    """
    Saves the content of a CSV file into an SQLite database.

    Args:
        csv_path (str): Path to the CSV file to be saved into the database.
        db_folder (str): Folder where the SQLite database will be stored.
    """
    # Construct the database path inside the db folder
    db_path = os.path.join(db_folder, "employee.db")

    # Load DataFrame from CSV
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='latin1')

    # Create SQLite engine
    engine = create_engine(f"sqlite:///{db_path}")

    # Load DataFrame into SQLite database
    df.to_sql("employee", engine, if_exists='replace', index=False)

    st.success(f"Database created and data loaded successfully into {db_path}")
