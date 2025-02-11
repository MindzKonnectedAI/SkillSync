import os
import pandas as pd

def get_csv_headers(folder_path):
    """
    Retrieves the headers (column names) for each CSV file in the specified folder.

    Args:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        dict: A dictionary where the keys are CSV file names and the values are lists of column headers.
    """
    headers_dict = {}
    
    # Get all CSV file names in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    # Loop through each CSV file and get headers
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        try:
            df = pd.read_csv(file_path)
            headers_dict = df.columns.tolist()
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    return headers_dict
