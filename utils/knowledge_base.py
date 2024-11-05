import streamlit as st
import os
import utils.csv_to_sql as csv_to_sql
import csv
import json

# Function to handle file upload
def upload_csv(uploaded_file,container):
    folder_path = "knowledge_base_csv"
    folder_path_json = "knowledge_base_json"
    # Delete all existing files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            container.error(f"Failed to delete file {file_name}: {str(e)}")
            return

    # Save the new file
    print("uploaded_file.name :",uploaded_file.name)
    file_path = os.path.join(folder_path, uploaded_file.name)
    file_path_json = os.path.join(folder_path_json,"knowledge_base.json")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    csv_to_json(file_path, file_path_json)

    container.success(f"CSV file saved successfully in {folder_path} as {uploaded_file.name}")

    # # Call the function to save CSV data into the database
    # csv_to_sql.save_csv_to_sql(file_path,container)

def csv_to_json(csv_file_path, json_file_path):
    """
    Convert CSV to JSON with each column as a key and values as an array of non-empty rows in that column.

    Args:
    - csv_file_path (str): Path to the CSV file.
    - json_file_path (str): Path where the JSON file will be saved.
    """
    data = {}

    # Open the CSV file and read it
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Initialize lists for each column header
        for header in csv_reader.fieldnames:
            data[header] = []
        
        # Populate lists with non-empty values from each column
        for row in csv_reader:
            for header, value in row.items():
                if value:  # Only add non-empty values
                    data[header].append(value)

    # Write the data to a JSON file
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

    print(f"Data successfully converted from {csv_file_path} to {json_file_path}")
