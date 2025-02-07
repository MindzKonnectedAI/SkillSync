import streamlit as st
import os
import utils.csv_to_sql as csv_to_sql

# Function to handle file upload
# def upload_csv(uploaded_file,container):
#     folder_path = "csv"

#     # Delete all existing files in the folder
#     for file_name in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, file_name)
#         try:
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#         except Exception as e:
#             container.error(f"Failed to delete file {file_name}: {str(e)}")
#             return

#     # Save the new file
#     file_path = os.path.join(folder_path, uploaded_file.name)
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     container.success(f"CSV file saved successfully in {folder_path} as {uploaded_file.name}")

#     # Call the function to save CSV data into the database
#     csv_to_sql.save_csv_to_sql(file_path,container)

import csv
from io import TextIOWrapper

def upload_csv(uploaded_file, container):
    folder_path = "csv"

    # Delete all existing files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            container.error(f"Failed to delete file {file_name}: {str(e)}")
            return

    # Process the uploaded CSV to remove spaces in headers
    try:
        # Read the uploaded file
        decoded_file = TextIOWrapper(uploaded_file, encoding="utf-8")
        csv_reader = csv.reader(decoded_file)

        # Extract and clean headers (remove spaces)
        headers = next(csv_reader)
        cleaned_headers = [header.strip().replace(" ", "") for header in headers]

        # Read remaining rows
        rows = list(csv_reader)

        # Save the cleaned CSV
        file_path = os.path.join(folder_path, uploaded_file.name)
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(cleaned_headers)  # Write cleaned headers
            csv_writer.writerows(rows)  # Write data rows

        container.success(f"CSV file saved successfully in {folder_path} as {uploaded_file.name}")

        # Call the function to save CSV data into the database
        csv_to_sql.save_csv_to_sql(file_path, container)

    except Exception as e:
        container.error(f"Error processing CSV: {str(e)}")