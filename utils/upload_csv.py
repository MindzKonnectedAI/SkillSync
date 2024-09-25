import streamlit as st
import os
import utils.csv_to_sql as csv_to_sql

# Function to handle file upload
def upload_csv(uploaded_file):
    folder_path = "csv"

    # Delete all existing files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            st.error(f"Failed to delete file {file_name}: {str(e)}")
            return

    # Save the new file
    file_path = os.path.join(folder_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.sidebar.success(f"CSV file saved successfully in {folder_path} as {uploaded_file.name}")

    # Call the function to save CSV data into the database
    csv_to_sql.save_csv_to_sql(file_path)
