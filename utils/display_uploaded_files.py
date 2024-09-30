import os
import streamlit as st

# Function to display already uploaded PDF files in the sidebar
def display_uploaded_files(index,folder_path,file_extension):
    # Get the list of PDF files in the directory
    files = [file for file in os.listdir(folder_path) if file.endswith(file_extension)]

    # Display the list of files
    if files:
        selected_file = st.sidebar.radio(
            label="Uploaded file",  # Empty label to collapse visibility
            options=files,
            key="file_selector"+index,
            label_visibility="collapsed",  # Collapse the label visibility
        )
        return selected_file
    else:
        st.sidebar.write("No files found in the upload directory.")
    return None