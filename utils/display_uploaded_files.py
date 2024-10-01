import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

# Function to view PDF content in a dialog
@st.dialog("View PDF")
def view_pdf(file_path):
    try:
        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()

        print("pages", pages)
        
        # Display the content of each page
        for page in pages:
            st.write(page.page_content)  # Assuming page_content holds the text content
            
    except Exception as e:
        st.error(f"An error occurred while reading the PDF: {str(e)}")

# Function to display uploaded files
def display_uploaded_files(index, folder_path, file_extension):
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        st.sidebar.error(f"Directory {folder_path} does not exist.")
        return None

    # Get the list of files in the directory
    files = [file for file in os.listdir(folder_path) if file.endswith(file_extension)]

    # Display files with an eye icon for PDFs
    if files:
        st.sidebar.subheader(f"{file_extension.upper()} Files:")
        for file in files:
            col1, col2 = st.sidebar.columns([3, 1])  # Two columns layout
            with col1:
                st.write(file)  # Display the file name
            with col2:
                # Eye icon button to show PDF in a dialog
                if file_extension == '.pdf':
                    if st.button("View", key=f"show_pdf_{file}"):
                        # Trigger PDF view in dialog
                        view_pdf(os.path.join(folder_path, file))  # Full file path

    if not files:
        st.sidebar.write(f"No {file_extension.upper()} files found in the upload directory.")
