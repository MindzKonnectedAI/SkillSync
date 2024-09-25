import os
from langchain.chains.summarize import load_summarize_chain
from llama_parse import LlamaParse
import joblib
import chardet
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI


llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini")

# Create the upload directory if it doesn't exist
folder_path = "./ruleData"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

folder_path = "./data"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def create_pkl_string(filename):
    file_name, extension = os.path.splitext(filename)
    new_string = file_name + ".pkl"
    return new_string


def load_or_parse_data(pdf_path, file_name, src_folder):
    # data_file = "data/Introduction-of-MS-Office-MS-Word-PDF-eng.pkl"

    changed_file_ext = create_pkl_string(file_name)
    print("changed_file_ext", changed_file_ext)
    data_file = f"data/{changed_file_ext}"

    if os.path.exists(data_file):
        # Load the parsed data from the file
        parsed_data = joblib.load(data_file)
    else:
        # Perform the parsing step and store the result in llama_parse_documents
        parsingInstructionUber10k = """The provided document is unstructured
        It contains many tables, text, image and list.
        Try to be precise each and every details in proper fromat"""
        parser = LlamaParse(
            api_key="llx-8MMHGFCJ5PKqyfZM6h5D8epMtjzG4OEOe6lMCEOvgu67YgIt",
            result_type="markdown",
            parsing_instruction=parsingInstructionUber10k,
            max_timeout=5000,
        )
        llama_parse_documents = parser.load_data(pdf_path)
        # llama_parse_documents = parser.load_data("data/Introduction-of-MS-Office-MS-Word-PDF-eng.pdf")
        print("llama_parse_documents", llama_parse_documents)
        # Save the parsed data to a file
        print("Saving the parse results in .pkl format ..........")
        joblib.dump(llama_parse_documents, f"{src_folder}/{file_name}")

        # Set the parsed data to the variable
        parsed_data = llama_parse_documents

    return parsed_data


# load_document
def load_document(pdf_path, file_name, outputFileMD, src_folder):
    """
    Creates a vector database using document loaders and embeddings.
    This function loads urls,
    splits the loaded documents into chunks, transforms them into embeddings using OllamaEmbeddings,
    and finally persists the embeddings into a Chroma vector database.
    """
    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data(pdf_path, file_name, src_folder)
    print("llama_parse_documents", llama_parse_documents[0].text[:300])

    with open(f"{src_folder}/{outputFileMD}", "w", encoding="utf-8") as f:
        for doc in llama_parse_documents:
            f.write(doc.text + "\n")
    return


def find_file_name_and_extract_text(src_folder, outputFileMD):
    # Find the PDF file in the src folder
    pdf_filename = None
    for file_name in os.listdir(src_folder):
        if file_name.lower().endswith(".pdf"):
            pdf_filename = file_name
            break

    if pdf_filename:
        pdf_path = os.path.join(src_folder, pdf_filename)

        # Extract text from the PDF
        extracted_text = load_document(pdf_path, pdf_filename, outputFileMD, src_folder)
    else:
        print("No PDF file found in the src folder.")


def detect_encoding(file_path):
    with open(file_path, "rb") as file:
        raw_data = file.read()
    result = chardet.detect(raw_data)
    return result["encoding"]

prompt_template = """
Write a detailed summary of the following text, delimited by triple backquotes. Extract and include all key elements such as dates, numbers, symbols, strings, and specific points. It is crucial to explicitly mention any "and" or "or" conditions present in the text.

Return your response as bullet points, using titles and headings to categorize each element effectively. The summary should be optimized for a large language model like GPT-4o-mini.

```{text}```

**BULLET POINT SUMMARY:**
"""


summarize_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

def summarize_data(load_markdown_path, create_markdown_path):
    print("load_markdown_path", load_markdown_path)

    encoding = detect_encoding(load_markdown_path)
    print(f"Detected encoding: {encoding}")

    try:
        loader = UnstructuredMarkdownLoader(load_markdown_path, encoding=encoding)
        documents = loader.load()
    except UnicodeDecodeError as e:
        print(f"Error loading file: {e}")
        return

    # neo_4j.create_graph(documents)
    stuff_chain = load_summarize_chain(llm, chain_type="stuff", prompt=summarize_prompt)
    summarize_text = stuff_chain.invoke(documents)

    print("summarize_text", summarize_text)

    try:
        with open(create_markdown_path, "w", encoding="utf-8") as file:
            file.write(summarize_text["output_text"])
    except UnicodeEncodeError as e:
        print(f"Error writing file: {e}")

# Function to handle file upload
def upload_rule_data(uploaded_file):
    folder_path = "./ruleData"

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

    # Perform additional processing
    try:
        find_file_name_and_extract_text(folder_path, "outputRuleData.md")
        summarize_data(
            "./ruleData/outputRuleData.md", "./data/summarizeOutputRuleData.md"
        )
        st.success("Data processed successfully")
    except Exception as e:
        st.error(f"Error in processing data: {str(e)}")


# Function to display already uploaded PDF files in the sidebar
def display_uploaded_files(folder_path):
    # Get the list of PDF files in the directory
    files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]

    # Display the list of files
    if files:
        selected_file = st.sidebar.radio(
            label="Upload job description",  # Empty label to collapse visibility
            options=files,
            key="file_pdfselector",
            label_visibility="collapsed",  # Collapse the label visibility
        )
        return selected_file
    else:
        st.sidebar.write("No PDF files found in the upload directory.")
    return None