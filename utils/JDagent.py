from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
import utils.create_image_func as create_image_func
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import os
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
import utils.create_image_func as create_image_func
import nltk
import streamlit as st
import os
from langchain.chains.summarize import load_summarize_chain
from llama_parse import LlamaParse
import joblib
import chardet
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from typing import Annotated, Literal
from langchain_core.messages import AIMessage


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
    data_file = f"{src_folder}/{changed_file_ext}"

    if os.path.exists(data_file):
        # Load the parsed data from the file
        return joblib.load(data_file)
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


def pdf_folder(uploaded_file,container):

    folder_path = "./pdf"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

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
    file_path = os.path.join(folder_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

def upload_job_description(uploaded_file,container):
    print("hello")
    folder_path = "./ruleData"

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
    file_path = os.path.join(folder_path, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    pdf_folder(uploaded_file,container)

    # Perform additional processing
    try:
        find_file_name_and_extract_text(folder_path, "outputRuleData.md")
        # summarize_data(
        #     "./ruleData/outputRuleData.md", "./data/summarizeOutputRuleData.md"
        # )
        jd_agent()
        container.success("Data processed successfully")
    except Exception as e:
        container.error(f"Error in processing data: {str(e)}")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    required_messages: Annotated[list[AnyMessage], add_messages]
    preferred_messages: Annotated[list[AnyMessage], add_messages]

def load_markdown(outputFile):

    # Debugging path
    print("Attempting to load file from:", os.path.abspath(outputFile))

    # Check file existence
    if not os.path.exists(outputFile):
        return f"Error: File not found at {outputFile}. Please check the path."

    try:
        loader = UnstructuredMarkdownLoader(outputFile, encoding="utf-8")
        documents = loader.load()
        texts = [d.page_content for d in documents]
        return texts[0] if texts else "Error: No content found in the markdown file."
    except Exception as e:
        return f"Error: An issue occurred while loading the file: {str(e)}"

import os

def get_file_content(file_name, folder_name):
    # Get the current working directory
    cwd = os.getcwd()

    # Create the full file path
    file_path = os.path.join(cwd, folder_name, file_name)

    try:
        # Open the file in read mode
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the content
            content = file.read()
        return content
    except FileNotFoundError:
        return f"Error: The file at '{file_path}' was not found."
    except Exception as e:
        return f"An error occurred: {e}"

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

folder_path = 'knowledge_base_csv'

# def find_required_point(state):
#     print("find_required_point", state)

#     job_description_file_content = get_file_content('outputRuleData.md', 'ruleData')
#     # print(job_description_file_content)

#     csv_headers = get_csv_headers(folder_path)


#     # Define a PromptTemplate
#     template="""
#         You are an AI assistant tasked with analyzing a job description to extract only **required** qualifications. Your goal is to determine which qualifications are critical for the job, even if they are not explicitly labeled as "required." Use context and phrasing to make this determination.

#         ### Job Description:
#         {job_description}

#         ### Instructions:
#         1. Extract only qualifications that are **required** for the job.
#         - A qualification is **required** if:
#             - It is explicitly stated as "required," "must have," "necessary," or "essential."
#             - It is strongly implied as critical to performing the job, based on context or role expectations.
#         - Ignore qualifications that are optional, nice-to-have, or described as "preferred."
#         2. Categorize the extracted qualifications under the provided table headers. If no qualification fits a header, leave it empty.
#         3. Ensure the extracted points match the exact wording in the job description.
#         4. Provide the output in a structured JSON format.
#         Ensure the output strictly adheres to the following JSON format:

#         ```json
#         {{
#         "Experience": ["..."],
#         "Frontend": ["..."],
#         "Backend": ["..."],
#         "DB": ["..."],
#         "Tools": ["..."],
#         "Miscellaneous": ["..."],
#         "Location": ["..."],
#         "Graduation": ["Bachelor's"/null],
#         "Post Graduation": ["Master's"/null]
#         }}

#         """

def find_required_point(state):
    print("find_required_point", state)

    job_description_file_content = get_file_content('outputRuleData.md', 'ruleData')
    # print(job_description_file_content)

    csv_headers = get_csv_headers(folder_path)


    # Define a PromptTemplate
    # template="""
    #     You are an AI assistant tasked with analyzing a job description to extract only **required** qualifications. Your goal is to determine which qualifications are critical for the job, even if they are not explicitly labeled as "required." Use context and phrasing to make this determination.

    #     ### Job Description:
    #     {job_description}

    #     ### Instructions:
    #     1. Extract only qualifications that are **required** for the job.
    #     - A qualification is **required** if:
    #         - It is explicitly stated as "required," "must have," "necessary," or "essential."
    #         - It is strongly implied as critical to performing the job, based on context or role expectations.
    #     - Ignore qualifications that are optional, nice-to-have, or described as "preferred."

    #     2. Categorize the extracted qualifications under the following table headers:
    #     - {table_header}

    #     For each header:
    #     - Include qualifications that fit directly under the category.
    #     - If no qualification fits a header, skip that header entirely and do not display it in the output.

    #     3. Ensure that:
    #     - The extracted points match the exact wording in the job description.
    #     - Qualifications are returned as a concise list of bullet points.

    #     """
        # ### Example Output:
        # - **Experience**:
        # - 3+ years in software development
        # - Proven experience in leading teams

        # - **Frontend**:
        # - Proficiency in React.js
        # - Strong knowledge of CSS and HTML

        # - **Backend**:
        # - Experience in building APIs using Node.js

        # - **DB**:
        # - Knowledge of relational databases like PostgreSQL

        # - **Tools**:
        # - Proficiency in Git and CI/CD pipelines

        # - **Location**:
        # - Chicago, IL

        # - **Graduation**:
        # - Bachelor's degree in Computer Science

    template="""
        You are an AI assistant tasked with analyzing a job description to extract only **required** qualifications. Your goal is to determine which qualifications are critical for the job, even if they are not explicitly labeled as "required." Use context and phrasing to make this determination.

        ### Job Description:
        {job_description}

        ### Instructions:
        1. Extract only qualifications that are **required** for the job.
        - A qualification is **required** if:
            - It is explicitly stated using terms like "required," "must have," "necessary," or "essential."
            - It is strongly implied as critical to performing the job, based on context or role expectations (e.g., "responsible for," "key function," "primary duty").
        - A qualification is **not required** if it is:
            - Labeled as "preferred," "nice-to-have," "desired," or "optional."
            - Mentioned as an advantage but not critical to the role.
            - Indirectly related to the job responsibilities without being explicitly critical.

        2. Categorize the extracted qualifications under the following table headers:
        - {table_header}

        For each header:
        - Include qualifications that directly fit under the category.
        - Skip any header for which no qualifications are found.

        3. Ensure that:
        - Extracted qualifications match the exact wording in the job description.
        - Qualifications are presented as a concise list of bullet points.

        ### Examples:
        - If the job description states, "A Bachelor's degree is required," include: **"Bachelor's degree."**
        - If it states, "Experience with Python is preferred," exclude: **"Experience with Python."**
        - If it states, "Must have excellent communication skills," include: **"Excellent communication skills."**
        - If it states, "Familiarity with cloud platforms like AWS is a plus," exclude: **"Familiarity with cloud platforms like AWS."**
        - Always consider location as required  


        ### Output:
        Return the qualifications as a structured list categorized under the provided table headers, ensuring that only **required** qualifications are included.

    """

    prompt = PromptTemplate(template=template, input_variables=["job_description", "table_header"])

    matching_points_llm = prompt | llm
    response = matching_points_llm.invoke({ "job_description": job_description_file_content, "table_header": csv_headers })
    # print("response: ", response)
    save_response_to_markdown(response.content, "./data/required_messages.md")

    return {"required_messages": [AIMessage(content=response.content)]}

def check_required_point(state):
    print("check_required_point", state)
    job_description_file_content = get_file_content('outputRuleData.md', 'ruleData')
    csv_headers = get_csv_headers(folder_path)

    # template="""
    #         You are an AI assistant tasked with verifying and improving the list of **required qualifications** extracted from a job description. Your goal is to:
    #         1. Validate the existing extracted points to ensure they align with the job description.
    #         2. Identify any required qualifications missing from the extracted list by analyzing the job description.
    #         3. Provide the final, complete list of required points.

    #         ### Job Description:
    #         {job_description}

    #         ### Existing Extracted Required Points:
    #         {extracted_required_points}

    #         ### Instructions:
    #         1. Cross-check the provided **extracted required points** with the job description.
    #         2. Identify any required qualifications explicitly stated or implied as critical in the job description that are missing from the provided list.
    #         - A point is **required** if it is explicitly marked as "required," "must have," "necessary," or "essential."
    #         - It may also be inferred as critical to performing the job based on the context or phrasing in the description.
    #         3. Ensure the final list is comprehensive, including all required points.

    #         Ensure the output strictly adheres to the following JSON format:

    #         ```json
    #         {{
    #         "Experience": ["..."],
    #         "Frontend": ["..."],
    #         "Backend": ["..."],
    #         "DB": ["..."],
    #         "Tools": ["..."],
    #         "Miscellaneous": ["..."],
    #         "Location": ["..."],
    #         "Graduation": ["Bachelor's"/null],
    #         "Post Graduation": ["Master's"/null]
    #         }}
    #     """

    # template = """
    # You are an AI assistant tasked with verifying and enhancing the list of **required qualifications** extracted from a job description. Your goal is to ensure the final list is accurate, comprehensive, and aligns with the job description.

    # ### Job Description:
    # {job_description}

    # ### Existing Extracted Required Points:
    # {extracted_required_points}

    # ### Instructions:
    # 1. Cross-check the **Existing Extracted Required Points** against the job description.
    # - Validate that each point aligns with the job description's explicit or strongly implied requirements.
    # - Ensure that points match the exact wording or phrasing used in the job description.
    # - Remove any points that are not explicitly required or strongly implied as critical.

    # 2. Identify any missing required qualifications:
    # - A qualification is **required** if:
    #     - It is explicitly stated as "required," "must have," "necessary," or "essential."
    #     - It is strongly implied as critical to performing the job based on context or role expectations.
    # - Add any qualifications meeting these criteria to the final list.

    # 3. Organize the final list under the following table headers:
    # - {table_header}
    # - Include qualifications that directly fit under each header.
    # - If no qualifications fit a header, omit it entirely from the final output.

    # 4. Present the final, complete list of required qualifications:
    # - Use bullet points for each qualification.
    # - Ensure the list is concise and comprehensive, covering all critical points.
    # """
    template = """
        You are an AI assistant tasked with verifying and enhancing the list of **required qualifications** extracted from a job description. Your goal is to ensure the final list is accurate, comprehensive, and aligns with the job description, focusing exclusively on **required** qualifications.

        ### Job Description:
        {job_description}

        ### Existing Extracted Required Points:
        {extracted_required_points}

        ### Instructions:
        1. Cross-check the **Existing Extracted Required Points** against the job description:
        - Validate that each point aligns with the job description's explicit or strongly implied **required** qualifications.
        - Ensure that points match the exact wording or phrasing used in the job description.
        - Remove any points that are not explicitly required or strongly implied as critical to the role.

        2. Identify any missing **required** qualifications:
        - A qualification is **required** if:
            - It is explicitly stated as "required," "must have," "necessary," or "essential."
            - It is strongly implied as critical to performing the job based on context or role expectations.
        - Do **not** include qualifications labeled as "preferred," "nice-to-have," or anything similar.

        3. Organize the final list under the following table headers:
        - {table_header}
        - Include qualifications that directly fit under each header.
        - If no qualifications fit a header, omit it entirely from the final output.

        4. Present the final, complete list of **required qualifications**:
        - Use bullet points for each qualification.
        - Ensure the list is concise and comprehensive, covering only the critical **required** qualifications.

        5. Always consider location as required  

        ### Important Note:
        - Exclude any points labeled as "preferred," "nice-to-have," or anything similar, even if they align with the role.

    """

    prompt = PromptTemplate(template=template, input_variables=["job_description", "extracted_required_points", "table_header"])

    matching_points_llm = prompt | llm
    response = matching_points_llm.invoke({"job_description": job_description_file_content,
     "extracted_required_points": state["required_messages"][-1].content,
     "table_header": csv_headers
     })
    save_response_to_markdown(response.content, "./data/check_required_messages.md")

    # # print("response: ", response)
    return {"required_messages": [AIMessage(content=response.content)]}
    # return matching_points_llm
    

def find_preferred_point(state):
    print("find_preferred_point", state)

    # job_description_file_content = get_file_content('outputRuleData.md', 'job_description')
    # job_description_file_content = get_file_content('summary.md', 'job_description')
    job_description_file_content = get_file_content('outputRuleData.md', 'ruleData')
    csv_headers = get_csv_headers(folder_path)

    # Define a PromptTemplate
    # template="""
    #     You are an AI assistant tasked with analyzing a job description to extract only **preferred** qualifications. Your goal is to identify qualifications that enhance a candidate's profile but are not explicitly required for the role.

    #     ### Job Description:
    #     {job_description}

    #     ### Instructions:
    #     1. Extract only qualifications that are **preferred** for the job.
    #     - A qualification is **preferred** if:
    #         - It is explicitly stated as "preferred," "nice to have," "a plus," or similar terms.
    #         - It is implied as desirable or beneficial but not critical for the role.
    #     - Do not include qualifications that are stated or implied as mandatory or required.
    #     2. Categorize the extracted qualifications under the provided table headers. If no qualification fits a header, leave it empty.
    #     3. Ensure the extracted points match the exact wording in the job description.

    #     Ensure the output strictly adheres to the following JSON format:

    #     ```json
    #     {{
    #     "Experience": ["..."],
    #     "Frontend": ["..."],
    #     "Backend": ["..."],
    #     "DB": ["..."],
    #     "Tools": ["..."],
    #     "Miscellaneous": ["..."],
    #     "Location": ["..."],
    #     "Graduation": ["Bachelor's"/null],
    #     "Post Graduation": ["Master's"/null]
    #     }}

    #     """
    template="""
        You are an AI assistant tasked with analyzing a job description to extract only **preferred** qualifications. Your goal is to determine which qualifications are desirable but not mandatory for the job, even if they are not explicitly labeled as "preferred." Use context and phrasing to make this determination.

        ### Job Description:
        {job_description}

        ### Instructions:
        1. Extract only qualifications that are **preferred** for the job.
        - A qualification is **preferred** if:
            - It is explicitly stated as "preferred," "nice-to-have," "desired," or "optional."
            - It is implied as beneficial or advantageous but not essential for performing the job.
        - Ignore qualifications that are explicitly described as "required," "must have," "necessary," or "essential."

        2. Categorize the extracted qualifications under the following table headers:
        - {table_header}

        For each header:
        - Include qualifications that fit directly under the category.
        - If no qualification fits a header, skip that header entirely and do not display it in the output.

        3. Ensure that:
        - The extracted points match the exact wording in the job description.
        - Qualifications are returned as a concise list of bullet points.


        """
        # ### Example Output:
        # - **Experience**:
        # - Experience in leading cross-functional teams
        # - Exposure to Agile methodologies

        # - **Frontend**:
        # - Familiarity with Vue.js
        # - Knowledge of Tailwind CSS

        # - **Backend**:
        # - Experience in building microservices architecture

        # - **DB**:
        # - Knowledge of Redis for caching

        # - **Tools**:
        # - Familiarity with Docker and Kubernetes

        # - **Miscellaneous**:
        # - Excellent presentation skills

        # - **Location**:
        # - Remote work flexibility preferred

        # - **Graduation**:
        # - Bachelor's degree in any field is sufficient, but a degree in Computer Science is preferred

        # - **Post Graduation**:
        # - Master's degree in Business Administration (MBA) preferred

    prompt = PromptTemplate(template=template, input_variables=["job_description", "table_header"])

    matching_points_llm = prompt | llm
    response = matching_points_llm.invoke({"job_description": job_description_file_content, "table_header": csv_headers })
    save_response_to_markdown(response.content, "./data/preferred_messages.md")

    # print("response: ", response)
    return {"preferred_messages": [AIMessage(content=response.content)]}

def check_preferred_point(state):
    print("Checking preferred point", state)
    job_description_file_content = get_file_content('outputRuleData.md', 'ruleData')
    csv_headers = get_csv_headers(folder_path)

    # template="""
    #         You are an AI assistant tasked with verifying and improving the list of **preferred qualifications** extracted from a job description. Your goal is to:
    #         1. Validate the existing extracted points to ensure they align with the job description.
    #         2. Identify any preferred qualifications missing from the extracted list by analyzing the job description.
    #         3. Provide the final, complete list of preferred points.

    #         ### Job Description:
    #         {job_description}

    #         ### Existing Extracted Preferred Points:
    #         {extracted_preferred_points}

    #         ### Instructions:
    #         1. Cross-check the provided **extracted preferred points** with the job description.
    #         2. Identify any preferred qualifications explicitly stated or implied as desirable or beneficial but not critical.
    #         - A point is **preferred** if it is explicitly marked with terms like "preferred," "nice to have," "a plus," or "advantageous."
    #         - It may also be inferred as a non-critical enhancement to a candidate's profile.
    #         3. Ensure the final list includes all relevant preferred qualifications.

    #         Ensure the output strictly adheres to the following JSON format:

    #         ```json
    #         {{
    #         "Experience": ["..."],
    #         "Frontend": ["..."],
    #         "Backend": ["..."],
    #         "DB": ["..."],
    #         "Tools": ["..."],
    #         "Miscellaneous": ["..."],
    #         "Location": ["..."],
    #         "Graduation": ["Bachelor's"/null],
    #         "Post Graduation": ["Master's"/null]
    #         }}
    #     """
    # template="""
    #     You are an AI assistant tasked with verifying and enhancing the list of **preferred qualifications** extracted from a job description. Your goal is to ensure the final list is accurate, comprehensive, and aligns with the job description.

    #     ### Job Description:
    #     {job_description}

    #     ### Existing Extracted Preferred Points:
    #     {extracted_preferred_points}

    #     ### Instructions:
    #     1. Cross-check the **Existing Extracted Preferred Points** against the job description.
    #     - Validate that each point aligns with qualifications explicitly stated or implied as "preferred," "nice-to-have," or "desirable."
    #     - Ensure that points match the exact wording or phrasing used in the job description.
    #     - Remove any points that are explicitly required or do not align with preferred qualifications.

    #     2. Identify any missing preferred qualifications:
    #     - A qualification is **preferred** if:
    #         - It is explicitly stated as "preferred," "nice-to-have," "desirable," or similar terminology.
    #         - It is implied as beneficial but not essential for the role, based on the job context.
    #     - Add any qualifications meeting these criteria to the final list.

    #     3. Organize the final list under the following table headers:
    #     - {table_header}
    #     - Include qualifications that directly fit under each header.
    #     - If no qualifications fit a header, omit it entirely from the final output.

    #     4. Present the final, complete list of preferred qualifications:
    #     - Use bullet points for each qualification.
    #     - Ensure the list is concise and comprehensive, covering all beneficial but non-essential points.

    #     """
    template="""
        You are an AI assistant tasked with verifying and enhancing the list of **preferred qualifications** extracted from a job description. Your goal is to ensure the final list is accurate, comprehensive, and aligns with the job description.

        ### Job Description:
        {job_description}

        ### Existing Extracted Preferred Points:
        {extracted_preferred_points}

        ### Instructions:
        1. **Validate the Existing Extracted Preferred Points**:
        - Confirm that each point in the list aligns explicitly with qualifications stated or implied as "preferred," "nice-to-have," "desirable," or similar terminology in the job description.
        - **Exclude any qualifications that are explicitly marked as "required," "essential," or implied as mandatory.**
        - Retain only those points that clearly fall under the "preferred" category based on the job description.

        2. **Identify Missing Preferred Qualifications**:
        - Look for qualifications described as "preferred," "nice-to-have," or "desirable," or qualifications implied as beneficial but not essential for the role.
        - **Do not include qualifications that are explicitly required or mandatory, even if they are implied to be beneficial.**
        - Add missing preferred qualifications that meet these criteria.

        3. **Organize the Final List Under the Following Table Headers**:
        - {table_header}
        - Assign qualifications to appropriate headers based on their relevance. 
        - Exclude headers with no applicable qualifications.

        4. **Output the Final List**:
        - Use bullet points under each table header.
        - Ensure the list is concise, comprehensive, and strictly focused on non-essential qualifications beneficial for the role.

        ### Additional Rules to Avoid Errors:
        - Use exact wording or phrasing from the job description wherever possible.
        - Double-check to ensure no explicitly required qualifications are included in the final preferred list.

        """

    prompt = PromptTemplate(template=template, input_variables=["job_description", "extracted_preferred_points", "table_header"])

    matching_points_llm = prompt | llm
    response = matching_points_llm.invoke({
        "job_description": job_description_file_content, 
        "extracted_preferred_points": state["preferred_messages"][-1].content, 
        "table_header": csv_headers
    })
    save_response_to_markdown(response.content, "./data/check_preferred_messages.md")

    # # print("response: ", response)
    return {"preferred_messages": [AIMessage(content=response.content)]}
    # return matching_points_llm
    

def generate_final_point_tool(state):
    """
    Use this tool to to generate final awsner.
    """

    print("generate_final_point_tool", state)

    # template="""
    #     You are an AI assistant tasked with creating an **optimized job description** for a specific role. The job description should be clear, concise, and tailored for querying via an SQL agent. Use the provided required and preferred points to ensure the description is comprehensive and well-structured.

    #     ### Input Data:
    #     - **Required Points**: {required_messages}
    #     - **Preferred Points**: {preferred_messages}

    #     ### Instructions:
    #     1. Create a professional and structured job description using the following sections:
    #     - **Job Title**: Include the role title.
    #     - **Overview**: Provide a brief introduction to the role, company, and objectives.
    #     - **Responsibilities**: List the key responsibilities of the role, ensuring alignment with the required and preferred points.
    #     - **Qualifications**:
    #         - **Required Qualifications**: List the required points in clear, bullet-point format.
    #         - **Preferred Qualifications**: List the preferred points in clear, bullet-point format.
    #     2. Ensure that:
    #     - Required qualifications are clearly marked as mandatory.
    #     - Preferred qualifications are marked as optional but desirable.
    #     - The wording is precise and avoids ambiguity to facilitate SQL agent queries.
    #     3. Use consistent and professional language suitable for job seekers.
    #     4. Return the result as a structured text output with clear headings and subheadings.
    #     5. Ensure the output is optimized for SQL agent use by maintaining logical and explicit formatting for each section.
    #     6. Focus on clarity and avoid overly verbose descriptions.

    #     ### Deliverable:
    #     Return the optimized job description as structured text. Do not include explanations or commentary.
        
    #     """

    # template="""
    #     You are tasked with creating a detailed job description based on the provided information. Use the following guidelines:

    #     Required Points: These are the mandatory qualifications, skills, and experience that are essential for the role. Please ensure that all items listed in the required_point section are clearly outlined as key qualifications.

    #     Preferred Points: These are the qualifications, skills, and experience that are desirable but not mandatory. Make sure to mention that these are additional qualifications that would give candidates an edge.

    #     Task: Based on the required_point and preferred_point, craft a professional job description. Make sure to clearly distinguish between what is required and what is preferred for potential candidates.

    #     Input:

    #     required_point = {required_messages}
    #     preferred_point = {preferred_messages}
        
    #     """
    # template="""
    #     Generate questions that guide a SQL agent to retrieve Required Points and Preferred Points based on the input context. The questions should focus on clarity, relevance, and adaptability to dynamic data.
    #     Input Parameters:

    #     Required Points: {required_messages}
    #     Preferred Points: {preferred_messages}
        
    #     """
    # template="""
    #     Generate a list of points based on the input context. Each point should be explicitly categorized as either Required or Preferred for clarity and ease of reference.

    #     Input Parameters:
    #     Required Points: {required_messages}
    #     Preferred Points: {preferred_messages}
        
    #     """
    template="""
            Purpose: Generate a list of points categorized explicitly as either "Required" or "Preferred" for clear distinction and reference.

            Instructions: Based on the provided input, clearly identify and categorize the points into "Required" and "Preferred" sections. The output should ensure proper alignment with SQL query requirements for accurate results.

            Input Parameters:

            Required Points: {required_messages}
            Preferred Points: {preferred_messages}
            Output Structure:

            Required:

            List all points under "Required" with exact names and descriptions as provided.
            Ensure that all required points are explicitly labeled and unambiguously stated.
            Preferred:

            List all points under "Preferred" separately.
            Clearly indicate that these points are optional but beneficial.
        """
        # Calculate the alignment score: Provide a score representing the percentage of alignment between the resume and job description.

    prompt = PromptTemplate(template=template, input_variables=["required_messages", "preferred_messages"])

    generateFinalAwsner = prompt | llm

    res = generateFinalAwsner.invoke({
        "required_messages": state["required_messages"][-1].content,
        "preferred_messages": state["preferred_messages"][-1].content
    })

    return {"messages": [res]}


def save_response_to_markdown(response, file_path):
    """
    Saves the AI response to a Markdown file.
    
    Args:
        response (str): The AI-generated response to save.
        file_path (str): The path where the Markdown file will be saved.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(response)
        print(f"Response successfully saved to {file_path}")
    except Exception as e:
        print(f"An error occurred while saving the response: {e}")

def jd_agent():

# Define a new graph
    workflow = StateGraph(State)

    workflow.add_node("find_required_point", find_required_point)
    workflow.add_node("check_required_point", check_required_point)
    workflow.add_node("find_preferred_point", find_preferred_point)
    workflow.add_node("check_preferred_point", check_preferred_point)
    workflow.add_node("generate_final_point", generate_final_point_tool)

    workflow.add_edge(START, "find_required_point")
    workflow.add_edge("find_required_point", "check_required_point")
    workflow.add_edge("check_required_point", "find_preferred_point")
    workflow.add_edge("find_preferred_point", "check_preferred_point")
    workflow.add_edge("check_preferred_point", "generate_final_point")
    # workflow.add_edge("generate_final_point", END)

    chain = workflow.compile()

    # Provide a valid input state

    create_image_func.create_graph_image(chain, "JDAGENT")

    job_description_file_content = get_file_content('outputRuleData.md', 'ruleData')

    input_state = {"messages": [job_description_file_content]}

    # Invoke the chain with the correct input
    response = chain.invoke(input_state)

    print("response", response["messages"][-1].content)
    file_name = "./data/ai_response.md"
    ai_response = response["messages"][-1].content
    save_response_to_markdown(ai_response, file_name)
    # save_response_to_markdown(response["required_messages"][-1].content, "./data/required_messages.md")
    # save_response_to_markdown(response["preferred_messages"][-1].content, "./data/preferred_messages.md")
    # print("response:", response)
    return response
