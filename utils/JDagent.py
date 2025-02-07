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
from langchain_core.output_parsers.json import JsonOutputParser


llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini")

# llm = ChatOpenAI(
#     api_key="sk-3f3074eaaa194d4c808bb90c3dedc257",  # Your API key
#     base_url="https://api.deepseek.com",  # Your custom API endpoint
#     model="deepseek-chat",  # The model you want to use
# )

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

def find_required_point(state):
    print("find_required_point", state)

    job_description_file_content = get_file_content('outputRuleData.md', 'ruleData')

    csv_headers = get_csv_headers(folder_path)

    # template="""
    #     Objective: Extract only the required information from the job description based on the provided table_header.

    #     Instructions:

    #     Read the job description carefully.

    #     Identify and extract only the mandatory or required details for each column in the table_header. Ignore any "preferred," "nice-to-have," or non-mandatory information.
        
    #     If found bachelor degree return only "Bachelor's"

    #     If a specific column has no required information, leave it blank or mark it as "N/A".

    #     Extract only the skill keywords from the following skills. Return them as a comma-separated list without any additional text.

    #     Always double check any information is not missing.

    #     Input:

    #     Table Header:
    #     {table_header}

    #     Job Description:
    #     {job_description}

    #     Output Format:
    #         Fill in the table with only the required information. Use "N/A" for missing or unspecified fields.
    #         json```{{
    #         "Required": {{
    #             "Location_required": "Extracted Required Location",
    #             "Experience_required": "Extracted Required Experience",
    #             "Graduation_required": "Extracted Required Graduation if found bachelor degree return only "Bachelor's" ",
    #             "Post_Graduation_required": "Extracted Required Post Graduation",
    #             "PhD_required": "Extracted Required PhD",
    #             "Skills_required": ["Extracted skills"]
    #         }}
    #         }}

    #     Explanation of Output:
    #         ###Required:
    #         json```{{
    #         "Required": {{
    #             "Location_required": "New York, NY",
    #             "Experience_required": "5+ years",
    #             "Graduation_required": "Bachelor's",
    #             "Post_Graduation_required": "N/A",
    #             "PhD_required": "N/A",
    #             "Skills_required": ["Manual Testing", "automation testing", "Selenium, TestNG"]
    #         }}
    #         }}

    # """
    template="""
        Objective: Extract only the required information from the job description based on the provided table_header.

        ### Instructions:

        1. **Read the job description carefully** and extract only the details explicitly marked as **mandatory** or **required**.  
        - Ignore any **preferred, nice-to-have, or non-mandatory** details.  

        2. **Extract Required Information for Each Column:**
        - Use the `table_header` to determine the required fields.
        - If a column does not contain required information, mark it as `"N/A"`.  

        3. **Degree Extraction:**
        - If a bachelor's degree is mentioned, **return only `"Bachelor's"`**.
        - Do **not** include alternative degree variations unless explicitly required.  

        4. **Skill Extraction:**
        - Extract **only the skill keywords** from the job description.  
        - Return them **as a comma-separated list** without any extra text.  
        - **Ensure no required skills are missing.**
        - **Example Output:** `["Python", "SQL", "Selenium", "TestNG"]`  

        5. **Final Validation:**
        - **Double-check** that all required details are included.  
        - Ensure that no mandatory field is missing.  

        ---

        ### **Input Format:**
        - **Table Header:**  
        ```{table_header}```  
        - **Job Description:**  
        ```{job_description}```  

        ---


        Output Format:
            json```{{
            "Required": {{
                "Location": ["Extracted Required Location"],
                "Experience": ["Extracted Required Experience In Number (don't add extra word )"],
                "Graduation": ["Extracted Required Graduation (if Bachelor's, return only 'Bachelor's')"],
                "Post_Graduation": ["Extracted Required Post Graduation"],
                "PhD": ["Extracted Required PhD"],
                "Skills": ["Extracted skills"]
                }}
            }}

        Explanation of Output:
            ###Required:
            json```{{
                "Required": {{
                "Location": ["New York, NY"],
                "Experience": ["5"],
                "Graduation": ["Bachelor's"],
                "Post_Graduation": ["N/A"],
                "PhD": ["N/A"],
                "Skills": ["Manual Testing", "Automation Testing", "Selenium", "TestNG"]
                }}
            }}

    """

    prompt = PromptTemplate(template=template, input_variables=["job_description", "table_header"])

    matching_points_llm = prompt | llm | JsonOutputParser()
    response = matching_points_llm.invoke({ "job_description": job_description_file_content, "table_header": csv_headers })
    # print("response: ", response)
    save_response_to_markdown(str(response), "./data/required_messages.md")

    return {"required_messages": [AIMessage(content=str(response))]}
    

def find_preferred_point(state):
    print("find_preferred_point", state)

    job_description_file_content = get_file_content('outputRuleData.md', 'ruleData')
    csv_headers = get_csv_headers(folder_path)

    # template="""  
    #     You are an AI assistant tasked with analyzing a job description to extract only **preferred** qualifications. Your goal is to determine which qualifications are desirable but not mandatory for the job, even if they are not explicitly labeled as "preferred." Use context and phrasing to make this determination.

    #     Instructions:  

    #     ### Instructions:
    #     - Read the job description carefully.  
    #     1. Extract only qualifications that are **preferred** for the job.
    #     - A qualification is **preferred** if:
    #         - It is explicitly stated as "preferred," "nice-to-have," "desired," or "optional."
    #         - It is implied as beneficial or advantageous but not essential for performing the job.
    #     - Ignore qualifications that are explicitly described as "required," "must have," "necessary," or "essential."
    #     - If a specific column has no preferred information, leave it blank or mark it as "N/A".
    #     - If the location is not marked as preferred, then it should not be considered as preferred.

    #     Input:  

    #     **Table Header:**  
    #     {table_header}  

    #     **Job Description:**  
    #     {job_description}  

    #     **Output Format:**  
    #         Fill in the table with only the preferred information. Use "N/A" for missing or unspecified fields.  

    #         ###Preferred:  
    #         ```json{{
    #         preferred:
    #         {{
    #             "Location_preferred": "Extracted Preferred Location",
    #             "Experience_preferred": "Extracted Preferred Experience",
    #             "Graduation_preferred": "Extracted Preferred Graduation",
    #             "Post_Graduation_preferred": "Extracted Preferred Post Graduation",
    #             "PhD_preferred": "Extracted Preferred PhD",
    #             "Skills_preferred": ["Extracted skills"]
    #         }}
    #     }}

    #     **Explanation of Output:**  
    #         ###Preferred: 
    #         ```json{{
    #         "preferred": {{
    #         "Location_preferred": "San Francisco, CA",
    #         "Experience_preferred": "3",
    #         "Graduation_preferred": "N/A",
    #         "Post_Graduation_preferred": "Master's",
    #         "PhD_preferred": "N/A"
    #         "Skills_preferred": ["Manual Testing", "API testing", "Postman"] 
    #         }}
    #         }}

    #     """

    # template="""  
    #     Objective: Extract only the preferred information from the job description based on the provided table_header.

    #     Instructions:  

    #     Read the job description carefully.  

    #     Identify and extract only the optional or preferred details for each column in the table_header. Ignore any "required," "must have," "necessary," "essential." or mandatory information.

    #     - If a specific column has no preferred information, leave it blank or mark it as "N/A".

    #     - If the location is not marked as preferred, then it should not be considered as preferred.

    #     Extract only the skill keywords from the following skills. Return them as a comma-separated list without any additional text.

    #     Always double check any information is not missing.

    #     Input:  

    #     **Table Header:**  
    #     {table_header}  

    #     **Job Description:**  
    #     {job_description}  

    #     **Output Format:**  
    #         Fill in the table with only the preferred information. Use "N/A" for missing or unspecified fields.  

    #         ###Preferred:  
    #         ```json{{
    #         preferred:
    #         {{
    #             "Location_preferred": "Extracted Preferred Location",
    #             "Experience_preferred": "Extracted Preferred Experience",
    #             "Graduation_preferred": "Extracted Preferred Graduation",
    #             "Post_Graduation_preferred": "Extracted Preferred Post Graduation",
    #             "PhD_preferred": "Extracted Preferred PhD",
    #             "Skills_preferred": ["Extracted skills"]
    #         }}
    #     }}

    #     **Explanation of Output:**  
    #         ###Preferred: 
    #         ```json{{
    #         "preferred": {{
    #         "Location_preferred": "San Francisco, CA",
    #         "Experience_preferred": "3",
    #         "Graduation_preferred": "N/A",
    #         "Post_Graduation_preferred": "Master's",
    #         "PhD_preferred": "N/A"
    #         "Skills_preferred": ["Manual Testing", "API testing", "Postman"] 
    #         }}
    #         }}

    #     """

    # template="""  
    #     Objective: Extract only the preferred (optional) information from the job description based on the provided table_header.

    #     Instructions:

    #     1. Read the job description carefully.
    #     2. Identify and extract **only** the optional or preferred details for each column in the table_header.
    #     - **Ignore** any information marked as "required," "must have," "necessary," or "essential."
    #     - If a specific column does not contain any preferred details, mark it as "N/A" or leave it blank.
    #     3. For the location field:
    #     - **Only** include location information if it is explicitly marked as preferred.
    #     4. Extract only the  optional or preferred skill keywords from the job description:
    #     - Return them as a **comma-separated list** (e.g., ["Manual Testing", "API testing", "Postman"]).
    #     - Remove any additional descriptive text; only extract the skill keywords.
    #     5. Always double-check that no preferred information is missing.

    #     Input:

    #     **Table Header:**
    #     {table_header}

    #     **Job Description:**
    #     {job_description}

    #     **Output Format:**  
    #         Fill in the table with only the preferred information. Use "N/A" for missing or unspecified fields.

    #         ###Preferred:  
    #         ```json{{
    #         preferred:
    #         {{
    #             "Location": ["Extracted Preferred Location"],
    #             "Experience": ["Extracted Preferred Experience"],
    #             "Graduation": ["Extracted Preferred Graduation"],
    #             "Post_Graduation": ["Extracted Preferred Post Graduation"],
    #             "PhD": ["Extracted Preferred PhD"],
    #             "Skills": ["Extracted Preferred skills"]
    #         }}
    #     }}

    #     **Explanation of Output:**  
    #         ###Preferred: 
    #         ```json{{
    #         "preferred": {{
    #         "Location": ["San Francisco, CA"],
    #         "Experience": ["3"],
    #         "Graduation": ["N/A"],
    #         "Post_Graduation": ["Master's"],
    #         "PhD": ["Phd"],
    #         "Skills": ["Manual Testing", "API testing", "Postman"] 
    #         }}
    #         }}

    #     """
    template="""  
         You are an AI assistant tasked with analyzing a job description to extract only **preferred** qualifications. Your goal is to determine which qualifications are desirable but not mandatory for the job, even if they are not explicitly labeled as "preferred." Use context and phrasing to make this determination.

        Instructions:

        1. Read the job description carefully.
        2. Identify and extract **only** the optional or preferred details for each column in the table_header.
        - **Ignore** any information marked as "required," "must have," "necessary," or "essential."
        - If a specific column does not contain any preferred details, mark it as "N/A" or leave it blank.
        3. For the location field:
        - **Only** include location information if it is explicitly marked as preferred.
        4. Extract only the  optional or preferred skill keywords from the job description:
        - Return them as a **comma-separated list** (e.g., ["Manual Testing", "API testing", "Postman"]).
        - Remove any additional descriptive text; only extract the skill keywords.
        5. Always double-check that no preferred information is missing.

        Input:

        **Table Header:**
        {table_header}

        **Job Description:**
        {job_description}

        **Output Format:**  
            Fill in the table with only the preferred information. Use "N/A" for missing or unspecified fields.

            ###Preferred:  
            ```json{{
            preferred:
            {{
                "Location": ["Extracted Preferred Location"],
                "Experience": ["Extracted Preferred Experience"],
                "Graduation": ["Extracted Preferred Graduation"],
                "Post_Graduation": ["Extracted Preferred Post Graduation"],
                "PhD": ["Extracted Preferred PhD"],
                "Skills": ["Extracted Preferred skills"]
            }}
        }}

        **Explanation of Output:**  
            ###Preferred: 
            ```json{{
            "preferred": {{
            "Location": ["San Francisco, CA"],
            "Experience": ["3"],
            "Graduation": ["N/A"],
            "Post_Graduation": ["Master's"],
            "PhD": ["Phd"],
            "Skills": ["Manual Testing", "API testing", "Postman"] 
            }}
            }}

        """
    prompt = PromptTemplate(template=template, input_variables=["job_description", "table_header"])

    matching_points_llm = prompt | llm | JsonOutputParser()
    response = matching_points_llm.invoke({"job_description": job_description_file_content, "table_header": csv_headers })
    save_response_to_markdown(str(response), "./data/preferred_messages.md")

    # print("response: ", response)
    return {"preferred_messages": [AIMessage(content=str(response))]}


import json

def write_to_json_file(data, filename="output.json"):
    """Writes a dictionary to a JSON file with ASCII encoding."""
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)  # Disable ASCII escaping
        print(f"Data successfully written to {filename}")
    except Exception as e:
        print(f"Error writing to JSON file: {e}")


def generate_final_point_tool(state):
    """
    Use this tool to to generate final awsner.
    """

    print("generate_final_point_tool", state)

    template="""
            Purpose: Generate a list of points categorized explicitly as either "Required" or "Preferred" for clear distinction and reference.

            Instructions: 
            Based on the provided input, clearly identify and categorize the points into "Required" and "Preferred" sections. The output should ensure proper alignment with SQL query requirements for accurate results.

            Exclude N/A, None or empty details in Location, Experience, Graduation, Post Graduation, PhD and Skills from output.
            
            Extract only the skill keywords from the following skills. Return them as a comma-separated list without any additional text.

            Input Parameters:

            Required Points: {required_messages}
            Preferred Points: {preferred_messages}

        Output Format:
           
            json```{{
                "Required": {{
                    "Location": ["Extracted Required Location"],
                    "Experience": ["Extracted Required Experience"],
                    "Graduation": ["Extracted Required Graduation. If found bachelor degree return only "Bachelor's""],
                    "Post_Graduation": ["Extracted Required Post Graduation"],
                    "PhD": ["Extracted Required PhD"],
                    "Skills": ["Extracted skills"]
                }}
                "preferred": {{
                    "Location": ["Extracted Preferred Location"],
                    "Experience": ["Extracted Preferred Experience"],
                    "Graduation": ["Extracted Preferred Graduation"],
                    "Post_Graduation": ["Extracted Preferred Post Graduation"],
                    "PhD": ["Extracted Preferred PhD"],
                    "Skills": ["Extracted skills"]
                }}
            }}

            Exclude Location, Experience, Graduation, Post Graduation, PhD and Skills from the output if they are N/A, None, or empty

            Clearly indicate that these points are optional but beneficial.
        """
        # Calculate the alignment score: Provide a score representing the percentage of alignment between the resume and job description.

    prompt = PromptTemplate(template=template, input_variables=["required_messages", "preferred_messages"])

    generateFinalAwsner = prompt | llm | JsonOutputParser()

    res = generateFinalAwsner.invoke({
        "required_messages": state["required_messages"][-1].content,
        "preferred_messages": state["preferred_messages"][-1].content
    })

    # print("res", res)
    # print("restype", type(res))

    file_name = "./data/ai_response.json"

    write_to_json_file(res, file_name)

    return {"messages": [str(res)]}


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

    # workflow.add_node("find_required_point", find_required_point)
    # workflow.add_node("check_required_point", check_required_point)
    # workflow.add_node("find_preferred_point", find_preferred_point)
    # workflow.add_node("check_preferred_point", check_preferred_point)
    # workflow.add_node("generate_final_point", generate_final_point_tool)

    # workflow.add_edge(START, "find_required_point")
    # workflow.add_edge("find_required_point", "check_required_point")
    # workflow.add_edge("check_required_point", "find_preferred_point")
    # workflow.add_edge("find_preferred_point", "check_preferred_point")
    # workflow.add_edge("check_preferred_point", "generate_final_point")
    # workflow.add_edge("generate_final_point", END)

    workflow.add_node("find_required_point", find_required_point)
    # workflow.add_node("check_required_point", check_required_point)
    workflow.add_node("find_preferred_point", find_preferred_point)
    # workflow.add_node("check_preferred_point", check_preferred_point)
    workflow.add_node("generate_final_point", generate_final_point_tool)

    workflow.add_edge(START, "find_required_point")
    workflow.add_edge("find_required_point", "find_preferred_point")
    # workflow.add_edge("check_required_point", "find_preferred_point")
    workflow.add_edge("find_preferred_point", "generate_final_point")
    # workflow.add_edge("check_preferred_point", "generate_final_point")
    workflow.add_edge("generate_final_point", END)

    chain = workflow.compile()

    # Provide a valid input state

    # create_image_func.create_graph_image(chain, "JDAGENT")

    job_description_file_content = get_file_content('outputRuleData.md', 'ruleData')

    input_state = {"messages": [job_description_file_content]}

    # Invoke the chain with the correct input
    response = chain.invoke(input_state)

    print("response", response["messages"][-1].content)
    file_name = "./data/ai_response.md"
    ai_response = response["messages"][-1].content
    save_response_to_markdown(str(ai_response), file_name)
    # save_response_to_markdown(response["required_messages"][-1].content, "./data/required_messages.md")
    # save_response_to_markdown(response["preferred_messages"][-1].content, "./data/preferred_messages.md")
    # print("response:", response)
    return response
