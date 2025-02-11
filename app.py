from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(override=True)
import streamlit as st
import os
from typing import Annotated, List
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from langchain_openai.chat_models import ChatOpenAI
from typing_extensions import TypedDict
import sql.sql_agent_team_supervisor as sql_agent_team_supervisor
import github.github_team_supervisor as github_team_supervisor
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,FewShotChatMessagePromptTemplate
import utils.csv_to_sql as csv_to_sql
import utils.create_image_func as create_image_func
import utils.create_team_supervisor_func as create_team_supervisor_func
from langchain_core.output_parsers.json import JsonOutputParser
# import utils.upload_job_description as upload_job_description
import utils.JDagent as upload_job_description
import utils.retreive_users as retreive_users
from langgraph.errors import GraphRecursionError
import utils.display_uploaded_files as display_uploaded_files
import utils.upload_csv as upload_csv
import utils.calculate_user_percentage as calculate_user_percentage
import utils.create_boolean_query as create_boolean_query
import utils.save_to_markdown as save_to_markdown
import utils.knowledge_base as knowledge_base
import time
import re
import json
import pandas as pd
import io
import time
import random
import uuid

# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_tracking_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_project = os.getenv("LANGCHAIN_PROJECT")
# llamaparse_api_key = "llx-8MMHGFCJ5PKqyfZM6h5D8epMtjzG4OEOe6lMCEOvgu67YgIt"

# Set environment variables if needed
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracking_v2
os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint
os.environ["LANGCHAIN_PROJECT"] = langchain_project
# os.environ["LLAMAPARSE_API_KEY"] = llamaparse_api_key

### Statefully manage chat history ###
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Define the agent node function
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

# Initialize the agents
github_chain = github_team_supervisor.github_team_supervisor(agent_node)
sql_chain = sql_agent_team_supervisor.sql_agent_team_supervisor()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Streamlit UI
st.title("SkillSync by DataCurve")
# st.title("Intelligent Recruitment Assistant")

def retrive():
    pass

# Define folder paths
csv_folder = "csv"

# Ensure the csv and db directories exist
os.makedirs(csv_folder, exist_ok=True)

st.sidebar.image("./image/DataCurvelogo.png", width=180, output_format="auto")

view = st.sidebar.selectbox(
    "View",
    ("User", "Admin"),
    0
)

# def load_markdown(path):
#     with open(path, "r",encoding="utf-8") as default_file:
#         default_text = default_file.read().strip()
#         return default_text

# @st.dialog("Boolean Query")
# def booleanQuery():
#     jd_path = "./data/summarizeOutputRuleData.md"

#     try:
#         # Read the template content
#         jd = load_markdown(jd_path)
#         st.write("Loaded template:", jd)  # Debug print to confirm content loading

#         # Start form
#         with st.form(key="prompt_form"):
#             # Display the template content in a text area
#     except Exception as e:
#         st.error(f"An error occurred: {str(e)}")

buttonVal = False   
agent_name = None

if(view=="User"):
    agent_name = st.sidebar.radio(
        "Get resource",
        ["ATS", "Github"]
    )
    agent_name2 = st.sidebar.radio(label="( Upcoming )",options=["LinkedIn","Reddit","CareerBuilder","Monster","Stack Overflow"],disabled=True,index=None)

    st.sidebar.button("Connect With APIs")

    # File uploader widget
    with st.sidebar.form("jd_pdf_upload_form", clear_on_submit=True):
        uploaded_checking_rule_file = st.file_uploader(
            "Upload your Job Description", type=["pdf"], key="pdf_uploader"
        )
        file_submitted = st.form_submit_button("Submit")
    if file_submitted and (uploaded_checking_rule_file is not None):
        container = st.empty()
        container.write("Processing the uploaded file...")
        upload_job_description.upload_job_description(uploaded_checking_rule_file,container)
        time.sleep(2)
        container.empty()
        st.rerun()


    display_uploaded_files.display_uploaded_files("1","./pdf",".pdf")

    buttonVal = st.sidebar.button(
        "Retrieve Users",
        on_click=retrive,  # Note the lack of parentheses here
        key="retreive_users",
    )

    boolean = st.sidebar.button(
        "Create boolean query",
        on_click=create_boolean_query.booleanQuery,  # Note the lack of parentheses here
        key="boolean",
    )

def read_prompt(custom_prompt_path, default_prompt_path):

    try:
        # Check if customprompt.md file has text
        with open(custom_prompt_path, "r",encoding="utf-8") as custom_file:
            custom_text = custom_file.read().strip()

            if custom_text:
                return custom_text
            else:
                # If customprompt.md is empty, read from defaultprompt.md
                with open(default_prompt_path, "r",encoding="utf-8") as default_file:
                    default_text = default_file.read().strip()
                    return default_text
    except FileNotFoundError as e:
        # If customprompt.md doesn't exist, read from defaultprompt.md
        with open(default_prompt_path, "r",encoding="utf-8") as default_file:
            default_text = default_file.read().strip()
            return default_text

# Function to clear the custom prompt file
def clear_custom_prompt(custom_path):
    with open(custom_path, "w") as file:
        file.write("")  # Clear the file by writing an empty string

# Function to update the default prompt file with the latest changes
def update_default_prompt(default_path, content):
    try:
        # Open the file with 'utf-8' encoding to handle special characters
        with open(default_path, "w", encoding="utf-8") as file:
            file.write(content)  # Write the latest content to the file
    except Exception as e:
        st.error(f"An error occurred while updating the file: {str(e)}")

@st.dialog("Prompt")
def jsonFilterPrompt():
    custom_prompt_path = "./filterPrompt/customPrompt.md"
    default_prompt_path = "./filterPrompt/defaultPrompt.md"

    try:
        # Read the template content
        template = read_prompt(custom_prompt_path, default_prompt_path)
        # st.write("Loaded template:", template)  # Debug print to confirm content loading

        # Start form
        with st.form(key="prompt_form"):
            # Display the template content in a text area
            updated_content = st.text_area("Template Content", template, height=200, key="template_text_area")

            # Create columns to place Submit and Reset buttons on the same line
            col1, col2 = st.columns([0.3, 1])

            # Place buttons in the respective columns
            with col1:
                submit_button = st.form_submit_button("Submit")
            with col2:
                reset_button = st.form_submit_button("Reset")

            # Check which button was clicked
            if submit_button:
                update_default_prompt(custom_prompt_path, updated_content)
                st.success("Template content updated successfully!")
                st.rerun()  # Close modal by reloading the app


            if reset_button:
                clear_custom_prompt(custom_prompt_path)
                st.success("Custom prompt file cleared!")
                st.rerun()  # Close modal by reloading the app


    except Exception as e:
        st.error(f"An error occurred: {str(e)}")



if(view=="Admin"):
    with st.sidebar.form("csv_upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"],key="csv_uploader")
        file_submitted = st.form_submit_button("Submit")

    if file_submitted and (uploaded_file is not None):
        container = st.empty()
        container.write("Processing the uploaded file...")
        upload_csv.upload_csv(uploaded_file,container)
        time.sleep(2)
        container.empty()

    display_uploaded_files.display_uploaded_files("2","./csv",".csv")

    with st.sidebar.form("knowledge_base", clear_on_submit=True):
        uploaded_file_knowledge_base = st.file_uploader("Upload knowledge base CSV File", type=["csv"], key="csv_uploader_knowledge_base")
        file_submitted_knowledge_base = st.form_submit_button("Submit")

    if file_submitted_knowledge_base and (uploaded_file_knowledge_base is not None):
        container = st.empty()
        container.write("Processing the uploaded file...")
        knowledge_base.upload_csv(uploaded_file_knowledge_base,container)
        time.sleep(2)
        container.empty()

    display_uploaded_files.display_uploaded_files("3","./knowledge_base_csv",".csv")

    filterButton = st.sidebar.button(
        "Fliter Prompt",
        on_click=jsonFilterPrompt,  # Note the lack of parentheses here
        key="Prompt",
    )

def get_agent_name(agent_name_here):
    if(agent_name_here=="ATS"):
        return "SQLTeam Agent"
    else:
        return "GithubTeam Agent"

# Filter for messages from SQLTeam Agent
agent_name_to_filter = get_agent_name(agent_name)  # Adjust this parameter as needed


# examples = [
#     {
#         "question": """For a 'Software Engineer' position located in Los Angeles, does the candidate meet these criteria:   
#         - 2-4 years of experience in software development (required)
#         - Bachelor’s degree in Computer Science (required)
#         - Proficiency in JavaScript (required)
#         - Strong understanding of Git (required)
#         - Master’s degree (preferred)
#         - PhD (preferred)
#         - Experience with cloud platforms such as AWS (preferred)
#         - Knowledge of Agile methodologies (preferred)""",
#         "answer": """
#         {"Required": {"Experience": [2], "Skills": ["JavaScript", "Git"], "Graduation": ["Bachelor's"]}, "Preferred": {"Post Graduation": ["Master's"], "PhD": ["PhD"], "Skills":["AWS", "Agile methodologies"]}}
#         """,
#     },
#     {
#         "question": """For a 'Software Engineer' position located in Austin, does the candidate meet these criteria:   
#         - 3 years of experience in software development (required)
#         - Bachelor’s degree in Computer Science (required)
#         - Proficiency in Python (required)
#         - Proficiency in SQL (required)
#         - Proficiency in Hadoop (required)
#         - Master’s degree (preferred)
#         - Knowledge of Agile methodologies (preferred)""",
#         "answer": """
#         {"Required": {"Experience": [3], "Skills": ["Python","SQL","Hadoop"], "Graduation": ["Bachelor's"]}, "Preferred": {"Post Graduation": ["Master's"], "Skills": ["Agile methodologies"]}}
#         """,
#     }
# ]

# prompt_template = """
# You are given a job description with specific required and preferred qualifications, along with a table of headers. 
# Your task is to extract and categorize the qualifications as either "Required" or "Preferred", using the table headers as a guide. 
# Ensure that no required fields from the job description are missed. 
# The output should be a dictionary with two keys: "Required" and "Preferred" 
# Under each key, list the relevant headers mentioned in the job description.

# Job Description:
# {job_description}

# Table Headers:
# {table}

# # Instructions:
# 1. Extract the qualifications from the job description.
# 2. Categorize them according to the table headers.
# 3. List "Preferred" qualifications under "Preferred" and all others under "Required."
# 4. If a qualification matches a value in the table rows, use the exact spelling from the table. Otherwise, use the spelling as found in the job description.
# 5. Ensure numeric values are presented as numbers only, without additional strings.
# 6. Ensure each section is clearly labeled, ordered, and separated by commas.
# 7. Experience column always has 1 point , ignore all others.
# 8. ENSURE that if any key contains multiple values separated by commas, they are always placed in a list. ALWAYS enforce this structure, and NEVER overlook this step.
# 9. ENSURE that the format strictly follows the provided example. Every field must exactly match the structure and format of the example, without exceptions.
# Output: 
# Ensure all table headers are addressed in the output.
# Only return dictionary nothing else.

# # Nerver forgot output format and must be dictionary:
# Output format: 
# dictionary: "{{"Required": {{}},"Preferred": {{}}}}"
# """

# Create a prompt with the correct input variable
# matchPrompt = PromptTemplate(template=prompt_template, input_variables=["job_description", "table"])

# matchPrompt = ChatPromptTemplate(messages=[
#     ("system","""
# You are given a job description with specific required and preferred qualifications, along with a table of headers. 
# Your task is to extract and categorize the qualifications as either "Required" or "Preferred", using the table headers as a guide. 
# Ensure that no required fields from the job description are missed. 
# The output should be a dictionary with two keys: "Required" and "Preferred" 
# Under each key, list the relevant headers mentioned in the job description.

# Job Description:
# {job_description}

# Table Headers:
# {table}

# # Instructions:
# 1. Extract the qualifications from the job description.
# 2. Categorize them according to the table headers.
# 3. List "Preferred" qualifications under "Preferred" and all others under "Required."
# 4. If a qualification matches a value in the table rows, use the exact spelling from the table. Otherwise, use the spelling as found in the job description.
# 5. Ensure numeric values are presented as numbers only, without additional strings.
# 6. Ensure each section is clearly labeled, ordered, and separated by commas.
# 7. ENSURE that if any key contains multiple values separated by commas, they are always placed in a list. ALWAYS enforce this structure, and NEVER overlook this step.
# 8. ENSURE that the 'Experience' field is always a list with exactly one element. If there are multiple elements in the list, keep only the first one and ignore the rest.
# 9. PhD should only appear under the PhD key and NOT under Post Graduation. Deduplicate "PhD" from the Post Graduation list if it appears there.

# Output: 
# Ensure all table headers are addressed in the output.
# Only return dictionary nothing else.

# # Nerver forgot output format and must be dictionary:
# Output format: 
# dictionary: "{{"Required": {{}},"Preferred": {{}}}}"
# """)
# ],input_variables=["job_description","table"])

# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=matchPrompt,
#     examples=examples,
# )

# matchChain = matchPrompt | llm | JsonOutputParser()

def extract_table_from_text(text):
    # print("***************************extract_table_from_text*********************************")
    # print("text :", text)
    
    # Updated regular expression to include the last row
    table_match = re.search(r'\|.*\n(\|.*\n)*\|.*', text)
    # print("table_match :", table_match)
    
    if not table_match:
        return []
    
    table_text = table_match.group(0)
    # print("table_text :", table_text)
    
    # Split the extracted table text into lines
    lines = table_text.strip().split('\n')
    # print("lines :", lines)
    
    # Extract the header and rows
    header = [col.strip() for col in lines[0].split('|') if col.strip()]
    # print("header :", header)
    
    rows = []
    for line in lines[2:]:  # Skip the separator line
        # Skip any separator lines (those with "---")
        if '---' in line:
            continue
        
        row = [col.strip() for col in line.split('|') if col.strip()]
        rows.append(row)
    
    # print("rows :", rows)
    
    # Combine header and rows into a table (list of lists)
    table = [header] + rows
    # print("final table to be returned:", table)
    
    return table

def extract_json_from_text(text):
    # Use a regular expression to find the JSON object in the text
    match = re.search(r'^\{.*\}$', text, re.DOTALL)
    
    if match:
        json_string = match.group(0)
        try:
            # Load the JSON string into a Python dictionary
            json_data = json.loads(json_string)
            return json_data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None
    else:
        print("No JSON object found at the beginning of the text.")
        return None
    
def update_table_in_res(res, user_data):
    """
    Update the table in the res string with the new user data and dynamic headers from the first row.
    Ensure all rows have the same length for consistent formatting.

    Parameters:
    - res (str): The string containing the old table.
    - user_data (list): The new table data, including headers and rows.

    Returns:
    - str: The updated string with the new table.
    """
    # Extract headers from the first row of user_data
    headers = user_data[0]
    
    # Calculate maximum column widths based on header and row data
    col_widths = [max(len(str(header)), max(len(str(item)) for item in column)) for header, column in zip(headers, zip(*user_data))]
    
    # Format headers and separator
    header_row = '| ' + ' | '.join([f'{header:<{col_widths[i]}}' for i, header in enumerate(headers)]) + ' |'
    separator_row = '| ' + ' | '.join(['-' * width for width in col_widths]) + ' |'
    
    # Format rows
    def format_row(row):
        return '| ' + ' | '.join([f'{str(cell):<{col_widths[i]}}' for i, cell in enumerate(row)]) + ' |'

    # Format user data rows
    formatted_rows = [format_row(row) for row in user_data[1:]]
    
    # Combine header, separator, and rows
    new_table_str = '\n'.join([header_row, separator_row] + formatted_rows)
    
    # Regex pattern to find the first table in the res string
    pattern = re.compile(
        r'\|.*?\|\n\|.*?\n(?:\|.*?\n)*',
        re.DOTALL
    )
    
    # Replace the first table with the new table
    updated_res = pattern.sub(
        f'{new_table_str}\n', res
    )
    
    return updated_res

def extract_required_preferred_fields(tableText):
    table = extract_table_from_text(tableText)
    # print("table", table)
    
    if len(table) > 0:
        
        # Create a DataFrame from the table for export
        df = pd.DataFrame(table[1:], columns=table[0])  # Skip header for data
        
        # Display 'Export' button and allow CSV download
        csv = df.to_csv(index=False)
        # Generate a unique key using current Unix epoch time
        buf = io.StringIO(csv)  # Use StringIO to handle CSV in memory
        # print("CSV :",csv)
        # print("buf :",buf)
        return buf 

    else:
        # If the table is empty, return the original table content
        print("Table is empty, returning the original content.")
        return tableText

def filter_names(table):
    # The first row is the header, so we skip it and extract the 'Name' column
    # We assume that the 'Name' column is the first one
    names = [row[0] for row in table[1:]]  # Skipping the header
    return names

# Send Emails Dialog Box
@st.dialog("Send Emails")
def send_emails(table):
    # print("table aaya :",table)
    options=filter_names(table)
    # print("options :",options)
    try:
        with st.form(key=str(uuid.uuid4()), clear_on_submit=True):
            multiselect_send_email = st.multiselect(
                "Select users to send emails", options=options,key=str(uuid.uuid4())
            )
            form_submitted = st.form_submit_button("Submit")
    except Exception as e:
        st.error(f"An error occurred while reading the PDF: {str(e)}")

# Conversation History
if 'chat_history' in st.session_state:
    for index, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage) and message.name == agent_name_to_filter:
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage) and message.name == agent_name_to_filter:
            with st.chat_message("AI"):
                # print("**********************************************************************")
                # print("message.content in chat history :",message.content)
                table = extract_table_from_text(message.content)
                # print("table in chat history :",table)
                if len(table) > 0:
                    st.markdown(message.content)
                    buf = extract_required_preferred_fields(message.content)
                    unique_file_name = f"{int(time.time())}.csv"
                    col1,col2 = st.columns([0.2,0.8])
                    with col1:
                        st.download_button(
                            label="Export as CSV",
                            data=buf.getvalue(),
                            file_name=unique_file_name,
                            mime='text/csv',
                            key=uuid.uuid4()
                        )
                    with col2:
                        st.button("Send Email", key=uuid.uuid4(),on_click=send_emails, args=(table,))
                else:
                    st.markdown(message.content)

# Check for existence of table in a text
# def checkForTable(tableText,question):
#     table = extract_table_from_text(tableText)
        
#     if len(table) > 0:
#         # print("JOB DESCRIPTION :",question)
#         # print("TABLE :",table)

#         matchChainResponse = matchChain.invoke({"job_description": question, "table": table})
#         # print("matchChainResponse :",matchChainResponse)
#         # Convert the string to a dictionary
#         # dict_obj = json.loads(matchChainResponse)
#         myres = calculate_user_percentage.calculate_user_percentage(table,matchChainResponse)
#         # print("myres :",myres)
#         updatedRes = update_table_in_res(tableText,myres)
#         # print("updatedRes :",updatedRes)
#         st.session_state.chat_history.append(AIMessage(content=updatedRes, name=get_agent_name(agent_name)))

#         unique_file_name = f"{int(time.time())}.csv"
#         buf = extract_required_preferred_fields(updatedRes)
#         if 'chat_history' in st.session_state:
#             # Check if chat_history is not empty
#             if st.session_state.chat_history:
#                 # Get the last message in the chat_history
#                 last_message = st.session_state.chat_history[-1]
                
#                 # Check if the last message is an AIMessage and filter by agent name
#                 if isinstance(last_message, AIMessage) and last_message.name == agent_name_to_filter:
#                     with st.chat_message("AI"):
#                         st.markdown(last_message.content)
#                         # st.button("export", key=random.randint(1, 10000))
#                         col1,col2 = st.columns([0.2,0.8])
#                         with col1:
#                             st.download_button(
#                                 label="Export as CSV",
#                                 data=buf.getvalue(),
#                                 file_name=unique_file_name,
#                                 mime='text/csv',
#                                 key=uuid.uuid4()
#                             )
#                         with col2:
#                             st.button("Send Email", key=uuid.uuid4(),on_click=send_emails, args=(table,))

#     else:
#         if 'chat_history' in st.session_state:
#             # Check if chat_history is not empty
#             if st.session_state.chat_history:
#                 # Get the last message in the chat_history
#                 last_message = st.session_state.chat_history[-1]
                
#                 # Check if the last message is an AIMessage and filter by agent name
#                 if isinstance(last_message, AIMessage) and last_message.name == agent_name_to_filter:
#                     with st.chat_message("AI"):
#                         st.markdown(last_message.content)

# Check for existence of table in a text
# def checkForTable1(tableText):
#     table = extract_table_from_text(tableText)
        
#     if len(table) > 0:
#         unique_file_name = f"{int(time.time())}.csv"
#         buf = extract_required_preferred_fields(tableText)
#         if 'chat_history' in st.session_state:
#             # Check if chat_history is not empty
#             if st.session_state.chat_history:
#                 # Get the last message in the chat_history
#                 last_message = st.session_state.chat_history[-1]
                
#                 # Check if the last message is an AIMessage and filter by agent name
#                 if isinstance(last_message, AIMessage) and last_message.name == agent_name_to_filter:
#                     with st.chat_message("AI"):
#                         st.markdown(last_message.content)
#                         # st.button("export", key=random.randint(1, 10000))
#                         col1,col2 = st.columns([0.2,0.8])
#                         with col1:
#                             st.download_button(
#                                 label="Export as CSV",
#                                 data=buf.getvalue(),
#                                 file_name=unique_file_name,
#                                 mime='text/csv',
#                                 key=uuid.uuid4()
#                             )
#                         with col2:
#                             st.button("Send Email", key=uuid.uuid4(),on_click=send_emails, args=(table,))

#     else:
#         if 'chat_history' in st.session_state:
#             # Check if chat_history is not empty
#             if st.session_state.chat_history:
#                 # Get the last message in the chat_history
#                 last_message = st.session_state.chat_history[-1]
                
#                 # Check if the last message is an AIMessage and filter by agent name
#                 if isinstance(last_message, AIMessage) and last_message.name == agent_name_to_filter:
#                     with st.chat_message("AI"):
#                         st.markdown(last_message.content)

# Define the path to your JSON configuration file
CONFIG_PATH = os.path.join('knowledge_base_json', 'knowledge_base.json')  # Adjust the path as needed

# @st.dialog("Query Filters")
# def query_filters_modal(matchChainResponse=None):
#     try:
#         # Load the JSON configuration
#         print("matchChainResponse :",matchChainResponse)
#         with open(CONFIG_PATH, 'r') as f:
#             form_config = json.load(f)

#         with st.form(key=str(uuid.uuid4()), clear_on_submit=True):
#             user_inputs = {}  # Dictionary to store user selections

#             # Create a multiselect for each key in the JSON file
#             for field_name, options in form_config.items():
#                 print("options :",options)
#                 print("field_name :",field_name)
#                 # if(type(matchChainResponse.get(field_name, []))==bool):
#                 #     default_values=[options[0]]
#                 # Check if the field is Graduation or Post Graduation
#                 if field_name in ["Graduation", "Post Graduation"]:
#                     # Set default to the first option if the value in matchChainResponse is True
#                     default_values = [options[0]] if matchChainResponse.get(field_name, False) is True else []

#                 else:
#                     default_values = [
#                         str(value) for value in matchChainResponse.get(field_name, [])
#                         if str(value) in options
#                     ]

#                 print("default_values :",default_values)

#                 user_inputs[field_name] = st.multiselect(
#                     label=field_name,
#                     options=options,
#                     default=default_values,
#                     key=str(uuid.uuid4())
#                 )

#             form_submitted = st.form_submit_button("Apply")

#             if form_submitted:
#                 st.success("Filters applied successfully!")
#                 st.write("Selected Filters:", user_inputs)

#     except FileNotFoundError:
#         st.error(f"Configuration file not found at path: {CONFIG_PATH}")
#     except json.JSONDecodeError:
#         st.error("Error decoding the JSON configuration file. Please check the file format.")
#     except Exception as e:
#         st.error(f"An error occurred while opening the Query Filters modal: {str(e)}")

def match_skills(ai_skills, options):
    # print("ai_skills", ai_skills)
    # print("options", options)
    # Convert both lists to lowercase for comparison
    ai_skills_lower = [str(skill).lower() for skill in ai_skills]
    options_lower = [option.lower() for option in options]

    # Find matches and return them in their original form
    matched_skills = [options[i] for i, option in enumerate(options_lower) if option in ai_skills_lower]
    return matched_skills


# promptForGenerateQuestion = ChatPromptTemplate(messages=[
#     ("system", """
#     As a recruiter using an AI SQL agent to search a candidate database, generate a concise question that can help retrieve candidates who meet the following criteria:

#     - Criteria: {parameter}

#     Phrase the question in a way that it’s both concise and can be easily translated into a SQL query by the AI agent, focusing on each specified criterion. The question should prompt the candidate (or filter results) to confirm or specify relevant details about each requirement.
#     """)
# ], input_variables=["parameter"])



# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=matchPrompt,
#     examples=examples,
# )

# generate_question_AI = promptForGenerateQuestion | llm 

import difflib
import re

def merge_requirements(requirements):
    merged_result = {
        'Location': [],
        'Skills': []
    }
    
    required = requirements.get('Required', {})
    preferred = requirements.get('preferred', {})

    # Merge location: prioritize required, fallback to preferred
    required_location = required.get('Location', [])
    preferred_location = preferred.get('Location', [])
    merged_result['Location'] = required_location if required_location else preferred_location
    
    # Merge skills: combine required and preferred, deduplicate
    required_skills = required.get('Skills', [])
    preferred_skills = preferred.get('Skills', [])
    merged_result['Skills'] = list(set(required_skills + preferred_skills))

    # Add other fields from 'Required' as lists (e.g., Experience, Graduation)
    for key in required:
        if key not in ['Location', 'Skills']:
            # lowercase_key = key.lower()
            merged_result[key] = required[key]  # Directly assign the list
    
    return merged_result

import json


# from difflib import get_close_matches

# def find_similar_matches(search_filter, database, similarity_threshold=0.6):
#     results = {}
    
#     for filter_key, filter_value in search_filter.items():
#         # Find similar keys in database
#         db_key_match = get_close_matches(
#             filter_key, 
#             database.keys(), 
#             n=1, 
#             cutoff=similarity_threshold
#         )
        
#         if not db_key_match:
#             continue  # Skip if no similar key found
            
#         db_key = db_key_match[0]
#         db_values = database[db_key]
        
#         # Handle different value types
#         if isinstance(filter_value, list):
#             value_matches = {}
#             for item in filter_value:
#                 matches = get_close_matches(
#                     item.lower().strip(),
#                     [v.lower().strip() for v in db_values],
#                     n=1,
#                     cutoff=similarity_threshold
#                 )
#                 if matches:
#                     value_matches[item] = matches[0]
#             if value_matches:
#                 results[filter_key] = {
#                     'database_key': db_key,
#                     'matches': value_matches
#                 }
                
#         elif isinstance(filter_value, str):
#             # Handle numerical ranges (e.g., "2+ years")
#             if db_key == 'Experience' and '+' in filter_value:
#                 min_years = int(filter_value.split('+')[0])
#                 numeric_values = [int(x) for x in db_values]
#                 matches = [str(x) for x in numeric_values if x >= min_years]
#                 if matches:
#                     results[filter_key] = {
#                         'database_key': db_key,
#                         'matches': matches
#                     }
#             else:
#                 # Handle string values
#                 matches = get_close_matches(
#                     filter_value.lower().strip(),
#                     [v.lower().strip() for v in db_values],
#                     n=1,
#                     cutoff=similarity_threshold
#                 )
#                 if matches:
#                     results[filter_key] = {
#                         'database_key': db_key,
#                         'matches': {filter_value: matches[0]}
#                     }

#     return results


from difflib import get_close_matches

def find_similar_matches(new_search, database, similarity_threshold=0.8, n=1):
    results = {
        "Required": {},
        "preferred": {}
    }

    # Process Required section
    for section in ['Required', 'preferred']:
        if section not in new_search:
            continue
            
        for filter_key, filter_value in new_search[section].items():
            # Find similar keys in database
            db_key_match = get_close_matches(
                filter_key.lower(),
                [k.lower() for k in database.keys()],
                n,
                cutoff=similarity_threshold
            )
            
            if not db_key_match:
                continue  # Skip if no similar key found
                
            # Get actual database key name
            db_key = [k for k in database.keys() if k.lower() == db_key_match[0]][0]
            db_values = database[db_key]
            
            # Handle different value types
            if isinstance(filter_value, list):
                value_matches = []
                for item in filter_value:
                    matches = get_close_matches(
                        item.strip(),
                        [v.strip() for v in db_values],
                        n,
                        cutoff=similarity_threshold
                    )
                    # print("matches", matches)
                    if matches:
                        value_matches.append(matches[0])

                # print("value_matches", value_matches)
                if value_matches:
                    results[section][filter_key] = value_matches
                    
                    
            elif isinstance(filter_value, str):
                # Handle numerical experience comparison
                if db_key.lower() == 'experience':
                    try:
                        req_experience = int(filter_value)
                        numeric_values = [int(v) for v in db_values]
                        matches = [str(x) for x in numeric_values if x >= req_experience]
                        if matches:
                            results[section][filter_key] =  matches
                            
                    except ValueError:
                        pass
                else:
                    # Handle string values
                    matches = get_close_matches(
                        filter_value.lower().strip(),
                        [v.lower().strip() for v in db_values],
                        n,
                        cutoff=similarity_threshold
                    )
                    if matches:
                        results[section][filter_key] = [matches[0]]
                        

    return results

# result = find_similar_matches(search_filter, database)

# # Pretty print results
# print("Key/Value Similarity Results:")
# for filter_key, match_data in result.items():
#     print(f"\nFilter Key: {filter_key}")
#     print(f"Mapped to Database Key: {match_data['database_key']}")
    
#     if isinstance(match_data['matches'], dict):
#         print("Value Matches:")
#         for filter_val, db_val in match_data['matches'].items():
#             print(f"  '{filter_val}' → '{db_val}'")
#     else:
#         print(f"Value Matches: {match_data['matches']}")


import copy

def update_required_dict(result, user_inputs):
    """
    Update the 'Required' section of the result dictionary based on user_inputs.
    
    For each key in user_inputs with a non-empty value:
      - If the key exists in result['Required'] and its value differs (ignoring order for lists),
        update it.
      - If the key does not exist, add it.
    
    Returns:
        tuple: (result, result_updated) where result is the updated dictionary and
               result_updated is True if any changes were made.
    """
    result_updated = False
    # Ensure the "Required" key exists.
    required = result.setdefault("Required", {})

    for key, new_value in user_inputs.items():
        if new_value:  # Only process non-empty values.
            if key in required:
                current_value = required[key]
                # For list values, compare as sets (ignoring order).
                if isinstance(current_value, list) and isinstance(new_value, list):
                    if set(current_value) != set(new_value):
                        required[key] = new_value
                        result_updated = True
                else:
                    if current_value != new_value:
                        required[key] = new_value
                        result_updated = True
            else:
                required[key] = new_value
                result_updated = True

    return result, result_updated


@st.dialog("Query Filters")
def query_filters_modal(matchChainResponse=None, requirements=None):
    try:
        print("requirements", requirements)
        # Remove the outer double quotes if they exist

        # if requirements.startswith('"') and requirements.endswith('"'):
        #     s = requirements[1:-1]

        # # Convert the string to a Python dictionary
        # data = ast.literal_eval(s)

        # with open('data.json', mode='w', encoding='utf-8') as json_file:
        #     json.dump(requirements, json_file, indent=4)

        # json_data = json.dumps(requirements, indent=4)
        # with open('data.json', 'w') as json_file:
        #     json.dump(json_data, json_file, indent=4)
        # print("json_data", type(json_data))

        # CONFIG_PATH = os.path.join('knowledge_base_json', 'knowledge_base.json') 
        with open("./data/ai_response.json", 'r') as f:
            json_requirements = json.load(f)

        # print("json_requirements", json_requirements)

        # Load the JSON configuration
        # print("matchChainResponse:", matchChainResponse)
        with open(CONFIG_PATH, 'r') as f:
            form_config = json.load(f)
        

        result = find_similar_matches(json_requirements, form_config)

        print("result", result)
        print("resultRequired", result["Required"])
        merge_details = merge_requirements(result)
        print("merge_details", merge_details)

        for key in form_config:
            if key not in merge_details:
                merge_details[key] = []
        # print("merge_requirementstype", type(merge_requirements))
        print("merge_details222", merge_details)

        # Initialize session state for user inputs if not already set
        if "user_inputs" not in st.session_state:
            st.session_state.user_inputs = {}

        # Initialize a flag to detect if the form was submitted
        form_submitted = False

        with st.form(key="my_key"):
            user_inputs = {}  # Dictionary to store user selections
            # Create a multiselect for each key in the JSON file
            # for field_name, options in form_config.items():
            #     # Set default values based on matchChainResponse
            #     if field_name.strip() in ["Graduation", "Post Graduation"]:
            #         # Check if the value in matchChainResponse is True
            #         if matchChainResponse.get(field_name.strip()) is True:
            #             # print("[options[0]] ", [options[0]] )
            #             default_values = [options[0]]  # First option if True
            #         else:
            #             default_values = []  # No default if not True
            #     else:
            #         # For other fields, filter default values based on matchChainResponse
            #         default_values = match_skills(matchChainResponse.get(field_name, []),options)
            #         print("default values", default_values)
                
            #     # Use session state to store and retrieve user inputs
            #     if field_name not in st.session_state.user_inputs:
            #         st.session_state.user_inputs[field_name] = default_values

            #     # Create the multiselect widget
            #     user_inputs[field_name] = st.multiselect(
            #         label=field_name,
            #         options=options,
            #         default=st.session_state.user_inputs[field_name],
            #         key=field_name
            #     )
            
            for field_name, options in form_config.items():
                # Set default values based on matchChainResponse
                # if field_name.strip() in ["Graduation", "Post Graduation"]:
                #     # Check if the value in matchChainResponse is True
                #     if matchChainResponse.get(field_name.strip()) is True:
                #         # print("[options[0]] ", [options[0]] )
                #         default_values = [options[0]]  # First option if True
                #     else:
                #         default_values = []  # No default if not True
                # else:
                    # For other fields, filter default values based on matchChainResponse
                default_values = merge_details[field_name.strip()]
                    # print("default values", default_values)

                # print("default values", options)
                
                # Use session state to store and retrieve user inputs
                if field_name not in st.session_state.user_inputs:
                    st.session_state.user_inputs[field_name.strip()] = default_values

                # Create the multiselect widget
                user_inputs[field_name] = st.multiselect(
                    label=field_name.strip(),
                    options=[skill for skill in options],
                    default=st.session_state.user_inputs[field_name.strip()],
                    key=field_name.strip()
                )

            form_submitted = st.form_submit_button(label="Apply")

            # st.markdown(jd_withfilter)
            # adjust_job_description_base_on_knowledge_base_details = adjust_job_description_base_on_knowledge_base(requirements, matchChainResponse)

            if form_submitted:
                # Update session state with the latest user inputs
                st.session_state.user_inputs = user_inputs
                # print("st.session_state.user_inputs after form submitted :",st.session_state.user_inputs)
                print("user_inputs", user_inputs)
                # jd_withfilter = add_filter_detail_in_optimize_jd_content(requirements, user_inputs)
                # print(jd_withfilter)
                # jd_withfilter = add_filter_detail_in_optimize_jd_content(requirements, user_inputs)

                # # st.markdown(jd_withfilter)
                
                config={"configurable": {"thread_id": "1"},"recursion_limit":40}
                with st.spinner("Processing your query..."):
                    # jd_withfilter = add_filter_detail_in_optimize_jd_content(requirements, user_inputs)


                    # jd_withfilter = add_filter_detail_in_optimize_jd_content(requirements, user_inputs)
                    # qq= requirements + """\n\n ALWAYS include these details at the time of generating SQL query-> """ + str(user_inputs)
                    # qq= requirements + """\n\nUse the following column-value mapping to generate an SQL query that accurately filters candidates based on the given criteria. """ + str(user_inputs)
                    # qq= jd_withfilter + """\n\n ALWAYS include these details at the time of generating SQL query-> """ + str(user_inputs)
                    # qq= requirements + """\n\nUse the following column-value mapping to accurately generate an SQL query for the above criteria. This is only to help you ensure the criteria match the details provided above. """ + str(user_inputs)

                    # print("Generated question :",qq)
                    # res = sql_chain.invoke({"messages": str(result)}, config)
                    # json_str = res["messages"][-1].tool_calls[0]["args"]["final_answer"]
                    # st.session_state.chat_history.append(AIMessage(content=json_str, name=get_agent_name(agent_name)))

                    def process_result(result, user_inputs, config):
                        # Make a deep copy of the original result to preserve it in case no updates occur.
                        original_result = copy.deepcopy(result)
                        updated_result, result_updated = update_required_dict(result, user_inputs)
                        
                        # If updated_result was changed, invoke sql_chain with updated_result; otherwise, with original_result.
                        # return sql_chain.invoke(str(updated_result) if result_updated else str(original_result), config)
                        if(result_updated):
                            print("updated_result", updated_result)
                            return sql_chain.invoke(str(updated_result), config)
                        else:
                            print("original_result", original_result)
                            return sql_chain.invoke(str(original_result), config)
                        
                    # res = sql_chain.invoke(str(result), config)
                    res = process_result(result, user_inputs, config)
                    st.session_state.chat_history.append(AIMessage(content=res["messages"][-1].content, name=get_agent_name(agent_name)))

                    st.session_state.user_inputs = {}
                    user_inputs = {}
                    st.rerun()

                return user_inputs


    except FileNotFoundError:
        st.error(f"Configuration file not found at path: {CONFIG_PATH}")
    except json.JSONDecodeError:
        st.error("Error decoding the JSON configuration file. Please check the file format.")
    except Exception as e:
        st.error(f"An error occurred while opening the Query Filters modal: {str(e)}")


# jd_examples = [
#     {
#         "job_description": """For a 'Software Engineer' position located in Los Angeles, does the candidate meet these criteria:   
#         - 2-4 years of experience in software development
#         - Bachelor’s degree in Computer Science
#         - Proficiency in JavaScript
#         - Strong understanding of Git
#         - Master’s degree 
#         - PhD
#         - Experience with cloud platforms such as AWS
#         - Knowledge of Agile methodologies""",
#         "answer": """
#         {"Experience": ["2"], "Skills": ["JavaScript", "Git"], "Graduation": True, "Post Graduation": False}
#         """,
#     },
#     {
#         "job_description": """For a 'Software Engineer' position located in Austin, does the candidate meet these criteria:   
#         - 3 years of experience in software development
#         - Bachelor’s degree in Computer Science 
#         - Proficiency in Python 
#         - Proficiency in SQL
#         - Proficiency in Hadoop 
#         - Knowledge of Agile methodologies""",
#         "answer": """
#         {"Experience": ["3"], "Skills": ["Python","SQL","Hadoop"], "Graduation": True, "Post Graduation": False}
#         """,
#     }
    # {
    #     "job_description": """For a 'Software Engineer' position located in Los Angeles, does the candidate meet these criteria:   
    #     - 2-4 years of experience in software development
    #     - Bachelor’s degree in Computer Science
    #     - Proficiency in JavaScript
    #     - Strong understanding of Git
    #     - Master’s degree 
    #     - PhD
    #     - Experience with cloud platforms such as AWS
    #     - Knowledge of Agile methodologies""",
    #     "answer": """
    #     {"Experience": ["2"], "Skills": ["JavaScript", "Git"], "Graduation": Bachelor's, "Post Graduation": Master's}
    #     """,
    # },
    # {
    #     "job_description": """For a 'Software Engineer' position located in Austin, does the candidate meet these criteria:   
    #     - 3 years of experience in software development
    #     - Bachelor’s degree in Computer Science 
    #     - Proficiency in Python 
    #     - Proficiency in SQL
    #     - Proficiency in Hadoop 
    #     - Knowledge of Agile methodologies""",
    #     "answer": """
    #     {"Experience": ["3"], "Skills": ["Python","SQL","Hadoop"], "Graduation": Bachelor's, "Post Graduation": Master's}
    #     """,
    # }
# ]


matchPrompt_kownledge_base = ChatPromptTemplate.from_messages([
("human","""
Job Description:
{job_description}
"""),
("ai","{answer}")
])

# few_shot_prompt = FewShotChatMessagePromptTemplate(
#     example_prompt=matchPrompt_kownledge_base,
#     examples=jd_examples,
# )

# final_prompt = ChatPromptTemplate.from_messages(
# [("system", """
# You are a highly skilled AI that extracts keywords from job descriptions. Please analyze the job description provided and structure the details into specific categories. Format your output in JSON with the following keys:

# **Double-check that all details are thoroughly covered for these categories, ensuring nothing is missing:**
# - **Experience**: Provide the minimum years of experience as a list of single value. Include relevant years if mentioned explicitly in the job description (e.g., '6').
# - **Skils**: List all skills.
# - **Location**: Extract the location(s) mentioned for this role.
# - **Graduation**: Return a list containing `"Bachelor's"` if a bachelor’s degree is required, otherwise return an empty list (`[]`).
# - **Post Graduation**: Return a list containing `"Master's"` if a master’s degree is required, otherwise return an empty list (`[]`).
# **Avoid suggesting experience unless explicitly mentioned.**

# Ensure the output strictly adheres to the following JSON format:

# ```json
# {{
#   "Experience": ["..."],
#   "Skills": ["..."],
#   "Phd": ["..."],
#   "Location": ["..."],
#   "Graduation": ["Bachelor's"/null],
#   "Post Graduation": ["Master's"/null]
# }}


# Job Description: {job_description}

# Table Headers: {table}
# """)]
# )

# final_prompt = ChatPromptTemplate.from_messages(
# [("system", """
# You are an expert in extracting key information from job descriptions. Your task is to analyze the following job description and extract the following details in a structured format:

# 1. **Experience**: Extract the required or preferred years of experience, return number like ["1"].
# 2. **Skills**: Extract the technical, soft, or domain-specific skills keywords mentioned in the job description.
# 3. **Phd**: Return a list containing `"Phd"` if a Ph.D. is required or preferred, otherwise return an empty list (`[]`).
# 4. **Location**: Extract the job location(s) mentioned in the description.
# 5. **Graduation**: Return a list containing `"Bachelor's"` if a bachelor’s degree is required, otherwise return an empty list (`[]`).
# 6. **Post Graduation**: Return a list containing `"Master's"` if a master’s degree is required, otherwise return an empty list (`[]`).

# If any of the above details are not explicitly mentioned in the job description, return an empty list (`[]`) for that field.

# **Job Description:**
# {job_description}

# **Output Format:**
# ```json
# {{
#   "Experience": ["..."],
#   "Skills": ["..."],
#   "Phd": ["..."],
#   "Location": ["..."],
#   "Graduation": ["Bachelor's"/null],
#   "Post Graduation": ["Master's"/null]
# }}

# Instructions:

# Be precise and extract only the relevant information.

# If a value is not explicitly mentioned, return an empty list ([]).

# Ensure the output is in valid JSON format.

# Table Headers: {table}
# """)]
# )

final_prompt = ChatPromptTemplate.from_messages(
[("system", """
You are an expert in extracting keywords from text. Your task is to analyze the provided job description and extract the relevant information in a structured JSON format. Follow the specific extraction criteria outlined below to ensure you capture only the most essential details.

### Extraction Criteria:
1. **Experience**: Extract only numerical values representing years of experience (e.g., ["1"]).
2. **Skills**: Extract only specific technology names, programming languages, or tools. Avoid generic categories, descriptions, or qualifiers (e.g., **do not include** phrases like "strong skills in," "experience with," "Familiarity with," "Experience with," etc.).
3. **PhD**: Return a list containing `"PhD"` if explicitly mentioned as required or preferred; otherwise, return an empty list (`[]`).
4. **Location**: Extract only the city name(s) mentioned in the context.
5. **Graduation**: Return a list containing `"Bachelor's"` if a bachelor's degree is mentioned; otherwise, return an empty list (`[]`).
6. **Post Graduation**: Return a list containing `"Master's"` if a master's degree is mentioned; otherwise, return an empty list (`[]`).

### Additional Instructions:
- **Do not include** any descriptive phrases or broad categories.
- If a value is not explicitly mentioned, return an empty list (`[]`).
- Ensure the output is in valid JSON format.

**context:**
{job_description}

**Output Format:**
```json
{{
"Experience": ["..."],
"Skills": ["..."],
"Phd": ["..."],
"Location": ["..."],
"Graduation": ["Bachelor's"/null],
"Post Graduation": ["Master's"/null]
}}
""")]
)

ai_filter = final_prompt | llm | JsonOutputParser()

def remove_extract_keywords_from_base_on_knowledge_base_using_job_description(matchChainResponse, word):

    # print("optimize_jd_content inside :", optimize_jd_content)
    # print("matchChainResponse inside :", matchChainResponse)
    template = """

    ### Task:  
    You are given two JSON objects:  

    1. **matchChainResponse**:
    {matchChainResponse}

    2. **word**:
    {word}

    Your task is to filter the `word` JSON based on `matchChainResponse`, following these steps:

    ### **Instructions:**
    1. **Extract Matching Values:**  
    - Retain only the values in `word` that are also present in `matchChainResponse`.  
    - Ignore any extra values in `word` that are not presnt in `matchChainResponse`.  

    2. **Preserve Important Details:**  
    - If a key exists in `matchChainResponse` but has an empty list (`[]`), remove it from the output.  
    - Ensure that the word **"never"** is never removed or altered.  

    3. **Output Format:**  
    - Return a JSON object with the same structure as `word`, but only containing the extracted values.  
 
    **Expected Output Format:**  

    ```json
    {{
        "Experience": ["..."],
        "Skills": ["..."],
        "Phd": ["..."],
        "Location": ["..."],
        "Graduation": ["Bachelor's"/null],
        "Post Graduation": ["Master's"/null]
    }}

    """

    

    prompt = PromptTemplate(template=template, input_variables=["word", "matchChainResponse"])

    matching_points_llm = prompt | llm | JsonOutputParser()
    response = matching_points_llm.invoke({
        "word": word, 
        "matchChainResponse": matchChainResponse, 
    })

    # save_to_markdown.save_to_markdown(response, "./data/adjust_job_description_base_on_knowledge_base.md")

    # print("response11111112222222222222222222222222222222222222222222: ", response)
    return response

def adjust_job_description_base_on_knowledge_base(optimize_jd_content, matchChainResponse):
    template = """

    ### Task:  
        You are given a matchChainResponse dictionary containing categorized skills and qualifications. Your task is to check if the details in the provided required_preferred_points text match the given matchChainResponse correctly. If any mismatch exists, correct it while maintaining formatting.  

        ### Inputs:
        matchChainResponse:
        {matchChainResponse}  

        required_preferred_points:
        {required_preferred_points}  

        ### Steps:
        1. **Check "Required" Fields:**  
        - Ensure all fields marked as required in `matchChainResponse` appear in the "Required" section of `required_preferred_points`.  
        - If a required value is missing or incorrect, update it.  

        2. **Check "Preferred" Fields:**  
        - Ensure that only values **not in "Required"** are placed under "Preferred".  
        - If a value is incorrectly placed in "Preferred", move it to "Required".  
        - Ensure no duplicate entries exist.  

        3. **Output the Corrected Text:**  
        - Maintain the original formatting and markdown structure.  
        - Ensure all corrections align with `matchChainResponse`.  

        Output Format:
            ###Required:
            Location (required): [Extracted Required Location](required)
            Experience (required): [Extracted Required Experience](required)
            Graduation (required): [Extracted Required Graduation](required)
            Post Graduation (required): [Extracted Required Post Graduation](required)
            PhD (required): [Extracted Required PhD](required)
            Skills (required): [Extracted Required Skills](required)

            ###Preferred:  
            Location (preferred): [Extracted Preferred Location](preferred)  
            Experience (preferred): [Extracted Preferred Experience](preferred)  
            Graduation (preferred): [Extracted Preferred Graduation](preferred)  
            Post Graduation (preferred): [Extracted Preferred Post Graduation](preferred)  
            PhD (preferred): [Extracted Preferred PhD](preferred)  
            Skills (preferred): [Extracted Preferred Skills](preferred)  
            
            Exclude Location, Experience, Graduation, Post Graduation, PhD, and Skills from the output if they are N/A, None, or empty

            Clearly indicate that these points are optional but beneficial.
        """

    

    prompt = PromptTemplate(template=template, input_variables=["required_preferred_points", "matchChainResponse"])

    matching_points_llm = prompt | llm
    response = matching_points_llm.invoke({
        "required_preferred_points": optimize_jd_content, 
        "matchChainResponse": matchChainResponse, 
    })

    save_to_markdown.save_to_markdown(response.content, "./data/adjust_job_description_base_on_knowledge_base.md")

    # print("response11111112222222222222222222222222222222222222222222: ", response)
    return response.content


def add_filter_detail_in_optimize_jd_content(optimize_jd_content, matchChainResponse):
    # print("Checking preferred point", state)
    # print("Checking user_inputs", matchChainResponse)

    # template= """
    #     Your task is to identify filter details not present in required_preferred_points, then add these details to the preferred points corresponding to their title. If the title is not present, create the title using key and value using its value and update the preferred points.        

    #     ### Input:
    #     - required_preferred_points: {job_description}
    #     - filter: {matchChainResponse}

    #     """
    # template= """
    #     You are a highly skilled assistant that analyzes and updates job descriptions. Your task is to ensure all details from a structured input are properly reflected in a plain-text job description. Follow these steps:

    #     Compare the details in the provided obj (structured input) with the points in required_preferred_points (plain-text job description).
    #     For any detail present in obj but missing from both "Required" and "Preferred" sections, add the detail to the appropriate category under the "Preferred" section.
    #     If a detail appears in "Required" under a specific title, but additional related details in obj are missing, add those missing details under the same title in the "Preferred" section.
    #     Maintain the structure of required_preferred_points with proper section headers and bullet points.

    #     ### Input:
    #      - required_preferred_points: {job_description}
    #      - obj: {matchChainResponse}

    #     """

    template = """
        You are a highly skilled assistant that analyzes and updates job descriptions. Your task is to ensure all details from a structured input are properly reflected in a plain-text job description. Follow these steps:

        1. Compare the details in the provided `obj` (structured input) with the points in `required_preferred_points` (plain-text job description).
        2. For any detail present in `obj` but missing from both "Required" and "Preferred" sections, add the detail to the appropriate category under the "Preferred" section.
        3. If a detail appears in "Required" under a specific title, but additional related details in `obj` are missing, add those missing details under the same title in the "Preferred" section.
        4. Use the keys from `obj` as titles when adding new information in the "Preferred" section, with the corresponding values listed as bullet points under those titles.
        5. Maintain the structure and formatting of `required_preferred_points` with proper section headers and bullet points.

        ### Input:
        - required_preferred_points (plain text): 
        {job_description}

        - obj (structured data): 
        {matchChainResponse}

        Output Format:
            ###Required:
            Location (required): [Extracted Required Location](required)
            Experience (required): [Extracted Required Experience](required)
            Graduation (required): [Extracted Required Graduation](required)
            Post Graduation (required): [Extracted Required Post Graduation](required)
            PhD (required): [Extracted Required PhD](required)
            Skills (required): [Extracted Required Skills](required)

            ###Preferred:  
            Location (preferred): [Extracted Preferred Location](preferred)  
            Experience (preferred): [Extracted Preferred Experience](preferred)  
            Graduation (preferred): [Extracted Preferred Graduation](preferred)  
            Post Graduation (preferred): [Extracted Preferred Post Graduation](preferred)  
            PhD (preferred): [Extracted Preferred PhD](preferred)  
            Skills (preferred): [Extracted Preferred Skills](preferred)   

            Exclude Location, Experience, Graduation, Post Graduation, PhD, and Skills from the output if they are N/A, None, or empty

            Clearly indicate that these points are optional but beneficial.

        """

    

    prompt = PromptTemplate(template=template, input_variables=["job_description", "matchChainResponse"])

    matching_points_llm = prompt | llm
    response = matching_points_llm.invoke({
        "job_description": optimize_jd_content, 
        "matchChainResponse": matchChainResponse, 
    })

    save_to_markdown.save_to_markdown(response.content, "./data/jd_with_filter.md")

    # print("response1111111: ", response)
    return response.content
    # return matching_points_llm


# print("ai_filter logged :",ai_filter)

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


prompt = st.chat_input("Find your next superstar")
if prompt is not None and prompt != "" :
    with st.chat_message("Human"):
        st.markdown(prompt)
        st.session_state.chat_history.append(HumanMessage(content=prompt, name=get_agent_name(agent_name)))
        folder_path = 'knowledge_base_csv'
        csv_headers = get_csv_headers(folder_path)
        # print(csv_headers)
        # print(type(csv_headers))
        # print(type(str(csv_headers)))
        matchChainResponse = ai_filter.invoke({"job_description": prompt, "table": csv_headers})
        # print(type(matchChainResponse))
        # Pass matchChainResponse to query_filters_modal
        # query = query_filters_modal(matchChainResponse=matchChainResponse, requirements=prompt)

        # st.session_state.user_inputs = user_inputs
        # print("st.session_state.user_inputs after form submitted :",st.session_state.user_inputs)
        holder = st.empty()
        config={"configurable": {"thread_id": "1"},"recursion_limit":40}
        with st.spinner("Processing your query..."):
            if(get_agent_name(agent_name) == "SQLTeam Agent"):
                # qq= prompt + """\n\n ALWAYS refer to this detail to check which details below to which coloum at the time of generating SQL query-> """ + str(matchChainResponse)
                qq= prompt + """\n\n Use the following column-value mapping to generate an SQL query that accurately filters candidates based on the given criteria. """ + str(matchChainResponse)
                # qq= prompt

                print("Generated question :",qq)
                res = sql_chain.invoke(qq, config)
                st.session_state.chat_history.append(AIMessage(content=res["messages"][-1].content, name=get_agent_name(agent_name)))
                # st.session_state.user_inputs = {}
                # user_inputs = {}
                st.rerun()
            else:
                config={"configurable": {"thread_id": "2"},"recursion_limit":40}
                res = github_chain.invoke(prompt,config)
                print("AI response hithub :",res["messages"])
                aiRes = res["messages"][-1].content
                holder.write(aiRes)            
                st.session_state.chat_history.append(AIMessage(content=aiRes, name=get_agent_name(agent_name)))

        # print("quert", query)

#     st.session_state.chat_history.append(HumanMessage(content=prompt, name=get_agent_name(agent_name)))
#         # create_image_func.create_graph_image(super_graph, "super_graph")
#     holder = st.empty()
#     with st.spinner("Processing your query..."):
#         print("the final prompt  :",prompt)
#         try:
#             if(get_agent_name(agent_name) == "SQLTeam Agent"):
#                 config={"configurable": {"thread_id": "1"},"recursion_limit":40}
#                 res = sql_chain.invoke(prompt, config)
#                 print("AI response :",res["messages"])
#                 tableText = res["messages"][-1].content
#                 st.session_state.chat_history.append(AIMessage(content=tableText, name=get_agent_name(agent_name)))
#                 checkForTable1(tableText)
#             else:
#                 config={"configurable": {"thread_id": "2"},"recursion_limit":40}
#                 res = github_chain.invoke(prompt,config)
#                 print("AI response :",res["messages"])
#                 aiRes = res["messages"][-1].content
#                 holder.write(aiRes)            
#                 st.session_state.chat_history.append(AIMessage(content=aiRes, name=get_agent_name(agent_name)))
#         except GraphRecursionError:
#             st.info("Graph recursion limit exceeded , try again!")


def get_cleared_question(question, Get_more_context):
    # Check if "No match found" is in the Get_more_context array
    if "No match found" in Get_more_context:
        return question
    else:
        cleared_question = f"{question} in which {Get_more_context} ?"
        return cleared_question


def load_knowledge_base():
    encodings_to_try = ['ISO-8859-1', 'Windows-1252', 'utf-16', 'utf-8-sig']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv("./Files/knowledgeBase/knowledgeBase.csv", encoding=encoding, dtype=str)
            print(f"Successfully loaded with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            print(f"Failed to decode using encoding: {encoding}")
        except Exception as e:
            print(f"An error occurred: {e}")
    return None  # Return None if all attempts fail

def find_match_in_csv(question):
    # Read the CSV file into a DataFrame
    # df = pd.read_csv("./Files/knowledgeBase/knowledgeBase.csv", encoding='utf-8', dtype=str)
    df = load_knowledge_base()

    # Split the question into words and create phrases of varying lengths
    clean_question = clean_text(question)
    words = clean_question.split()
    
    phrases = []
    for i in range(len(words)):
        for j in range(i+1, len(words)+1):
            phrase = " ".join(words[i:j])
            phrases.append(phrase)
    
    matches = []
    
    # Loop through each column and each value to find exact matches
    for column in df.columns:
        for value in df[column].astype(str).unique():
            for phrase in phrases:
                if phrase.lower() == value.lower():
                    matches.append(f'{value} is a {column}')
                    break  # Avoid duplicate matches for the same value
    
    return matches if matches else ["No match found"]


def find_similar_words(input_dict, database):
    result = {}
    for category in input_dict:
        if category not in database:
            continue
        input_words = [word.strip().lower() for word in input_dict[category]]
        db_words = database[category]
        matched_words = []
        for db_word in db_words:
            db_word_lower = db_word.lower()
            for input_word in input_words:
                if input_word in db_word_lower:
                    matched_words.append(db_word)
                    break  # Avoid adding duplicates for the same db_word
        result[category] = matched_words
    return result
import json

if(buttonVal):
    question = retreive_users.retreive_users_fnc()
    job_discription_markdown = retreive_users.load_markdown("./ruleData/outputRuleData.md") 
    optimize_jd_content = retreive_users.load_markdown("./data/ai_response.md") 
    # job_discription_markdown = retreive_users.load_markdown("./data/summarizeOutputRuleData.md") 
    with st.chat_message("Human"):
        st.markdown(job_discription_markdown)
    st.session_state.chat_history.append(HumanMessage(content=job_discription_markdown, name=get_agent_name(agent_name)))
    # create_image_func.create_graph_image(super_graph, "super_graph")
    holder = st.empty()
    with st.spinner("Processing your query..."):
        try:
            if(get_agent_name(agent_name) == "SQLTeam Agent"):
                folder_path = 'knowledge_base_csv'
                csv_headers = get_csv_headers(folder_path)
                # print(csv_headers)
                # print(type(csv_headers))
                # print(type(str(csv_headers)))
                
                matchChainResponse = ai_filter.invoke({"job_description": optimize_jd_content, "table": csv_headers})
                print("matchChainResponse111111111111: ", matchChainResponse)

                with open(CONFIG_PATH, 'r') as f:
                    form_config = json.load(f)
                # word = find_similar_words(matchChainResponse, form_config)
                # print("word: ", word)
                # filter_word = remove_extract_keywords_from_base_on_knowledge_base_using_job_description(matchChainResponse, word)

                # print("filter_word: ", filter_word)

                # adjust_job_description_base_on_knowledge_base_details = adjust_job_description_base_on_knowledge_base(optimize_jd_content, matchChainResponse)
                # json_data = json.dumps(optimize_jd_content)

                # # matches = find_similarget(json_data, form_config)
                # print("Match results:")
                # from pprint import pprint
                # pprint(matches)
                # query = query_filters_modal(matchChainResponse=matchChainResponse, requirements=adjust_job_description_base_on_knowledge_base_details)
                query = query_filters_modal(matchChainResponse=matchChainResponse, requirements=optimize_jd_content)

                # config={"configurable": {"thread_id": "1"},"recursion_limit":40}
                # res = sql_chain.invoke(question, config)
                # print("AI response :",res["messages"])
                # tableText = res["messages"][-1].content
                # # st.session_state.chat_history.append(AIMessage(content=aiRes, name=get_agent_name(agent_name)))
                # checkForTable(tableText,question)
            else:
                config={"configurable": {"thread_id": "2"},"recursion_limit":40}
                res = github_chain.invoke(question,config)
                print("AI response :",res["messages"])
                aiRes = res["messages"][-1].content
                holder.write(aiRes)            
                st.session_state.chat_history.append(AIMessage(content=aiRes, name=get_agent_name(agent_name)))
        except GraphRecursionError:
            st.info("Graph recursion limit exceeded , try again!")


# from openai import OpenAI

# def deepseekfn():
#     client = OpenAI(api_key="sk-3f3074eaaa194d4c808bb90c3dedc257", base_url="https://api.deepseek.com")

#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant"},
#             {"role": "user", "content": "Hello"},
#         ],
#         stream=False
#     )

#     print("hello MR",response)