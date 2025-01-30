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


examples = [
    {
        "question": """For a 'Software Engineer' position located in Los Angeles, does the candidate meet these criteria:   
        - 2-4 years of experience in software development (required)
        - Bachelor’s degree in Computer Science (required)
        - Proficiency in JavaScript (required)
        - Strong understanding of Git (required)
        - Master’s degree (preferred)
        - PhD (preferred)
        - Experience with cloud platforms such as AWS (preferred)
        - Knowledge of Agile methodologies (preferred)""",
        "answer": """
        {"Required": {"Experience": [2], "Skills": ["JavaScript", "Git"], "Graduation": ["Bachelor's"]}, "Preferred": {"Post Graduation": ["Master's"], "PhD": ["PhD"], "Skills":["AWS", "Agile methodologies"]}}
        """,
    },
    {
        "question": """For a 'Software Engineer' position located in Austin, does the candidate meet these criteria:   
        - 3 years of experience in software development (required)
        - Bachelor’s degree in Computer Science (required)
        - Proficiency in Python (required)
        - Proficiency in SQL (required)
        - Proficiency in Hadoop (required)
        - Master’s degree (preferred)
        - Knowledge of Agile methodologies (preferred)""",
        "answer": """
        {"Required": {"Experience": [3], "Skills": ["Python","SQL","Hadoop"], "Graduation": ["Bachelor's"]}, "Preferred": {"Post Graduation": ["Master's"], "Skills": ["Agile methodologies"]}}
        """,
    }
]

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

matchPrompt = ChatPromptTemplate(messages=[
    ("system","""
You are given a job description with specific required and preferred qualifications, along with a table of headers. 
Your task is to extract and categorize the qualifications as either "Required" or "Preferred", using the table headers as a guide. 
Ensure that no required fields from the job description are missed. 
The output should be a dictionary with two keys: "Required" and "Preferred" 
Under each key, list the relevant headers mentioned in the job description.

Job Description:
{job_description}

Table Headers:
{table}

# Instructions:
1. Extract the qualifications from the job description.
2. Categorize them according to the table headers.
3. List "Preferred" qualifications under "Preferred" and all others under "Required."
4. If a qualification matches a value in the table rows, use the exact spelling from the table. Otherwise, use the spelling as found in the job description.
5. Ensure numeric values are presented as numbers only, without additional strings.
6. Ensure each section is clearly labeled, ordered, and separated by commas.
7. ENSURE that if any key contains multiple values separated by commas, they are always placed in a list. ALWAYS enforce this structure, and NEVER overlook this step.
8. ENSURE that the 'Experience' field is always a list with exactly one element. If there are multiple elements in the list, keep only the first one and ignore the rest.
9. PhD should only appear under the PhD key and NOT under Post Graduation. Deduplicate "PhD" from the Post Graduation list if it appears there.

Output: 
Ensure all table headers are addressed in the output.
Only return dictionary nothing else.

# Nerver forgot output format and must be dictionary:
Output format: 
dictionary: "{{"Required": {{}},"Preferred": {{}}}}"
""")
],input_variables=["job_description","table"])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=matchPrompt,
    examples=examples,
)

matchChain = matchPrompt | llm | JsonOutputParser()

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



@st.dialog("Query Filters")
def query_filters_modal(matchChainResponse=None, requirements=None):
    try:
        # Load the JSON configuration
        print("matchChainResponse:", matchChainResponse)
        with open(CONFIG_PATH, 'r') as f:
            form_config = json.load(f)
        
        # Initialize session state for user inputs if not already set
        if "user_inputs" not in st.session_state:
            st.session_state.user_inputs = {}

        # Initialize a flag to detect if the form was submitted
        form_submitted = False

        with st.form(key="my_key"):
            user_inputs = {}  # Dictionary to store user selections
            # Create a multiselect for each key in the JSON file
            for field_name, options in form_config.items():
                # Set default values based on matchChainResponse
                if field_name.strip() in ["Graduation23", "Post Graduation23"]:
                    # Check if the value in matchChainResponse is True
                    if matchChainResponse.get(field_name.strip()) is True:
                        print("[options[0]] ", [options[0]] )
                        default_values = [options[0]]  # First option if True
                    else:
                        default_values = []  # No default if not True
                else:
                    # For other fields, filter default values based on matchChainResponse
                    default_values = match_skills(matchChainResponse.get(field_name, []),options)
                    print("default values", default_values)
                
                # Use session state to store and retrieve user inputs
                if field_name not in st.session_state.user_inputs:
                    st.session_state.user_inputs[field_name] = default_values

                # Create the multiselect widget
                user_inputs[field_name] = st.multiselect(
                    label=field_name,
                    options=options,
                    default=st.session_state.user_inputs[field_name],
                    key=field_name
                )

            form_submitted = st.form_submit_button(label="Apply")

            # st.markdown(jd_withfilter)
                

            if form_submitted:
                # Update session state with the latest user inputs
                st.session_state.user_inputs = user_inputs
                # print("st.session_state.user_inputs after form submitted :",st.session_state.user_inputs)
                # print("user_inputs", user_inputs)
                # jd_withfilter = add_filter_detail_in_optimize_jd_content(requirements, user_inputs)
                # print(jd_withfilter)
                # jd_withfilter = add_filter_detail_in_optimize_jd_content(requirements, user_inputs)

                # # st.markdown(jd_withfilter)
                
                config={"configurable": {"thread_id": "1"},"recursion_limit":40}
                with st.spinner("Processing your query..."):
                    jd_withfilter = add_filter_detail_in_optimize_jd_content(requirements, user_inputs)

                    # jd_withfilter = add_filter_detail_in_optimize_jd_content(requirements, user_inputs)
                    # qq= requirements + """\n\n ALWAYS include these details at the time of generating SQL query-> """ + str(user_inputs)
                    qq= requirements + """\n\nUse the following column-value mapping to generate an SQL query that accurately filters candidates based on the given criteria. """ + str(user_inputs)

                    # print("Generated question :",qq)
                    res = sql_chain.invoke(qq, config)
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


jd_examples = [
    {
        "job_description": """For a 'Software Engineer' position located in Los Angeles, does the candidate meet these criteria:   
        - 2-4 years of experience in software development
        - Bachelor’s degree in Computer Science
        - Proficiency in JavaScript
        - Strong understanding of Git
        - Master’s degree 
        - PhD
        - Experience with cloud platforms such as AWS
        - Knowledge of Agile methodologies""",
        "answer": """
        {"Experience": ["2"], "Skills": ["JavaScript", "Git"], "Graduation": True, "Post Graduation": False}
        """,
    },
    {
        "job_description": """For a 'Software Engineer' position located in Austin, does the candidate meet these criteria:   
        - 3 years of experience in software development
        - Bachelor’s degree in Computer Science 
        - Proficiency in Python 
        - Proficiency in SQL
        - Proficiency in Hadoop 
        - Knowledge of Agile methodologies""",
        "answer": """
        {"Experience": ["3"], "Skills": ["Python","SQL","Hadoop"], "Graduation": True, "Post Graduation": False}
        """,
    }
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
]


matchPrompt_kownledge_base = ChatPromptTemplate.from_messages([
("human","""
Job Description:
{job_description}
"""),
("ai","{answer}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=matchPrompt_kownledge_base,
    examples=jd_examples,
)

final_prompt = ChatPromptTemplate.from_messages(
[("system", """
You are a highly skilled AI that extracts job details from job descriptions. Please analyze the job description provided and structure the details into specific categories. Format your output in JSON with the following keys:

**Double-check that all details are thoroughly covered for these categories, ensuring nothing is missing:**
- **Experience**: Provide the minimum years of experience as a list of single value. Include relevant years if mentioned explicitly in the job description (e.g., '6').
- **Frontend**: List all frontend-related skills and technologies mentioned in the job description, such as frameworks, libraries, and tools specific to frontend development.
- **Backend**: List all backend-related skills and technologies mentioned in the job description, such as programming languages, frameworks, and platforms specific to backend development.
- **DB**: List all database-related technologies mentioned in the job description, including database systems, query languages, and relevant tools.
- **Tools**: Include any additional tools or software mentioned in the job description that are not specific to frontend, backend, or database.
- **Miscellaneous**: List any other skills, methodologies, or attributes mentioned in the job description that don't fall into the above categories.
- **Location**: Extract the location(s) mentioned for this role.
- **Graduation**: Return a list containing `"Bachelor's"` if a bachelor’s degree is required, otherwise return an empty list (`[]`).
- **Post Graduation**: Return a list containing `"Master's"` if a master’s degree is required, otherwise return an empty list (`[]`).
**Avoid suggesting experience unless explicitly mentioned.**

Ensure the output strictly adheres to the following JSON format:

```json
{{
  "Experience": ["..."],
  "Frontend": ["..."],
  "Backend": ["..."],
  "DB": ["..."],
  "Tools": ["..."],
  "Phd": ["..."],
  "Location": ["..."],
  "Miscellaneous": ["..."],
  "Graduation": ["Bachelor's"/null],
  "Post Graduation": ["Master's"/null]
}}


Job Description: {job_description}

Table Headers: {table}
""")]
)

ai_filter = final_prompt | llm | JsonOutputParser()


def add_filter_detail_in_optimize_jd_content(optimize_jd_content, matchChainResponse):
    # print("Checking preferred point", state)
    print("Checking user_inputs", matchChainResponse)

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

        ### Output:
        Provide the updated plain-text job description with:
        1. Missing details from `obj` added to the "Preferred" section, using keys from `obj` as titles if not already present.
        2. Existing details in the "Required" section unchanged.
        3. Proper formatting and consistency throughout.
        4. Don't return explanation. Only return the updated job description.

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