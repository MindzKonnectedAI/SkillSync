from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
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
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
import utils.csv_to_sql as csv_to_sql
import utils.create_image_func as create_image_func
import utils.create_team_supervisor_func as create_team_supervisor_func
from langchain_core.output_parsers.json import JsonOutputParser
import utils.upload_job_description as upload_job_description
import utils.retreive_users as retreive_users
import os
from langgraph.errors import GraphRecursionError
import utils.display_uploaded_files as display_uploaded_files
import utils.upload_csv as upload_csv
import utils.calculate_user_percentage as calculate_user_percentage
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
st.title("Intelligent Recruitment Assistant")

def retrive():
    pass

# Define folder paths
csv_folder = "csv"

# Ensure the csv and db directories exist
os.makedirs(csv_folder, exist_ok=True)

view = st.sidebar.selectbox(
    "View",
    ("User", "Admin"),
    0
)

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
        upload_job_description.upload_rule_data(uploaded_checking_rule_file,container)
        time.sleep(2)
        container.empty()


    display_uploaded_files.display_uploaded_files("1","./pdf",".pdf")

    buttonVal = st.sidebar.button(
        "Retrieve Users",
        on_click=retrive,  # Note the lack of parentheses here
        key="retreive_users",
    )

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

def get_agent_name(agent_name_here):
    if(agent_name_here=="ATS"):
        return "SQLTeam Agent"
    else:
        return "GithubTeam Agent"

# Filter for messages from SQLTeam Agent
agent_name_to_filter = get_agent_name(agent_name)  # Adjust this parameter as needed

prompt_template = """
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
3. List "Preferred" qualifications under "Preferred" and all others under "Required".
4. Qualification key name is ALWAYS the column name of the table.
5. If a qualification matches a value in the table rows, use the exact spelling from the table. Otherwise, use the spelling as found in the job description.
6. Ensure numeric values are presented as numbers only, without additional strings.
7. Ensure each section is clearly labeled, ordered, and separated by commas.
8. ALWAYS RETURN values of all keys as arrays.
Output: 
Ensure all table headers are addressed in the output.
Only return dictionary nothing else.

# Nerver forgot output format and must be dictionary:
Output format: 
dictionary: "{{"Required": {{}},"Preferred": {{}}}}"
"""

# Create a prompt with the correct input variable
matchPrompt = PromptTemplate(template=prompt_template, input_variables=["job_description", "table"])

matchChain = matchPrompt | llm | JsonOutputParser()

def extract_table_from_text(text):
    print("***************************extract_table_from_text*********************************")
    print("text :", text)
    
    # Updated regular expression to include the last row
    table_match = re.search(r'\|.*\n(\|.*\n)*\|.*', text)
    print("table_match :", table_match)
    
    if not table_match:
        return []
    
    table_text = table_match.group(0)
    print("table_text :", table_text)
    
    # Split the extracted table text into lines
    lines = table_text.strip().split('\n')
    print("lines :", lines)
    
    # Extract the header and rows
    header = [col.strip() for col in lines[0].split('|') if col.strip()]
    print("header :", header)
    
    rows = []
    for line in lines[2:]:  # Skip the separator line
        # Skip any separator lines (those with "---")
        if '---' in line:
            continue
        
        row = [col.strip() for col in line.split('|') if col.strip()]
        rows.append(row)
    
    print("rows :", rows)
    
    # Combine header and rows into a table (list of lists)
    table = [header] + rows
    print("final table to be returned:", table)
    
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
    print("table", table)
    
    if len(table) > 0:
        
        # Create a DataFrame from the table for export
        df = pd.DataFrame(table[1:], columns=table[0])  # Skip header for data
        
        # Display 'Export' button and allow CSV download
        csv = df.to_csv(index=False)
        # Generate a unique key using current Unix epoch time
        buf = io.StringIO(csv)  # Use StringIO to handle CSV in memory
        print("CSV :",csv)
        print("buf :",buf)
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
    print("table aaya :",table)
    options=filter_names(table)
    print("options :",options)
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
                print("**********************************************************************")
                print("message.content in chat history :",message.content)
                table = extract_table_from_text(message.content)
                print("table in chat history :",table)
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
def checkForTable(tableText,question):
    table = extract_table_from_text(tableText)
        
    if len(table) > 0:
        print("JOB DESCRIPTION :",question)
        print("TABLE :",table)
        matchChainResponse = matchChain.invoke({"job_description": question, "table": table})
        print("matchChainResponse :",matchChainResponse)
        # Convert the string to a dictionary
        # dict_obj = json.loads(matchChainResponse)
        myres = calculate_user_percentage.calculate_user_percentage(table,matchChainResponse)
        print("myres :",myres)
        updatedRes = update_table_in_res(tableText,myres)
        print("updatedRes :",updatedRes)
        st.session_state.chat_history.append(AIMessage(content=updatedRes, name=get_agent_name(agent_name)))

        unique_file_name = f"{int(time.time())}.csv"
        buf = extract_required_preferred_fields(updatedRes)
        if 'chat_history' in st.session_state:
            # Check if chat_history is not empty
            if st.session_state.chat_history:
                # Get the last message in the chat_history
                last_message = st.session_state.chat_history[-1]
                
                # Check if the last message is an AIMessage and filter by agent name
                if isinstance(last_message, AIMessage) and last_message.name == agent_name_to_filter:
                    with st.chat_message("AI"):
                        st.markdown(last_message.content)
                        # st.button("export", key=random.randint(1, 10000))
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
        if 'chat_history' in st.session_state:
            # Check if chat_history is not empty
            if st.session_state.chat_history:
                # Get the last message in the chat_history
                last_message = st.session_state.chat_history[-1]
                
                # Check if the last message is an AIMessage and filter by agent name
                if isinstance(last_message, AIMessage) and last_message.name == agent_name_to_filter:
                    with st.chat_message("AI"):
                        st.markdown(last_message.content)

# Check for existence of table in a text
def checkForTable1(tableText):
    table = extract_table_from_text(tableText)
        
    if len(table) > 0:
        unique_file_name = f"{int(time.time())}.csv"
        buf = extract_required_preferred_fields(tableText)
        if 'chat_history' in st.session_state:
            # Check if chat_history is not empty
            if st.session_state.chat_history:
                # Get the last message in the chat_history
                last_message = st.session_state.chat_history[-1]
                
                # Check if the last message is an AIMessage and filter by agent name
                if isinstance(last_message, AIMessage) and last_message.name == agent_name_to_filter:
                    with st.chat_message("AI"):
                        st.markdown(last_message.content)
                        # st.button("export", key=random.randint(1, 10000))
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
        if 'chat_history' in st.session_state:
            # Check if chat_history is not empty
            if st.session_state.chat_history:
                # Get the last message in the chat_history
                last_message = st.session_state.chat_history[-1]
                
                # Check if the last message is an AIMessage and filter by agent name
                if isinstance(last_message, AIMessage) and last_message.name == agent_name_to_filter:
                    with st.chat_message("AI"):
                        st.markdown(last_message.content)

prompt = st.chat_input("Find your next superstar")
if prompt is not None and prompt != "" :
    with st.chat_message("Human"):
        st.markdown(prompt)
    
    st.session_state.chat_history.append(HumanMessage(content=prompt, name=get_agent_name(agent_name)))
        # create_image_func.create_graph_image(super_graph, "super_graph")
    holder = st.empty()
    with st.spinner("Processing your query..."):
        print("the final prompt  :",prompt)
        try:
            if(get_agent_name(agent_name) == "SQLTeam Agent"):
                config={"configurable": {"thread_id": "1"},"recursion_limit":40}
                res = sql_chain.invoke(prompt, config)
                print("AI response :",res["messages"])
                tableText = res["messages"][-1].content
                st.session_state.chat_history.append(AIMessage(content=tableText, name=get_agent_name(agent_name)))
                checkForTable1(tableText)
            else:
                config={"configurable": {"thread_id": "2"},"recursion_limit":40}
                res = github_chain.invoke(prompt,config)
                print("AI response :",res["messages"])
                aiRes = res["messages"][-1].content
                holder.write(aiRes)            
                st.session_state.chat_history.append(AIMessage(content=aiRes, name=get_agent_name(agent_name)))
        except GraphRecursionError:
            st.info("Graph recursion limit exceeded , try again!")


if(buttonVal):
    question = retreive_users.retreive_users_fnc()
    with st.chat_message("Human"):
        st.markdown(question)
    
    st.session_state.chat_history.append(HumanMessage(content=question, name=get_agent_name(agent_name)))
    # create_image_func.create_graph_image(super_graph, "super_graph")
    holder = st.empty()
    with st.spinner("Processing your query..."):
        try:
            if(get_agent_name(agent_name) == "SQLTeam Agent"):
                config={"configurable": {"thread_id": "1"},"recursion_limit":40}
                res = sql_chain.invoke(question, config)
                print("AI response :",res["messages"])
                tableText = res["messages"][-1].content
                # st.session_state.chat_history.append(AIMessage(content=aiRes, name=get_agent_name(agent_name)))
                checkForTable(tableText,question)
            else:
                config={"configurable": {"thread_id": "2"},"recursion_limit":40}
                res = github_chain.invoke(question,config)
                print("AI response :",res["messages"])
                aiRes = res["messages"][-1].content
                holder.write(aiRes)            
                st.session_state.chat_history.append(AIMessage(content=aiRes, name=get_agent_name(agent_name)))
        except GraphRecursionError:
            st.info("Graph recursion limit exceeded , try again!")