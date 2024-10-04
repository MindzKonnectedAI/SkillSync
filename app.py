import streamlit as st
from dotenv import load_dotenv
import os
from typing import Annotated, List
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from langchain_openai.chat_models import ChatOpenAI
from typing_extensions import TypedDict
import sql.sql_agent_team_supervisor as sql_agent_team_supervisor
import github.github_team_supervisor as github_team_supervisor

import utils.csv_to_sql as csv_to_sql
import utils.create_image_func as create_image_func
import utils.create_team_supervisor_func as create_team_supervisor_func
import utils.upload_job_description as upload_job_description
import utils.retreive_users as retreive_users
import os
from langgraph.errors import GraphRecursionError
import utils.display_uploaded_files as display_uploaded_files
import utils.upload_csv as upload_csv
import time
import re
import json
import pandas as pd
import io
import time
import random
import uuid
# Load environment variables from .env file
load_dotenv()

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
llm = ChatOpenAI(model="gpt-4o-mini")

# # Define the supervisor node
# supervisor_node = create_team_supervisor_func.create_team_supervisor_func(
#     llm,
#     "You are a supervisor tasked with managing a conversation between the following teams: {team_members}. "
#     "Given the following user request, respond with the worker to act next. For a user request , you can only call one worker , so carefully analyze the user request and then assign the worker. Each worker will perform a "
#     "task and respond with their results and status. When finished, respond with FINISH."
#     "When giving results back to the user , also mention the worker that was used.",
#     ["SqlTeam", "GithubTeam"],
# )

# # Define the top-level state
# class State(TypedDict):
#     messages: Annotated[List[BaseMessage], operator.add]
#     next: str

# # Helper functions
# def get_last_message(state: State) -> str:
#     return state["messages"][-1].content

# def join_graph(response: dict):
#     return {"messages": [response["messages"][-1]]}

# # Define the graph
# super_graph = StateGraph(State)
# super_graph.add_node("GithubTeam", get_last_message | github_chain | join_graph)
# super_graph.add_node("SqlTeam", get_last_message | sql_chain | join_graph)
# super_graph.add_node("super_supervisor", supervisor_node)

# super_graph.add_edge("GithubTeam", "super_supervisor")
# super_graph.add_edge("SqlTeam", "super_supervisor")

# def next_step(x):
#     return x["next"]

# super_graph.add_conditional_edges(
#     "super_supervisor",
#     next_step,
#     {
#         "SqlTeam": "SqlTeam",
#         "GithubTeam": "GithubTeam",
#         "FINISH": END,
#     },
# )

# super_graph.add_edge(START, "super_supervisor")
# super_graph = super_graph.compile()

# import pdfplumber  # Use this for more reliable PDF text extraction
# import fitz  # PyMuPDF

# def show_pdf(file_path):
#     if file_path is not None:
#         try:
#             # Open the PDF with PyMuPDF
#             doc = fitz.open(file_path)
#             content = ""
#             for page_num in range(doc.page_count):
#                 page = doc.load_page(page_num)
#                 content += page.get_text()

#             # Display the content
#             st.write(content if content else "No extractable text found.")
        
#         except Exception as e:
#             st.error(f"An error occurred: {str(e)}")


# file_path = os.path.join("./ruleData", "Name.pdf")
# print("file_path", file_path)
# show_pdf(file_path)  # Trigger PDF display with selected file


# Streamlit UI
st.title("Intelligent Recruitment Assistant")

def retrive():
    pass
    # question = retreive_users.retreive_users_fnc()
    # return question
    # print("question", question)

    # with st.chat_message("Human"):
    #     st.markdown(question)
    
    # st.session_state.chat_history.append(HumanMessage(question))
    # with st.chat_message("AI"):
    #     # Add a spinner with "Processing your query..." text
    #     # with st.spinner("Processing your query..."):
    #         # Generate the graph image and save it to the temporary file
    #     create_image_func.create_graph_image(super_graph, "super_graph")
    #     final_prompt = question +" *** using SQLTeam Agent *** "
    #     res = super_graph.invoke(input={"messages": [HumanMessage(content=final_prompt)]})
    #     print("AI response :",res["messages"])
    #     aiRes = res["messages"][-1].content
    #     st.write(aiRes)            
    #     st.session_state.chat_history.append(AIMessage(aiRes))
    # return question

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

    # Use a lambda to delay the function call until the button is clicked
    buttonVal = st.sidebar.button(
        "Retrieve Users",
        on_click=retrive,  # Note the lack of parentheses here
        key="retreive_users",
        # help="collectible_button",
    )



        # st.sidebar.divider()

        # uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"],key="csv_uploader")
        # display_uploaded_files.display_uploaded_files("2","./csv",".csv")

        # if uploaded_file is not None:
        #     container = st.empty()
        #     container.write("Processing the uploaded file...")
        #     upload_csv.upload_csv(uploaded_file,container)
        #     time.sleep(2)
        #     container.empty()

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

# # Conversation History
# for message in st.session_state.chat_history:
#     print("st.session_state.chat_history", st.session_state.chat_history)
#     if isinstance(message,HumanMessage):
#         with st.chat_message("Human"):
#             st.markdown(message.content)
#     else:
#         with st.chat_message("AI"):
#             st.markdown(message.content)

# user_input = st.text_input("Enter your query:", "Total number of users in SQL Database?")


# Function to add a column with email button and return the modified table as a string
# def display_with_email_button(data):
#     # Add a new header for the "Send Email" column
#     data[0].append('Action')
    
#     # Create a new table with the "Send Email" button placeholder
#     modified_data = [data[0]]  # Start with the headers

#     for row in data[1:]:
#         # Create button label with the candidate's name
#         if st.button(f"Send Email to {row[0]}"):
#             st.write(f"Sending email to {row[0]}")  # Placeholder action when the button is clicked
#         # Add a placeholder for the button action in the table
#         modified_row = row + ['Send Email Button']
#         modified_data.append(modified_row)

#     # Convert the modified table to a string
#     table_as_string = ""
#     for row in modified_data:
#         table_as_string += ", ".join(row) + "\n"
    
#     # Return the table as a string
#     return table_as_string


# Function to create a markdown table with email buttons
# def display_with_email_button(data):
#     # Add a new header for the "Send Email" column
#     data[0].append('Action')

#     # Create a new table with the "Send Email" button placeholder
#     modified_data = [data[0]]  # Start with the headers

#     for row in data[1:]:
#         # Create button label with the candidate's name
#         if st.button(f"Send Email to {row[0]}"):
#             st.write(f"Sending email to {row[0]}")  # Placeholder action when the button is clicked
#         # Add a placeholder for the button action in the table
#         modified_row = row + ['Send Email Button']
#         modified_data.append(modified_row)

#     # Convert the modified table to a markdown-like table string
#     table_as_string = ""
    
#     # Add table headers
#     headers = f"| {' | '.join(modified_data[0])} |\n"
#     separator = "| " + " | ".join(["---"] * len(modified_data[0])) + " |\n"
#     table_as_string += headers + separator

#     # Add table rows
#     for row in modified_data[1:]:
#         table_as_string += f"| {' | '.join(row)} |\n"
    
#     # Return the table in a format that can be matched by the regex
#     return table_as_string


# # Function to display the table with buttons inside each row
# def display_with_email_button(data):
#     # Display the headers manually
#     st.write(f"| {' | '.join(data[0])} |")
#     st.write("| " + " | ".join(["---"] * len(data[0])) + " |")
    
#     # Loop through the rows to create columns for each row of data
#     for row in data[1:]:
#         col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 4, 2])
        
#         # Fill each column with the respective data
#         col1.write(row[0])  # Name
#         col2.write(row[1])  # Location
#         col3.write(row[2])  # Experience
#         col4.write(row[3])  # Skills
        
#         # Add the button in the last column
#         if col5.button(f"Send Email to {row[0]}"):
#             st.write(f"Email sent to {row[0]}")

# Function to display the table with buttons inside each row
# def display_with_email_button(data):
#     # Display the headers using the same column structure
#     header_cols = st.columns([2, 2, 1, 4, 2])
#     headers = data[0] + ['Action']
    
#     for i, header in enumerate(headers):
#         header_cols[i].write(f"**{header}**")  # Display headers in bold

#     # Loop through the rows to create columns for each row of data
#     for row in data[1:]:
#         col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 4, 2])
        
#         # Fill each column with the respective data
#         col1.write(row[0])  # Name
#         col2.write(row[1])  # Location
#         col3.write(row[2])  # Experience
#         col4.write(row[3])  # Skills
        
#         # Add the button in the last column
#         if col5.button(f"Send Email to {row[0]}"):
#             st.write(f"Email sent to {row[0]}")
        

# Function to display the table with buttons inside each row
# def display_with_email_button(data):
#     # Display the headers using the same column structure
#     header_cols = st.columns([2, 2, 1, 4, 2])
#     headers = data[0] + ['Action']
    
#     for i, header in enumerate(headers):
#         header_cols[i].write(f"**{header}**")  # Display headers in bold

#     # Loop through the rows to create columns for each row of data
#     for index, row in enumerate(data[1:]):
#         col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 4, 2])
        
#         # Fill each column with the respective data
#         col1.write(row[0])  # Name
#         col2.write(row[1])  # Location
#         col3.write(row[2])  # Experience
#         col4.write(row[3])  # Skills
        
#         # Add the button in the last column, with a unique key
#         if col5.button(f"Send Email to {row[0]}", key=f"email_button_{index}"):
#             st.write(f"Email sent to {row[0]}")

# def extract_table_from_text(text):
#     print("***************************extract_table_from_text*********************************")
#     print("text :",text)
#     # Extract the table part of the text using regular expression
#     table_match = re.search(r'\|.*\n(\|.*\n)+', text)
#     print("table_match :",table_match)
#     if not table_match:
#         return []
    
#     table_text = table_match.group(0)
#     print("table_text :",table_text)
#     # Split the extracted table text into lines
#     lines = table_text.strip().split('\n')
#     print("lines :",lines)
#     # Extract the header and rows
#     header = [col.strip() for col in lines[0].split('|') if col.strip()]
#     print("header :",header)
#     rows = []
#     for line in lines[2:]:  # Skip the separator line
#         row = [col.strip() for col in line.split('|') if col.strip()]
#         rows.append(row)
#     print("rows :",rows)
#     # Combine header and rows into a table (list of lists)
#     table = [header] + rows
#     print("final table to be returned:",table)
#     return table

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
    header_row = ' | '.join([f'{header:<{col_widths[i]}}' for i, header in enumerate(headers)]) + ' |'
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
    # print("table type", type(table))
    
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
            # selectbox_send_email = st.selectbox(
            #     "Select users to send emails",options=options
            # )
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
                        st.button("Send Email", key=uuid.uuid4(),on_click=lambda:send_emails(table))
                else:
                    st.markdown(message.content)



# Check for existence of table in a text
def checkForTable(tableText):
    table = extract_table_from_text(tableText)
        
    if len(table) > 0:
        unique_file_name = f"{int(time.time())}.csv"
        buf = extract_required_preferred_fields(aiRes)
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
                            st.button("Send Email", key=uuid.uuid4(),on_click=lambda:send_emails(table))

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
        #final_prompt = prompt +" using "+ "***" +get_agent_name(agent_name)+"***"
        #print("the final prompt to go to supervisor :",final_prompt)
        print("the final prompt  :",prompt)
        try:
            if(get_agent_name(agent_name) == "SQLTeam Agent"):
                config={"configurable": {"thread_id": "1"},"recursion_limit":40}
                res = sql_chain.invoke(prompt, config)
                print("AI response :",res["messages"])
                aiRes = res["messages"][-1].content
                # tableLength = extract_table_from_text(aiRes)
                # newAIRes = extract_required_preferred_fields(aiRes)
                # if len(tableLength)>0:
                #     holder.write(aiRes)
                #     st.button("eXport",key=random.randint(1,10000))
                # else:
                #     holder.write(aiRes)
                st.session_state.chat_history.append(AIMessage(content=aiRes, name=get_agent_name(agent_name)))
                checkForTable(aiRes)
                # unique_key = f"download_button_{int(time.time())}"
                # unique_file_name = f"latest_{int(time.time())}.csv"
                # buf = extract_required_preferred_fields(aiRes)
                # if 'chat_history' in st.session_state:
                #     # Check if chat_history is not empty
                #     if st.session_state.chat_history:
                #         # Get the last message in the chat_history
                #         last_message = st.session_state.chat_history[-1]
                        
                #         # Check if the last message is an AIMessage and filter by agent name
                #         if isinstance(last_message, AIMessage) and last_message.name == agent_name_to_filter:
                #             with st.chat_message("AI"):
                #                 st.markdown(last_message.content)
                #                 # st.button("export", key=random.randint(1, 10000))
                #                 st.download_button(
                #                     label="Export as CSV",
                #                     data=buf.getvalue(),
                #                     file_name=unique_file_name,
                #                     mime='text/csv',
                #                     key=unique_key
                #                 )

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
                aiRes = res["messages"][-1].content
                st.session_state.chat_history.append(AIMessage(content=aiRes, name=get_agent_name(agent_name)))
                checkForTable(aiRes)
            else:
                config={"configurable": {"thread_id": "2"},"recursion_limit":40}
                res = github_chain.invoke(question,config)
                print("AI response :",res["messages"])
                aiRes = res["messages"][-1].content
                holder.write(aiRes)            
                st.session_state.chat_history.append(AIMessage(content=aiRes, name=get_agent_name(agent_name)))
        except GraphRecursionError:
            st.info("Graph recursion limit exceeded , try again!")