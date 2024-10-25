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

# Define the supervisor node
supervisor_node = create_team_supervisor_func.create_team_supervisor_func(
    llm,
    "You are a supervisor tasked with managing a conversation between the following teams: {team_members}. "
    "Given the following user request, respond with the worker to act next. For a user request , you can only call one worker , so carefully analyze the user request and then assign the worker. Each worker will perform a "
    "task and respond with their results and status. When finished, respond with FINISH."
    "When giving results back to the user , also mention the worker that was used.",
    ["SqlTeam", "GithubTeam"],
)

# Define the top-level state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

# Helper functions
def get_last_message(state: State) -> str:
    return state["messages"][-1].content

def join_graph(response: dict):
    return {"messages": [response["messages"][-1]]}

# Define the graph
super_graph = StateGraph(State)
super_graph.add_node("GithubTeam", get_last_message | github_chain | join_graph)
super_graph.add_node("SqlTeam", get_last_message | sql_chain | join_graph)
super_graph.add_node("super_supervisor", supervisor_node)

super_graph.add_edge("GithubTeam", "super_supervisor")
super_graph.add_edge("SqlTeam", "super_supervisor")

def next_step(x):
    return x["next"]

super_graph.add_conditional_edges(
    "super_supervisor",
    next_step,
    {
        "SqlTeam": "SqlTeam",
        "GithubTeam": "GithubTeam",
        "FINISH": END,
    },
)

super_graph.add_edge(START, "super_supervisor")
super_graph = super_graph.compile()


# Streamlit UI
st.title("Multi-Agent Supervisor System")

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
        ["Internally", "Github","LinkedIn","Reddit"]
    )

    if(agent_name=="Internally"):
        # File uploader widget
        # st.sidebar.title('File Upload and Processing')
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


        display_uploaded_files.display_uploaded_files("1","./ruleData",".pdf")

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
    if(agent_name_here=="Internally"):
        return "SQLTeam Agent"
    else:
        return "GithubTeam Agent"


# Conversation History
for message in st.session_state.chat_history:
    if isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# user_input = st.text_input("Enter your query:", "Total number of users in SQL Database?")
prompt = st.chat_input("Enter your query")
if prompt is not None and prompt != "" :
    with st.chat_message("Human"):
        st.markdown(prompt)
    
    st.session_state.chat_history.append(HumanMessage(prompt))
    with st.chat_message("AI"):
        # create_image_func.create_graph_image(super_graph, "super_graph")
        holder = st.empty()
        with st.spinner("Processing your query..."):
            final_prompt = prompt +" using "+ "***" +get_agent_name(agent_name)+"***"
            print("the final prompt to go to supervisor :",final_prompt)
            try:
                res = super_graph.invoke(input={"messages": [HumanMessage(content=final_prompt)]},config={"recursion_limit":40})
                print("AI response :",res["messages"])
                aiRes = res["messages"][-1].content
                holder.write(aiRes)            
                st.session_state.chat_history.append(AIMessage(aiRes))
            except GraphRecursionError:
                st.info("Graph recursion limit exceeded , try again!")

if(buttonVal):
    question = retreive_users.retreive_users_fnc()
    with st.chat_message("Human"):
        st.markdown(question)
    
    st.session_state.chat_history.append(HumanMessage(question))
    with st.chat_message("AI"):
        # create_image_func.create_graph_image(super_graph, "super_graph")
        holder = st.empty()
        with st.spinner("Processing your query..."):
            final_prompt = question +" *** using SQLTeam Agent *** "
            # print("the final prompt to go to supervisor :",final_prompt)
            try:
                res = super_graph.invoke(input={"messages": [HumanMessage(content=final_prompt)]},config={"recursion_limit":40})
                print("AI response :",res["messages"])
                aiRes = res["messages"][-1].content
                holder.write(aiRes)            
                st.session_state.chat_history.append(AIMessage(aiRes))
            except GraphRecursionError:
                st.info("Graph recursion limit exceeded , try again!")
