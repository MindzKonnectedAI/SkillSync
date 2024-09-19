import streamlit as st
from dotenv import load_dotenv
import os
from typing import Annotated, List
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import BaseMessage, HumanMessage
import operator
from langchain_openai.chat_models import ChatOpenAI
from typing_extensions import TypedDict
import sql.sql_agent_team_supervisor as sql_agent_team_supervisor
import github.github_team_supervisor as github_team_supervisor

import utils.csv_to_sql as csv_to_sql
import utils.create_image_func as create_image_func
import utils.create_team_supervisor_func as create_team_supervisor_func

import os

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_tracking_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_project = os.getenv("LANGCHAIN_PROJECT")

# Set environment variables if needed
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracking_v2
os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint
os.environ["LANGCHAIN_PROJECT"] = langchain_project

# Define the agent node function
def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}

# Initialize the agents
# research_chain = research_team_supervisor_agent.research_team_supervisor_agent(agent_node)
# authoring_chain = document_team_supervisor.document_team_supervisor(agent_node)
github_chain = github_team_supervisor.github_team_supervisor(agent_node)
sql_chain = sql_agent_team_supervisor.sql_agent_team_supervisor()
llm = ChatOpenAI(model="gpt-4o-mini")

# # Define the supervisor node
# supervisor_node = create_team_supervisor_func.create_team_supervisor_func(
#     llm,
#     "You are a supervisor tasked with managing a conversation between the following teams: {team_members}. "
#     "Given the following user request, respond with the worker to act next. "
#     "Carefully analyze the query and then respond with the worker to act next as for a user request , you can only call one worker "
#     "Each worker will perform a task and respond with their results and status. When finished, respond with FINISH.",
#     ["SqlTeam", "GithubTeam"],
# )

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
# super_graph.add_node("ResearchTeam", get_last_message | research_chain | join_graph)
# super_graph.add_node("PaperWritingTeam", get_last_message | authoring_chain | join_graph)
super_graph.add_node("GithubTeam", get_last_message | github_chain | join_graph)
super_graph.add_node("SqlTeam", get_last_message | sql_chain | join_graph)
super_graph.add_node("super_supervisor", supervisor_node)

# super_graph.add_edge("ResearchTeam", "super_supervisor")
# super_graph.add_edge("PaperWritingTeam", "super_supervisor")
super_graph.add_edge("GithubTeam", "super_supervisor")
super_graph.add_edge("SqlTeam", "super_supervisor")
super_graph.add_conditional_edges(
    "super_supervisor",
    lambda x: x["next"],
    {
        "SqlTeam": "SqlTeam",
        "GithubTeam":"GithubTeam",
        "FINISH": END,
    },
)
super_graph.add_edge(START, "super_supervisor")
super_graph = super_graph.compile()

# Streamlit UI
st.title("Multi-Agent Supervisor System")

# Define folder paths
csv_folder = "csv"

# Ensure the csv and db directories exist
os.makedirs(csv_folder, exist_ok=True)

# File uploader widget
st.sidebar.title('File Upload and Processing')
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
agent_name = st.sidebar.radio(
    "Search using",
    ["SQL", "Github"]
)

def get_agent_name(agent_name_here):
    if(agent_name_here=="SQL"):
        return "SQLTeam Agent"
    else:
        return "GithubTeam Agent"

if uploaded_file is not None:
    st.sidebar.write("Processing the uploaded file...")

    # Define the path to save the CSV in the csv folder
    csv_path = os.path.join(csv_folder, uploaded_file.name)

    # Save the uploaded file to the csv folder
    with open(csv_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    st.sidebar.success(f"CSV file saved successfully in {csv_folder} as {uploaded_file.name}")

    # Call the function to save CSV data into the database
    csv_to_sql.save_csv_to_sql(csv_path)


# User input
# user_input = st.text_input("Enter your query:", "Total number of users in SQL Database?")
prompt = st.chat_input("Enter your query")
if prompt is not None and prompt != "":
    with st.chat_message("Human"):
        st.markdown(prompt)
    
    with st.chat_message("AI"):
        # Add a spinner with "Processing your query..." text
        with st.spinner("Processing your query..."):
            # Generate the graph image and save it to the temporary file
            create_image_func.create_graph_image(super_graph, "super_graph")
            final_prompt = prompt +" using "+ get_agent_name(agent_name)
            print("the final prompt to go to supervisor :",final_prompt)
            res = super_graph.invoke(input={"messages": [HumanMessage(content=final_prompt)]})
            print("AI response :",res["messages"])
            st.write(res["messages"][-1].content)            