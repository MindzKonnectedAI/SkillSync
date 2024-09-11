from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

import create_image_func
from langchain_core.messages import BaseMessage, HumanMessage
import create_team_supervisor_func

# import csv_to_sql
# path="./csv/B&M (2).csv"
# csv_to_sql.save_csv_to_sql(path)

# import requests
# url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"

# response = requests.get(url)

# if response.status_code == 200:
#     # Open a local file in binary write mode
#     with open("Chinook.db", "wb") as file:
#         # Write the content of the response (the file) to the local file
#         file.write(response.content)
#     print("File downloaded and saved as Chinook.db")
# else:
#     print(f"Failed to download the file. Status code: {response.status_code}")


from langchain_community.utilities import SQLDatabase
# db_path="./db/employee.db"
db_folder = "./db"

# Define the path to the SQLite database file
db_path = os.path.join(db_folder, "employee.db")

db = SQLDatabase.from_uri(f"sqlite:///{db_path}")

print(db.dialect)
print(db.get_usable_table_names())
# db.run("SELECT * FROM  LIMIT 10;")

from typing import Any

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    """
    Create a ToolNode with a fallback to handle errors and surface them to the agent.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI(model="gpt-4o-mini"))
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")

# print(list_tables_tool.invoke(""))

# print(get_schema_tool.invoke("Chinook"))


from langchain_core.tools import tool


@tool
def db_query_tool(query: str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    # print("query11111111111111111111111:", query)
    result = db.run_no_throw(query)
    # n=1
    # if(n==1):
    #     result = db.run_no_throw("SELECT COUNT(*) AS total_users FROM employeeX")
    #     return result
    # else:
    #     result = db.run_no_throw(query)
    #     return result
    # n=2
    if not result:
        return "Error: Query failed. Please rewrite your query and try again."
    return result


# print(db_query_tool.invoke("SELECT * FROM Chinook LIMIT 10;"))
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

query_check_system = """You are a SQL expert with a strong attention to detail.
Double check the SQLite query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.

You will call the appropriate tool to execute the query after running this check."""

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", query_check_system), ("placeholder", "{messages}")]
)

query_check = query_check_prompt | ChatOpenAI(
    model="gpt-4o-mini", temperature=0
).bind_tools([db_query_tool], tool_choice="required")

# query_check = query_check_prompt | ChatOpenAI(
#     model="gpt-4o-mini", temperature=0
# ).bind(functions=[convert_to_openai_function(db_query_tool)], function_call={"name":"db_query_tool"})

# query_check.invoke({"messages": [("user", "SELECT * FROM Chinook LIMIT 10;")]})

from typing import Annotated, Literal

from langchain_core.messages import AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages


# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    team_members: str
    next: str


# Add a node for the first tool call
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }


def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    """
    Use this tool to double-check if your query is correct before executing it.
    """
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}


# Add a node for a model to choose the relevant tables based on the question and available tables
# model_get_schema = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
#     functions=[convert_to_openai_function(get_schema_tool)], function_call={"name":"get_schema_tool"}
# )

model_get_schema = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(
    [get_schema_tool]
)


# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    """Submit the final answer to the user based on the query results."""

    final_answer: str = Field(..., description="The final answer to the user")


# Add a node for a model to generate a query based on the question and schema
query_gen_system = """You are a SQL expert with a strong attention to detail.

Given an input question, output a syntactically correct SQLite query to run, then look at the results of the query and return the answer.

DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.

When generating the query:

Output the SQL query that answers the input question without a tool call.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

If you get an error while executing a query, rewrite the query and try again.

If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

"""
# query_gen_prompt = ChatPromptTemplate.from_messages(
#     [("system", query_gen_system), MessagesPlaceholder(variable_name="messages", optional=True) ]
# )
query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}"), ]
)


query_gen = query_gen_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)
# query_gen = query_gen_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(
#     [SubmitFinalAnswer]
# )

# SubmitFinalAnswer_function = convert_to_openai_function(SubmitFinalAnswer)

# query_gen = query_gen_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
#     functions=[convert_to_openai_function(SubmitFinalAnswer)], function_call={"name":"SubmitFinalAnswer"}
# )


def query_gen_node(state: State):
    # print("query_gen_node_message", state)
    # print("query_genaaaaaaaaaaaa", query_gen)
    message = query_gen.invoke(state)
    print("query_gen_node", message)
    # # print("query_gen_prompt", query_gen_prompt)

    # # Sometimes, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
    # tool_messages = []
    # if message.tool_calls:
    #     for tc in message.tool_calls:
    #         if tc["name"] != "SubmitFinalAnswer":
    #             tool_messages.append(
    #                 ToolMessage(
    #                     content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
    #                     tool_call_id=tc["id"],
    #                 )
    #             )
    # else:
    #     tool_messages = []

    return {"messages": [message]}


# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(
    state: State,
) -> Literal["SqlTeam", "query_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    print("last_message", last_message)
    # If there is a tool call, then we finish
    # if getattr(last_message, "tool_calls", None):
    #     return "SqlTeam"
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "SqlTeam"


# messages = app.invoke(
#     {"messages": [("user", "user which have skills python?")]}
# )
# json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]
# print("json_str", json_str)


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str):
    print("message", message)
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


def sql_agent_team_supervisor() -> str:

    llm = ChatOpenAI(model="gpt-4o-mini")

    def supervisor_agent(state):

        print("state1111111", state["messages"][-1])

        supervisor_agent = create_team_supervisor_func.create_team_supervisor_func(
            llm,
            "You are a supervisor tasked with managing a conversation between the"
            " following workers:  SqlTeam. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH.",
            ["SqlTeam"],
        )
        return supervisor_agent.invoke({"messages": [state["messages"][-1]]})

    def tool_call_fn(state):
        print("Calling_tool_call", state["messages"])
        # tool_call = state["messages"][-1]  # Access the first tool call
        # final_answer = state["messages"][-1].tool_calls[0]["args"]["final_answer"]
        final_answer = state["messages"][-1].content
        print("final_answer", final_answer)
        results = {"messages": [HumanMessage(content=final_answer, name="SqlTeam")]}
        # print("results", results)
        return results

    def finalResponse(state):
        print("state[messages][-1]", state["messages"][-1])

        SubmitFinalAnswerLLM = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(
            [SubmitFinalAnswer], tool_choice="required"
        )
        # print("Calling_tool_call", state["messages"][-1])
        # tool_call = state["messages"][-1]  # Access the first tool call
        # final_answer = state["messages"][-1].tool_calls[0]["args"]["final_answer"]
        # results = SubmitFinalAnswerLLM.invoke({state["messages"][-1].content})
        print("Calling", {"messages": [HumanMessage(content=state["messages"][-1].content)]})
        results = SubmitFinalAnswerLLM.invoke([HumanMessage(content=state["messages"][-1].content)])
        # results = {"messages": [HumanMessage(content=state["messages"][-1].content)]}
        print("results", results)
        return {"messages": results}

    # Define a new graph
    workflow = StateGraph(State)

    workflow.add_node("first_tool_call", first_tool_call)

    # Add nodes for the first two tools
    workflow.add_node(
        "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
    )
    workflow.add_node(
        "get_schema_tool", create_tool_node_with_fallback([get_schema_tool])
    )

    workflow.add_node(
        "model_get_schema",
        lambda state: {
            "messages": [model_get_schema.invoke(state["messages"])],
        },
    )

    workflow.add_node("query_gen", query_gen_node)

    # Add a node for the model to check the query before executing it
    workflow.add_node("correct_query", model_check_query)

    # Add node for executing the query
    workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))
    # workflow.add_node("finalResponse", finalResponse)
    workflow.add_node("SqlTeam", tool_call_fn)
    workflow.add_node("supervisor", supervisor_agent)
    # workflow.add_node("supervisor", final_response)

    # Specify the edges between the nodes
    workflow.add_edge(START, "first_tool_call")
    workflow.add_edge("first_tool_call", "list_tables_tool")
    workflow.add_edge("list_tables_tool", "model_get_schema")
    workflow.add_edge("model_get_schema", "get_schema_tool")
    workflow.add_edge("get_schema_tool", "query_gen")
    workflow.add_edge("query_gen","correct_query")
    # workflow.add_conditional_edges("query_gen","correct_query")
    workflow.add_edge("correct_query", "execute_query")
    workflow.add_conditional_edges("execute_query", should_continue)
    # workflow.add_edge("execute_query", "finalResponse")
    # workflow.add_edge("finalResponse", "query_gen")
    workflow.add_edge("SqlTeam", "supervisor")
    workflow.add_edge("supervisor", END)

    # Compile the workflow into a runnable
    chain = workflow.compile()

    # sql_chain = enter_chain | chain
    sql_chain = enter_chain | chain

    # print("Sql: ", sql_chain)
    create_image_func.create_graph_image(chain, "sql_graph_image3")

    # messages = sql_chain.invoke("user which have skills python?")

    # json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]
    # print("json_str", json_str)
    # print("messageshjjjj", messages)

    # for s in sql_chain.stream(
    #     "Total user have skills python?", {"recursion_limit": 100}
    # ):
    #     if "__end__" not in s:
    #         print(s)
    #         print("---")

    return sql_chain
