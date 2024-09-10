import github_agent
from typing import Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage
import operator
from typing_extensions import TypedDict
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from create_team_supervisor_func import create_team_supervisor_func
from langgraph.graph import END, StateGraph, START
import create_image_func

# GithubTeamState graph state
class GithubTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str


# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results

def github_team_supervisor(agent_node)-> str:

    llm = ChatOpenAI(model="gpt-4o-mini")

    query_param_node = github_agent.query_param_generator_node(agent_node)

    fetch_node = github_agent.fetch_users_node(agent_node)

    def supervisor_agent(state):

        github_supervisor_agent = create_team_supervisor_func(
            llm,
            "You are a supervisor tasked with managing a conversation between the"
            " following workers:  query_param_generator, fetch_users. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH.",
            ["query_param_generator", "fetch_users"],
        )
        return github_supervisor_agent

    github_graph = StateGraph(GithubTeamState)
    github_graph.add_node("query_param_generator", query_param_node)
    github_graph.add_node("fetch_users", fetch_node)
    github_graph.add_node("supervisor", supervisor_agent)

    # Define the control flow
    github_graph.add_edge("query_param_generator", "supervisor")
    github_graph.add_edge("fetch_users", "supervisor")
    github_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"query_param_generator": "query_param_generator", "fetch_users": "fetch_users", "FINISH": END},
    )


    github_graph.add_edge(START, "supervisor")
    chain = github_graph.compile()

    github_chain = enter_chain | chain
    create_image_func.create_graph_image(chain, "github_graph_image")
    return github_chain
