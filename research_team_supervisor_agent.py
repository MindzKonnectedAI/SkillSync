
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
import search_agent
import research_agent
import create_team_supervisor_func
import operator
from typing import Annotated, List
import create_image_func

# ResearchTeam graph state
class ResearchTeamState(TypedDict):
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

def research_team_supervisor_agent(agent_node) -> str:
    
    llm = ChatOpenAI(model="gpt-4o-mini")

    search_node = search_agent.search_agent_node(agent_node)
    # print("search_agent.py",search_agent.search_agent_node(agent_node))
    research_node = research_agent.research_agent_node(agent_node)

    def supervisor_agent(state):
        # print("state111", state["messages"])
    
        supervisor_agent1 = create_team_supervisor_func.create_team_supervisor_func(
            llm,
            "You are a supervisor tasked with managing a conversation between the"
            " following workers:  Search, WebScraper. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH.",
            ["Search", "WebScraper"],
        )
        # print("supervisor_agent10", supervisor_agent1.invoke({"messages": state["messages"]}))
        # print("supervisor_agent10", supervisor_agent1.invoke({"messages": state["messages"]}))
        return supervisor_agent1
    
    research_graph = StateGraph(ResearchTeamState)
    research_graph.add_node("Search", search_node)
    research_graph.add_node("WebScraper", research_node)
    research_graph.add_node("supervisor", supervisor_agent)
    
    # Define the control flow
    research_graph.add_edge("Search", "supervisor")
    research_graph.add_edge("WebScraper", "supervisor")
    research_graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {"Search": "Search", "WebScraper": "WebScraper", "FINISH": END},
    )

    research_graph.add_edge(START, "supervisor")
    chain = research_graph.compile()

    research_chain = enter_chain | chain

    # print("research_chain", research_chain)

    create_image_func.create_graph_image(chain, "research_graph_image2")

    # for s in research_chain.stream(
    #     "when is Taylor Swift's next tour?", {"recursion_limit": 100}
    # ):
    #     if "__end__" not in s:
    #         print(s)
    #         print("---")

    # inputs = {"messages": [HumanMessage(content="when is Taylor Swift's next tour?")]}
    # for s in chain.stream(inputs,config={"configurable": {"thread_id": 42}}, stream_mode="values"):
    #     message = s["messages"][-1]
    #     if isinstance(message, tuple):
    #         print(message)
    #     else:
    #         message.pretty_print()
            
    return research_chain
