import functools
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults


def search_agent_node(agent_node):

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tavily_tool = TavilySearchResults(max_results=5)

    search_agent = create_react_agent(llm, tools=[tavily_tool])
    search_node = functools.partial(agent_node, agent=search_agent, name="Search")
    
    return search_node