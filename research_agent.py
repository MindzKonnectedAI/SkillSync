import functools
from langchain_openai.chat_models import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from typing import Annotated, List

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

def research_agent_node(agent_node):

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    research_agent = create_react_agent(llm, tools=[scrape_webpages])
    research_node = functools.partial(agent_node, agent=research_agent, name="WebScraper")

    return research_node