from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import requests
import functools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage


@tool("query_param_generator")
def query_param_generator(query:str):
    """
    Takes a natural language query as input and returns the appropriate query parameters based on the rules defined in system_message
    """
    pass
    # query_gen_system = """
    # Strictly follow all the rules below to generate query parameters for GitHub User Search . Ensure all rules are followed to generate accurate search queries.
    # Only return the query , no extra information .
    # # GitHub User Search - Query Parameter Generation Rules

    # The following guidelines define how to construct query parameters for the GitHub User Search. Ensure all rules are followed to generate accurate search queries.

    # # Overview

    # 1. Search Scope: Applies to public personal GitHub accounts (not organizations).
    # 2. Query Components: Queries can include:
    # 3. Keywords: For general information like usernames, names, emails, and bios.
    # 4. Qualifiers: For searching specific fields.
    # 5. Sort Parameters: Optional but can be added for ordering results.
    # 6. Case Sensitivity: Keywords are case-insensitive.
    # 7. Result Limit: The search returns the first 1000 results, sorted by best match (by default).

    # # Qualifiers & Usage

    # Each qualifier targets a specific field in the GitHub user data. These cannot be mixed with regular keywords.

    # user:NAME: Matches exact usernames.
    # Example: user:braingain

    # in:login: Searches within usernames (non-exact matches allowed).
    # Example: braingain in:login

    # in:email: Searches within users' email addresses.
    # Example: irina in:email

    # in:name: Searches within users' full names.
    # Example: Irina in:name

    # fullname:NAME: Similar to in:name, searches users' full names.
    # Example: fullname:john smith

    # location:NAME: Searches users based on location.
    # Example: location:Boston

    # language:NAME: Finds users based on the primary language of their public repositories.
    # Example: language:python

    # repos:n: Searches users by the number of public repositories.
    # Example: repos:>1000

    # followers:n: Searches users by the number of followers.
    # Example: followers:>1000

    # created:DATE: Finds users by their GitHub account creation date.
    # Example: created:>2020-01-01

    # is:sponsorable: Finds users with a GitHub Sponsors profile.
    # Example: is:sponsorable

    # sort:: Sorts users based on specific attributes.
    # Example: repos:>10000 sort:followers

    # # Boolean Operators
    # You can combine keywords and qualifiers using Boolean operators to refine the search. Follow these rules:

    # AND (implied): Combining two different qualifiers or a qualifier and a keyword automatically implies AND.
    # Example: location:"San Francisco" language:python (Finds users in San Francisco who primarily use Python).

    # OR: Explicitly use OR between keywords or in: qualifiers only. For other qualifiers, using the same qualifier twice implies OR.
    # Example: "front-end developer" OR "ui developer"
    # Example: location:"new jersey" location:"new york" (Finds users in either New Jersey or New York).

    # NOT (-): Use the minus sign (-) to exclude certain terms or qualifiers.
    # Example: location:iceland -location:Reykjavik (Finds users in Iceland but not Reykjavik).

    # # Key Limitations & Constraints

    # Character Limit: Queries must not exceed 256 characters.

    # No Parentheses: Do not use parentheses in queries.

    # AND/OR/NOT Limits: You cannot use more than five AND, OR, or NOT operators in a single query.
    # For example: location:"silicon valley" -language:java -language:c++ -language:python -language:javascript -language:html is valid (5 negations).
    # Special Notes on Combining Operators

    # AND is implied for combining qualifiers and keywords but cannot be used explicitly with certain fields like location, language, etc.

    # OR cannot be used explicitly between different qualifiers.
    # Example: fullname:irina user:braingain is interpreted as AND, while fullname:irina OR user:braingain is invalid.

    # You cannot combine keywords and qualifiers in OR statements.
    # Example: language:java OR "java developer" is invalid, while language:java "java developer" is interpreted as AND.
    # """
    
    # query_gen_prompt = ChatPromptTemplate.from_messages(
    #     [("system", query_gen_system),("user","{input}")]
    # )
    # query_gen = query_gen_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # res = query_gen.invoke(input=query)
    # # print("response from query param generator tool :",res.content)
    # return res.content


# @tool("fetch_users")
# def fetch_users(query_param:str):
#     """Returns a list of Github users based on query parameters generated by the query_param_generator tool."""
#     # get paper page in html
#     print("query_param :",query_param)
#     res = requests.get(
#         f"https://api.github.com/search/users?q={query_param}"
#     )
#     print("response logged :",res.json())
#     return res.json()

@tool("fetch_users")
def fetch_users(query_param: str, per_page: int = 10):
    """Returns a list of Github users based on query parameters and results per page."""

    try:
        # Ensure the per_page value is within the allowed range
        if per_page > 100:
            per_page = 100
        elif per_page < 1:
            per_page = 1

        # Log query parameters
        print("query_param:", query_param)
        print("per_page:", per_page)
        print("the url :",f"https://api.github.com/search/users?q={query_param}&per_page={per_page}")
        # Make the API request with query parameters and per_page limit
        res = requests.get(
            f"https://api.github.com/search/users?q={query_param}&per_page={per_page}"
        )
        print("fetch_users api res :",res)
        
        return res.json()
    except:
        print("inside exception !!!c")
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: please fix your mistakes.",
                    tool_call_id=["id"],
                )
            ]
        }



llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def query_param_generator_node(agent_node):
    # give prompt to agent
    query_gen_system = """
    Strictly follow all the rules below to generate query parameters for GitHub User Search . Ensure all rules are followed to generate accurate search queries.
    Only return the query , no extra information .
    # GitHub User Search - Query Parameter Generation Rules

    The following guidelines define how to construct query parameters for the GitHub User Search. Ensure all rules are followed to generate accurate search queries.

    # Overview

    1. Search Scope: Applies to public personal GitHub accounts (not organizations).
    2. Query Components: Queries can include:
    3. Keywords: For general information like usernames, names, emails, and bios.
    4. Qualifiers: For searching specific fields.
    5. Sort Parameters: Optional but can be added for ordering results.
    6. Case Sensitivity: Keywords are case-insensitive.
    7. Result Limit: The search returns the first 1000 results, sorted by best match (by default).

    # Qualifiers & Usage

    Each qualifier targets a specific field in the GitHub user data. These cannot be mixed with regular keywords.

    user:NAME: Matches exact usernames.
    Example: user:braingain

    in:login: Searches within usernames (non-exact matches allowed).
    Example: braingain in:login

    in:email: Searches within users' email addresses.
    Example: irina in:email

    in:name: Searches within users' full names.
    Example: Irina in:name

    fullname:NAME: Similar to in:name, searches users' full names.
    Example: fullname:john smith

    location:NAME: Searches users based on location.
    Example: location:Boston

    language:NAME: Finds users based on the primary language of their public repositories.
    Example: language:python

    repos:n: Searches users by the number of public repositories.
    Example: repos:>1000

    followers:n: Searches users by the number of followers.
    Example: followers:>1000

    created:DATE: Finds users by their GitHub account creation date.
    Example: created:>2020-01-01

    is:sponsorable: Finds users with a GitHub Sponsors profile.
    Example: is:sponsorable

    sort:: Sorts users based on specific attributes.
    Example: repos:>10000 sort:followers

    # Boolean Operators
    You can combine keywords and qualifiers using Boolean operators to refine the search. Follow these rules:

    AND (implied): Combining two different qualifiers or a qualifier and a keyword automatically implies AND.
    Example: location:"San Francisco" language:python (Finds users in San Francisco who primarily use Python).

    OR: Explicitly use OR between keywords or in: qualifiers only. For other qualifiers, using the same qualifier twice implies OR.
    Example: "front-end developer" OR "ui developer"
    Example: location:"new jersey" location:"new york" (Finds users in either New Jersey or New York).

    NOT (-): Use the minus sign (-) to exclude certain terms or qualifiers.
    Example: location:iceland -location:Reykjavik (Finds users in Iceland but not Reykjavik).

    # Key Limitations & Constraints

    Character Limit: Queries must not exceed 256 characters.

    No Parentheses: Do not use parentheses in queries.

    AND/OR/NOT Limits: You cannot use more than five AND, OR, or NOT operators in a single query.
    For example: location:"silicon valley" -language:java -language:c++ -language:python -language:javascript -language:html is valid (5 negations).
    Special Notes on Combining Operators

    AND is implied for combining qualifiers and keywords but cannot be used explicitly with certain fields like location, language, etc.

    OR cannot be used explicitly between different qualifiers.
    Example: fullname:irina user:braingain is interpreted as AND, while fullname:irina OR user:braingain is invalid.

    You cannot combine keywords and qualifiers in OR statements.
    Example: language:java OR "java developer" is invalid, while language:java "java developer" is interpreted as AND.
    
    # Output format

    Never forget to follow output format when returning answer

    Below are examples of some output formats 
    1. AI Engineer location:India created:<2019-01-01 repos:>50
    
    
    """
    
    # query_gen_prompt = ChatPromptTemplate.from_messages(
    #     [("system", query_gen_system),("user","{input}")]
    # )
    query_param_generator_agent = create_react_agent(llm,tools=[query_param_generator],state_modifier=query_gen_system)
    query_param_node = functools.partial(agent_node, agent=query_param_generator_agent, name="query_param_generator")
    return query_param_node


def fetch_users_node(agent_node):

    fetch_users_agent = create_react_agent(llm, tools=[fetch_users])
    fetch_node = functools.partial(agent_node, agent=fetch_users_agent, name="fetch_users")
    return fetch_node
