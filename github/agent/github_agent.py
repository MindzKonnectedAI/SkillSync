from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import requests
import functools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

@tool("supported_queries")
def supported_queries(query:str):
    """
    Takes a natural language user query as input and only returns the parameters acceptable by the Github User Search API 
    """
    print("query in supported_queries node :",query)
    pass

@tool("query_param_generator")
def query_param_generator(query:str):
    """
    Takes a natural language query as input and returns the appropriate query parameters based on the rules defined in system_message
    """
    print("query in query_param_generator node :",query)
    pass


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
        print("query_param :", query_param)
        print("per_page :", per_page)
        print("the url :",f"https://api.github.com/search/users?q={query_param}&per_page={per_page}")
        
        # Make the API request with query parameters and per_page limit
        res = requests.get(
            f"https://api.github.com/search/users?q={query_param}&per_page={per_page}"
        )
        print("fetch_users github user search api response :",res.json())
        print("status code of res :",res.status_code)
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

def supported_queries_node(agent_node):
    # give prompt to agent
    query_gen_system = """
    You are a specialized Query Parameters Filter Agent. Your task is to analyze a Job Description written in natural language and extract only the relevant fields supported by the GitHub User Search API. Focus exclusively on fields that are searchable via the API.

    Your output should be a list of strings containing only those fields from the job description that align with the GitHub User Search API's capabilities.

    Below are the fields supported by the GitHub User Search API:
    1. Job Title: Title or role of the candidate
    2. Location: Candidate’s location
    3. Repositories: Number of GitHub repositories the candidate has
    4. Followers: Number of GitHub followers the candidate has
    5. Number of candidates: How many candidates to retrieve
    6. Language: Programming languages used by the candidate
    7. Email: Candidate’s email address
    8. Bio: Information from the candidate's bio
    9. Username: Candidate’s GitHub username
    10. is:sponsorable: Boolean indicating whether the candidate is sponsorable
    """

    # query_gen_system = """
    # You are an expert Query Parameters Filter Agent .
    # Your job is to take a natural language user question and find out the key fields which are supported by the Github User Search API.
    # You will be given a Job Description written in natural language .
    # You need to extract only the fields that are supported by Github User Search API from the job description and return them as a list of strings.
    # ALWAYS need to make sure these key fields can be searched using the Github User Search API .
    # Below are some of the fields supported by the Github User Search API :
    # 1. Job Title : Job title of the candidate 
    # 2. Location : Location of the candidate
    # 3. Repositories : Number of Github Repositories of the candidate
    # 4. Followers : Number of Github Followers of the candidate
    # 5. Number of candidates : Number of candidates you want, if mentioned in the job description
    # 6. Language : Programming language of the candidate
    # 7. Email : Email of the candidate
    # 8. Bio : Bio of the candidate
    # 9. Username : Username of the candidate
    # 10. is:sponsorable : Boolean value denoting if the candidate is sponsorable
    # """
    supported_queries_agent = create_react_agent(llm,tools=[supported_queries],state_modifier=query_gen_system)
    supported_queries_agent_node = functools.partial(agent_node, agent=supported_queries_agent, name="supported_queries")
    return supported_queries_agent_node


def query_param_generator_node(agent_node):
    # give prompt to agent
    query_gen_system = """
    Strictly follow all the rules below to generate query parameters for GitHub User Search . Ensure all rules are followed to generate accurate search queries.
    Only return the query , no extra information .
    Prefix every query you generate with 'type:user' 
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

    Always return ONLY the query parameters you generated WITHOUT any extra text . 
    Below are examples of some output that will be considered correct : 

    1. type:user AI Engineer location:India created:<2019-01-01 repos:>50
    2. type:user Blockchain developer location:Ukraine sort:followers
    3. type:user Full Stack Developer location:USA created:>2020-01-01 
    4. type:user UX Engineer location:Delhi 
    """
    query_param_generator_agent = create_react_agent(llm,tools=[query_param_generator],state_modifier=query_gen_system)
    query_param_node = functools.partial(agent_node, agent=query_param_generator_agent, name="query_param_generator")
    return query_param_node


def fetch_users_node(agent_node):

    fetch_users_agent = create_react_agent(llm, tools=[fetch_users])
    fetch_node = functools.partial(agent_node, agent=fetch_users_agent, name="fetch_users")
    return fetch_node
