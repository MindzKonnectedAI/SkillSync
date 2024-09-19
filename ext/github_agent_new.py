from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import requests
import functools
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool("query_param_generator")
def query_param_generator(query:str):
    """
    Takes a natural language query as input and returns the appropriate query parameters based on the rules defined in system_message
    """

    query_gen_system = """
    # Strictly follow all the rules below to generate query parameters for GitHub User Search . Ensure all rules are followed to generate accurate search queries.
    # Only return the query parameters , no extra information.
    # Searching users
    You can search for users on GitHub and narrow the results using these user search qualifiers in any combination.

    ## Search only users or organizations
    By default, searching users will return both personal and organizations. However, you can use the type qualifier to restrict search results to personal accounts or organizations only.

    Qualifier                                                           Example
    ---------------------------------------------------------------------------
    type:user	                                                        mike in:name created:<2011-01-01 type:user matches personal accounts named "mike" that were created before 2011.

    type:org	                                                        data in:email type:org matches organizations with the word "data" in their email.

    ## Search by account name, full name, or public email
    You can filter your search to the personal user or organization account name with user or org qualifiers.
    With the in qualifier you can restrict your search to the username (login), full name, public email, or any combination of these. When you omit this qualifier, only the username and email address are searched. For privacy reasons, you cannot search by email domain name.

    Qualifier	                                                        Example
    ---------------------------------------------------------------------------
    user:name	                                                        user:octocat matches the user with the username "octocat".

    org:name	                                                        org:electron type:users matches the Electron organization's account name.

    in:login	                                                        kenya in:login matches users with the word "kenya" in their username.

    in:name	                                                            bolton in:name matches users whose real name contains the word "bolton."

    fullname:firstname lastname	                                        fullname:nat friedman matches a user with the full name "Nat Friedman." Note: This search qualifier is sensitive to spacing.

    in:email	                                                        data in:email matches users with the word "data" in their email.

    ## Search by number of repositories a user owns
    You can filter users based on the number of repositories they own, using the repos qualifier and greater than, less than, and range qualifiers.

    Qualifier	                                                        Example
    ---------------------------------------------------------------------------
    repos:n	                                                            repos:>9000 matches users whose repository count is over 9,000.
    
    name repos:n	                                                    bert repos:10..30 matches users with the word "bert" in their username or real name who own 10 to 30 repositories.

    ## Search by location
    You can search for users by the location indicated in their profile.

    Qualifier	                                                        Example
    ---------------------------------------------------------------------------
    location:LOCATION	repos:1 location:iceland matches users with exactly one repository that live in Iceland.

    ## Search by repository language
    Using the language qualifier you can search for users based on the languages of repositories they own.

    Qualifier	                                                        Example
    ---------------------------------------------------------------------------
    language:LANGUAGE location:LOCATION	                                language:javascript location:russia matches users in Russia with a majority of their repositories written in JavaScript.
    name language:LANGUAGE in:fullname	                                jenny language:javascript in:fullname matches users with JavaScript repositories whose full name contains the word "jenny."

    ## Search by when a personal account was created
    You can filter users based on when they joined GitHub with the created qualifier. This takes a date as its parameter. Date formatting must follow the ISO8601 standard, which is YYYY-MM-DD (year-month-day). You can also add optional time information THH:MM:SS+00:00 after the date, to search by the hour, minute, and second. That's T, followed by HH:MM:SS (hour-minutes-seconds), and a UTC offset (+00:00).
    When you search for a date, you can use greater than, less than, and range qualifiers to further filter results.

    Qualifier	                                                        Example
    ---------------------------------------------------------------------------
    created:YYYY-MM-DD	                                                created:<2011-01-01 matches users that joined before 2011.
    created:>=YYYY-MM-DD	                                            created:>=2013-05-11 matches users that joined at or after May 11th, 2013.
    created:YYYY-MM-DD location:LOCATION	                            created:2013-03-06 location:london matches users that joined on March 6th, 2013, who list their location as London.
    created:YYYY-MM-DD..YYYY-MM-DD name in:login	                    created:2010-01-01..2011-01-01 john in:login matches users that joined between 2010 and 2011 with the word "john" in their username.
    
    ## Search by number of followers
    You can filter users based on the number of followers that they have, using the followers qualifier with greater than, less than, and range qualifiers.

    Qualifier	                                                        Example
    ---------------------------------------------------------------------------
    followers:n	                                                        followers:>=1000 matches users with 1,000 or more followers.
    name followers:n	                                                sparkle followers:1..10 matches users with between 1 and 10 followers, with the word "sparkle" in their name.

    ## Search based on ability to sponsor
    You can search for users and organizations who can be sponsored on GitHub Sponsors with the is:sponsorable qualifier. For more information, see "About GitHub Sponsors."

    Qualifier	                                                        Example
    ---------------------------------------------------------------------------
    is:sponsorable	                                                    is:sponsorable matches users and organizations who have a GitHub Sponsors profile.

    """

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    examples = [
        {
            "question": "Solidity developer located in Gurugram",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is there any programming language ?
    Intermediate answer: Solidity
    Follow up: What is the role?
    Intermediate answer: Solidity developer
    Follow up: What is the location?
    Intermediate answer: Gurugram
    Follow up: Any additional parameters ?
    Intermediate answer: No
    So the final answer is: type:user "Solidity developer" language:solidity location:"Gurugram"
    """,
        },
        {
            "question": "Backend developer skilled in NodeJS located in England",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is there any programming language ?
    Intermediate answer: NodeJS
    Follow up: What is the role?
    Intermediate answer: Backend developer
    Follow up: What is the location?
    Intermediate answer: England
    Follow up: Any additional parameters ?
    Intermediate answer: No
    So the final answer is: type:user "Backend developer" language:NodeJS location:"England"
    """,
        },
        {
            "question": "Frontend developer skilled in ReactJS based in India with 100+ repos and account created before 1 Jan 2019",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is there any programming language ?
    Intermediate answer: ReactJS
    Follow up: What is the role?
    Intermediate answer: Frontend developer
    Follow up: What is the location?
    Intermediate answer: India 
    Follow up: Any additional parameters ?
    Intermediate answer: repos:>100
    So the final answer is: type:user "Frontend developer" language:ReactJS location:"India" repos:>100
    """,
        },
        {
            "question": "Full stack developer from Ukraine with more than 50 followers",
            "answer": """
    Are follow up questions needed here: Yes.
    Follow up: Is there any programming language?
    Intermediate Answer: No
    Follow up: What is the role?
    Intermediate Answer: Full stack developer
    Follow up: What is the location?
    Intermediate Answer: Ukraine
    Follow up: Any additional parameters ?
    Intermediate Answer: followers:>50
    So the final answer is: type:user "Full stack developer" location:"Ukraine" followers:>50
    """,
        },
    ]

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", query_gen_system),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
    chain = final_prompt | llm
    res = chain.invoke({"input": query})

    print("response from query param generator tool :",res.content)
    return res.content



    # query_gen_prompt = ChatPromptTemplate.from_messages(
    #     [("system", query_gen_system),("user","{input}")]
    # )
    # query_gen = query_gen_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # res = query_gen.invoke(input=query)
    # # print("response from query param generator tool :",res.content)
    # return res.content


# @tool("query_param_checker")
# def query_param_checker(query_parameters):
#     """
#     This tool checks if the query parameters generated by query_param_generator are valid.
#     """
#     example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")
#     examples = [
#         {
#             "question": "Solidity developer located in Gurugram",
#             "answer": """
#     Are follow up questions needed here: Yes.
#     Follow up: Is there any programming language ?
#     Intermediate answer: Solidity
#     Follow up: What is the role?
#     Intermediate answer: Solidity developer
#     Follow up: What is the location?
#     Intermediate answer: Gurugram
#     Follow up: Any additional parameters ?
#     Intermediate answer: No
#     So the final answer is: type:user "Solidity developer" language:solidity location:"Gurugram"
#     """,
#         },
#         {
#             "question": "Backend developer skilled in NodeJS located in England",
#             "answer": """
#     Are follow up questions needed here: Yes.
#     Follow up: Is there any programming language ?
#     Intermediate answer: NodeJS
#     Follow up: What is the role?
#     Intermediate answer: Backend developer
#     Follow up: What is the location?
#     Intermediate answer: England
#     Follow up: Any additional parameters ?
#     Intermediate answer: No
#     So the final answer is: type:user "Backend developer" language:NodeJS location:"England"
#     """,
#         },
#         {
#             "question": "Frontend developer skilled in ReactJS based in India with 100+ repos and account created before 1 Jan 2019",
#             "answer": """
#     Are follow up questions needed here: Yes.
#     Follow up: Is there any programming language ?
#     Intermediate answer: ReactJS
#     Follow up: What is the role?
#     Intermediate answer: Frontend developer
#     Follow up: What is the location?
#     Intermediate answer: India 
#     Follow up: Any additional parameters ?
#     Intermediate answer: repos:>100
#     So the final answer is: type:user "Frontend developer" language:ReactJS location:"India" repos:>100
#     """,
#         },
#         {
#             "question": "Full stack developer from Ukraine with more than 50 followers",
#             "answer": """
#     Are follow up questions needed here: Yes.
#     Follow up: Is there any programming language?
#     Intermediate Answer: No
#     Follow up: What is the role?
#     Intermediate Answer: Full stack developer
#     Follow up: What is the location?
#     Intermediate Answer: Ukraine
#     Follow up: Any additional parameters ?
#     Intermediate Answer: followers:>50
#     So the final answer is: type:user "Full stack developer" location:"Ukraine" followers:>50
#     """,
#         },
#     ]

#     prompt = FewShotPromptTemplate(
#         examples=examples,
#         example_prompt=example_prompt,
#         suffix="Question: {input}",
#         input_variables=["input"],
#     )


#     query_check_system = """
#     ## Constructing a search query

#     Each endpoint for searching uses query parameters to perform searches on GitHub. 
#     A query can contain any combination of search qualifiers supported on GitHub. The format of the search query is:
#     SEARCH_KEYWORD_1 SEARCH_KEYWORD_N QUALIFIER_1 QUALIFIER_N
#     For example, if you wanted to search for all repositories owned by defunkt that contained the word GitHub and Octocat in the README file, you would use the following query with the search repositories endpoint:
#     `GitHub Octocat in:readme user:defunkt`

    
#     ## Limitations on query length

#     You cannot use queries that:

#     are longer than 256 characters (not including operators or qualifiers).
#     have more than five AND, OR, or NOT operators.
#     These search queries will return a "Validation failed" error message.

#     ## Query for values greater or less than another value
#     You can use >, >=, <, and <= to search for values that are greater than, greater than or equal to, less than, and less than or equal to another value.

#     Query	                                                            Example
#     ---------------------------------------------------------------------------
#     >n	                                                                cats stars:>1000 matches repositories with the word "cats" that have more than 1000 stars.
#     >=n	                                                                cats topics:>=5 matches repositories with the word "cats" that have 5 or more topics.
#     <n	                                                                cats size:<10000 matches code with the word "cats" in files that are smaller than 10 KB.
#     <=n	                                                                cats stars:<=50 matches repositories with the word "cats" that have 50 or fewer stars.
#     n..*	                                                            cats stars:10..* is equivalent to stars:>=10 and matches repositories with the word "cats" that have 10 or more stars.
#     *..n	                                                            cats stars:*..10 is equivalent to stars:<=10 and matches repositories with the word "cats" that have 10 or fewer stars.

    
#     ## Use quotation marks for queries with whitespace
#     If your search query contains whitespace, you will need to surround it with quotation marks. For example:

#     cats NOT "hello world" matches repositories with the word "cats" but not the words "hello world."
#     build label:"bug fix" matches issues with the word "build" that have the label "bug fix."
#     """

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
    # Ensure the per_page value is within the allowed range
    if per_page > 100:
        per_page = 100
    elif per_page < 1:
        per_page = 1

    # Log query parameters
    print("query_param:", query_param)
    print("per_page:", per_page)

    # Make the API request with query parameters and per_page limit
    res = requests.get(
        f"https://api.github.com/search/users?q={query_param}&per_page={per_page}"
    )

    return res.json()


def query_param_generator_node(agent_node):

    query_param_generator_agent = create_react_agent(llm, tools=[query_param_generator])
    query_param_node = functools.partial(agent_node, agent=query_param_generator_agent, name="query_param_generator")
    return query_param_node

# def query_param_checker_node(agent_node):

#     query_param_checker_agent = create_react_agent(llm, tools=[query_param_checker])
#     query_checker_node = functools.partial(agent_node, agent=query_param_checker_agent, name="query_param_checker")
#     return query_checker_node

def fetch_users_node(agent_node):

    fetch_users_agent = create_react_agent(llm, tools=[fetch_users])
    fetch_node = functools.partial(agent_node, agent=fetch_users_agent, name="fetch_users")
    return fetch_node
