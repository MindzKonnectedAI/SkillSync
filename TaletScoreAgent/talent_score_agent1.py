
from typing import Annotated, Literal

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
import utils.create_image_func as create_image_func
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import os
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")


# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def load_markdown(outputFile):

    # Debugging path
    print("Attempting to load file from:", os.path.abspath(outputFile))

    # Check file existence
    if not os.path.exists(outputFile):
        return f"Error: File not found at {outputFile}. Please check the path."

    try:
        loader = UnstructuredMarkdownLoader(outputFile, encoding="utf-8")
        documents = loader.load()
        texts = [d.page_content for d in documents]
        return texts[0] if texts else "Error: No content found in the markdown file."
    except Exception as e:
        return f"Error: An issue occurred while loading the file: {str(e)}"


# def find_matching_point(state):
#     # For demonstration, using a simple response
#     response = llm.invoke("HELLO AI")
#     # Ensure the return value updates the 'messages' field in the state
#     return {"messages": state["messages"] + [AIMessage(content=response.content)]}


import os

def get_file_content(file_name, folder_name):
    # Get the current working directory
    cwd = os.getcwd()

    # Create the full file path
    file_path = os.path.join(cwd, folder_name, file_name)

    try:
        # Open the file in read mode
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read the content
            content = file.read()
        return content
    except FileNotFoundError:
        return f"Error: The file at '{file_path}' was not found."
    except Exception as e:
        return f"An error occurred: {e}"


def find_matching_point(state):
    print("state", state)

    job_description_file_content = get_file_content('outputRuleData.md', 'job_description')
    print(job_description_file_content)

    resume_file_content = get_file_content('outputRuleData.md', 'resume')
    print(resume_file_content)

    # Define a PromptTemplate
    template="""
        Input:

        Job Description:
        {job_description}

        Resume:
        {resume}

        Task:
        Analyze the resume against the job description and categorize the findings into:

        Match: Points where the resume aligns with the job description.
        Not Match: Points where the resume does not meet the job description's requirements.

        ***Never forgot to follow output format.***
        Output Format:
        **Match**
        list of points where the resume aligns with the job description.
        **Not Match**
        list of points where the resume don't aligns with the job description.
        Profile match: Bad, Average or Good.
        Score: score resume align with job description.

        """

    prompt = PromptTemplate(template=template, input_variables=["job_description","resume"])

    matching_points_llm = prompt | llm
    response = matching_points_llm.invoke({"job_description": job_description_file_content, "resume": resume_file_content})
    # print("response: ", response)
    return {"messages": [AIMessage(content=response.content)]}

def check_matching_point(state):
    print("state", state)

    job_description_file_content = get_file_content('outputRuleData.md', 'job_description')
    print(job_description_file_content)

    resume_file_content = get_file_content('outputRuleData.md', 'resume')
    print(resume_file_content)

    template="""

        Task:
        Analyze the resume against the job description and categorize the findings into:
        Must mention titles which detail correspond to which tittle.

        Match: Points where the resume aligns with the job description.
        Not Match: Points where the resume does not meet the job description's requirements.

        ***Never forgot to follow output format.***
        Output Format:
        **Match**
        list of points where the resume aligns with the job description
        **Not Match**
        list of points where the resume don't aligns with the job description

        Input:

        Job Description:
        {job_description}

        Resume:
        {resume}
        """
    

    prompt = PromptTemplate(template=template, input_variables=["job_description","resume"])

    matching_points_llm = prompt | llm
    response = matching_points_llm.invoke({"job_description": job_description_file_content, "resume": resume_file_content})
    # print("response: ", response)
    return {"messages": [AIMessage(content=response.content)]}
    

def talent_score_agent():

# Define a new graph
    workflow = StateGraph(State)

    workflow.add_node("find_matching_point", find_matching_point)
    workflow.add_edge(START, "find_matching_point")
    workflow.add_edge("find_matching_point", END)
    chain = workflow.compile()

    # Provide a valid input state
    input_state = {"messages": []}

    create_image_func.create_graph_image(chain, "talentScore")


    # Invoke the chain with the correct input
    response = chain.invoke(input_state)

    print("response:", response)
    return response

    

# # Define the state for the agent
# class State(TypedDict):
#     messages: Annotated[list[AnyMessage], add_messages]
#     # matching_points: [list[AnyMessage]]
#     # not_matching_points: [list[AnyMessage]]
#     # matching_points: [list[AnyMessage]]
#     # not_matching_points: [list[AnyMessage]]

# # Define a new graph
# workflow = StateGraph(State)


# def load_markdown(outputFile):
#     markdown_path = outputFile
#     # print("markdown_path", markdown_path)
#     loader = UnstructuredMarkdownLoader(markdown_path, encoding="utf-8")
#     documents = loader.load()
#     # print("UnstructuredMarkdownLoaderdocuments", documents)
#     # print(f"length of UnstructuredMarkdownLoader documents loaded: {len(documents)}")

#     texts = [d.page_content for d in documents]

#     # print(f"ltexts: ", texts[0])
#     return texts[0]


# def find_matching_point(state):

#     # job_description = load_markdown("../job_description/outputRuleData.md")
#     # resume = load_markdown("../resume/outputRuleData.md")

#     # prompt = """
#     #     You are given a Job Description and a Resume. Your task is to identify every single piece of information that matches between the two documents. Simply list the matched items as a numbered list without providing additional details or context.

#     #     Input:
#     #     Job Description:
#     #     {{job_description}}

#     #     Resume:
#     #     {{resume}}

#     #     Output:
#     #     [List all matching information as a numbered list]
#     # """
#     # matching_points_llm = prompt | llm
#     # response = matching_points_llm.invoke({"job_description": job_description, "resume": resume})
#     # return {"messages": [AIMessage(content=response.content)]}
#     response = llm.invoke("HELLO AI")
#     return {"messages": [llm.invoke("HELLO AI")]}


# workflow.add_node("find_matching_point", find_matching_point)
# workflow.add_edge(START, "find_matching_point")
# workflow.add_edge("find_matching_point", END)

# chain = workflow.compile()
# # create_image_func.create_graph_image(chain, "talentScore")

# response = chain.invoke("find matching points")

# print("response: " + response)