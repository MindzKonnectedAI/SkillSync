
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
from langchain_core.tools import tool

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


def find_matching_and_not_matching_point(state):
    # print("find_matching_point_state", state)

    # job_description_file_content = get_file_content('outputRuleData.md', 'job_description')
    job_description_file_content = get_file_content('summary.md', 'job_description')
    # print(job_description_file_content)

    # resume_file_content = get_file_content('outputRuleData.md', 'resume')
    resume_file_content = get_file_content('summary.md', 'resume')
    # print(resume_file_content)

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
        """

    prompt = PromptTemplate(template=template, input_variables=["job_description","resume"])

    matching_points_llm = prompt | llm
    response = matching_points_llm.invoke({"job_description": job_description_file_content, "resume": resume_file_content})
    # print("response: ", response)
    return {"messages": response}

def check_matching_point():
    # print("check_matching_point_state", state)

    # template="""

    #     You are an expert assistant with knowledge of resume analysis and job description matching. Your task is to verify if all the matching and not matching points between the resume and job description have been correctly identified. Follow these instructions:

    #     Input Details:

    #     Resume: {resume}
    #     Job Description: {job_description}
    #     Identified Matches and Not Matches: {points}

    #     Your Tasks:

    #     Cross-check the Matches with both the Resume and the Job Description to ensure no matching points are missing.
    #     Cross-check the Not Matches with both the Resume and the Job Description to ensure no mismatched points are missing.
    #     Identify any points in the Resume that could align with the Job Description but are not listed in Matches.
    #     Identify any requirements in the Job Description that are unmet by the Resume and are not listed in Not Matches.

    #     """
    template="""

        You are an expert assistant with knowledge of resume analysis and job description matching. Your task is to verify if all the matching and not matching points between the resume and job description have been correctly identified. Follow these instructions:

            Input Details:

            Resume: {resume}
            Job Description: {job_description}
            Identified Matches and Not Matches: {points}

            Your Tasks:

            1. Cross-check the **Matches** with both the **Resume** and the **Job Description** to ensure no matching points are missing.
            2. Cross-check the **Not Matches** with both the **Resume** and the **Job Description** to ensure no mismatched points are missing.
            3. Identify any points in the **Resume** that could align with the **Job Description** but are not listed in **Matches**.
            4. Identify any requirements in the **Job Description** that are unmet by the **Resume** and are not listed in **Not Matches**.

            If any matching or not matching points are missing, please add them to the corresponding list:

            - Add any missing points where the **Resume** aligns with the **Job Description** to the **Matches** list.
            - Add any missing points where the **Resume** does not meet the **Job Description**'s requirements to the **Not Matches** list.

        """

    prompt = PromptTemplate(template=template, input_variables=["job_description","resume", "points"])

    matching_points_llm = prompt | llm
    # response = matching_points_llm.invoke({"job_description": job_description_file_content, "resume": resume_file_content})
    # # print("response: ", response)
    # return {"messages": [AIMessage(content=response.content)]}
    return matching_points_llm
    

# @tool
def check_matching_and_not_matching_point(state):
    """
    Use this tool to double check if matching and not mathcing point.
    """
    # print("query logged :",state)
    # print("points[messages][-1].content :",state["messages"][-1])
    
    # job_description_file_content = get_file_content('outputRuleData.md', 'job_description')
    job_description_file_content = get_file_content('summary.md', 'job_description')
    # print(job_description_file_content)

    # resume_file_content = get_file_content('outputRuleData.md', 'resume')
    resume_file_content = get_file_content('summary.md', 'resume')
    # print(resume_file_content)

    res = check_matching_point().invoke({
        "job_description": job_description_file_content, 
        "resume": resume_file_content,
        "points": state["messages"][-1].content})
    print("res", res)
    return {"messages": res}


def generate_final_point_tool(state):
    """
    Use this tool to to generate final awsner.
    """
    template="""
        You have a cross-checked document that compares a resume to a job description. Your task is to generate the following:

        A list of matched points, where the resume aligns with the job description.
        A list of not matched points, where the resume does not align with the job description.
        Evaluate the overall profile match: classify it as Bad, Average, or Good based on the alignment.
        Provide the score, which represents how well the resume aligns with the job description.
        Input:
        Cross-Checked Document: Contains the comparison between the resume and job description.
        {cross-checked-document}
        Instructions:
        Extract points that match: List all the points where the resume aligns with the job description.
        Extract points that don't match: List all the points where the resume does not align with the job description.
        Evaluate profile match:
        Bad: If there is little to no alignment.
        Average: If there is moderate alignment.
        Good: If the alignment is strong.
        
        ***Never forgot to follow output format.***
        Output Format:
        **Match**
        list of points where the resume aligns with the job description.
        **Not Match**
        list of points where the resume don't aligns with the job description.
        Profile match: Bad, Average or Good.
        Score: score resume align with job description.

        """
        # Calculate the alignment score: Provide a score representing the percentage of alignment between the resume and job description.

    prompt = PromptTemplate(template=template, input_variables=["cross-checked-document"])

    generateFinalAwsner = prompt | llm

    res = generateFinalAwsner.invoke({"cross-checked-document": state["messages"][-1].content})

    return {"messages": res}

def talent_score_agent():

# Define a new graph
    workflow = StateGraph(State)

    workflow.add_node("find_matching_and_not_matching_point", find_matching_and_not_matching_point)
    workflow.add_node("check_matching_and_not_matching_point", check_matching_and_not_matching_point)
    workflow.add_node("generate_final_point", generate_final_point_tool)
    
    workflow.add_edge(START, "find_matching_and_not_matching_point")
    workflow.add_edge("find_matching_and_not_matching_point", "check_matching_and_not_matching_point")
    workflow.add_edge("check_matching_and_not_matching_point", "generate_final_point")
    # workflow.add_edge("generate_final_point", END)

    chain = workflow.compile()

    # Provide a valid input state
    input_state = {"messages": []}

    create_image_func.create_graph_image(chain, "talentScore")


    # Invoke the chain with the correct input
    response = chain.invoke(input_state)

    # print("response:", response)
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