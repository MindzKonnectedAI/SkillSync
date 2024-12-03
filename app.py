from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
import streamlit as st
import os
from typing import Annotated, List
from langgraph.graph import END, StateGraph, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import operator
from langchain_openai.chat_models import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,FewShotChatMessagePromptTemplate
import utils.create_image_func as create_image_func
from langchain_core.output_parsers.json import JsonOutputParser
import utils.upload_job_description as upload_job_description
from langgraph.errors import GraphRecursionError
import utils.display_uploaded_files as display_uploaded_files
import time
import re
import json
import pandas as pd
import io
import time
import random
import uuid
import TaletScore.talentScore as talentScore

# Access the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_tracking_v2 = os.getenv("LANGCHAIN_TRACING_V2")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")
langchain_project = os.getenv("LANGCHAIN_PROJECT")
# llamaparse_api_key = "llx-8MMHGFCJ5PKqyfZM6h5D8epMtjzG4OEOe6lMCEOvgu67YgIt"

# Set environment variables if needed
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracking_v2
os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint
os.environ["LANGCHAIN_PROJECT"] = langchain_project
# os.environ["LLAMAPARSE_API_KEY"] = llamaparse_api_key

### Statefully manage chat history ###
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Streamlit UI
st.title("Talent Score")

def retrive():
    pass

buttonVal = False   
# if(view=="User"):

# File uploader widget
with st.sidebar.form("jd_pdf_upload_form", clear_on_submit=True):
    uploaded_jd_file = st.file_uploader(
        "Upload your Job Description", type=["pdf"], key="pdf_uploader"
    )
    file_submitted = st.form_submit_button("Submit")

if file_submitted and (uploaded_jd_file is not None):
    container = st.empty()
    container.write("Processing the uploaded file...")
    upload_job_description.upload_rule_data(uploaded_jd_file,container,"./job_description", "./view_jd")
    time.sleep(2)
    container.empty()
    st.rerun()

display_uploaded_files.display_uploaded_files("1","./view_jd",".pdf")

with st.sidebar.form("resume_pdf_upload_form", clear_on_submit=True):
    uploaded_resume_file = st.file_uploader(
        "Upload your resume", type=["pdf"], key="resume_file"
    )
    file_submitted = st.form_submit_button("Submit")

if file_submitted and (uploaded_resume_file is not None):
    container = st.empty()
    container.write("Processing the uploaded file...")
    upload_job_description.upload_rule_data(uploaded_resume_file,container,"./resume","./view_resume")
    time.sleep(2)
    container.empty()
    st.rerun()

display_uploaded_files.display_uploaded_files("2","./view_resume",".pdf")

buttonVal = st.sidebar.button(
    "Check Talent Match",
    on_click=retrive,  # Note the lack of parentheses here
    key="retreive_users",
)

# Conversation History
if 'chat_history' in st.session_state:
    for index, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage) and message.name == agent_name_to_filter:
            with st.chat_message("Human"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage) and message.name == agent_name_to_filter:
            with st.chat_message("AI"):
                # print("**********************************************************************")
                # print("message.content in chat history :",message.content)
                table = extract_table_from_text(message.content)
                # print("table in chat history :",table)
                if len(table) > 0:
                    st.markdown(message.content)
                    buf = extract_required_preferred_fields(message.content)
                    unique_file_name = f"{int(time.time())}.csv"
                    col1,col2 = st.columns([0.2,0.8])
                    with col1:
                        st.download_button(
                            label="Export as CSV",
                            data=buf.getvalue(),
                            file_name=unique_file_name,
                            mime='text/csv',
                            key=uuid.uuid4()
                        )
                    with col2:
                        st.button("Send Email", key=uuid.uuid4(),on_click=send_emails, args=(table,))
                else:
                    st.markdown(message.content)

# prompt = st.chat_input("Find your next superstar")

if(buttonVal):
    holder = st.empty()
    with st.spinner("Processing your query..."):
        try:
            aiRes = talentScore.talent_score_agent()
            st.session_state.chat_history.append(AIMessage(content=aiRes["messages"][-1].content))
            holder.write(aiRes["messages"][-1].content) 
        except GraphRecursionError:
            st.info("Graph recursion limit exceeded , try again!")