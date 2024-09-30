from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
import streamlit as st
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini")

def read_prompt(custom_prompt_path, default_prompt_path):

    try:
        # Check if customprompt.md file has text
        with open(custom_prompt_path, "r") as custom_file:
            custom_text = custom_file.read().strip()

            if custom_text:
                return custom_text
            else:
                # If customprompt.md is empty, read from defaultprompt.md
                with open(default_prompt_path, "r") as default_file:
                    default_text = default_file.read().strip()
                    return default_text
    except FileNotFoundError as e:
        # If customprompt.md doesn't exist, read from defaultprompt.md
        with open(default_prompt_path, "r") as default_file:
            default_text = default_file.read().strip()
            return default_text


def check_prompt():
    template = read_prompt("./data/customPrompt.md", "./data/defaultPrompt.md")
    # print("template: ", template)
    return template


def get_prompt():

    template = check_prompt()

    # print("template: ", load_markdown("./data/defaultPrompt.md"))

    prompt = PromptTemplate(template=template, input_variables=["requirements"])

    return prompt


def load_markdown(outputFile):
    markdown_path = outputFile
    # print("markdown_path", markdown_path)
    loader = UnstructuredMarkdownLoader(markdown_path, encoding="utf-8")
    documents = loader.load()
    # print("UnstructuredMarkdownLoaderdocuments", documents)
    # print(f"length of UnstructuredMarkdownLoader documents loaded: {len(documents)}")

    texts = [d.page_content for d in documents]

    # print(f"ltexts: ", texts[0])
    return texts[0]


def retreive_users_fnc():

    prompt = get_prompt()
    requirements = load_markdown("./data/summarizeOutputRuleData.md")
    # print("requirements: ", requirements)

    querychain = prompt | llm
    queryres = querychain.invoke({"requirements": requirements})

    user_query = queryres.content
    # if user_query is not None and user_query != "":
    #     st.session_state.chat_history.append(HumanMessage(content=user_query))

    return user_query