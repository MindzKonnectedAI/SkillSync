from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
import os
from langchain_core.messages import AIMessage
import ast

llm = ChatOpenAI(model="gpt-4o-mini")

def read_prompt(custom_prompt_path, default_prompt_path):

    try:
        # Check if customprompt.md file has text
        with open(custom_prompt_path, "r",encoding="utf-8") as custom_file:
            custom_text = custom_file.read().strip()

            if custom_text:
                return custom_text
            else:
                # If customprompt.md is empty, read from defaultprompt.md
                with open(default_prompt_path, "r",encoding="utf-8") as default_file:
                    default_text = default_file.read().strip()
                    return default_text
    except FileNotFoundError as e:
        # If customprompt.md doesn't exist, read from defaultprompt.md
        with open(default_prompt_path, "r",encoding="utf-8") as default_file:
            default_text = default_file.read().strip()
            return default_text


def check_prompt():
    template = read_prompt("./data/customPrompt.md", "./data/defaultPrompt.md")
    # print("template: ", template)
    return template


def get_prompt():

    template = check_prompt()

    # print("template: ", load_markdown("./data/defaultPrompt.md"))

    prompt = PromptTemplate(template=template, input_variables=["requirements","column_headers","example_rows"])

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

# Add a node for the first tool call
def first_tool_call() -> dict[str, list[AIMessage]]:
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "sql_db_schema",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

def string_to_tuple(string):
    try:
        # Convert the string to a tuple
        result = ast.literal_eval(string)
        
        # Ensure the result is a tuple
        if isinstance(result, tuple):
            return result
        else:
            raise ValueError("The string does not represent a tuple.")
    except (SyntaxError, ValueError) as e:
        print(f"Error converting string to tuple: {e}")
        return None


def retreive_users_fnc():

    prompt = get_prompt()
    
    db_folder = "./db"

    # Define the path to the SQLite database file
    db_path = os.path.join(db_folder, "employee.db")

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tableName = "employee"
    valHere = db.get_context()
    table_schema = db.get_table_info([tableName])
    # Extract just the column names (headers)
    # column_headers = [col.split()[0] for col in table_schema.splitlines() if col]
    # Run the PRAGMA query to get the column headers
    query = f"PRAGMA table_info({tableName});"
    result = db.run(query)
    print("result :",result)
    newresult = ast.literal_eval(result)
    print("newresult :",newresult)
    print("type of newresult :",type(newresult))
    print("newresult logged :",newresult)
    # # Extract the column names from the result
    column_headers = [row[1] for row in newresult]  

    print(f"Column Headers for {tableName}: {column_headers}")

    print("table_schema :",table_schema)
    # Extract only the example rows
    example_rows = table_schema.split("/*")[-1].strip("*/").strip()
    print("example_rows :",example_rows)
    print("column_headers :",column_headers)
    print("valHere :",valHere)
    
    # valhere = db.get_table_info_no_throw(
    #         [t.strip() for t in tableName.split(",")]
    #     )

    # tools = toolkit.get_tools()
    # get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    # model_get_schema = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(
    #     [get_schema_tool]
    # )
    # valHere = model_get_schema.invoke([
    #         AIMessage(
    #             content="",
    #             tool_calls=[
    #                 {
    #                     "name": "sql_db_schema",
    #                     "args": {},
    #                     "id": "tool_abcd123",
    #                 }
    #             ],
    #         )
    #     ])
    # print("valHere :",valHere)
    requirements = load_markdown("./data/summarizeOutputRuleData.md")
    # print("requirements: ", requirements)

    querychain = prompt | llm
    queryres = querychain.invoke({"requirements": requirements,"column_headers":column_headers,"example_rows":example_rows})

    user_query = queryres.content
    # if user_query is not None and user_query != "":
    #     st.session_state.chat_history.append(HumanMessage(content=user_query))

    return user_query