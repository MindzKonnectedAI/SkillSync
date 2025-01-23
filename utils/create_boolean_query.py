import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.load_markdown import load_markdown

# def load_markdown(file_path):
#     """Load content from markdown file."""
#     with open(file_path, 'r') as file:
#         return file.read()

def generate_boolean_query(jd_content):
    """Generate a boolean query based on JD content using LangChain and OpenAI."""
    # Define the prompt template for generating the boolean query

    print("js", jd_content)
    prompt_template = """
    You are a recruitment assistant. Given the following job description, generate a boolean query that can be used
    to search candidates who match the qualifications and skills required in the job description.

    Job Description:
    {job_description}

    Boolean Query:
    """
    # Create the prompt using the JD content
    prompt = PromptTemplate(input_variables=["job_description"], template=prompt_template)

    # Initialize the LLM (OpenAI) and the chain
    llm = ChatOpenAI(temperature=0)
    chain = prompt | llm

    # Generate the boolean query
    return chain.invoke({"job_description":jd_content})


import os

@st.dialog("Boolean Query")
def booleanQuery():
    # # Get the directory where the current script is located (i.e., the 'util' folder)
    # script_dir = os.path.dirname(__file__)

    # # Go up one level from the 'util' folder to the root, then access the 'data' folder
    # data_dir = os.path.join(script_dir, '..', 'data')  # Go one level up and then into 'data'
    # jd_path = os.path.join(data_dir, 'ai_response.md')
    # requirements = load_markdown("./data/ai_response.md")
    jd_path = "ruleData"

    try:
        # Read the template content
        # Verify the file exists and load the content
        if os.path.exists(jd_path):
            jd_content = load_markdown("./ruleData/outputRuleData.md")
            optimize_jd_content = load_markdown("./data/ai_response.md")
            print("Markdown file loaded successfully!")
                    # Start form
            with st.form(key="prompt_form"):
                # Display the template content in a text area
                st.text_area("Job Description", jd_content, height=300)
                
                # Submit button to generate the boolean query
                submit_button = st.form_submit_button(label="Generate Boolean Query")

                if submit_button:
                    # Generate the boolean query using LangChain and OpenAI
                    boolean_query = generate_boolean_query(optimize_jd_content)
                    st.write("Generated Boolean Query:", boolean_query.content)
        else:
            print(f"File not found: {jd_path}")
        # st.write("Loaded template:", jd_content)  # Debug print to confirm content loading

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


