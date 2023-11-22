import os
import pandas as pd
import openai
import streamlit as st
from dotenv import load_dotenv
from utils import init_agent, query_agent,decode_response
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.llms.openai import AzureOpenAI

load_dotenv()
st.title("Analysis on Food Data for diseases")
st.header("Please upload your Food Data CSV file:")

# Capture the CSV file
data = st.file_uploader("Upload CSV file",type="csv")

query = st.text_area("Enter your query")
button = st.button("Generate Response")

def write_answer(response_dict: dict):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

     # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        print(data)
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

if button:
    agent =  init_agent(data,query)
    response = query_agent(agent=agent, query=query)
    # Decode the response.
    decoded_response = decode_response(response)
    # Write the response to the Streamlit app.
    write_answer(decoded_response)


