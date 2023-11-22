import os
import pandas as pd
import json
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.llms.openai import AzureOpenAI

def init_agent(data, query):
    # Parse the CSV file and create a Pandas DataFrame from its contents.
    df = pd.read_csv(data)

    #llm = AzureOpenAI
    llm = AzureChatOpenAI(
                openai_api_key = os.getenv("OPENAI_API_KEY"),
                openai_api_base = os.getenv("OPENAI_API_BASE"),
                deployment_name=os.getenv("DEPLOYMENT_NAME"),
                openai_api_version="2023-05-15", temperature=0)
    
    # Create a Pandas DataFrame agent.
    agent = create_pandas_dataframe_agent(llm, df, verbose=True)
    return agent

def query_agent(agent, query):
    """
    Query an agent and return the response as a string.

    Args:
        agent: The agent to query.
        query: The query to ask the agent.
        agent: the tool name which should be "python_repl_ast"
        query = query.replace("```", "")

    Returns:
        The response from the agent as a string.
    """
    # Prepare the prompt with query guidelines and formatting
    prompt = (
        """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

        1. If the query requires a table, format your answer like this:
           {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        2. For a plain question that doesn't need a chart or table, your response should be:
           {"answer": "Your answer goes here"}

        For example:
           {"answer": "The food milk is not treated for the disease 'cardiovasular disease'"}

        3. If the answer is not known or available, respond with:
           {"answer": "I do not know."}

        Return all output as a string.

        All strings in "columns" list and data list, should be in double quotes,

        For example: {"columns": ["Food", "Disease", "istreat"], "data": [["milk", "cadiaattack", "no"], ["coffee", "metabolic", "no"]]}

        Lets think step by step.

        Below is the query.
        Query: 
        """
        + query 
    )
    
    # Run the prompt through the agent.
    response = agent.run(prompt)
    # Convert the response to a string.
    return str(response)

def decode_response(response: str) -> dict:
    """This function converts the string response from the model to a dictionary object.

    Args:
        response (str): response from the model

    Returns:
        dict: dictionary with response data
    """
    ### ADDED this loop to overcome the false Dictionaries load/loads creates!  ####
    return json.loads(response)
    
    #if type(response) == str:
    #    return json.loads(response)
    #else:
    #   return response


