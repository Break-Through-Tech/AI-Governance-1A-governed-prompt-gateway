

import os
from typing import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["NEMOGUARDRAILS_LLM_FRAMEWORK"] = "langchain"
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
from nemoguardrails import RailsConfig
from nemoguardrails.integrations.langchain.runnable_rails import RunnableRails
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# this is the API key so that the guardrails can access the model

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = API_KEY

# these are the configurations for the guardrails, including the model and the input flows
config = RailsConfig.from_path("config")
guardrails = RunnableRails(config=config)
# this is the model that will be used for the guardrails
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    max_output_tokens=1024
)
# this is the state graph used in langgraph to define the input and output of the guardrails
class Agent_state(TypedDict):
    input_text: str
    is_safe: bool
    output_text: str

async def guardrail_node(state:Agent_state):
    # this is the input text that will be passed to the guardrails
    user_input = state["input_text"]
# getst the result of the guardrails, which will be a string that indicates whether the input is safe or not
    result = await guardrails.ainvoke(user_input)
# this is the output of the guardrails, which will be a string that indicates whether the input is safe or not
    output_text = result if isinstance(result, str) else str(result)
    if "can not assist" in output_text.lower() or "sorry" in output_text.lower():
        return {
        "is_safe":False,
        "output_text":output_text
      }
    return{ 
    "is_safe":True,
    "output_text": ""
    }

async def generator_node(state:Agent_state):
    """
    this funciton is responsible for generating a response to the users text after it has passed the guardrails 
    """
    prompt =[
            
        {"role": "system", "content": "You are a helpfull Bank Assistant."},
        {"role": "user", "content": state["input_text"]}


    ]
    """ 
    the prompt: lets us custimaize how the model will response 
    as well as it passess the users input in to the model 
    """
    # the response from the model will be stored in the response variable

    response = await llm.ainvoke(prompt)

    return{
        "output_text": response.content
    }


def check_saftey_condition(state:Agent_state)-> str:
    if state["is_safe"]:
        return "generator_node"
    return END


"""
we are making our our stategraph that will communicate with all of 
the other nodes that we created so it will communicate weatehr 
a node has decided that the prompt is toxict or a jailbreak 

"""

workflow = StateGraph(Agent_state)
workflow.add_node("guardrail_node", guardrail_node)
workflow.add_node("generator_node", generator_node)

workflow.add_edge(START,"guardrail_node")
workflow.add_conditional_edges(
    "guardrail_node",
    check_saftey_condition,
    {
     "generator_node": "generator_node" , 
      END:END
    }


)
workflow.add_edge("generator_node", END)

app = workflow.compile()









