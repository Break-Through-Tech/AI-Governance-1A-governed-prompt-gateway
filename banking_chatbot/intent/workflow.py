from langgraph.graph import StateGraph, END
from banking_chatbot.intent.classifier import estimate_llm_cost_and_route
from banking_chatbot.intent.executor import execute_llm_call
from typing import TypedDict, Optional

# update to match other LangGraph setups
class GatewayState(TypedDict):
    query_raw: str
    intent_label: Optional[str]
    intent_confidence: Optional[float]
    system_prompt: Optional[str]
    model_tier: Optional[str]
    model_name: Optional[str]
    estimated_input_tokens: Optional[int]
    estimated_output_tokens: Optional[int]
    projected_cost_usd: Optional[float]
    response_content: Optional[str]
    actual_input_tokens: Optional[int]
    actual_output_tokens: Optional[int]
    actual_cost_usd: Optional[float]

def build_gateway_graph():
    workflow = StateGraph(GatewayState)
    
    # Register nodes
    workflow.add_node("intent_and_cost_node", estimate_llm_cost_and_route)
    workflow.add_node("llm_execution_node", execute_llm_call)
    
    # Wire sequential edges: START -> Classifier -> Executor -> END
    workflow.set_entry_point("intent_and_cost_node")
    workflow.add_edge("intent_and_cost_node", "llm_execution_node")
    workflow.add_edge("llm_execution_node", END)
    
    return workflow.compile()