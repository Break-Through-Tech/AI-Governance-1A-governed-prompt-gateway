# banking_chatbot/intent/workflow.py

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from banking_chatbot.intent.classifier import estimate_llm_cost_and_route

# Shared state schema flowing through the graph nodes
class GatewayState(TypedDict):
    query_raw: str
    query_masked: Optional[str]
    guardrail_status: Optional[str]
    intent_label: Optional[str]
    intent_confidence: Optional[float]
    system_prompt: Optional[str]
    model_tier: Optional[str]
    model_name: Optional[str]
    estimated_input_tokens: Optional[int]
    estimated_output_tokens: Optional[int]
    projected_cost_usd: Optional[float]
    cache_ttl_seconds: Optional[int]
    cacheable_intent: Optional[bool]

def build_gateway_graph():
    workflow = StateGraph(GatewayState)
    
    # Register intent prediction & token cost estimation node
    workflow.add_node("intent_and_cost_node", estimate_llm_cost_and_route)
    
    # Single node graph pipeline
    workflow.set_entry_point("intent_and_cost_node")
    workflow.add_edge("intent_and_cost_node", END)
    
    return workflow.compile()