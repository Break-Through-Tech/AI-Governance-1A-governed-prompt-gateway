"""The pipeline as a LangGraph state graph: which stage runs next, and nothing else.

Every box in the graph wraps a stage function from ``stages.py``, ``router.py``
or ``llm.py``. Every arrow is either fixed (``add_edge``) or a small function
that reads the state and names the next box (``add_conditional_edges``). No
routing rules, cost maths or model calls live here; those stay in the stage
modules, where they are tested on their own.

    python -m gateway.graph        # print the graph as a Mermaid diagram
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from . import stages
from .config import GatewayConfig
from .llm import Provider
from .prompts import system_prompt
from .router import route
from .schema import (
    CacheDecision,
    ComplianceDecision,
    GenerationResult,
    GuardrailDecision,
    IntentDecision,
    RouterDecision,
)
from .stages import Cache


# ---- state: the one dictionary that travels through a run -----------------------------

class GatewayState(TypedDict, total=False):
    """Keys mirror ``DecisionRecord``. A missing stage key means that stage never ran."""

    # inputs, set once before the run starts
    query: str
    intent_override: str | None
    # working value: produced by the intent node, read by the router and generate nodes
    system_prompt: str
    # one block per stage, written only by that stage's node
    input_guardrails: GuardrailDecision
    cache: CacheDecision
    intent: IntentDecision
    router: RouterDecision
    generation: GenerationResult
    output_compliance: ComplianceDecision
    # written by whichever node ends the run
    final_status: str


# The stage blocks ``process()`` copies from the final state into the record.
STAGE_KEYS = ("input_guardrails", "cache", "intent", "router", "generation", "output_compliance")


# ---- context: the tools a run uses, as opposed to data about the request -------------

@dataclass
class Deps:
    """Passed to ``invoke`` once per run; nodes read it through ``runtime.context``."""

    provider: Provider
    config: GatewayConfig
    cache: Cache | None = None


# ---- nodes: each wraps one stage function and returns only the keys it produced -------

def guardrails_node(state: GatewayState) -> dict:
    decision = stages.check_guardrails(state["query"])
    update: dict = {"input_guardrails": decision}
    if decision.status == "block":
        update["final_status"] = "blocked"  # the run ends here, see after_guardrails
    return update


def cache_node(state: GatewayState, runtime: Runtime[Deps]) -> dict:
    deps = runtime.context
    decision, cached = stages.lookup_cache(state["query"], deps.cache, deps.provider.name)
    update: dict = {"cache": decision}
    if decision.status == "hit":
        update["generation"] = cached  # the stored answer, with zero model tokens
        update["final_status"] = "cached"
    return update


def intent_node(state: GatewayState) -> dict:
    decision = stages.classify_intent(state["query"], state.get("intent_override"))
    return {"intent": decision, "system_prompt": system_prompt(decision.prompt_template)}


def router_node(state: GatewayState, runtime: Runtime[Deps]) -> dict:
    decision = route(
        state["query"],
        intent=state["intent"].label,
        risk=state["input_guardrails"].status,
        system_prompt=state["system_prompt"],
        config=runtime.context.config,
    )
    return {"router": decision}


def generate_node(state: GatewayState, runtime: Runtime[Deps]) -> dict:
    deps = runtime.context
    tier = deps.config.tier(state["router"].tier)
    result = deps.provider.generate(
        state["query"], state["system_prompt"], tier.model, thinking_level=tier.thinking_level,
    )
    update: dict = {"generation": result}
    if result.status != "generated":
        update["final_status"] = "error"
    return update


def compliance_node(state: GatewayState) -> dict:
    decision = stages.check_compliance(state["generation"].answer)
    return {
        "output_compliance": decision,
        "final_status": "answered" if decision.status == "pass" else "blocked",
    }


def cache_store_node(state: GatewayState, runtime: Runtime[Deps]) -> dict:
    """Only answers that passed compliance reach this node, so everything here is reusable."""
    cache = runtime.context.cache
    if cache is not None:
        cache.store(state["query"], state["generation"])
    return {}  # nothing new for the record


# ---- edges: a conditional edge reads the state and names the next node (or END) ------

def after_guardrails(state: GatewayState) -> str:
    return END if state["input_guardrails"].status == "block" else "cache"


def after_cache(state: GatewayState) -> str:
    return END if state["cache"].status == "hit" else "intent"


def after_generate(state: GatewayState) -> str:
    return END if state["generation"].status != "generated" else "compliance"


def after_compliance(state: GatewayState) -> str:
    return "cache_store" if state["output_compliance"].status == "pass" else END


# ---- the graph ---------------------------------------------------------------------

def build_graph():
    builder = StateGraph(GatewayState, context_schema=Deps)

    builder.add_node("guardrails", guardrails_node)
    builder.add_node("cache", cache_node)
    builder.add_node("intent", intent_node)
    builder.add_node("router", router_node)
    builder.add_node("generate", generate_node)
    builder.add_node("compliance", compliance_node)
    builder.add_node("cache_store", cache_store_node)

    builder.add_edge(START, "guardrails")
    # The dict after each edge function lists every answer it can give. LangGraph uses it
    # to draw the diagram; without it every node would appear reachable from every other.
    builder.add_conditional_edges("guardrails", after_guardrails, {"cache": "cache", END: END})
    builder.add_conditional_edges("cache", after_cache, {"intent": "intent", END: END})
    builder.add_edge("intent", "router")
    builder.add_edge("router", "generate")
    builder.add_conditional_edges("generate", after_generate, {"compliance": "compliance", END: END})
    builder.add_conditional_edges("compliance", after_compliance, {"cache_store": "cache_store", END: END})
    builder.add_edge("cache_store", END)

    return builder.compile()


GRAPH = build_graph()


def mermaid() -> str:
    """The graph as a Mermaid diagram, for the README and the slides."""
    return GRAPH.get_graph().draw_mermaid()


if __name__ == "__main__":
    print(mermaid())
