"""Private synchronous graph: state contains sanitized findings, never drafts."""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .compliance_contracts import CheckResult


class GateState(TypedDict):
    results: list[CheckResult]


def run_gate(service, candidate, context):
    def local(_):
        return {'results': service.local_checks(candidate, context)}

    def route(state):
        return 'finish' if any(f.outcome != 'pass' for r in state['results'] for f in r.findings) else 'review'

    def review(state):
        return {'results': [*state['results'], service.semantic_check(candidate, context)]}

    graph = StateGraph(GateState)
    graph.add_node('local', local)
    graph.add_node('review', review)
    graph.add_edge(START, 'local')
    graph.add_conditional_edges('local', route, {'finish': END, 'review': 'review'})
    graph.add_edge('review', END)
    # No checkpointer, stream API, model callbacks, or private inputs in graph state.
    return graph.compile().invoke({'results': []}, config={'callbacks': []})['results']
