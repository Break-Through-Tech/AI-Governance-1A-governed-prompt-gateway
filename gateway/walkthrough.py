"""Turn one run of the graph into a picture and a story, for the demo page.

The graph's shape comes from ``GRAPH.get_graph()``: the same nodes and edges
LangGraph compiled, so the drawing can never drift from the wiring in
``graph.py``. This module adds only what LangGraph does not know: a title for
each node, a word for each exit, a colour for the nodes a run visited, and one
plain sentence per step.

Nothing here touches Streamlit. The page feeds ``process(on_step=...)`` events
into ``narrate`` and ``diagram`` and displays what comes back, so both can be
tested without a browser.
"""

from __future__ import annotations

from collections.abc import Sequence

from langgraph.graph import END, START

from .graph import GRAPH, GatewayState

TITLES = {
    START: "start",
    "guardrails": "1. Input guardrails",
    "cache": "2. Cache lookup",
    "intent": "3. Intent",
    "router": "4. Router",
    "generate": "5. LLM response",
    "compliance": "6. Output compliance",
    "cache_store": "7. Store answer",
    END: "end",
}

# What each conditional exit means, keyed (from node, to node) exactly as the compiled graph has them.
EXIT_LABELS = {
    ("guardrails", "cache"): "allow",
    ("guardrails", END): "block",
    ("cache", "intent"): "miss",
    ("cache", END): "hit",
    ("generate", "compliance"): "generated",
    ("generate", END): "error",
    ("compliance", "cache_store"): "pass",
    ("compliance", END): "fail",
}

STYLES = (
    "classDef pending fill:#f4f4f6,stroke:#b8b8c4,color:#8a8a99",
    "classDef visited fill:#dcefe0,stroke:#1b7f3b,color:#0f3d1f",
    "classDef current fill:#fff1c2,stroke:#c77700,stroke-width:3px,color:#5a3600",
)
TAKEN_EDGE = "stroke:#1b7f3b,stroke-width:3px"


def _why(reason: str) -> str:
    """A stage's reason without its final full stop, so it can sit inside a sentence."""
    return reason.strip().rstrip(".")


def narrate(node: str, state: GatewayState) -> str:
    """One sentence about what ``node`` just decided, read from the state it left behind."""
    if node == "guardrails":
        g = state["input_guardrails"]
        masked = f" Removed before anything else ran: {', '.join(g.masked)}." if g.masked else ""
        if g.status == "block":
            return f"Input guardrails blocked the question: {_why(g.reason)}.{masked} The run stops here."
        return f"Input guardrails let the question through: {_why(g.reason)}.{masked} Next: the cache."

    if node == "cache":
        c = state["cache"]
        if c.status == "hit":
            score = f", similarity {c.similarity:.2f}" if c.similarity is not None else ""
            return f"Cache hit ({c.match_type}{score}): the stored answer is reused and no model is called."
        if c.status == "bypass":
            return f"Cache skipped: {_why(c.reason)}. Next: the intent classifier."
        return f"Cache {c.status}: {_why(c.reason)}. Next: the intent classifier."

    if node == "intent":
        i = state["intent"]
        return (f"Intent: {i.label} ({i.confidence:.0%} confidence), so the '{i.prompt_template}' "
                f"system prompt is used. Next: the router.")

    if node == "router":
        r = state["router"]
        return f"Router picked {r.tier} ({r.model}): {_why(r.reason)}. Next: the model."

    if node == "generate":
        g = state["generation"]
        if g.status != "generated":
            return f"The model call failed: {_why(g.message or g.status)}. The run stops here."
        if g.simulated:
            return "The simulated provider returned a placeholder answer (no network). Next: output compliance."
        return (f"{g.model} answered in {g.elapsed_ms:.0f} ms "
                f"({g.input_tokens} in / {g.output_tokens} out tokens). Next: output compliance.")

    if node == "compliance":
        c = state["output_compliance"]
        if c.status == "pass":
            ran = f" ({', '.join(c.checks)})" if c.checks else ""
            return f"Output compliance passed{ran}, so the answer is released. Next: store it in the cache."
        failed = f" ({', '.join(c.flags)})" if c.flags else ""
        return f"Output compliance failed{failed}: the answer is withheld. The run stops here."

    if node == "cache_store":
        c = state["cache"]
        if c.status == "bypass":
            return f"Nothing stored: {_why(c.reason)}. Done."
        return "The answer is stored in the cache for the next similar question. Done."

    return f"{node} ran."


def path_taken(visited: Sequence[str], *, finished: bool) -> list[str]:
    """The nodes a run passed through, with START in front and END behind once it has finished."""
    if not visited:
        return []  # nothing has run yet, not even the first step
    return [START, *visited, *([END] if finished else [])]


def diagram(visited: Sequence[str] = (), *, finished: bool = False) -> str:
    """The compiled graph as a Mermaid flowchart, with this run's nodes and edges highlighted.

    ``visited`` is the list of node names in the order they ran so far. Until
    ``finished`` the last one is drawn as "current"; afterwards the edge into
    END is highlighted too. With no visited nodes, the whole graph is grey.
    """
    graph = GRAPH.get_graph()
    path = path_taken(visited, finished=finished)
    current = None if finished or not visited else visited[-1]
    taken = set(zip(path, path[1:]))

    lines = ["graph LR;"]
    for name in graph.nodes:
        title = TITLES.get(name, name)
        shape = f'(["{title}"])' if name in (START, END) else f'["{title}"]'
        style = "current" if name == current else "visited" if name in path else "pending"
        lines.append(f"    {name}{shape}:::{style}")
    for index, edge in enumerate(graph.edges):
        label = EXIT_LABELS.get((edge.source, edge.target))
        arrow = ("-.->" if edge.conditional else "-->") + (f"|{label}|" if label else "")
        lines.append(f"    {edge.source} {arrow} {edge.target}")
        if (edge.source, edge.target) in taken:
            lines.append(f"    linkStyle {index} {TAKEN_EDGE}")
    lines.extend(f"    {style}" for style in STYLES)
    return "\n".join(lines)
