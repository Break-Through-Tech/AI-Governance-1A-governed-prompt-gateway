"""Turn one run of the graph into a picture and a story, for the demo page.

The graph's shape comes from ``GRAPH.get_graph()``: the same nodes and edges
LangGraph compiled, so the drawing can never drift from the wiring in
``graph.py``. This module adds only what LangGraph does not know: a title for
each node, a word for each exit, where to put each box so the picture is a
straight row, a colour for the nodes a run visited, and one plain sentence per
step. The picture is plain SVG, so it needs no library and works offline.

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
    "guardrails": "1. Guardrails",
    "cache": "2. Cache",
    "intent": "3. Intent",
    "router": "4. Router",
    "generate": "5. LLM",
    "compliance": "6. Compliance",
    "cache_store": "7. Store",
    END: "end",
}

# How a run that leaves from each node into END is shown. The compiled graph has one END;
# the drawing shows it once per exit, as the outcome that exit means.
OUTCOMES = {
    "guardrails": "blocked",
    "cache": "cached",
    "generate": "error",
    "compliance": "withheld",
    "cache_store": "answered",
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

# Colours per node state, and for the edges a run took.
COLOURS = {
    "pending": ("#f4f4f6", "#b8b8c4", "#8a8a99"),   # fill, stroke, text
    "visited": ("#dcefe0", "#1b7f3b", "#0f3d1f"),
    "current": ("#fff1c2", "#c77700", "#5a3600"),
    "outcome": ("#ffffff", "#b8b8c4", "#8a8a99"),
    "reached": ("#1b7f3b", "#1b7f3b", "#ffffff"),
}
EDGE_GREY, EDGE_GREEN = "#9a9aa6", "#1b7f3b"

# Canvas: a straight main row, and a second row for the exits that end a run early.
WIDTH, HEIGHT = 1200, 250
COLUMN, BOX_W, BOX_H, START_W = 128, 106, 50, 72
ROW_MAIN, ROW_EXIT = 60, 170
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
    """The compiled graph as an SVG picture, with this run's nodes and edges highlighted.

    The nodes are read from ``GRAPH.get_graph()`` in the order they were added, which
    for this pipeline is also the order they run: they form one straight row. Every
    edge into END is drawn as a small outcome box: under the node it leaves for the
    early exits, at the end of the row for the normal one. ``visited`` is the list of
    node names in the order they ran so far; until ``finished`` the last one is drawn
    as "current", afterwards the outcome box reached is filled in.
    """
    graph = GRAPH.get_graph()
    path = path_taken(visited, finished=finished)
    current = None if finished or not visited else visited[-1]
    taken = set(zip(path, path[1:]))

    chain = [name for name in graph.nodes if name != END]
    column = {name: index for index, name in enumerate(chain)}
    out: list[str] = []

    def centre(name: str) -> float:
        return 20 + column[name] * COLUMN + BOX_W / 2

    def box(x: float, y: float, width: float, title: str, style: str, *, rounded: bool, extra: str = "") -> None:
        fill, stroke, text = COLOURS[style]
        radius = BOX_H / 2 if rounded else 8
        out.append(f'<rect class="{style}" x="{x - width / 2:.0f}" y="{y - BOX_H / 2:.0f}" width="{width:.0f}" '
                   f'height="{BOX_H}" rx="{radius:.0f}" fill="{fill}" stroke="{stroke}" stroke-width="2"{extra}/>')
        out.append(f'<text x="{x:.0f}" y="{y + 6:.0f}" text-anchor="middle" font-size="15" '
                   f'font-weight="600" fill="{text}">{title}</text>')

    def arrow(x1: float, y1: float, x2: float, y2: float, *, label: str | None, dashed: bool, lit: bool,
              label_at: tuple[float, float]) -> None:
        colour, width = (EDGE_GREEN, 3) if lit else (EDGE_GREY, 1.6)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        out.append(f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="{colour}" '
                   f'stroke-width="{width}"{dash} marker-end="url(#{"lit" if lit else "dim"})"/>')
        if label:
            lx, ly = label_at
            out.append(f'<text x="{lx:.0f}" y="{ly:.0f}" text-anchor="middle" font-size="14" '
                       f'fill="{EDGE_GREEN if lit else "#6b6b78"}">{label}</text>')

    # the main row
    for name in chain:
        style = "current" if name == current else "visited" if name in path else "pending"
        pulse = ' style="animation: pulse 1s ease-in-out infinite"' if style == "current" else ""
        box(centre(name), ROW_MAIN, START_W if name == START else BOX_W, TITLES.get(name, name), style,
            rounded=name == START, extra=pulse)

    # the edges, each placed from the compiled graph's source and target
    for edge in graph.edges:
        lit = (edge.source, edge.target) in taken
        label = EXIT_LABELS.get((edge.source, edge.target))
        sx = centre(edge.source)
        if edge.target != END:
            tx = centre(edge.target)
            half = START_W / 2 if edge.source == START else BOX_W / 2
            arrow(sx + half + 2, ROW_MAIN, tx - BOX_W / 2 - 2, ROW_MAIN, label=label, dashed=edge.conditional,
                  lit=lit, label_at=((sx + tx) / 2, ROW_MAIN - BOX_H / 2 - 8))
            continue
        outcome = OUTCOMES.get(edge.source, "end")
        style = "reached" if lit else "outcome"
        if edge.source == chain[-1]:  # the normal end: last on the main row
            x = sx + COLUMN
            arrow(sx + BOX_W / 2 + 2, ROW_MAIN, x - BOX_W / 2 - 2, ROW_MAIN, label=label, dashed=edge.conditional,
                  lit=lit, label_at=((sx + x) / 2, ROW_MAIN - BOX_H / 2 - 8))
            box(x, ROW_MAIN, BOX_W, outcome, style, rounded=True)
        else:  # an early exit: straight down from the node it leaves
            arrow(sx, ROW_MAIN + BOX_H / 2 + 2, sx, ROW_EXIT - BOX_H / 2 - 2, label=label, dashed=edge.conditional,
                  lit=lit, label_at=(sx + 28, (ROW_MAIN + ROW_EXIT) / 2 + 5))
            box(sx, ROW_EXIT, BOX_W, outcome, style, rounded=True)

    body = "\n".join(out)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" width="100%"
     font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" role="img" aria-label="gateway graph">
  <style>@keyframes pulse {{ 50% {{ opacity: .45; }} }}</style>
  <defs>
    <marker id="dim" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{EDGE_GREY}"/></marker>
    <marker id="lit" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="{EDGE_GREEN}"/></marker>
  </defs>
{body}
</svg>"""
