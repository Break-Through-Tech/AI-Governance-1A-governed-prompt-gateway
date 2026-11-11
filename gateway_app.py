"""Governed Prompt Gateway demo page.

    .venv/bin/python -m streamlit run gateway_app.py

Tab 1: type a question and watch it travel through the LangGraph: each node lights up
as it runs, with one line of narration per step, then the answer, the stage cards and
the decision record.
Tab 2: aggregate numbers over the audit log (block rate, tier split, cost, savings).

This page only *displays* what ``gateway.pipeline.process`` returns and reports through
``on_step``. No routing or cost logic lives here, so the CLI, the tests, and this page
always agree. The drawing comes from ``gateway.walkthrough``, which reads the compiled
graph itself.
"""

from __future__ import annotations

import html
import os
import time

import streamlit as st

from gateway.audit import DEFAULT_LOG_PATH, AuditLog
from gateway.cache_adapter import GatewayCache
from gateway.guardrails_adapter import InputGuardrails
from gateway.config import load_config
from gateway.env import load_dotenv
from gateway.llm import GeminiProvider, SimulatedProvider
from gateway.pipeline import process
from gateway.walkthrough import diagram, narrate
from gateway.workload import load_queries, run_workload, summarize

MERMAID_JS = "https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.12.0/mermaid.min.js"
GRAPH_HEIGHT = 200  # px: the left-to-right diagram scaled to the page width

STAGES = [
    ("input_guardrails", "1. Input guardrails"),
    ("cache", "2. Semantic cache"),
    ("intent", "3. Intent"),
    ("router", "4. Router"),
    ("generation", "5. LLM response"),
    ("output_compliance", "6. Output compliance"),
]
# status value -> (Streamlit box, label). Anything else is shown neutrally.
STATUS_STYLE = {
    "allow": ("success", "ALLOW"), "pass": ("success", "PASS"), "generated": ("success", "GENERATED"),
    "miss": ("info", "MISS"), "hit": ("success", "HIT"), "bypass": ("info", "BYPASS"),
    "escalate": ("warning", "ESCALATE"), "block": ("error", "BLOCK"), "fail": ("error", "FAIL"),
    "error": ("error", "ERROR"),
}


@st.cache_resource(show_spinner=False)
def config():
    return load_config()


@st.cache_resource(show_spinner=False)
def gateway_cache(mode: str) -> GatewayCache:
    return GatewayCache(mode=mode, config=config())


@st.cache_resource(show_spinner=False)
def input_guardrails() -> InputGuardrails:
    return InputGuardrails()


def stage_status(name: str, block: dict | None) -> str | None:
    if block is None:
        return None
    if name == "intent":
        return "pass"
    if name == "router":
        return "pass"
    return block.get("status")


def render_stage(name: str, title: str, block: dict | None) -> None:
    status = stage_status(name, block)
    box, label = STATUS_STYLE.get(status, ("info", (status or "").upper()))
    with st.container(border=True):
        st.markdown(f"**{title}**")
        if block is None:
            st.caption("did not run")
            return
        getattr(st, box)(label)
        if name == "router":
            st.markdown(f"**{block['tier']}** · `{block['model']}`")
            st.caption(block["reason"])
            if block["signals"]:
                st.caption("signals: " + ", ".join(block["signals"]))
        elif name == "generation":
            thinking = f" + {block['thinking_tokens']} thinking" if block.get("thinking_tokens") else ""
            st.caption(f"{block['provider']} · {block['input_tokens']} in / {block['output_tokens']} out{thinking} · {block['elapsed_ms']:.0f} ms")
            if block.get("message"):
                st.caption(block["message"])
        elif name == "intent":
            st.caption(f"{block['label']} ({block['confidence']:.0%}) · template: {block['prompt_template']}")
            if block.get("reason"):
                st.caption(block["reason"])
        elif name == "input_guardrails":
            if block.get("masked"):
                st.caption("masked: " + ", ".join(block["masked"]))
            st.caption(block.get("reason", ""))
            if block.get("model"):
                st.caption(f"{block['model']} · {block.get('input_tokens')} in / {block.get('output_tokens')} out · "
                           f"{block.get('elapsed_ms') or 0:.0f} ms")
        elif name == "output_compliance":
            if block.get("flags"):
                st.caption("failed rules: " + ", ".join(block["flags"]))
            st.caption(block.get("reason", ""))
        else:
            st.caption(block.get("reason", ""))


def render_question(query: str, removed: list[str]) -> None:
    with st.chat_message("user"):
        st.write(query)
        if removed:
            st.caption("For your security, this was removed before your question was processed: "
                       + ", ".join(removed) + ".")


def render_answer(record: dict) -> None:
    """What the customer sees after their question: the answer, or why there is none."""
    status = record["final_status"]
    generation = record.get("generation") or {}
    router = record.get("router") or {}
    text = record.get("response") or generation.get("answer") or ""
    with st.chat_message("assistant"):
        if status == "blocked":
            st.error(f"**Blocked.** {text}")
        elif status == "error":
            st.warning(text)
        else:
            st.write(text)
            if status == "cached":
                st.caption("Served from the response cache. No model was called.")
            elif generation.get("simulated"):
                st.caption("Simulated provider: this is a placeholder, not a model's answer. "
                           "Turn on “Call Gemini for real” in the sidebar for an actual reply.")
            else:
                st.caption(f"Answered by {generation.get('model', '?')} ({router.get('tier', '?')})")


def render_graph(visited: list[str], *, finished: bool) -> None:
    """Draw the compiled graph with this run's path highlighted, via Mermaid in a small iframe."""
    text = html.escape(diagram(visited, finished=finished))
    page = f"""<!doctype html><html><body style="margin:0;background:transparent">
<pre class="mermaid" style="margin:0;font-family:sans-serif">{text}</pre>
<style>
  g.current rect {{ animation: pulse 1s ease-in-out infinite; }}
  @keyframes pulse {{ 50% {{ opacity: .45; }} }}
</style>
<script src="{MERMAID_JS}"
        onerror="document.querySelector('pre').style.whiteSpace='pre';
                 document.body.insertAdjacentHTML('afterbegin','<p>Could not load the diagram library; this is the graph as text.</p>')"></script>
<script>
  if (window.mermaid) mermaid.initialize({{startOnLoad: true, theme: "neutral",
                                          themeVariables: {{fontSize: "22px"}},
                                          flowchart: {{curve: "basis", useMaxWidth: true, nodeSpacing: 30, rankSpacing: 40, padding: 12}}}});
</script>
</body></html>"""
    st.iframe(page, height=GRAPH_HEIGHT)


class Slots:
    """One ``st.empty`` placeholder per element that is redrawn during a run.

    Each placeholder holds exactly one element, so replacing it is reliable. (A single
    placeholder holding a container keeps stale children when the next drawing has fewer.)
    """

    def __init__(self) -> None:
        self.question = st.empty()
        self.answer = st.empty()
        st.subheader("How the gateway decided")
        self.graph = st.empty()
        self.story = st.empty()
        self.status = st.empty()

    def walkthrough(self, visited: list[str], story: list[str], *, finished: bool) -> None:
        """The graph plus one numbered line per node that has run."""
        with self.graph:
            render_graph(visited, finished=finished)
        if story:
            self.story.markdown("\n".join(f"{n}. {line}" for n, line in enumerate(story, 1)))
        else:
            self.story.empty()
        if finished:
            self.status.empty()
        else:
            self.status.caption("Running…")


def render_record(record: dict) -> None:
    """The stage cards, the cost and the raw record for the last run."""
    st.subheader("Stage details")
    columns = st.columns(3)
    for index, (name, title) in enumerate(STAGES):
        with columns[index % 3]:
            render_stage(name, title, record.get(name))

    router = record.get("router")
    if router:
        st.subheader("Cost")
        a, b, c, d = st.columns(4)
        a.metric("Tier", router["tier"])
        b.metric("This request", f"${router['estimated_cost_usd']:.6f}")
        c.metric("No gateway", f"${router['baseline_cost_usd']:.6f}", help=f"Same tokens sent to {router['baseline_tier']}")
        d.metric("Saved", f"${router['savings_usd']:.6f}")

    st.subheader(f"Decision record · final status: {record['final_status']}")
    st.json(record, expanded=False)


def tab_single() -> None:
    cfg = config()
    with st.sidebar:
        st.header("Settings")
        load_dotenv()
        has_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        live = st.toggle("Call Gemini for real", value=has_key, disabled=not has_key,
                         help="Needs GEMINI_API_KEY in .env. Off = simulated provider, no network.")
        if not has_key:
            st.caption("No GEMINI_API_KEY found in .env, so only the simulated provider is available.")
        guard = st.toggle("Input guardrails", value=has_key, disabled=not has_key,
                          help="A policy check on the question. It is one extra Gemini call, "
                               "made before the cache and the model.")
        intents = ["(let the classifier decide)"] + sorted(cfg.rules.intent_tiers)
        intent_choice = st.selectbox("Intent override", intents,
                                     help="Leave this alone to let the intent classifier label the question, "
                                          "or pick a label to force one.")
        intent = None if intent_choice.startswith("(") else intent_choice
        cache_mode = st.selectbox("Response cache", ["semantic", "exact", "semantic-shadow", "off"],
                                  help="Ask the same question twice to see a hit; "
                                       "semantic also matches close paraphrases.")
        log = st.toggle("Write to audit log", value=True)
        pace = st.slider("Pause between graph steps (s)", 0.0, 2.0, 0.6, 0.1,
                         help="Only for watching: slows the walkthrough down so each node is visible. "
                              "Set to 0 for the real speed.")
        st.divider()
        st.caption(f"Price table {cfg.version} · tiers: " + ", ".join(f"{t.name}={t.model}" for t in cfg.tiers.values()))

    st.write("Type a banking question and watch it travel through the graph.")
    query = st.chat_input("Ask a banking question…")

    # Placeholders in page order. During a run they are redrawn after every node; afterwards
    # (and on any later rerun of the page) they show the finished run from session state.
    slots = Slots()

    if query:
        provider = GeminiProvider() if live else SimulatedProvider(cfg.rules.expected_output_tokens)
        visited: list[str] = []
        story: list[str] = []
        with slots.question:
            render_question(query, [])
        slots.walkthrough(visited, story, finished=False)

        def on_step(node: str, state: dict) -> None:
            visited.append(node)
            story.append(narrate(node, state))
            if node == "guardrails":  # from here on the state holds the masked question
                with slots.question:
                    render_question(state["query"], state["input_guardrails"].masked)
            slots.walkthrough(visited, story, finished=False)
            if pace:
                time.sleep(pace)

        record = process(query, intent=intent, provider=provider, config=cfg,
                         audit=AuditLog(DEFAULT_LOG_PATH) if log else None,
                         cache=None if cache_mode == "off" else gateway_cache(cache_mode),
                         guardrails=input_guardrails() if guard else None,
                         on_step=on_step)
        st.session_state["last_record"] = record.to_dict()
        st.session_state["last_walk"] = {"visited": visited, "story": story}

    if "last_record" in st.session_state:
        record = st.session_state["last_record"]
        walk = st.session_state.get("last_walk", {"visited": [], "story": []})
        with slots.question:
            render_question(record["query"], (record.get("input_guardrails") or {}).get("masked") or [])
        with slots.answer:
            render_answer(record)
        slots.walkthrough(walk["visited"], walk["story"], finished=True)
        render_record(record)
    else:
        slots.walkthrough([], [], finished=True)
        slots.status.caption("No question yet. The graph above is the pipeline LangGraph compiled; "
                             "ask something to see the path it takes.")


def tab_workload() -> None:
    cfg = config()
    log = AuditLog(DEFAULT_LOG_PATH)
    st.write("Aggregate numbers over every record in the audit log.")
    left, right = st.columns([3, 1])
    with right:
        if st.button("Run bundled sample (30 queries, simulated)", use_container_width=True):
            with st.spinner("Running workload…"):
                run_workload(load_queries("data/workload_sample.csv"), config=cfg, audit=log)
            st.rerun()
    summary = summarize(log.read(), cfg)
    if summary["requests"] == 0:
        st.info("The audit log is empty. Ask a question in the first tab or run the sample workload.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Requests", summary["requests"])
    b.metric("Model calls", summary["model_calls"])
    c.metric("Cost through gateway", f"${summary['cost']['gateway_usd']:.4f}")
    d.metric("Saved vs no gateway", f"${summary['cost']['savings_usd']:.4f}", f"{summary['cost']['savings_pct']}%")
    left, right = st.columns(2)
    with left:
        st.markdown("**Outcomes**")
        st.bar_chart(summary["outcomes"])
    with right:
        st.markdown("**Tier split**")
        st.bar_chart(summary["tier_split"])
    st.caption(f"Tokens: {summary['tokens']['input']} in / {summary['tokens']['output']} out · price table {summary['config_version']} · log: {log.path}")


def main() -> None:
    st.set_page_config(page_title="Governed Prompt Gateway", page_icon="🛂", layout="wide")
    st.title("Governed Prompt Gateway")
    st.caption("Safety and cost control for LLMs in retail banking support · Team 1A")
    one, two = st.tabs(["Single request", "Workload"])
    with one:
        tab_single()
    with two:
        tab_workload()


if __name__ == "__main__":
    main()
