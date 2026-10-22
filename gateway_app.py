"""Governed Prompt Gateway demo page.

    .venv/bin/python -m streamlit run gateway_app.py

Tab 1: type a question and watch the six stages light up, with the decision record.
Tab 2: aggregate numbers over the audit log (block rate, tier split, cost, savings).

This page only *displays* what ``gateway.pipeline.process`` returns. No routing or
cost logic lives here, so the CLI, the tests, and this page always agree.
"""

from __future__ import annotations

import os

import streamlit as st

from gateway.audit import DEFAULT_LOG_PATH, AuditLog
from gateway.cache_adapter import GatewayCache
from gateway.config import load_config
from gateway.env import load_dotenv
from gateway.llm import GeminiProvider, SimulatedProvider
from gateway.pipeline import process
from gateway.workload import load_queries, run_workload, summarize

STAGES = [
    ("input_guardrails", "1. Input guardrails", "Team A"),
    ("cache", "2. Semantic cache", "Team B"),
    ("intent", "3. Intent", "Team C"),
    ("router", "4. Router", "us"),
    ("generation", "5. LLM response", "us"),
    ("output_compliance", "6. Output compliance", "Team D"),
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


def stage_status(name: str, block: dict | None) -> str | None:
    if block is None:
        return None
    if name == "intent":
        return "pass"
    if name == "router":
        return "pass"
    return block.get("status")


def render_stage(name: str, title: str, owner: str, block: dict | None) -> None:
    status = stage_status(name, block)
    box, label = STATUS_STYLE.get(status, ("info", (status or "").upper()))
    with st.container(border=True):
        st.markdown(f"**{title}**  \n<small>{owner}</small>", unsafe_allow_html=True)
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
        else:
            st.caption(block.get("reason", ""))


def render_record(record: dict) -> None:
    columns = st.columns(3)
    for index, (name, title, owner) in enumerate(STAGES):
        with columns[index % 3]:
            render_stage(name, title, owner, record.get(name))

    router = record.get("router")
    if router:
        st.subheader("Cost")
        a, b, c, d = st.columns(4)
        a.metric("Tier", router["tier"])
        b.metric("This request", f"${router['estimated_cost_usd']:.6f}")
        c.metric("No gateway", f"${router['baseline_cost_usd']:.6f}", help=f"Same tokens sent to {router['baseline_tier']}")
        d.metric("Saved", f"${router['savings_usd']:.6f}")

    generation = record.get("generation")
    if generation and generation["status"] == "generated":
        st.subheader("Answer")
        if generation["provider"] == "cache":
            st.caption(generation["message"] + " No model was called.")
        elif generation["simulated"]:
            st.caption("Simulated provider: no model was called.")
        st.write(generation["answer"])

    st.subheader(f"Decision record · final status: {record['final_status']}")
    st.json(record, expanded=False)


def tab_single() -> None:
    cfg = config()
    with st.sidebar:
        st.header("Settings")
        load_dotenv()
        has_key = bool(os.environ.get("GEMINI_API_KEY", "").strip())
        live = st.toggle("Call Gemini for real", value=False, disabled=not has_key,
                         help="Needs GEMINI_API_KEY in .env. Off = simulated provider, no network.")
        if not has_key:
            st.caption("No GEMINI_API_KEY found in .env, so only the simulated provider is available.")
        intents = ["(let the classifier decide)"] + sorted(cfg.rules.intent_tiers)
        intent_choice = st.selectbox("Intent override", intents,
                                     help="The intent classifier is a stub, so you can supply the label here.")
        intent = None if intent_choice.startswith("(") else intent_choice
        cache_mode = st.selectbox("Response cache", ["semantic", "exact", "semantic-shadow", "off"],
                                  help="Joohye's cache. Ask the same question twice to see a hit; "
                                       "semantic also matches close paraphrases.")
        log = st.toggle("Write to audit log", value=True)
        st.divider()
        st.caption(f"Price table {cfg.version} · tiers: " + ", ".join(f"{t.name}={t.model}" for t in cfg.tiers.values()))

    st.write("Type a banking question and watch each stage decide.")
    examples = {
        "Simple": "How do I reset my PIN?",
        "Numeric": "If I deposit $500 a month at 4% interest how much will I have after 5 years?",
        "Comparison": "What is the difference between a current account and a savings account and which should I pick?",
    }
    cols = st.columns(len(examples))
    picked = None
    for col, (label, text) in zip(cols, examples.items()):
        if col.button(label, use_container_width=True):
            picked = text
    typed = st.chat_input("Ask a banking question…")
    query = picked or typed
    if query:
        provider = GeminiProvider() if live else SimulatedProvider(cfg.rules.expected_output_tokens)
        with st.spinner("Calling Gemini…" if live else "Running the gateway…"):
            record = process(query, intent=intent, provider=provider, config=cfg,
                             audit=AuditLog(DEFAULT_LOG_PATH) if log else None,
                             cache=None if cache_mode == "off" else gateway_cache(cache_mode))
        st.session_state["last_record"] = record.to_dict()
    if "last_record" in st.session_state:
        st.markdown(f"**Query:** {st.session_state['last_record']['query']}")
        render_record(st.session_state["last_record"])


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
