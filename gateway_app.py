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
from gateway.guardrails_adapter import InputGuardrails
from gateway.config import load_config
from gateway.env import load_dotenv
from gateway.llm import GeminiProvider, SimulatedProvider
from gateway.pipeline import process
from gateway.workload import load_queries, run_workload, summarize

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
        elif name == "cache":
            if block.get("match_type"):
                st.caption(f"Match type · {block['match_type']}")
            if block.get("similarity") is not None:
                st.caption(f"Similarity · {block['similarity']:.3f}")
            st.caption(block.get("reason", ""))
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


def render_response(record: dict) -> None:
    """The exchange as the customer sees it: their question, then the answer or why there is none."""
    status = record["final_status"]
    generation = record.get("generation") or {}
    router = record.get("router") or {}
    text = record.get("response") or generation.get("answer") or ""
    with st.chat_message("user"):
        st.write(record["query"])
        removed = (record.get("input_guardrails") or {}).get("masked") or []
        if removed:
            st.caption("For your security, this was removed before your question was processed: "
                       + ", ".join(removed) + ".")
    with st.chat_message("assistant"):
        if status == "blocked":
            st.error(f"**Blocked.** {text}")
        elif status == "error":
            st.warning(text)
        else:
            st.write(text)
            if status == "cached":
                st.caption("Served from the response cache. Answer generation was skipped.")
            elif generation.get("simulated"):
                st.caption("Simulated provider: this is a placeholder, not a model's answer. "
                           "Turn on “Call Gemini for real” in the sidebar for an actual reply.")
            else:
                st.caption(f"Answered by {generation.get('model', '?')} ({router.get('tier', '?')})")


def render_record(record: dict) -> None:
    render_response(record)
    st.subheader("How the gateway decided")
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
        st.divider()
        st.caption(f"Price table {cfg.version} · tiers: " + ", ".join(f"{t.name}={t.model}" for t in cfg.tiers.values()))

    st.write("Type a banking question and watch each stage decide.")
    query = st.chat_input("Ask a banking question…")
    if query:
        provider = GeminiProvider() if live else SimulatedProvider(cfg.rules.expected_output_tokens)
        with st.spinner("Calling Gemini…" if live else "Running the gateway…"):
            record = process(query, intent=intent, provider=provider, config=cfg,
                             audit=AuditLog(DEFAULT_LOG_PATH) if log else None,
                             cache=None if cache_mode == "off" else gateway_cache(cache_mode),
                             guardrails=input_guardrails() if guard else None)
        st.session_state["last_record"] = record.to_dict()
    if "last_record" in st.session_state:
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
