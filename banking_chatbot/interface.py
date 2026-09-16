"""A session-local chat view of the existing retriever; no model or generated text."""

import hashlib
import re
from pathlib import Path
from time import perf_counter

import streamlit as st

from .data import DEFAULT_CLEAN
from .retrieval import DEFAULT_THRESHOLD, Retriever

EXAMPLES = {
    "Activate a card": "How do I activate a new debit card?",
    "Check my balance": "How can I check my account balance?",
    "Branch hours": "What are your branch hours?",
}
FLAG_LABELS = {
    "contact_detail": "Contains demo contact details",
    "fictional_bank_claim": "Contains a fictional bank's service or policy claim",
    "multiple_answers_for_question": "The source contains multiple answers to this question",
}


def plain_markdown(text: str) -> str:
    """Show user and dataset text literally, including Markdown links and images."""
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|~$-])", r"\\\1", text)


@st.cache_resource(show_spinner=False, max_entries=2)
def load_retriever(knowledge_path: str, content_digest: str) -> Retriever:
    # Hash is part of the cache key so preparing a changed dataset rebuilds the index.
    return Retriever(Path(knowledge_path))


def reset_chat() -> None:
    st.session_state["turns"] = []


def render_result(turn: dict) -> None:
    result = turn["result"]
    with st.chat_message("assistant", avatar=":material/search:"):
        if not result["matches"]:
            st.info(result["message"])
        else:
            count = len(result["matches"])
            st.markdown(f"**{count} matching reference{'s' if count != 1 else ''}**")
            st.caption("These are answers from the dataset, shown without an LLM rewrite.")
            for rank, match in enumerate(result["matches"], start=1):
                with st.container(border=True):
                    st.caption(f"MATCH {rank} · Similarity {match['score']:.3f} / 1.000")
                    st.markdown(f"**{plain_markdown(match['question'])}**")
                    st.markdown(plain_markdown(match["answer"]))
                    with st.expander(f"Source details · match {rank}"):
                        rows = ", ".join(str(row) for row in match["source_rows"])
                        st.caption(f"File: {match['source_file']} · Data rows: {rows}")
                        st.caption(f"Reference ID: {match['id']}")
                        for flag in match.get("review_flags", []):
                            st.caption(FLAG_LABELS.get(flag, flag))
        st.caption(
            f"Search time: {turn['elapsed_ms']:.0f} ms · "
            f"Minimum similarity: {result['threshold']:.2f} · "
            f"Maximum results: {turn['top_k']}"
        )


def main() -> None:
    st.set_page_config(page_title="Banking help desk", page_icon="🏦", layout="centered")
    st.session_state.setdefault("turns", [])
    st.title("Banking help desk")
    st.caption("Synthetic banking demo · Retrieval only")

    with st.sidebar:
        st.header("Your conversation")
        st.button("Reset chat", key="reset_chat", on_click=reset_chat, use_container_width=True)
        st.caption("History stays in this browser session. Each question is searched independently.")
        st.divider()
        st.subheader("About this demo")
        st.write("Search sample banking questions and see the original answers with their sources.")
        st.caption("All banking information and contact details are fictional. Similarity measures wording overlap, not correctness.")
        with st.expander("Search settings"):
            top_k = st.slider("Maximum results", 1, 5, 3, key="top_k")
            threshold = st.slider(
                "Minimum similarity", 0.0, 1.0, DEFAULT_THRESHOLD, 0.05, key="threshold",
                help="Higher values return fewer, more closely worded matches. Applies to new questions only.",
            )

    try:
        digest = hashlib.sha256(DEFAULT_CLEAN.read_bytes()).hexdigest()
        retriever = load_retriever(str(DEFAULT_CLEAN), digest)
    except (OSError, ValueError):
        st.error("The banking reference data could not be loaded. Prepare the dataset, then reload this page.")
        st.code("python -m banking_chatbot prepare", language="bash")
        st.stop()

    st.sidebar.caption(f"{len(retriever.records)} reference pairs available")
    example = None
    if not st.session_state["turns"]:
        st.subheader("What would you like to find?")
        st.write("Ask a complete banking question, or start with an example.")
        columns = st.columns(len(EXAMPLES))
        for column, (label, query) in zip(columns, EXAMPLES.items()):
            if column.button(label, use_container_width=True):
                example = query

    prompt = st.chat_input("Ask a banking question…", max_chars=1000, key="question")
    submitted = example if example is not None else prompt
    if submitted is not None:
        if not submitted.strip():
            st.warning("Please enter a banking question.")
        else:
            try:
                started = perf_counter()
                result = retriever.search(submitted, top_k=top_k, threshold=threshold)
                turn = {
                    "query": submitted, "result": result, "top_k": top_k,
                    "elapsed_ms": (perf_counter() - started) * 1000,
                }
            except (OSError, ValueError):
                st.error("This search could not be completed. Please try another question.")
            else:
                st.session_state["turns"].append(turn)
                st.rerun()

    for turn in st.session_state["turns"]:
        with st.chat_message("user"):
            st.markdown(plain_markdown(turn["query"]))
        render_result(turn)
