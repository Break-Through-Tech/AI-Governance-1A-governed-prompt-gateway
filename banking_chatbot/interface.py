"""Session-local banking chat with retrieval and optional Gemini generation."""

import hashlib
import re
from pathlib import Path

import streamlit as st

from .cache import SQLiteResponseCache, cache_metrics
from .data import DEFAULT_CLEAN
from .chat import chat
from .gemini import DEFAULT_MODEL, SUPPORTED_MODELS, GenerationError, local_settings
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


@st.cache_resource(show_spinner=False)
def get_response_cache() -> SQLiteResponseCache:
    return SQLiteResponseCache()


def reset_chat() -> None:
    st.session_state["turns"] = []


def disconnect_gemini() -> None:
    st.session_state["gemini_key"] = ""
    st.session_state["use_gemini"] = False
    st.session_state["use_local_key"] = False
    reset_chat()


def render_result(turn: dict) -> None:
    result = turn["result"]
    with st.chat_message("assistant", avatar=":material/search:"):
        generation = turn.get("generation")
        if generation:
            if generation["status"] == "generated":
                st.markdown("**Gemini answer**")
                st.markdown(plain_markdown(generation["answer"]))
                usage = generation["usage"]
                input_tokens = usage["input_tokens"] if usage["input_tokens"] is not None else "unavailable"
                output_tokens = usage["output_tokens"] if usage["output_tokens"] is not None else "unavailable"
                timing_label = "Original generation" if turn.get("cache", {}).get("status") == "hit" else "Generation"
                st.caption(
                    f"{generation['model']} · Input tokens: {input_tokens} · "
                    f"Output tokens: {output_tokens} · {timing_label}: {generation['elapsed_ms'] / 1000:.1f} s"
                )
                if generation["finish_reason"] == "MAX_TOKENS":
                    st.warning("This answer reached the response-length limit and may be incomplete.")
                st.caption("AI-generated from the references below. Check the sources; this is fictional banking information.")
                st.divider()
            else:
                st.warning(generation["message"])
        cache = turn.get("cache") or {}
        cache_status = cache.get("status")
        if cache_status == "hit":
            st.success(
                f"Exact cache hit · {cache.get('tokens_saved', 0)} tokens and "
                f"{cache.get('latency_saved_ms', 0) / 1000:.1f} s of generation avoided"
            )
        elif cache_status == "error":
            st.warning(f"Cache unavailable · {cache.get('reason', 'Generation continued without cache reuse.')}")
        elif cache_status in {"miss", "stale", "bypass"}:
            st.caption(f"Response cache: {cache_status} · {cache.get('reason', '')}")
        if not result["matches"]:
            st.info(result["message"])
        else:
            count = len(result["matches"])
            st.markdown(f"**{count} matching reference{'s' if count != 1 else ''}**")
            st.caption("Original dataset answers, shown without rewriting.")
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
    st.caption("Synthetic banking demo · Local retrieval + optional Gemini answers")

    with st.sidebar:
        st.header("Your conversation")
        st.button("Reset chat", key="reset_chat", on_click=reset_chat, use_container_width=True)
        st.caption("History stays in this browser session. Retrieval searches each question independently.")
        st.divider()
        st.subheader("Gemini connection")
        try:
            configured_key, configured_model = local_settings()
        except GenerationError as exc:
            st.warning(str(exc))
            configured_key, configured_model = "", DEFAULT_MODEL
        entered_key = st.text_input("Gemini API key", type="password", key="gemini_key",
                                    help="Kept only in this session. It is not saved to files or chat history.")
        use_local = False
        if configured_key:
            use_local = st.checkbox("Use local configured key", key="use_local_key", value=False)
        api_key = entered_key.strip() or (configured_key if use_local else "")
        st.button("Disconnect Gemini", key="disconnect_gemini", on_click=disconnect_gemini,
                  help="Clears the session key and chat, and disables AI answers. A configured local key stays on disk.")
        model = st.selectbox("Gemini model", SUPPORTED_MODELS,
                             index=SUPPORTED_MODELS.index(configured_model), key="gemini_model")
        enabled = st.toggle("Generate answers with Gemini", key="use_gemini", disabled=not api_key)
        use_gemini = bool(api_key) and enabled
        if not api_key:
            st.caption("No key configured for this session. Retrieval remains available.")
        st.caption("When enabled, your question, retrieved references, and up to 3 recent AI exchanges are sent to Google.")
        with st.expander("Free API setup"):
            st.markdown("[Create a Gemini key in Google AI Studio](https://aistudio.google.com/api-keys)")
            st.caption("Use a project on the Free tier. Billing and quota follow your Google project; the app cannot enforce free billing. No automatic model switch or billing upgrade is performed.")
            st.caption("Google may use free-tier content to improve its products. Use synthetic test questions.")
        metrics = cache_metrics(st.session_state["turns"])
        st.divider()
        st.subheader("Response cache")
        cache_columns = st.columns(2)
        cache_columns[0].metric("Exact hits", metrics["exact_hits"])
        cache_columns[1].metric("Eligible hit rate", f"{metrics['eligible_hit_rate']:.0%}")
        st.caption(
            f"Eligible lookups: {metrics['eligible_lookups']} · Misses: {metrics['misses']} · "
            f"Bypasses: {metrics['bypasses']} · Stale: {metrics['stale']} · Errors: {metrics['errors']}"
        )
        st.caption(
            f"Avoided: {metrics['tokens_avoided']} tokens and "
            f"{metrics['generation_latency_avoided_ms'] / 1000:.1f} s generation latency. "
            "Estimated cost avoided is unavailable until a versioned price model is configured."
        )
        st.caption("The temporary banking-demo policy caches only stable, retrieval-backed general FAQs.")
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
                with st.spinner("Generating a Gemini answer…" if use_gemini else "Searching references…"):
                    response_cache = get_response_cache() if use_gemini else None
                    turn = chat(
                        submitted, st.session_state["turns"], retriever=retriever,
                        api_key=api_key, use_gemini=use_gemini, model=model,
                        top_k=top_k, threshold=threshold, cache=response_cache,
                    )
            except (OSError, ValueError):
                st.error("This search could not be completed. Please try another question.")
            else:
                st.session_state["turns"].append(turn)
                st.rerun()

    for turn in st.session_state["turns"]:
        with st.chat_message("user"):
            st.markdown(plain_markdown(turn["query"]))
        render_result(turn)
