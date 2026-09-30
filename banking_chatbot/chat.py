"""Reusable chat operation shared by the UI and tests."""

from time import perf_counter

from .gemini import DEFAULT_MODEL, GenerationError, generate_answer
from .retrieval import DEFAULT_THRESHOLD, Retriever


def chat(message: str, history: list[dict], *, retriever: Retriever,
         api_key: str = "", use_gemini: bool = False, model: str = DEFAULT_MODEL,
         top_k: int = 3, threshold: float = DEFAULT_THRESHOLD) -> dict:
    started = perf_counter()
    result = retriever.search(message, top_k=top_k, threshold=threshold)
    turn = {
        "query": message, "result": result, "top_k": top_k,
        "elapsed_ms": (perf_counter() - started) * 1000,
        "generation": None,
    }
    if use_gemini and result["matches"]:
        try:
            turn["generation"] = generate_answer(message, result["matches"], history, api_key, model)
        except GenerationError as exc:
            turn["generation"] = {"status": "error", "message": str(exc), "model": model}
    return turn
