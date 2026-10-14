"""Reusable chat operation shared by the UI and tests."""

from time import perf_counter

from .cache import (
    CacheAudit,
    CacheContext,
    CacheMode,
    CacheStatus,
    ResponseCache,
    SourceFingerprint,
    TEMPORARY_CACHE_POLICY_VERSION,
    TEMPORARY_POLICY_CLASSIFICATION,
    classify_cache_query,
    is_cacheable,
    stable_digest,
)
from .data import normalize
from .gemini import (
    DEFAULT_MODEL,
    GENERATION_CONFIG,
    SYSTEM_PROMPT_VERSION,
    GenerationError,
    generate_answer,
    recent_conversation,
)
from .retrieval import DEFAULT_THRESHOLD, Retriever


def _bypass(reason: str) -> CacheAudit:
    return CacheAudit(status=CacheStatus.BYPASS, reason=reason)


def _cache_context(
    query: str,
    matches: list[dict],
    history: list[dict],
    retriever: Retriever,
    model: str,
    embedding_model_version: str | None = None,
) -> CacheContext:
    sources = tuple(
        SourceFingerprint(
            match["id"],
            stable_digest({"question": match["question"], "answer": match["answer"]}),
        )
        for match in matches[:5]
    )
    return CacheContext(
        normalized_query=query,
        tenant_or_jurisdiction="banking-demo",
        policy_classification=TEMPORARY_POLICY_CLASSIFICATION,
        policy_version=TEMPORARY_CACHE_POLICY_VERSION,
        knowledge_base_digest=stable_digest(retriever.records),
        sources=sources,
        system_prompt_version=SYSTEM_PROMPT_VERSION,
        model=model,
        generation_settings_digest=stable_digest(GENERATION_CONFIG),
        conversation_context_digest=stable_digest(recent_conversation(history)),
        embedding_model_version=embedding_model_version,
    )


def _with_reason(audit: CacheAudit, reason: str) -> CacheAudit:
    return CacheAudit(
        status=audit.status,
        reason=f"{audit.reason} {reason}",
        match_type=audit.match_type,
        entry_id=audit.entry_id,
        similarity=audit.similarity,
        tokens_saved=audit.tokens_saved,
        latency_saved_ms=audit.latency_saved_ms,
    )


def chat(message: str, history: list[dict], *, retriever: Retriever,
         api_key: str = "", use_gemini: bool = False, model: str = DEFAULT_MODEL,
         top_k: int = 3, threshold: float = DEFAULT_THRESHOLD,
         cache: ResponseCache | None = None,
         cache_mode: CacheMode | str = CacheMode.EXACT) -> dict:
    try:
        mode = cache_mode if isinstance(cache_mode, CacheMode) else CacheMode(cache_mode)
    except ValueError as error:
        raise ValueError("cache_mode is invalid.") from error
    started = perf_counter()
    normalized_message = normalize(message)
    query_eligibility = classify_cache_query(
        normalized_message,
        policy_classification=TEMPORARY_POLICY_CLASSIFICATION,
        history_digest_included=True,
    )
    result = retriever.search(normalized_message, top_k=top_k, threshold=threshold)
    turn = {
        "query": message, "result": result, "top_k": top_k,
        "elapsed_ms": (perf_counter() - started) * 1000,
        "generation": None,
        "cache": _bypass("Gemini generation is disabled for this request.").to_dict(),
    }
    if not use_gemini:
        return turn
    if not result["matches"]:
        turn["cache"] = _bypass("No retrieved references support generation or reuse.").to_dict()
        return turn

    eligibility = query_eligibility
    if eligibility.cacheable:
        eligibility = is_cacheable(
            result["query"],
            result["matches"],
            policy_classification=TEMPORARY_POLICY_CLASSIFICATION,
            history_digest_included=True,
        )
    context = None
    lookup = None
    if not eligibility.cacheable:
        turn["cache"] = _bypass(eligibility.reason).to_dict()
    elif mode is CacheMode.OFF:
        turn["cache"] = _bypass("Response caching is disabled by CACHE_MODE.").to_dict()
    elif cache is None:
        turn["cache"] = _bypass("Response caching is not configured.").to_dict()
    else:
        embedder = getattr(cache, "embedder", None)
        embedding_version = getattr(embedder, "version", None)
        context = _cache_context(
            result["query"], result["matches"], history, retriever, model,
            embedding_model_version=embedding_version,
        )
        lookup = cache.lookup_exact(context)
        turn["cache"] = lookup.audit.to_dict()
        if lookup.audit.status is CacheStatus.HIT:
            turn["generation"] = dict(lookup.generation or {})
            return turn
        if mode in {CacheMode.SEMANTIC, CacheMode.SEMANTIC_SHADOW}:
            semantic = cache.lookup_semantic(context)
            lookup = semantic
            turn["cache"] = semantic.audit.to_dict()
            if semantic.audit.status is CacheStatus.HIT:
                candidate_eligibility = is_cacheable(
                    result["query"],
                    result["matches"],
                    policy_classification=TEMPORARY_POLICY_CLASSIFICATION,
                    history_digest_included=True,
                    generation=semantic.generation,
                )
                if candidate_eligibility.cacheable and mode is CacheMode.SEMANTIC:
                    turn["generation"] = dict(semantic.generation or {})
                    return turn
                if mode is CacheMode.SEMANTIC_SHADOW:
                    turn["cache"] = CacheAudit(
                        status=CacheStatus.MISS,
                        reason="Semantic shadow candidate observed; fresh generation was used.",
                    ).to_dict()
                    turn["cache"]["semantic_shadow"] = semantic.audit.to_dict()
                else:
                    turn["cache"] = CacheAudit(
                        status=CacheStatus.STALE,
                        reason=f"Semantic candidate failed policy revalidation: {candidate_eligibility.reason}",
                    ).to_dict()

    try:
        turn["generation"] = generate_answer(message, result["matches"], history, api_key, model)
    except GenerationError as exc:
        turn["generation"] = {"status": "error", "message": str(exc), "model": model}
        return turn

    if (
        context is not None
        and lookup is not None
        and turn["cache"].get("status") != CacheStatus.HIT.value
    ):
        output_eligibility = is_cacheable(
            result["query"],
            result["matches"],
            policy_classification=TEMPORARY_POLICY_CLASSIFICATION,
            history_digest_included=True,
            generation=turn["generation"],
        )
        if output_eligibility.cacheable:
            cache.put(context, turn["generation"])
        else:
            audit = lookup.audit
            if audit.status is CacheStatus.HIT:
                audit = CacheAudit(
                    status=CacheStatus.MISS,
                    reason="A fresh generation was required.",
                )
            turn["cache"] = _with_reason(
                audit,
                f"Generated response was not stored: {output_eligibility.reason}",
            ).to_dict()
    return turn
