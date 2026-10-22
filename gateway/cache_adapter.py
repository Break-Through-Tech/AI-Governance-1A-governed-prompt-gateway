"""Stage 2 adapter: plug Joohye's response cache into the gateway.

Her cache (``banking_chatbot.cache``) was designed for a RAG chatbot: every
stored answer is tied to the retrieved FAQ rows that backed it, so editing the
knowledge base invalidates the entry. The gateway runs non-RAG (an intent-picked
system prompt plus the model's own knowledge), so there are no retrieved rows.
This module bridges that gap without editing her package:

* The only "source" a gateway answer depends on is the system prompt, so the
  adapter fingerprints the prompt text where she fingerprints FAQ rows. Team C
  rewording the prompt therefore invalidates every cached answer, which is the
  behaviour we want.
* The cache stage runs *before* the router picks a tier, so the scope carries a
  fixed model label instead of a tier model. Routing is deterministic, so the
  same query always lands on the same tier anyway.
* The provider name is part of the scope, so simulated placeholder answers and
  live Gemini answers can never be served for each other.

Her eligibility policy (``is_cacheable``) runs unchanged on both lookup and
store: personal data, transactional, time-sensitive and personalized queries
are never cached, and only complete, successful generations are stored.
"""

from __future__ import annotations

from pathlib import Path

from banking_chatbot.cache import (
    TEMPORARY_CACHE_POLICY_VERSION,
    TEMPORARY_POLICY_CLASSIFICATION,
    CacheContext,
    CacheMode,
    CacheResult,
    CacheStatus,
    SQLiteResponseCache,
    SourceFingerprint,
    configured_cache_mode,
    is_cacheable,
    stable_digest,
)
from banking_chatbot.data import normalize
from banking_chatbot.embeddings import LocalHashingEmbedder

from .config import GatewayConfig, load_config
from .prompts import system_prompt
from .schema import CacheDecision, GenerationResult

# Separate file from her ``var/response_cache.sqlite3`` so the two apps never
# evict each other's entries.
DEFAULT_CACHE_PATH = Path("var/gateway_cache.sqlite3")
# The cache runs before routing, so no tier model is known yet (see module docstring).
SCOPE_MODEL = "gateway-any-tier"
SEMANTIC_MODES = {CacheMode.SEMANTIC, CacheMode.SEMANTIC_SHADOW}


class GatewayCache:
    """Adapter with the two calls the pipeline needs: ``lookup`` and ``store``."""

    name = "semantic-cache"

    def __init__(
        self,
        path: Path | str = DEFAULT_CACHE_PATH,
        *,
        mode: str | CacheMode | None = None,
        config: GatewayConfig | None = None,
        prompt_template: str = "default",
    ) -> None:
        # ``None`` falls back to the CACHE_MODE environment variable (her convention), then "exact".
        self.mode = configured_cache_mode(mode.value if isinstance(mode, CacheMode) else mode)
        config = config or load_config()
        prompt = system_prompt(prompt_template)
        self._sources = (
            SourceFingerprint(f"system-prompt:{prompt_template}", stable_digest(prompt)),
        )
        # ``is_cacheable`` only checks that *some* reference exists; this is that reference.
        self._matches = [{"id": self._sources[0].source_id}]
        self._prompt_version = stable_digest(prompt)[:12]
        self._tiers = {t.name: [t.model, t.thinking_level] for t in config.tiers.values()}
        self.embedder = LocalHashingEmbedder() if self.mode in SEMANTIC_MODES else None
        self.backend = SQLiteResponseCache(Path(path), embedder=self.embedder)

    # ---- scope ---------------------------------------------------------------

    def _context(self, query: str, provider_name: str) -> CacheContext:
        """Everything that can invalidate a cached answer, in her ``CacheContext`` shape."""
        return CacheContext(
            normalized_query=normalize(query),
            tenant_or_jurisdiction="banking-demo",
            policy_classification=TEMPORARY_POLICY_CLASSIFICATION,
            policy_version=TEMPORARY_CACHE_POLICY_VERSION,
            knowledge_base_digest="parametric-only",  # non-RAG: there is no knowledge base
            sources=self._sources,
            system_prompt_version=self._prompt_version,
            model=SCOPE_MODEL,
            generation_settings_digest=stable_digest({"provider": provider_name, "tiers": self._tiers}),
            conversation_context_digest=stable_digest([]),  # the gateway is single-turn
            embedding_model_version=self.embedder.version if self.embedder else None,
        )

    def _eligibility(self, query: str, generation: dict | None = None):
        return is_cacheable(
            query,
            self._matches,
            policy_classification=TEMPORARY_POLICY_CLASSIFICATION,
            history_digest_included=True,
            generation=generation,
        )

    # ---- translating her result into our record ---------------------------------

    @staticmethod
    def _decision(result: CacheResult, **overrides) -> CacheDecision:
        audit = result.audit
        fields = dict(
            status=audit.status.value,
            similarity=audit.similarity,
            entry_id=audit.entry_id,
            match_type=audit.match_type.value if audit.match_type else None,
            reason=audit.reason,
        )
        fields.update(overrides)
        return CacheDecision(**fields)

    @staticmethod
    def _generation(result: CacheResult, provider_name: str) -> GenerationResult:
        """A cache hit costs no model tokens, so both counts are zero, not None."""
        stored = result.generation or {}
        return GenerationResult(
            status="generated",
            provider="cache",
            model=SCOPE_MODEL,
            answer=str(stored.get("answer", "")),
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
            simulated=provider_name == "simulated",
            finish_reason="STOP",
            message=f"Served from cache entry {result.audit.entry_id}.",
        )

    # ---- the two pipeline calls --------------------------------------------------

    def lookup(self, query: str, *, provider_name: str) -> tuple[CacheDecision, GenerationResult | None]:
        """Return the stage-2 decision and, on a hit, the stored answer as a generation."""
        if self.mode is CacheMode.OFF:
            return CacheDecision(status="bypass", reason="cache mode is off"), None
        if not query.strip():
            return CacheDecision(status="bypass", reason="empty query"), None
        eligibility = self._eligibility(query)
        if not eligibility.cacheable:
            return CacheDecision(status="bypass", reason=f"{eligibility.classification}: {eligibility.reason}"), None

        context = self._context(query, provider_name)
        result = self.backend.lookup_exact(context)
        if result.audit.status is CacheStatus.HIT:
            return self._decision(result), self._generation(result, provider_name)
        if self.mode not in SEMANTIC_MODES:
            return self._decision(result), None

        semantic = self.backend.lookup_semantic(context)
        if semantic.audit.status is not CacheStatus.HIT:
            return self._decision(semantic), None
        # Same safety net as her chat loop: re-check the stored answer before reusing it.
        revalidation = self._eligibility(query, dict(semantic.generation or {}))
        if not revalidation.cacheable:
            return self._decision(semantic, status="stale",
                                  reason=f"semantic candidate failed revalidation: {revalidation.reason}"), None
        if self.mode is CacheMode.SEMANTIC_SHADOW:
            # Shadow mode measures what semantic matching *would* do without trusting it yet.
            return self._decision(semantic, status="miss",
                                  reason="shadow mode: semantic candidate observed, fresh answer generated"), None
        return self._decision(semantic), self._generation(semantic, provider_name)

    def store(self, query: str, generation: GenerationResult) -> bool:
        """Store an answer the pipeline has already passed through compliance. Returns True if stored."""
        if self.mode is CacheMode.OFF or generation.status != "generated" or not query.strip():
            return False
        inp, out = generation.input_tokens, generation.output_tokens
        # Her lookup revalidates a stored entry against the scope: the stored "model" and
        # "source_ids" must equal the context's, so these carry the scope values. The tier
        # that produced the answer is still in the audit log's original record.
        payload = {
            "status": "generated",
            "answer": generation.answer,
            "model": SCOPE_MODEL,
            "source_ids": [source.source_id for source in self._sources],
            "finish_reason": generation.finish_reason,
            "usage": {
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": inp + out if isinstance(inp, int) and isinstance(out, int) else None,
            },
            "elapsed_ms": generation.elapsed_ms,
        }
        if not self._eligibility(query, payload).cacheable:
            return False
        self.backend.put(self._context(query, generation.provider), payload)
        return True
