"""Contracts for governed response caching.

It defines the versioned request context, conservative temporary eligibility
policy, auditable lookup result, and local SQLite implementation.
"""

import math
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Callable

import numpy as np

from .data import ROOT
from .embeddings import Embedder, validate_embeddings


DEFAULT_CACHE_PATH = ROOT / "var/response_cache.sqlite3"
DEFAULT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MAX_ENTRIES = 1_000
DEFAULT_SEMANTIC_THRESHOLD = 0.90
DEFAULT_SEMANTIC_MARGIN = 0.05
TEMPORARY_CACHE_POLICY_VERSION = "banking-demo-cache-policy-v1"
TEMPORARY_POLICY_CLASSIFICATION = "temporary-banking-demo-allow"

_PERSONAL_DATA = re.compile(
    r"(?:[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b(?:\d[ -]?){8,19}\b)", re.IGNORECASE,
)
_GOVERNANCE_RISK = re.compile(
    r"\b(?:ignore (?:all |the )?(?:previous|prior|system)|jailbreak|system prompt|"
    r"bypass (?:the )?(?:rules|safety|policy)|steal|launder|fraud)\b",
    re.IGNORECASE,
)
_TIME_SENSITIVE = re.compile(
    r"\b(?:today|right now|currently|current|latest|live|exchange rate|interest rate|stock price)\b",
    re.IGNORECASE,
)
_PERSONALIZED = re.compile(
    r"\b(?:should i|best for me|recommend(?:ation)? for me|am i eligible|my credit score|"
    r"approve my|my (?:account )?balance|my transactions?|my statement|my payment|my transfer|"
    r"my account details|my card number)\b",
    re.IGNORECASE,
)
_TRANSACTIONAL = re.compile(
    r"(?:\b(?:send|transfer|pay|withdraw|deposit)\b.{0,40}(?:[$€£]\s*\d|\b\d+(?:\.\d{1,2})?\b|\bto\b)|"
    r"\b(?:cancel|reverse) (?:my |the )?(?:payment|transfer|transaction)\b)",
    re.IGNORECASE,
)

class CacheStatus(str, Enum):
    """Auditable outcomes for every cache decision."""

    HIT = "hit"
    MISS = "miss"
    BYPASS = "bypass"
    STALE = "stale"
    ERROR = "error"


class CacheMatchType(str, Enum):
    """Supported cache match strategies."""

    EXACT = "exact"
    SEMANTIC = "semantic"


class CacheMode(str, Enum):
    """Runtime modes for staged semantic-cache rollout."""

    OFF = "off"
    EXACT = "exact"
    SEMANTIC_SHADOW = "semantic-shadow"
    SEMANTIC = "semantic"


def configured_cache_mode(value: str | None = None) -> CacheMode:
    """Read and validate CACHE_MODE, defaulting to the safe exact-only mode."""

    raw = value if value is not None else os.getenv("CACHE_MODE", CacheMode.EXACT.value)
    try:
        return CacheMode(raw.strip().casefold())
    except (AttributeError, ValueError) as error:
        choices = ", ".join(mode.value for mode in CacheMode)
        raise ValueError(f"CACHE_MODE must be one of: {choices}.") from error


def configured_semantic_settings() -> tuple[float, float]:
    """Read conservative threshold and ambiguity margin from the environment."""

    try:
        threshold = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", str(DEFAULT_SEMANTIC_THRESHOLD)))
        margin = float(os.getenv("SEMANTIC_CACHE_MARGIN", str(DEFAULT_SEMANTIC_MARGIN)))
    except ValueError as error:
        raise ValueError("Semantic cache threshold and margin must be numbers.") from error
    for name, value in (("threshold", threshold), ("margin", margin)):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Semantic cache {name} must be between 0 and 1.")
    return threshold, margin


@dataclass(frozen=True)
class CacheEligibility:
    """Deterministic temporary-policy decision for response reuse."""

    cacheable: bool
    classification: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.cacheable, bool):
            raise ValueError("cacheable must be a boolean.")
        _require_text("classification", self.classification)
        _require_text("reason", self.reason)


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")


def classify_cache_query(
    query: str,
    *,
    policy_classification: str,
    history_digest_included: bool,
) -> CacheEligibility:
    """Classify query-level reuse risk before retrieval runs."""

    if policy_classification not in {"allow", TEMPORARY_POLICY_CLASSIFICATION}:
        return CacheEligibility(False, "policy_not_allowed", "Policy classification does not allow reuse.")
    if not history_digest_included:
        return CacheEligibility(False, "context_unversioned", "Conversation context is not versioned.")
    if _PERSONAL_DATA.search(query):
        return CacheEligibility(False, "personal_data", "Possible personal or account data was detected.")
    if _GOVERNANCE_RISK.search(query):
        return CacheEligibility(False, "governance_risk", "Potentially unsafe instructions require fresh handling.")
    if _TIME_SENSITIVE.search(query):
        return CacheEligibility(False, "time_sensitive", "Time-sensitive requests are not reusable.")
    if _PERSONALIZED.search(query):
        return CacheEligibility(False, "personalized", "Personalized or account-specific requests are not reusable.")
    if _TRANSACTIONAL.search(query):
        return CacheEligibility(False, "transactional", "Transactional requests are not reusable.")
    return CacheEligibility(True, "stable_faq_candidate", "Query passed the temporary cache policy.")


def is_cacheable(
    query: str,
    matches: Sequence[Mapping[str, object]],
    *,
    policy_classification: str,
    history_digest_included: bool,
    generation: Mapping[str, object] | None = None,
) -> CacheEligibility:
    """Apply the conservative temporary banking-demo cache policy.

    This is not a substitute for the planned risk-classification layer.  It
    permits only retrieval-backed general FAQs and rejects data or wording that
    signals personal, transactional, time-sensitive, or unsafe requests.
    """

    query_decision = classify_cache_query(
        query,
        policy_classification=policy_classification,
        history_digest_included=history_digest_included,
    )
    if not query_decision.cacheable:
        return query_decision
    if not matches:
        return CacheEligibility(False, "no_references", "No retrieved references support reuse.")

    if generation is not None:
        if generation.get("status") != "generated":
            return CacheEligibility(False, "generation_error", "Only successful generations can be stored.")
        if generation.get("finish_reason") != "STOP":
            return CacheEligibility(False, "generation_incomplete", "Only complete generations can be stored.")
        answer = generation.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            return CacheEligibility(False, "generation_empty", "Empty generations cannot be stored.")

    return CacheEligibility(
        True,
        "stable_general_faq",
        "Temporary banking-demo policy allows this stable, retrieval-backed FAQ.",
    )


@dataclass(frozen=True)
class SourceFingerprint:
    """Opaque source identity used to invalidate cached generations."""

    source_id: str
    content_hash: str

    def __post_init__(self) -> None:
        _require_text("source_id", self.source_id)
        _require_text("content_hash", self.content_hash)


@dataclass(frozen=True)
class CacheContext:
    """Complete non-secret scope needed to identify reusable generations.

    Raw conversation history and credentials are deliberately absent.  Callers
    supply only a digest for conversation context and stable identifiers or
    digests for every input that can invalidate an answer.
    """

    normalized_query: str
    tenant_or_jurisdiction: str
    policy_classification: str
    policy_version: str
    knowledge_base_digest: str
    sources: tuple[SourceFingerprint, ...]
    system_prompt_version: str
    model: str
    generation_settings_digest: str
    conversation_context_digest: str
    embedding_model_version: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "normalized_query",
            "tenant_or_jurisdiction",
            "policy_classification",
            "policy_version",
            "knowledge_base_digest",
            "system_prompt_version",
            "model",
            "generation_settings_digest",
            "conversation_context_digest",
        ):
            _require_text(name, getattr(self, name))
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("sources must be a nonempty tuple of source fingerprints.")
        if not all(isinstance(source, SourceFingerprint) for source in self.sources):
            raise ValueError("sources must contain only SourceFingerprint values.")
        if self.embedding_model_version is not None:
            _require_text("embedding_model_version", self.embedding_model_version)


@dataclass(frozen=True)
class CacheAudit:
    """User-safe metadata describing a cache decision."""

    status: CacheStatus
    reason: str
    match_type: CacheMatchType | None = None
    entry_id: str | None = None
    similarity: float | None = None
    tokens_saved: int = 0
    latency_saved_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.status, CacheStatus):
            raise ValueError("status must be a CacheStatus value.")
        _require_text("reason", self.reason)
        if self.match_type is not None and not isinstance(self.match_type, CacheMatchType):
            raise ValueError("match_type must be a CacheMatchType value or None.")
        if self.entry_id is not None:
            _require_text("entry_id", self.entry_id)
        if self.similarity is not None and (
            isinstance(self.similarity, bool)
            or not isinstance(self.similarity, (int, float))
            or not math.isfinite(self.similarity)
            or not 0 <= self.similarity <= 1
        ):
            raise ValueError("similarity must be a finite number between 0 and 1.")
        if isinstance(self.tokens_saved, bool) or not isinstance(self.tokens_saved, int) or self.tokens_saved < 0:
            raise ValueError("tokens_saved must be a nonnegative integer.")
        if (
            isinstance(self.latency_saved_ms, bool)
            or not isinstance(self.latency_saved_ms, (int, float))
            or not math.isfinite(self.latency_saved_ms)
            or self.latency_saved_ms < 0
        ):
            raise ValueError("latency_saved_ms must be a finite nonnegative number.")
        if self.status is CacheStatus.HIT:
            if self.match_type is None or self.entry_id is None or self.similarity is None:
                raise ValueError("cache hits require match_type, entry_id, and similarity.")
            if self.match_type is CacheMatchType.EXACT and self.similarity != 1.0:
                raise ValueError("exact cache hits must have similarity 1.0.")
        elif self.tokens_saved or self.latency_saved_ms:
            raise ValueError("only cache hits may report avoided tokens or latency.")

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Return the stable public audit shape used by chat turns."""

        return {
            "status": self.status.value,
            "match_type": self.match_type.value if self.match_type is not None else None,
            "entry_id": self.entry_id,
            "similarity": self.similarity,
            "reason": self.reason,
            "tokens_saved": self.tokens_saved,
            "latency_saved_ms": self.latency_saved_ms,
        }


@dataclass(frozen=True)
class CacheResult:
    """Cache lookup result containing a generation only on a valid hit."""

    audit: CacheAudit
    generation: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.audit, CacheAudit):
            raise ValueError("audit must be a CacheAudit value.")
        if self.audit.status is CacheStatus.HIT:
            if not isinstance(self.generation, Mapping) or not self.generation:
                raise ValueError("cache hits require a nonempty generation mapping.")
        elif self.generation is not None:
            raise ValueError("only cache hits may include a generation.")


class ResponseCache(ABC):
    """Persistence-independent cache interface shared by exact and semantic phases."""

    @abstractmethod
    def lookup_exact(self, context: CacheContext) -> CacheResult:
        """Return an exact-match decision for the supplied governed context."""

    @abstractmethod
    def lookup_semantic(self, context: CacheContext) -> CacheResult:
        """Return a semantic-match decision when that strategy is available."""

    @abstractmethod
    def put(self, context: CacheContext, generation: Mapping[str, object]) -> None:
        """Persist an eligible generated response for later reuse."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_digest(value: object) -> str:
    """Return a deterministic digest for JSON-compatible versioned inputs."""

    return _sha256(_canonical_json(value))


def source_content_digest(context: CacheContext) -> str:
    """Return a stable digest of ordered source IDs and their content hashes."""

    return stable_digest([
        {"source_id": source.source_id, "content_hash": source.content_hash}
        for source in context.sources
    ])


def scope_fingerprint(context: CacheContext) -> str:
    """Hash every non-query input that can invalidate a cached answer."""

    scope = {
        "tenant_or_jurisdiction": context.tenant_or_jurisdiction,
        "policy_classification": context.policy_classification,
        "policy_version": context.policy_version,
        "knowledge_base_digest": context.knowledge_base_digest,
        "sources": [
            {"source_id": source.source_id, "content_hash": source.content_hash}
            for source in context.sources
        ],
        "system_prompt_version": context.system_prompt_version,
        "model": context.model,
        "generation_settings_digest": context.generation_settings_digest,
        "conversation_context_digest": context.conversation_context_digest,
        "embedding_model_version": context.embedding_model_version,
    }
    return stable_digest(scope)


def exact_key(context: CacheContext) -> str:
    """Return the exact-match key for a normalized query and governed scope."""

    return _sha256(f"{scope_fingerprint(context)}\n{context.normalized_query}")


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _optional_nonnegative_number(value: object) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        return None
    return float(value)


def _safe_generation(generation: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(generation, Mapping):
        raise ValueError("generation must be a mapping.")
    if generation.get("status") != "generated":
        raise ValueError("generation must have generated status.")
    answer = generation.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("generation must contain a nonempty answer.")
    safe: dict[str, object] = {"status": "generated", "answer": answer}

    for name in ("model", "finish_reason"):
        value = generation.get(name)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"generation {name} must be a string.")
            safe[name] = value

    source_ids = generation.get("source_ids")
    if source_ids is not None:
        if not isinstance(source_ids, (list, tuple)) or not all(isinstance(value, str) for value in source_ids):
            raise ValueError("generation source_ids must be a list of strings.")
        safe["source_ids"] = list(source_ids)

    usage = generation.get("usage")
    if usage is not None:
        if not isinstance(usage, Mapping):
            raise ValueError("generation usage must be a mapping.")
        safe["usage"] = {
            name: _optional_nonnegative_int(usage.get(name))
            for name in ("input_tokens", "output_tokens", "total_tokens")
        }

    attempts = generation.get("attempts")
    if isinstance(attempts, int) and not isinstance(attempts, bool) and attempts >= 0:
        safe["attempts"] = attempts
    elapsed_ms = _optional_nonnegative_number(generation.get("elapsed_ms"))
    if elapsed_ms is not None:
        safe["elapsed_ms"] = elapsed_ms
    return safe


class SQLiteResponseCache(ResponseCache):
    """Bounded exact-response cache backed by a local SQLite database.

    Storage failures are converted into cache errors or ignored writes so the
    governed generation path can continue without depending on cache health.
    """

    def __init__(
        self,
        path: Path = DEFAULT_CACHE_PATH,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        embedder: Embedder | None = None,
        semantic_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
        semantic_margin: float = DEFAULT_SEMANTIC_MARGIN,
        clock: Callable[[], float] = time.time,
    ):
        if (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be a finite positive number.")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer.")
        if not isinstance(path, Path):
            raise ValueError("path must be a pathlib.Path.")
        for name, value in (
            ("semantic_threshold", semantic_threshold),
            ("semantic_margin", semantic_margin),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be a finite number between 0 and 1.")
        self.path = path
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = max_entries
        self.embedder = embedder
        self.semantic_threshold = float(semantic_threshold)
        self.semantic_margin = float(semantic_margin)
        self._clock = clock
        self._lock = RLock()
        self._initialization_error = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._connect()) as connection, connection:
                self._create_schema(connection)
        except (OSError, sqlite3.Error):
            self._initialization_error = True

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = WAL")
        except BaseException:
            connection.close()
            raise
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS response_cache (
                exact_key TEXT PRIMARY KEY,
                entry_id TEXT NOT NULL UNIQUE,
                normalized_query TEXT NOT NULL,
                generation_json TEXT NOT NULL,
                answer TEXT NOT NULL,
                source_ids_json TEXT NOT NULL,
                source_content_digest TEXT NOT NULL,
                model TEXT NOT NULL,
                system_prompt_version TEXT NOT NULL,
                policy_classification TEXT NOT NULL,
                policy_version TEXT NOT NULL,
                knowledge_base_digest TEXT NOT NULL,
                generation_settings_digest TEXT NOT NULL,
                conversation_context_digest TEXT NOT NULL,
                tenant_or_jurisdiction TEXT NOT NULL,
                eligibility_classification TEXT NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                generation_latency_ms REAL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0,
                last_used_at REAL NOT NULL
            )
        """)
        existing = {
            row[1] for row in connection.execute("PRAGMA table_info(response_cache)").fetchall()
        }
        additions = {
            "scope_fingerprint": "TEXT",
            "embedding": "BLOB",
            "embedding_dimensions": "INTEGER",
            "embedding_model_version": "TEXT",
        }
        for column, declaration in additions.items():
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE response_cache ADD COLUMN {column} {declaration}"
                )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS response_cache_last_used "
            "ON response_cache(last_used_at, created_at)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS response_cache_semantic_scope "
            "ON response_cache(scope_fingerprint, embedding_model_version, expires_at)"
        )

    @staticmethod
    def _error(reason: str) -> CacheResult:
        return CacheResult(CacheAudit(status=CacheStatus.ERROR, reason=reason))

    def lookup_exact(self, context: CacheContext) -> CacheResult:
        key = exact_key(context)
        now = float(self._clock())
        if self._initialization_error:
            return self._error("Cache storage is unavailable; generation should continue.")
        try:
            with self._lock, closing(self._connect()) as connection, connection:
                row = connection.execute(
                    "SELECT entry_id, generation_json, expires_at, total_tokens, "
                    "generation_latency_ms, source_ids_json, source_content_digest, model, "
                    "system_prompt_version, policy_classification, policy_version, "
                    "knowledge_base_digest, generation_settings_digest, "
                    "conversation_context_digest, tenant_or_jurisdiction "
                    "FROM response_cache WHERE exact_key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    return CacheResult(CacheAudit(
                        status=CacheStatus.MISS,
                        reason="No exact entry exists for this governed scope.",
                    ))
                (
                    entry_id, generation_json, expires_at, total_tokens, latency_ms,
                    source_ids_json, stored_source_digest, stored_model, stored_prompt_version,
                    stored_policy_classification, stored_policy_version, stored_knowledge_digest,
                    stored_settings_digest, stored_context_digest, stored_tenant,
                ) = row
                if expires_at <= now:
                    connection.execute("DELETE FROM response_cache WHERE exact_key = ?", (key,))
                    return CacheResult(CacheAudit(
                        status=CacheStatus.STALE,
                        entry_id=entry_id,
                        reason="The exact cache entry expired and was removed.",
                    ))
                expected_scope = (
                    _canonical_json([source.source_id for source in context.sources]),
                    source_content_digest(context),
                    context.model,
                    context.system_prompt_version,
                    context.policy_classification,
                    context.policy_version,
                    context.knowledge_base_digest,
                    context.generation_settings_digest,
                    context.conversation_context_digest,
                    context.tenant_or_jurisdiction,
                )
                if row[5:] != expected_scope:
                    connection.execute("DELETE FROM response_cache WHERE exact_key = ?", (key,))
                    return CacheResult(CacheAudit(
                        status=CacheStatus.STALE,
                        entry_id=entry_id,
                        reason="The exact cache entry failed scope revalidation and was removed.",
                    ))
                try:
                    generation = json.loads(generation_json)
                    generation = _safe_generation(generation)
                except (json.JSONDecodeError, ValueError, TypeError):
                    connection.execute("DELETE FROM response_cache WHERE exact_key = ?", (key,))
                    return self._error("The exact cache entry was invalid and was removed.")
                expected_source_ids = [source.source_id for source in context.sources]
                if generation.get("model") != context.model or generation.get("source_ids") != expected_source_ids:
                    connection.execute("DELETE FROM response_cache WHERE exact_key = ?", (key,))
                    return CacheResult(CacheAudit(
                        status=CacheStatus.STALE,
                        entry_id=entry_id,
                        reason="The exact cache entry failed generation revalidation and was removed.",
                    ))
                connection.execute(
                    "UPDATE response_cache SET hit_count = hit_count + 1, last_used_at = ? "
                    "WHERE exact_key = ?",
                    (now, key),
                )
            return CacheResult(
                CacheAudit(
                    status=CacheStatus.HIT,
                    match_type=CacheMatchType.EXACT,
                    entry_id=entry_id,
                    similarity=1.0,
                    reason="Exact eligible match.",
                    tokens_saved=total_tokens if isinstance(total_tokens, int) and total_tokens >= 0 else 0,
                    latency_saved_ms=(
                        latency_ms
                        if isinstance(latency_ms, (int, float)) and math.isfinite(latency_ms) and latency_ms >= 0
                        else 0.0
                    ),
                ),
                generation,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError):
            return self._error("Cache storage could not be read; generation should continue.")

    def lookup_semantic(self, context: CacheContext) -> CacheResult:
        if self.embedder is None:
            return CacheResult(CacheAudit(
                status=CacheStatus.BYPASS,
                reason="No semantic embedder is configured.",
            ))
        if context.embedding_model_version != self.embedder.version:
            return CacheResult(CacheAudit(
                status=CacheStatus.BYPASS,
                reason="The request embedding version does not match the configured embedder.",
            ))
        if self._initialization_error:
            return self._error("Cache storage is unavailable; generation should continue.")

        now = float(self._clock())
        try:
            query_vector = validate_embeddings(
                self.embedder.embed([context.normalized_query]), 1,
            )[0]
            with self._lock, closing(self._connect()) as connection, connection:
                connection.execute("DELETE FROM response_cache WHERE expires_at <= ?", (now,))
                rows = connection.execute(
                    "SELECT exact_key, entry_id, generation_json, total_tokens, "
                    "generation_latency_ms, embedding, embedding_dimensions, "
                    "source_ids_json, source_content_digest, model, system_prompt_version, "
                    "policy_classification, policy_version, knowledge_base_digest, "
                    "generation_settings_digest, conversation_context_digest, "
                    "tenant_or_jurisdiction FROM response_cache "
                    "WHERE scope_fingerprint = ? AND embedding_model_version = ? "
                    "AND embedding IS NOT NULL AND exact_key != ? AND expires_at > ?",
                    (
                        scope_fingerprint(context),
                        self.embedder.version,
                        exact_key(context),
                        now,
                    ),
                ).fetchall()
                scored: list[tuple[float, tuple[object, ...]]] = []
                for row in rows:
                    dimensions = row[6]
                    if not isinstance(dimensions, int) or dimensions != query_vector.shape[0]:
                        continue
                    candidate = np.frombuffer(row[5], dtype=np.float32)
                    if candidate.shape != query_vector.shape or not np.all(np.isfinite(candidate)):
                        continue
                    score = float(np.dot(query_vector, candidate))
                    if math.isfinite(score):
                        scored.append((max(0.0, min(1.0, score)), row))

                if not scored:
                    return CacheResult(CacheAudit(
                        status=CacheStatus.MISS,
                        reason="No semantic candidate exists in this governed scope.",
                    ))
                scored.sort(key=lambda item: item[0], reverse=True)
                best_score, best = scored[0]
                second_score = scored[1][0] if len(scored) > 1 else 0.0
                if best_score < self.semantic_threshold:
                    return CacheResult(CacheAudit(
                        status=CacheStatus.MISS,
                        reason="The best semantic candidate was below the similarity threshold.",
                        similarity=best_score,
                    ))
                if len(scored) > 1 and best_score - second_score < self.semantic_margin:
                    return CacheResult(CacheAudit(
                        status=CacheStatus.MISS,
                        reason="Semantic candidates were too ambiguous to reuse safely.",
                        similarity=best_score,
                    ))

                expected_scope = (
                    _canonical_json([source.source_id for source in context.sources]),
                    source_content_digest(context),
                    context.model,
                    context.system_prompt_version,
                    context.policy_classification,
                    context.policy_version,
                    context.knowledge_base_digest,
                    context.generation_settings_digest,
                    context.conversation_context_digest,
                    context.tenant_or_jurisdiction,
                )
                if best[7:] != expected_scope:
                    connection.execute("DELETE FROM response_cache WHERE exact_key = ?", (best[0],))
                    return CacheResult(CacheAudit(
                        status=CacheStatus.STALE,
                        entry_id=str(best[1]),
                        reason="The semantic candidate failed scope revalidation and was removed.",
                    ))
                try:
                    generation = _safe_generation(json.loads(str(best[2])))
                except (json.JSONDecodeError, ValueError, TypeError):
                    connection.execute("DELETE FROM response_cache WHERE exact_key = ?", (best[0],))
                    return self._error("The semantic candidate was invalid and was removed.")
                expected_source_ids = [source.source_id for source in context.sources]
                if generation.get("model") != context.model or generation.get("source_ids") != expected_source_ids:
                    connection.execute("DELETE FROM response_cache WHERE exact_key = ?", (best[0],))
                    return CacheResult(CacheAudit(
                        status=CacheStatus.STALE,
                        entry_id=str(best[1]),
                        reason="The semantic candidate failed generation revalidation and was removed.",
                    ))
                connection.execute(
                    "UPDATE response_cache SET hit_count = hit_count + 1, last_used_at = ? "
                    "WHERE exact_key = ?",
                    (now, best[0]),
                )
            total_tokens, latency_ms = best[3], best[4]
            return CacheResult(
                CacheAudit(
                    status=CacheStatus.HIT,
                    match_type=CacheMatchType.SEMANTIC,
                    entry_id=str(best[1]),
                    similarity=best_score,
                    reason="Constrained semantic match passed threshold, margin, and scope checks.",
                    tokens_saved=total_tokens if isinstance(total_tokens, int) and total_tokens >= 0 else 0,
                    latency_saved_ms=(
                        latency_ms
                        if isinstance(latency_ms, (int, float)) and math.isfinite(latency_ms) and latency_ms >= 0
                        else 0.0
                    ),
                ),
                generation,
            )
        except (OSError, sqlite3.Error, TypeError, ValueError, RuntimeError, OverflowError):
            return self._error("Semantic lookup failed; generation should continue.")

    def put(self, context: CacheContext, generation: Mapping[str, object]) -> None:
        if self._initialization_error:
            return
        try:
            safe = _safe_generation(generation)
            usage = safe.get("usage") if isinstance(safe.get("usage"), Mapping) else {}
            input_tokens = _optional_nonnegative_int(usage.get("input_tokens"))
            output_tokens = _optional_nonnegative_int(usage.get("output_tokens"))
            total_tokens = _optional_nonnegative_int(usage.get("total_tokens"))
            latency_ms = _optional_nonnegative_number(safe.get("elapsed_ms"))
            now = float(self._clock())
            key = exact_key(context)
            embedding_blob: bytes | None = None
            embedding_dimensions: int | None = None
            embedding_version: str | None = None
            if (
                self.embedder is not None
                and context.embedding_model_version == self.embedder.version
            ):
                try:
                    vector = validate_embeddings(
                        self.embedder.embed([context.normalized_query]), 1,
                    )[0]
                    embedding_blob = vector.astype(np.float32, copy=False).tobytes()
                    embedding_dimensions = int(vector.shape[0])
                    embedding_version = self.embedder.version
                except (TypeError, ValueError, RuntimeError):
                    # Exact caching must remain available when embedding fails.
                    pass
            values = (
                key,
                uuid.uuid4().hex,
                context.normalized_query,
                _canonical_json(safe),
                safe["answer"],
                _canonical_json([source.source_id for source in context.sources]),
                source_content_digest(context),
                context.model,
                context.system_prompt_version,
                context.policy_classification,
                context.policy_version,
                context.knowledge_base_digest,
                context.generation_settings_digest,
                context.conversation_context_digest,
                context.tenant_or_jurisdiction,
                "eligible",
                input_tokens,
                output_tokens,
                total_tokens,
                latency_ms,
                now,
                now + self.ttl_seconds,
                now,
                scope_fingerprint(context),
                embedding_blob,
                embedding_dimensions,
                embedding_version,
            )
            with self._lock, closing(self._connect()) as connection, connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute("""
                    INSERT INTO response_cache (
                        exact_key, entry_id, normalized_query, generation_json, answer,
                        source_ids_json, source_content_digest, model, system_prompt_version,
                        policy_classification, policy_version, knowledge_base_digest,
                        generation_settings_digest, conversation_context_digest,
                        tenant_or_jurisdiction, eligibility_classification, input_tokens,
                        output_tokens, total_tokens, generation_latency_ms, created_at,
                        expires_at, last_used_at, scope_fingerprint, embedding,
                        embedding_dimensions, embedding_model_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(exact_key) DO UPDATE SET
                        entry_id = excluded.entry_id,
                        generation_json = excluded.generation_json,
                        answer = excluded.answer,
                        input_tokens = excluded.input_tokens,
                        output_tokens = excluded.output_tokens,
                        total_tokens = excluded.total_tokens,
                        generation_latency_ms = excluded.generation_latency_ms,
                        created_at = excluded.created_at,
                        expires_at = excluded.expires_at,
                        scope_fingerprint = excluded.scope_fingerprint,
                        embedding = excluded.embedding,
                        embedding_dimensions = excluded.embedding_dimensions,
                        embedding_model_version = excluded.embedding_model_version,
                        hit_count = 0,
                        last_used_at = excluded.last_used_at
                """, values)
                connection.execute("DELETE FROM response_cache WHERE expires_at <= ?", (now,))
                connection.execute("""
                    DELETE FROM response_cache
                    WHERE exact_key IN (
                        SELECT exact_key FROM response_cache
                        ORDER BY last_used_at DESC, created_at DESC, exact_key DESC
                        LIMIT -1 OFFSET ?
                    )
                """, (self.max_entries,))
        except (OSError, sqlite3.Error, TypeError, ValueError, OverflowError):
            return


def cache_metrics(turns: Sequence[Mapping[str, object]]) -> dict[str, int | float | None]:
    """Aggregate auditable response-cache outcomes from chat turns."""

    counts = {status.value: 0 for status in CacheStatus}
    exact_hits = 0
    semantic_hits = 0
    semantic_shadow_matches = 0
    tokens_saved = 0
    latency_saved_ms = 0.0
    total = 0
    eligible_lookups = 0
    for turn in turns:
        audit = turn.get("cache")
        if not isinstance(audit, Mapping):
            continue
        status = audit.get("status")
        if status not in counts:
            continue
        total += 1
        counts[status] += 1
        if status != CacheStatus.BYPASS.value:
            eligible_lookups += 1
        if status == CacheStatus.HIT.value and audit.get("match_type") == CacheMatchType.EXACT.value:
            exact_hits += 1
        if status == CacheStatus.HIT.value and audit.get("match_type") == CacheMatchType.SEMANTIC.value:
            semantic_hits += 1
        shadow = audit.get("semantic_shadow")
        if isinstance(shadow, Mapping) and shadow.get("status") == CacheStatus.HIT.value:
            semantic_shadow_matches += 1
        saved = audit.get("tokens_saved")
        if isinstance(saved, int) and not isinstance(saved, bool) and saved >= 0:
            tokens_saved += saved
        latency = audit.get("latency_saved_ms")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool) and math.isfinite(latency):
            latency_saved_ms += max(0.0, float(latency))
    return {
        "requests": total,
        "eligible_lookups": eligible_lookups,
        "exact_hits": exact_hits,
        "semantic_hits": semantic_hits,
        "semantic_shadow_matches": semantic_shadow_matches,
        "misses": counts[CacheStatus.MISS.value],
        "bypasses": counts[CacheStatus.BYPASS.value],
        "stale": counts[CacheStatus.STALE.value],
        "errors": counts[CacheStatus.ERROR.value],
        "tokens_avoided": tokens_saved,
        "generation_latency_avoided_ms": latency_saved_ms,
        "estimated_cost_avoided_usd": None,
        "eligible_hit_rate": (exact_hits + semantic_hits) / eligible_lookups if eligible_lookups else 0.0,
        "overall_hit_rate": (exact_hits + semantic_hits) / total if total else 0.0,
    }
