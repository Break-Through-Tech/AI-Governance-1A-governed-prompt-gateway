import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from banking_chatbot.cache import (
    CacheContext,
    CacheMatchType,
    CacheMode,
    CacheStatus,
    SQLiteResponseCache,
    SourceFingerprint,
    cache_metrics,
    configured_cache_mode,
)
from banking_chatbot.chat import chat
from banking_chatbot.embeddings import LocalHashingEmbedder


class MapEmbedder:
    def __init__(self, vectors, version="test-embed-v1"):
        self.vectors = vectors
        self.version = version

    def embed(self, texts):
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class FailingEmbedder:
    version = "test-embed-v1"

    def embed(self, _texts):
        raise RuntimeError("embedding service unavailable")


class FixedRetriever:
    records = [{"id": "faq_pin", "question": "How do I change my PIN?", "answer": "Use the banking app."}]

    def search(self, query, *, top_k, threshold):
        return {
            "query": query.casefold().rstrip("?"),
            "matches": [{
                "id": "faq_pin", "question": "How do I change my PIN?",
                "answer": "Use the banking app.", "score": 1.0,
            }],
            "threshold": threshold,
        }


def context(query, **overrides):
    values = {
        "normalized_query": query,
        "tenant_or_jurisdiction": "banking-demo",
        "policy_classification": "allow",
        "policy_version": "policy-v1",
        "knowledge_base_digest": "knowledge-v1",
        "sources": (SourceFingerprint("faq_pin", "pin-content-v1"),),
        "system_prompt_version": "prompt-v1",
        "model": "gemini-2.5-flash-lite",
        "generation_settings_digest": "settings-v1",
        "conversation_context_digest": "empty-history-v1",
        "embedding_model_version": "test-embed-v1",
    }
    values.update(overrides)
    return CacheContext(**values)


def generation(answer="Use the banking app."):
    return {
        "status": "generated",
        "answer": answer,
        "model": "gemini-2.5-flash-lite",
        "source_ids": ["faq_pin"],
        "usage": {"input_tokens": 100, "output_tokens": 12, "total_tokens": 112},
        "finish_reason": "STOP",
        "elapsed_ms": 80.5,
    }


class SemanticCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "semantic.sqlite3"
        self.vectors = {
            "how do i change my pin": [1.0, 0.0, 0.0],
            "what is the process to reset my pin": [0.98, 0.15, 0.0],
            "how do i activate a card": [0.0, 1.0, 0.0],
            "another close pin question": [0.97, 0.16, 0.0],
        }
        self.embedder = MapEmbedder(self.vectors)
        self.cache = SQLiteResponseCache(
            self.path,
            embedder=self.embedder,
            semantic_threshold=0.9,
            semantic_margin=0.05,
        )

    def test_paraphrase_hits_but_non_equivalent_query_misses(self):
        self.cache.put(context("how do i change my pin"), generation())
        hit = self.cache.lookup_semantic(context("what is the process to reset my pin"))
        miss = self.cache.lookup_semantic(context("how do i activate a card"))
        self.assertEqual(hit.audit.status, CacheStatus.HIT)
        self.assertEqual(hit.audit.match_type, CacheMatchType.SEMANTIC)
        self.assertGreater(hit.audit.similarity, 0.9)
        self.assertEqual(hit.generation["answer"], "Use the banking app.")
        self.assertEqual(miss.audit.status, CacheStatus.MISS)

    def test_complete_scope_and_embedding_version_isolate_candidates(self):
        self.cache.put(context("how do i change my pin"), generation())
        query = "what is the process to reset my pin"
        variants = [
            {"policy_version": "policy-v2"},
            {"knowledge_base_digest": "knowledge-v2"},
            {"sources": (SourceFingerprint("faq_pin", "changed"),)},
            {"conversation_context_digest": "history-v2"},
            {"tenant_or_jurisdiction": "other-tenant"},
            {"embedding_model_version": "test-embed-v2"},
        ]
        for overrides in variants:
            with self.subTest(overrides=overrides):
                result = self.cache.lookup_semantic(context(query, **overrides))
                self.assertIn(result.audit.status, {CacheStatus.MISS, CacheStatus.BYPASS})

    def test_margin_rejects_ambiguous_candidates(self):
        self.cache.put(context("how do i change my pin"), generation("First"))
        self.cache.put(context("another close pin question"), generation("Second"))
        result = self.cache.lookup_semantic(context("what is the process to reset my pin"))
        self.assertEqual(result.audit.status, CacheStatus.MISS)
        self.assertIn("ambiguous", result.audit.reason)

    def test_exact_cache_survives_embedding_failure(self):
        cache = SQLiteResponseCache(self.path, embedder=FailingEmbedder())
        value = context("how do i change my pin")
        cache.put(value, generation())
        self.assertEqual(cache.lookup_exact(value).audit.status, CacheStatus.HIT)
        self.assertEqual(cache.lookup_semantic(
            context("what is the process to reset my pin")
        ).audit.status, CacheStatus.ERROR)

    def test_metrics_distinguish_semantic_and_shadow_matches(self):
        semantic = self.cache.lookup_semantic(context("how do i activate a card")).audit.to_dict()
        turns = [
            {"cache": {
                "status": "hit", "match_type": "semantic", "entry_id": "one",
                "similarity": 0.95, "reason": "hit", "tokens_saved": 12,
                "latency_saved_ms": 10,
            }},
            {"cache": {
                "status": "miss", "reason": "shadow", "semantic_shadow": {
                    "status": "hit", "match_type": "semantic",
                },
            }},
            {"cache": semantic},
        ]
        metrics = cache_metrics(turns)
        self.assertEqual(metrics["semantic_hits"], 1)
        self.assertEqual(metrics["semantic_shadow_matches"], 1)

    def test_local_embedder_is_deterministic_normalized_and_versioned(self):
        embedder = LocalHashingEmbedder()
        first = embedder.embed(["How do I change my PIN?", "Reset my PIN"])
        second = embedder.embed(["How do I change my PIN?", "Reset my PIN"])
        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), np.ones(2), atol=1e-6)
        self.assertIn("v1", embedder.version)

    def test_cache_mode_validation(self):
        self.assertEqual(configured_cache_mode("semantic").value, "semantic")
        with self.assertRaises(ValueError):
            configured_cache_mode("unsafe-mode")

    @patch("banking_chatbot.chat.generate_answer")
    def test_chat_serves_semantic_mode_and_only_observes_shadow_mode(self, generate):
        generate.return_value = generation("Fresh answer")
        retriever = FixedRetriever()
        chat(
            "how do i change my pin", [], retriever=retriever,
            use_gemini=True, api_key="fake", cache=self.cache,
            cache_mode=CacheMode.SEMANTIC,
        )
        served = chat(
            "what is the process to reset my pin", [], retriever=retriever,
            use_gemini=True, api_key="fake", cache=self.cache,
            cache_mode=CacheMode.SEMANTIC,
        )
        self.assertEqual(served["cache"]["match_type"], "semantic")
        self.assertEqual(generate.call_count, 1)

        shadow_cache = SQLiteResponseCache(
            Path(self.temp.name) / "shadow.sqlite3",
            embedder=self.embedder,
            semantic_threshold=0.9,
        )
        chat(
            "how do i change my pin", [], retriever=retriever,
            use_gemini=True, api_key="fake", cache=shadow_cache,
            cache_mode=CacheMode.SEMANTIC_SHADOW,
        )
        shadow = chat(
            "what is the process to reset my pin", [], retriever=retriever,
            use_gemini=True, api_key="fake", cache=shadow_cache,
            cache_mode=CacheMode.SEMANTIC_SHADOW,
        )
        self.assertEqual(shadow["cache"]["status"], "miss")
        self.assertEqual(shadow["cache"]["semantic_shadow"]["status"], "hit")
        self.assertEqual(shadow["generation"]["answer"], "Fresh answer")
        self.assertEqual(generate.call_count, 3)


if __name__ == "__main__":
    unittest.main()
