# Cache Implementation Plan

## Overview

Implement one versioned response-cache service with two lookup strategies. Phase 1 establishes safe storage, invalidation, eligibility, auditing, and metrics through exact-match caching. Phase 2 adds embedding-based semantic matching without changing those controls.

The cache belongs between retrieval and the Gemini call in `banking_chatbot/chat.py`. A cached response must never bypass input governance, authorization, source validation, or output safety checks.

```text
User query
  -> normalize and classify
  -> policy and risk check
  -> retrieve references
  -> optimize prompt and select model
  -> cache lookup
       -> hit: revalidate cached response
       -> miss: call LLM -> validate output -> cache eligible response
  -> audit decision
  -> return response
```

The Streamlit resource cache in `banking_chatbot/interface.py` should remain unchanged. It caches the constructed TF-IDF index, not generated responses.

## Shared Design

### Cache interface

Add `banking_chatbot/cache.py` with a small interface shared by both phases:

```python
class ResponseCache:
    def lookup_exact(self, context) -> CacheResult: ...
    def lookup_semantic(self, context) -> CacheResult: ...
    def put(self, context, generation) -> None: ...
```

Keep persistence behind this interface so the local SQLite implementation can later be replaced by Redis or a vector database without changing the request flow.

### Cache identity

Construct two fingerprints:

```text
scope fingerprint =
  tenant or jurisdiction
  + policy version
  + knowledge-base digest
  + retrieved source IDs and content hashes
  + system-prompt version
  + model and generation settings
  + conversation-context digest
  + embedding-model version, when applicable

exact key = SHA-256(scope fingerprint + normalized query)
```

The scope fingerprint prevents an answer from being reused after a relevant policy, source, prompt, model, configuration, or conversation change.

### Storage

For the local prototype, use SQLite through Python's standard library. Store the database in an ignored runtime directory such as `var/response_cache.sqlite3`.

Each entry should contain:

- Hashed key and opaque entry ID
- Normalized query
- Generated answer
- Source IDs and source-content digest
- Model, prompt, policy, and knowledge versions
- Original token usage and generation latency
- Creation and expiration timestamps
- Eligibility classification
- Hit count and last-used timestamp
- Embedding and embedding-model version in phase 2

Never store API keys, credentials, or raw conversation history.

## Phase 1: Exact-Match Caching

### 1. Add cache eligibility rules

Create a deterministic `is_cacheable()` policy. Initially cache only:

- Successful, complete generations
- Questions backed by retrieved references
- Stable, general FAQ requests
- Entries containing no detected personal or transactional data

Bypass caching for:

- Errors, blocked responses, and `MAX_TOKENS` results
- Personalized, account-specific, transactional, or time-sensitive questions
- Requests that fail governance checks
- Context-dependent conversations unless the history digest is included
- Requests whose policy classification is unknown

The application does not yet have the full risk-classification layer described by the project. Until that exists, use conservative eligibility and label it as a temporary banking-demo policy.

### 2. Implement persistence and invalidation

Create a SQLite repository with:

- Primary-key lookup on `exact_key`
- Expiration checks on read
- Atomic insert or update operations
- Configurable time to live
- Bounded size with oldest or least-used eviction
- Safe handling of corrupt or unavailable cache storage

A cache failure should bypass caching and continue through the governed generation path. It must not disable governance.

### 3. Integrate the cache into the request flow

Update `banking_chatbot/chat.py` to:

1. Normalize and classify the query.
2. Run retrieval.
3. Build the cache context and eligibility result.
4. Look up the exact key.
5. On a hit, revalidate entry versions and return the cached answer.
6. On a miss, call Gemini.
7. Apply output checks.
8. Store the response if eligible.

### 4. Expose auditing and metrics

Add cache metadata to every turn:

```python
"cache": {
    "status": "hit" | "miss" | "bypass" | "stale" | "error",
    "match_type": "exact" | None,
    "entry_id": "...",
    "similarity": 1.0,
    "reason": "...",
    "tokens_saved": 112,
    "latency_saved_ms": 420,
}
```

Update the Streamlit interface to display cache status, tokens saved, and bypass reasons. Keep cache keys and stored queries out of user-visible logs.

Track at least:

- Eligible cache lookups
- Exact hits and misses
- Bypassed and stale requests
- Cache errors
- Tokens and estimated cost avoided
- Generation latency avoided

Calculate hit rate over eligible lookups as well as over all requests so policy-driven bypasses do not obscure cache performance.

### 5. Test phase 1

Add `tests/test_cache.py` and integration coverage proving that:

- A repeated query with identical scope avoids the second Gemini call.
- Query, model, source, prompt, policy, or context changes cause a miss.
- Expired entries are not served.
- Errors and truncated answers are not cached.
- API keys and raw history never enter the cache.
- Cache-storage failures fall back safely.
- Concurrent writes do not produce malformed entries.

### Phase 1 completion criteria

Phase 1 is complete when repeated identical requests reliably avoid generation, cache invalidation works across all versioned inputs, and every cache decision is auditable.

## Phase 2: Semantic Matching with Embeddings

### 1. Add an embedding abstraction

Introduce an embedding interface:

```python
class Embedder:
    @property
    def version(self) -> str: ...

    def embed(self, texts: list[str]) -> np.ndarray: ...
```

Use a compact local sentence-embedding model for the prototype so regulated prompts do not need to be sent to another external service. Normalize vectors and compare them using cosine similarity.

### 2. Extend storage

When inserting a cache entry:

- Embed the normalized query.
- Store the vector and embedding-model version.
- Preserve the exact key so exact lookup always runs first.

For the prototype's expected cache size, SQLite plus a NumPy linear scan is sufficient. Introduce a dedicated vector database only if measured cache size or lookup latency justifies it.

### 3. Add constrained semantic lookup

The lookup sequence becomes:

1. Try an exact match.
2. Filter semantic candidates by the complete scope fingerprint.
3. Compare the new query embedding only with eligible candidates.
4. Select the highest-scoring candidate.
5. Require a calibrated similarity threshold and, optionally, a minimum margin over the second candidate.
6. Re-run policy and output checks before serving the cached response.
7. Otherwise, treat the lookup as a miss and call the LLM.

Do not use semantic matching for personalized, transactional, high-risk, or strongly conversation-dependent requests.

### 4. Calibrate the similarity threshold

Build an evaluation set containing:

- Equivalent banking paraphrases
- Similar-looking but meaningfully different questions
- Negation pairs
- Different products with overlapping vocabulary
- Unsafe and jailbreak prompts
- Context-dependent follow-up questions

Sweep similarity thresholds and prioritize reuse precision over hit rate. A proposed release target is at least 95% semantic-reuse precision on a held-out set, with zero known unsafe cross-policy reuse. Record the selected threshold, embedding-model version, dataset digest, and evaluation results.

### 5. Roll out in shadow mode

Add a configuration setting:

```text
CACHE_MODE=off | exact | semantic-shadow | semantic
```

In `semantic-shadow` mode, calculate and audit the proposed semantic match but still call Gemini. Compare the candidate response with the newly generated response before enabling semantic reuse.

### 6. Test phase 2

Verify that:

- Known paraphrases hit the same entry.
- Similar but non-equivalent questions miss.
- Scope or embedding-version differences prevent reuse.
- Exact matching still works when embedding generation fails.
- Semantic hits never cross tenants, policy profiles, or source versions.
- Metrics distinguish exact hits, semantic hits, misses, and bypasses.

### Phase 2 completion criteria

Phase 2 is complete when semantic matching improves hit rate on repeated and paraphrased workloads while meeting the calibrated precision target and all governance isolation requirements.

## Proposed File Changes

| File | Change |
|---|---|
| `banking_chatbot/cache.py` | Cache contracts, SQLite repository, key construction, eligibility, and lookup logic |
| `banking_chatbot/embeddings.py` | Phase 2 embedding abstraction and local implementation |
| `banking_chatbot/chat.py` | Governance-aware cache lookup and insertion around generation |
| `banking_chatbot/gemini.py` | Expose stable prompt and generation-configuration fingerprints |
| `banking_chatbot/interface.py` | Cache configuration, status, and savings display |
| `tests/test_cache.py` | Unit tests for storage, invalidation, eligibility, and exact matching |
| `tests/test_semantic_cache.py` | Embedding and semantic-match tests |
| `tests/test_gemini.py` | Verify hits suppress external generation calls |
| `.gitignore` | Exclude runtime cache databases and sidecar files |
| `requirements.txt` | Add the selected local embedding dependency in phase 2 |

## Recommended Delivery Order

1. Define cache context, results, and audit schemas.
2. Implement SQLite storage and exact keys.
3. Add eligibility and invalidation rules.
4. Integrate exact lookup into `chat()`.
5. Add phase 1 tests, UI status, and metrics.
6. Establish an exact-cache performance baseline.
7. Add the embedding abstraction and vector persistence.
8. Build and label the semantic evaluation set.
9. Calibrate semantic matching in shadow mode.
10. Enable semantic reuse after it meets the precision and governance criteria.
