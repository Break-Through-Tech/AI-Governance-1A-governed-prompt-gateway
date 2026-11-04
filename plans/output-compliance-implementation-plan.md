# Output Compliance Implementation Plan

Planning baseline: October 28, 2026. Recommendations below are proposed defaults, not measured results or implemented behavior.

## Objective and scope

Add a release gate that checks a candidate answer against the fictional banking demo's policies before display, conversation-history reuse, or response-cache insertion. Both fresh generations and cache hits must pass through it. Passing means the configured checks passed; it is not certification of factual or legal correctness.

Build this as a small LangGraph subgraph called from `chat()`. Keep retrieval and the existing Gemini REST client outside it initially. No model training, autonomous tools, external browsing, or human-review service is required for the first release.

Current gaps verified in the repository:

- `gemini.py` instructs the generator to use references, avoid invented banking facts, avoid personalized recommendations, and never claim account access or completed transactions.
- `chat.py` returns exact and semantic cache hits early. These paths need to converge with fresh generation before release.
- `cache.py:is_cacheable()` checks storage eligibility, including completeness. It does not determine whether an answer may be shown.
- `interface.py` displays generated answers and source records. A blocked answer must not remain in history or leak through source previews.
- `requirements.txt` does not currently include LangGraph.

## Decisions to settle

| Decision | Recommended first-release choice | Alternative / reason to revisit |
|---|---|---|
| Policy owner and scope | Team agrees a versioned fictional-banking policy with labeled examples; English text only initially | Real-bank or jurisdiction-specific rules need a separately owned policy and evidence base |
| Architecture | Deterministic checks followed by one structured semantic review | Separate grounding and safety models only if evaluation shows a benefit |
| Release decisions | `allow`, `block`, `unverified`; all but `allow` withhold the candidate | Add repair and human review later |
| Personal information | Reject known secrets and disallowed identifiers; distinguish authorized public contacts from personal data | Span redaction if useful answers are frequently lost; names alone should not imply a violation |
| Grounding | Every material factual claim must be supported by the provided source text; policy overrides source content | Add current authoritative sources later; the synthetic FAQ dataset cannot verify a real bank's facts |
| Semantic evaluator | Benchmark Gemini 3.8 Flash against Gemini 3.5 Flash-Lite; provisional reference evaluator is 3.8 Flash | Select the cheaper model only if it meets the same release gates |
| Generator | Preserve existing generator during baseline experiments if the account can use it | Benchmark migration to 3.5 Flash-Lite separately so generator and judge changes are distinguishable |
| Failure behavior | Timeout, malformed verdict, refusal, missing checks, or missing required evidence yields `unverified` and fixed fallback | No implicit allow or unreviewed model fallback |
| Caching | Review every candidate in v1, including exact and semantic hits | Reuse an exact compliance verdict only when all relevant identities and freshness requirements match |
| UI boundary | Display only the final answer, safe reason, and approved/sanitized reference previews | A restricted developer inspection view requires a separate access and retention design |
| Runtime mode | Offline evaluation, then synthetic-data shadow testing, then enforced gate | No ordinary user-facing switch that disables enforcement |
| Latency and budget | Agree limits after measuring two reviewer candidates; record total pipeline cost | Smaller judge or safe verdict caching if cost/latency exceed the agreed budget |
| Human review | Defer; show fallback for uncertainty | Add only when there is an actual reviewer, queue, and response expectation |

## Models and detection components

| Role | Proposed component | Selection rationale and limitations |
|---|---|---|
| Answer generation baseline | Existing `gemini-2.5-flash-lite` | Already integrated; use only after verifying account availability. Keep baseline stable during judge comparison. |
| Generator migration candidate | `gemini-3.5-flash-lite` | Current stable Flash-Lite option; benchmark answer quality and adapt model-specific settings before migration. |
| Semantic reviewer, reference candidate | `gemini-3.8-flash` | Use a separate call with a review-only prompt and JSON Schema. Proposed candidate, not a proven superior banking judge. |
| Semantic reviewer, economy candidate | `gemini-3.5-flash-lite` | Compare against the same human-labeled set; may reduce cost if it passes the same gates. |
| Legacy reviewer baseline, optional | `gemini-2.5-flash` | Already allowed by the app; useful if accessible, but do not make new deployment depend on legacy access. |
| Format, secrets, identifiers, permitted contacts | Python rules, checksum validation, explicit allowlists | Predictable and cheap. Regex alone does not provide comprehensive personal-information detection. |
| Broader personal-information detection, optional | Presidio Analyzer with explicitly configured recognizers and an evaluated NLP backend | A detection framework rather than a single model; evaluate entity coverage and false positives before adding its dependencies. |
| Harmful-content classifier, optional | `meta-llama/Llama-Guard-4-12B` | Specialized safety candidate if harmful-content coverage becomes a distinct need. Requires hosting/license review and benchmarking; cannot replace banking grounding and business-rule evaluation. |

Google currently lists 3.5 Flash-Lite and 3.8 Flash as stable and recommends them for new projects, while limiting 2.5 access to prior users. Verify actual access with a small synthetic smoke test during implementation; documentation availability does not establish availability for this account. [Model catalog](https://ai.google.dev/gemini-api/docs/models), [deprecation and access guidance](https://ai.google.dev/gemini-api/docs/deprecations).

Use the current REST integration pattern rather than introducing a second agent framework. Give the reviewer its own model, prompt version, output schema, output-token limit, timeout, and validated generation settings. Do not blindly reuse the generator's 384-token cap or model-specific thinking configuration. Record configured and returned model/version identifiers where available.

Gemini supports schema-constrained outputs, including through REST. Parse and validate locally anyway: schema validity does not establish correctness of the verdict. [Structured output documentation](https://ai.google.dev/gemini-api/docs/structured-output).

Same-provider generator and reviewer models can share errors. Separate prompts and model sizes are useful engineering choices, not proof of independence. Human labels remain the evaluation reference. The optional Llama Guard model addresses a safety taxonomy, not verification of arbitrary banking claims. [Meta model card](https://huggingface.co/meta-llama/Llama-Guard-4-12B).

## Policy specification

Create a JSON policy file with stable rule IDs, descriptions, severity, applicable scope, evidence requirements, and allowed actions. Keep executable detection logic in Python. Begin with:

| Rule | What fails | Enforcement |
|---|---|---|
| `OUTPUT_COMPLETE` | Empty, malformed, or truncated candidate | Deterministic block |
| `SENSITIVE_DATA` | Secrets, disallowed account/card identifiers, or unauthorized personal information | Local detectors first; block known high-severity findings |
| `SUPPORTED_FACTS` | Unsupported rates, fees, eligibility decisions, procedures, or other material factual claims | Semantic review against actual supplied evidence |
| `NO_ACCOUNT_ACTION_CLAIM` | Unsupported claims of account access, transfers, approvals, or completed changes | Semantic review; phrase rules can flag examples but cannot establish full coverage |
| `NO_PERSONAL_RECOMMENDATION` | Personalized financial recommendations rather than general fictional procedures | Semantic review, with paired allowed/blocked examples |
| `FICTIONAL_CONTEXT` | Presents synthetic banking information as real-bank policy | Semantic review |
| `CONTACT_POLICY` | Presents fictional or unapproved contact details as real support | Deterministic contact allowlist plus contextual review |
| `CONTENT_SAFETY` | Assistance prohibited by the team's explicit safety policy | Semantic review, with optional specialized classifier later |
| `RESPONSE_FORMAT` | Exceeds agreed length or violates required output shape | Deterministic check; repair may be permitted in phase 2 |

Keep source provenance separate from grounding: a statement can match a synthetic source and still be inappropriate to present as a real-world fact. Presence of a source ID or high TF-IDF similarity is not claim verification. Allow supported partial answers that explicitly state what the references do not establish.

## Modular interfaces and dependency boundaries

Expose one application entry point: `compliance_service.evaluate(candidate, context) -> ComplianceResult`. `chat()`, the cache, and the UI depend on this public contract, not on Gemini, individual detectors, or LangGraph state. A model or validator swap changes an adapter and its configuration; it does not change the caller or graph topology.

Define provider-independent contracts in `compliance_contracts.py`. The following signatures are the proposed interface specification; named request/result types will be implemented as validated immutable data objects:

```python
from typing import Protocol


class OutputValidator(Protocol):
    @property
    def descriptor(self) -> ComponentDescriptor: ...

    def validate(self, request: ValidationRequest) -> CheckResult: ...


class ComplianceReviewer(Protocol):
    @property
    def descriptor(self) -> ComponentDescriptor: ...

    def review(self, request: ReviewRequest) -> ReviewResult: ...


class ComplianceService(Protocol):
    def evaluate(
        self, candidate: Candidate, context: ComplianceContext
    ) -> ComplianceResult: ...
```

Use these common data contracts:

| Contract | Required contents and boundary |
|---|---|
| `Candidate` | Text, completion metadata, and origin; never credentials |
| `ComplianceContext` | Request ID, question, bounded relevant history, evidence, trusted policy and scope |
| `ComponentDescriptor` | Stable component ID, implementation version, contract version, supported rule IDs, non-secret configuration digest, and local/external execution type; reviewers also identify provider, model, prompt, and schema versions |
| `ValidationRequest` | Immutable candidate, applicable policy rules, and minimum required context |
| `ReviewRequest` | Sanitized, bounded question/candidate/evidence and assigned review rules; built by the service after local checks |
| `Finding` | Rule ID, component ID, severity, safe reason code, and optional private evidence spans/source IDs; no release authority |
| `CheckResult` | Component identity, explicit per-rule `pass`, `fail`, `uncertain`, or `error` outcomes, findings, sanitized error code, elapsed time, and normalized usage |
| `ReviewResult` | Same check-result semantics plus claim assessments and source support; contains no provider response object |
| `ComplianceResult` | Final released answer or fixed fallback, `allow`/`block`/`unverified`, safe reason codes, sanitized audit metadata, and aggregated usage; excludes drafts and private evidence |

An empty findings list does not mean approval: every required check must explicitly complete. Keep execution errors distinct from content violations. Known adapter errors such as timeout, refusal, unsupported model, or parse failure become normalized error outcomes. Unexpected adapter exceptions are caught at the service boundary, recorded without raw exception text, and withhold the candidate. Validate returned objects at runtime; Python protocols alone do not validate values.

Keep the policy decision function independent of adapters: `decide(policy, check_results) -> PolicyDecision`. It verifies coverage and combines results using the mandatory-rule precedence already specified below. The LLM cannot directly set the public release status. Local validators report findings without rewriting the candidate; any later sanitizer/repairer must return a new candidate that goes through the complete gate.

### Construction and configuration

Add an application composition function that resolves a trusted configuration into a policy, a tuple of `OutputValidator` implementations, and one `ComplianceReviewer`. Inject these into `build_compliance_service(...)`; its internal graph factory captures the dependencies. Graph nodes call protocols and never import concrete providers. Keep credentials in injected clients, not configuration fingerprints or graph state.

Initial implementations are `FormatValidator`, `SensitiveDataValidator`, `ContactPolicyValidator`, `GeminiReviewer`, and a test-only `FakeReviewer`. A later `PresidioValidator` or local-model reviewer implements the same contracts. Optional libraries are loaded only by their adapter, so baseline tests and other adapters do not require their installation.

Configuration selects registered component IDs and their validated settings. For example, changing the Gemini reviewer model setting from `gemini-3.8-flash` to `gemini-3.5-flash-lite` must not require graph, UI, or cache code edits. Switching providers adds an adapter and registration; model-specific prompts, schema translation, generation settings, and usage parsing stay inside that adapter. The policy remains provider-independent and is supplied to the adapter as trusted input.

Validate configuration at startup:

- Reject unknown component IDs, duplicate IDs, incompatible contract versions, and unsupported required rules.
- Map every mandatory rule to explicit required check IDs. Where several checks cover a rule, run all configured required checks; one pass must not erase another failure or uncertainty.
- Confirm external-processing restrictions, timeout limits, and payload limits. A component swap must not silently move local data processing to an external service.
- Do not permit a test fake or a missing reviewer to enable production release. Startup validation failure disables generated-answer release with a safe configuration error.

Supported-rule declarations establish compatibility, not accuracy. Any new model, prompt, validator, or material setting must pass the relevant evaluation gates before deployment. Derive a compliance configuration fingerprint from the policy, contract, all active component descriptors, and required-check mapping. Use it for audits and cache invalidation; never retain an old approval merely because a replacement uses the same interface.

### Contract and substitution tests

Create a shared adapter contract suite and run it against every implementation. Verify normalized result shape, explicit rule coverage, valid source references, bounded/sanitized failures, and absence of provider objects or credentials. Use mocked transport for ordinary tests and budgeted live smoke tests separately.

Run the same graph/application fixtures with interchangeable fake reviewers and validators to prove that callers and routing remain unchanged. Cover a swapped model, a replaced validator, a missing mandatory check, duplicate check configuration, an exception, contradictory results, and a configuration-fingerprint change. Assert that hard failures always win, errors never release drafts, and changed components invalidate prior approvals. Substitution tests establish interface compatibility; human-labeled evaluation establishes whether the replacement is acceptable.

## Contracts and LangGraph flow

Private input/state fields:

- Request ID, normalized question, minimum necessary conversation context.
- Candidate text, origin (`fresh`, `exact_cache`, `semantic_cache`), completion metadata.
- References with IDs and the exact text used as evidence, source hashes, policy version.
- Deterministic findings, semantic findings, evaluator status, repair count, timing and token usage.

Do not put API keys in state. Inject clients/configuration through application dependencies. Start with an in-process synchronous graph without persistent checkpoints, keeping drafts transient.

The semantic verdict schema should include `status = pass | fail | uncertain`, per-rule results, claim assessments (`supported | contradicted | insufficient_evidence`), valid source IDs, and concise evidence explanations. Keep any necessary evidence spans private. Reject unknown IDs, missing required checks, contradictory fields, and invalid enum values. Do not use model-reported confidence as a calibrated probability or ask for hidden reasoning.

The policy reducer owns the final outcome:

- Any mandatory-rule failure -> `block`.
- Any required check unavailable or uncertain -> `unverified`.
- All required checks completed and passed -> `allow`.
- A semantic pass cannot override a deterministic hard failure.

```text
candidate from fresh generation or cache
  -> validate candidate and local rules
       -> hard failure: fixed fallback
       -> otherwise: semantic review
            -> parse and validate verdict
            -> policy decision
                 -> allow: release candidate
                 -> block/unverified: fixed fallback
  -> record sanitized audit event
  -> return public result
```

Use `StateGraph`, ordinary nodes, and conditional edges. Keep the decision reducer a pure function that can be tested without a network or graph runtime. Return a restricted output schema containing `final_answer`, release status, safe reason codes, and usage metadata. [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api).

An output schema is not a security boundary for streams, tracing, or checkpoints. Do not stream draft tokens or whole graph state to the browser. Do not attach callbacks that export raw drafts. Scope this explicitly in tests.

The reviewer receives policy as trusted instructions and the question/draft/references as bounded untrusted data. It has no tools. Detect disallowed sensitive content locally before a second external call; minimize and sanitize the review payload, and return `unverified` when necessary context cannot safely be supplied. This output component does not substitute for input-side data governance.

## Integration and cache behavior

1. Refactor `chat()` to obtain a candidate without returning it from cache branches.
2. Send every candidate through the same compliance entry point. On reviewer failure, return a fixed fallback; on a rejected cache candidate, invalidate/quarantine it so it does not repeatedly trigger review. Fresh regeneration after a rejected hit can be a later bounded recovery path.
3. Store only released, cache-eligible answers. Keep compliance decisions distinct from cache eligibility.
4. Extend cache scope/version metadata with compliance policy, evaluator model, evaluator prompt, ruleset, and schema versions. Migrate/invalidate old entries explicitly; existing cache policy version is not sufficient on its own.
5. Review semantic hits against the current question and evidence, even if an older answer previously passed. Similarity is not approval.
6. Store only the released answer in chat history. A block/unverified result must not leave the original candidate under `generation.answer` or another UI-accessible field.
7. Apply a separate display policy to retrieved excerpts and links. For the demo, allow vetted synthetic records with visible fictional labeling and sanitization of disallowed sensitive/contact data. Hide unapproved previews after a sensitive-data finding. Retrieval-only mode must retain its offline behavior; it must not claim semantic approval without running that review.
8. Update savings metrics: cache hits avoid generation, but still incur reviewer cost in v1. Report net cost and latency, not gross generation savings alone.

Later, exact verdict caching may key on question, candidate, relevant history, source text hashes, scope, all policy/evaluator versions, and an expiry. Re-run local checks and invalidate after changed inputs or policy. Do not reuse verdicts across merely similar questions.

## Audit and user experience

Record request ID, candidate origin, decision, rule IDs, check status, policy/ruleset/prompt/model versions, source identifiers/digests, latency, tokens, estimated cost, and retry/repair counts. No raw prompts, drafts, reviewer quotations, credentials, or personal identifiers in routine logs. Define retention and access before introducing persistence; default to session-only sanitized metadata for the demo.

Display a normal answer on allow. For block/unverified, display a fixed message explaining that the response could not be verified and inviting a more specific question. Do not fabricate a human escalation or real support contact. Use “Passed configured checks” rather than “Guaranteed safe” or “Legally compliant.” Detailed internal findings belong in sanitized developer metrics, not the ordinary customer flow.

## Evaluation and release gates

Build roughly 240 synthetic, human-labeled examples initially: 120 for development and 120 held out, grouped by source FAQ, scenario, and paraphrase family to avoid leakage. Expand high-severity coverage before broader use. Have a second team member review ambiguous/high-severity labels. The judge must not label its own benchmark.

Cover compliant responses, near-miss false positives, invented fees/rates, unsupported transactions, recommendations, fictional contacts, sensitive identifiers, malicious instructions inside drafts/references, conflicting evidence, legitimate uncertainty, truncation, and mixed compliant/noncompliant claims. Include paraphrases and negations such as “I cannot transfer money” so crude phrase rules do not look artificially successful.

Compare deterministic-only, rules + Flash-Lite reviewer, and rules + Flash reviewer. Measure per-rule violation recall and precision, false-block rate, uncertain/error rate, unsafe-release rate, repeated-run consistency, p50/p95 latency, tokens, and cost per completed turn. Report uncertainty/error separately so a system that blocks everything cannot appear successful.

Proposed demo release targets, to agree before tuning:

- No observed high-severity unsafe releases on the held-out set; report counts and uncertainty, not a universal zero-risk claim.
- At least 95% violation recall overall and no more than 5% false blocks on compliant held-out cases; inspect per-rule results and sample sizes.
- All routing/failure-path integration tests pass; unknown/error verdicts never release candidates.
- Meet an agreed measured p95 latency and cost budget. Select the cheapest candidate satisfying the quality gates, rather than assuming a specific model is best.

Automated tests must cover all three origins, cache disabled, generation failure, invalid reviewer JSON, missing evidence, provider refusal, timeout, unrecognized model, policy/version changes, sensitive source previews, and history leakage. Mock external calls in the existing unittest suite. Run live synthetic evaluation separately and explicitly budget it.

## Delivery sequence and proposed files

| Phase | Work | Completion evidence |
|---|---|---|
| 1. Policy and baseline | Agree decisions, author policy and labels, verify model access, measure current behavior | Reviewed policy, fixtures, baseline report |
| 2. Local gate | Add modular contracts, validator adapters, composition/configuration validation, service, graph, fixed fallback, common release path, cache/UI/history protections | Offline routing and substitution tests prove that all candidates pass the gate and blocked drafts stay private |
| 3. Semantic review | Add structured Gemini reviewer; compare two candidates; validate fail-closed behavior | Human-labeled evaluation report with quality, cost, latency, and model choice |
| 4. Enforced demo | Run synthetic shadow experiments, then enable enforcement; document limitations | Release gates met; regression suite and manual UI checks pass |
| 5. Optional recovery | Add one repair attempt or a genuine human-review workflow if measured needs justify it | Recovery improves useful answers without increasing unsafe releases |

| File | Planned change |
|---|---|
| `banking_chatbot/compliance_contracts.py` | Provider-independent request/result schemas, descriptors, and reviewer/validator/service protocols |
| `banking_chatbot/compliance.py` | Public service implementation, policy loading, pure decision reducer, public-result projection |
| `banking_chatbot/compliance_config.py` | Adapter registry/composition, required-check mapping, startup validation, configuration fingerprints |
| `banking_chatbot/compliance_validators.py` | Initial format, sensitive-data, and contact-policy validator adapters |
| `banking_chatbot/compliance_graph.py` | Dependency-injected LangGraph nodes and routing; no concrete provider imports |
| `banking_chatbot/compliance_reviewer.py` | Gemini implementation of reviewer protocol, provider translation, schema validation, normalized failures |
| `banking_chatbot/policies/banking_demo_v1.json` | Versioned rules, scope, actions, fixed response templates |
| `banking_chatbot/chat.py` | Single gate for fresh/cached candidates; released history only |
| `banking_chatbot/cache.py` | Approval metadata/versioning, invalidation, insertion controls |
| `banking_chatbot/gemini.py` | Model availability/settings configuration; preserve generation contract |
| `banking_chatbot/interface.py` | Safe rendering, vetted reference previews, accurate review/cache cost display |
| `banking_chatbot/compliance_evaluation.py` | Reproducible offline/live evaluation entry point |
| `tests/test_compliance.py`, `tests/test_compliance_graph.py`, `tests/test_compliance_reviewer.py` | Rules, routing, schema/error contracts |
| `tests/test_compliance_contracts.py`, `tests/test_compliance_config.py` | Shared adapter conformance, substitution tests, coverage validation, and fingerprint invalidation |
| Existing cache, Gemini, and interface tests | Gate bypass, history, reference display, and failure regressions |
| `data/compliance/` and `reports/` | Synthetic fixtures, split manifest, evaluation summaries |
| `requirements.txt` | Tested pinned LangGraph and explicit schema-validation dependency; optional detectors only after selection |
| `docs/banking-chatbot.md` | Policy behavior, setup, model configuration, limitations, evaluation commands |

For phase 5, allow at most one repair followed by all checks again. Secrets and hard policy violations should go directly to fallback in the initial design. Pass trusted rule IDs and evidence constraints to the repair prompt, not unrestricted evaluator instructions. A human-review feature needs authenticated reviewers, a durable checkpointer, protected drafts, thread ownership, expiry, and revalidation of edited responses. LangGraph supports interruption/resumption but does not provide those application controls automatically. [Interrupt documentation](https://docs.langchain.com/oss/python/langgraph/interrupts).

The first implementable milestone is phases 1–2 with mocked semantic verdicts; it proves the release boundary. It should not be presented as full semantic compliance until phase 3 has passed evaluation.

## Implementation status on output-compliance

The first-release service, contracts, factory registry, local validators, structured
Gemini adapter, LangGraph routing, cache revalidation/eviction, private-history
handling, source-preview sanitization, and UI status are implemented. The existing
cache implementation was brought from `feature/semantic-cache`, because this branch
predated it. Component fingerprints extend the existing cache policy scope without
requiring a new SQLite column. The default reviewer is provisionally Gemini 3.8 Flash.

The evaluation CLI and 24 synthetic seed cases are available. Their labels are drafts,
not independent human labels. The offline fixture report validates data shape and
split separation only. No live benchmark, production accuracy claim, or final model
selection has been completed. Expand/adjudicate the planned dataset and run the
budgeted live comparisons before treating the gate as production validated.

Local checks use conservative pattern/checksum detection. Broader Presidio detection,
Llama Guard hosting, automatic repair, and durable human review remain optional future
work. Audit metadata is session-local and contains policy/component versions, evidence
digest, decisions, usage, and timing. The UI reports generation savings and review
usage separately; monetary/net savings require actual model prices and workload
measurement rather than inferred prices.

Release 2 implements optional one-attempt evidence-based regeneration behind the
`OutputRepairer` protocol (`COMPLIANCE_REPAIR_MODE=once`, off by default). Every
replacement is fully revalidated; hard violations and unavailable checks do not
trigger repair. Repair identity invalidates cached approvals, and repaired cache
candidates are evicted. Human review and independent live model evaluation remain
pending. See the release 2 documentation for supported rules and billing limits.

Release 3 adds blank independent annotation templates, digest-bound agreement checks,
paired reviewer/repair comparisons, repeated-run consistency, and conservative quality
and latency gates. The comparison command defaults to a no-network workload preview.
It never promotes draft labels, counts repeats as extra distinct cases, evaluates a
repaired answer using its original draft label, or automatically changes the model.
Actual human labels, a larger held-out dataset, live benchmark measurements, cost
sign-off, and application human-review infrastructure remain outstanding.
