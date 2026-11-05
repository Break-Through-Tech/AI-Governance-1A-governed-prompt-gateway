# Local banking chatbot: steps 1, 2, and 4

## Scope and data flow

```text
Bundled Kaggle CSV -> validate / normalize / deduplicate -> knowledge.jsonl
                                                       -> quality reports

Chat interface or CLI -> question-only TF-IDF -> cosine ranking -> threshold -> QA pairs
                                                                          -> no-match fallback
```

The foundation runs locally without a GPU, model download, API key, or network call.
The Streamlit interface displays retrieval results in a chat with session history.
It does not include an LLM or gateway policies. Search uses only the current question;
previous messages are displayed but are not used to interpret follow-ups.
A similarity score describes word overlap, not answer confidence.

## Setup and commands

Tested with Python 3.13 on macOS. From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m banking_chatbot prepare
python -m banking_chatbot search "How do I activate my debit card?"
python -m banking_chatbot search "Explain quantum entanglement"
python -m banking_chatbot search "Where can I see my account balance?" --top-k 2 --threshold 0.6
python -m banking_chatbot evaluate
python -m unittest discover -s tests -v
```

Default file locations are resolved relative to the package, not the current working
directory. Run module commands from the repository root, or put it on `PYTHONPATH`.
Dependencies are pinned in `requirements.txt`; the virtual environment is ignored by Git.

## Chat interface

After setup, start the app from the project root:

```bash
python -m streamlit run app.py
```

Open http://127.0.0.1:8501. Stop the server with Ctrl+C in its terminal.

- Enter a complete question or choose one of three example buttons.
- See the retrieved questions and complete original answers, ranked by similarity.
  These are dataset excerpts, not generated chatbot answers.
- Expand each result's source details to see its file, row numbers, reference ID,
  and any review flags. No-match responses suggest rephrasing.
- Change maximum results (1–5) and minimum similarity under Search settings.
  Settings apply to new searches only; historical results retain their original settings.
- Reset chat clears the conversation for the current session, while retaining search settings.
  Sessions are independent. History is kept in server memory for that browser session,
  is not written to disk, and is lost when the session ends.
- Search time measures retrieval only, not page rendering or index initialization.

The shared read-only index is cached by the knowledge file's content hash, so a
changed dataset is picked up on the next app rerun. Results already in chat history
remain snapshots. If the data is missing or malformed, the page provides the
preparation command instead of a broken chat input.

The app binds to localhost by default and Streamlit usage telemetry is disabled
in `.streamlit/config.toml`. No model, external inference API, or model key is involved.
Markdown in user messages and reference text is escaped for literal display.

To prepare a replacement CSV with `Query,Response` columns without overwriting the
bundled outputs:

```bash
python -m banking_chatbot prepare --input /path/to/input.csv \
  --output /path/to/knowledge.jsonl --report /path/to/quality.json
python -m banking_chatbot search "Your question" --knowledge /path/to/knowledge.jsonl
```

The report command also writes a Markdown report beside the JSON report. Encoding
defaults to strict UTF-8, falling back to Windows-1252; use `--encoding` for a known
different encoding. Invalid schemas, malformed records, and empty datasets fail
with a readable error. Preparation refuses overlapping input/output paths.

## Preparation choices

- Preserve the downloaded CSV and its checksum.
- Normalize Unicode typography and whitespace, then write UTF-8 JSON Lines.
- Drop records with empty questions or answers. None occur in the bundled source.
- Deduplicate case-insensitive normalized question/answer pairs. Preserve the first
  wording and all source rows. One duplicate is removed from the bundled source.
- Use a hash of the normalized pair for IDs that do not depend on source ordering.
- Keep answers paired with their questions; no chunking or generated replacements.
- Retain and flag different answers for identical questions. Also flag possible
  contact details and fictional-bank claims; these flags do not certify other rows.

Generated reports are deterministic for the same source and configuration.
The manual review and provenance documents are separate from generated reports.

## Retrieval choices and integration

`Retriever` uses scikit-learn word unigram/bigram TF-IDF with English stop words,
sublinear term frequency, and cosine similarity. Only questions enter the index;
complete answers and provenance are returned from matching records. The small
144-record index is rebuilt in memory when a retriever is created. Keep one instance
alive in a future app; no database or unsafe serialized Python index is needed.

```python
from banking_chatbot.retrieval import Retriever

retriever = Retriever()
result = retriever.search("How do I activate my debit card?")
# result: status, query, threshold, message, matches
# each match: id, question, answer, score, source_file, source_rows, review_flags
```

Default `top_k` is 3 and threshold is 0.50. Every returned match must meet the threshold
and have positive similarity. Zero-overlap queries never return arbitrary records,
even if the threshold is zero. Ties are ordered by stable record ID. Empty queries
and invalid bounds raise a clear error. A no-match result supplies a clarification
message. Results are reference candidates, not a generated response or a guarantee
of relevance; multiple candidate topics may still be returned.

## Threshold development and limitations

`evaluate` runs 20 manually authored cases (14 supported, 6 unsupported), sweeps
thresholds from 0.10 through 0.80 in 0.05 steps, and writes the full results with
knowledge/case checksums. It selects the maximum mean of supported recall@3 and
unsupported rejection rate; ties prefer greater rejection, then a lower threshold.
The selected threshold is recorded in code; evaluation does not silently change it.

At 0.50, the current development set retrieves an expected reference for 12/14
supported questions and rejects 6/6 unsupported ones. The two supported misses are
"How do I send money abroad?" and "Hello". The source uses different words, including
"internationally" and "Hi". A lower threshold improves coverage but also accepts some
unsupported banking questions. These are calibration results on a tiny set, not a
held-out benchmark or a guarantee about unseen prompts.

Do not use similarity to decide whether a request is safe or whether the bank offers
a product. English stop-word removal and lexical matching can lose negation and
meaning. Ambiguous queries, follow-ups, synonyms, and unsupported questions sharing
banking vocabulary remain limitations. An independent evaluation set and semantic
retrieval could be added later; the current work intentionally has no model dependency.

## Files

- `app.py`: Streamlit entry point.
- `banking_chatbot/interface.py`: chat interface, session history, and index cache.
- `.streamlit/config.toml`: local server settings and visual theme.
- `banking_chatbot/data.py`: preparation, validation, normalization, IDs, reports.
- `banking_chatbot/retrieval.py`: reusable retrieval component.
- `banking_chatbot/evaluation.py`: development-set threshold sweep and metrics.
- `banking_chatbot/__main__.py`: command-line interface.
- `data/banking/`: original source, cleaned knowledge, and development cases.
- `reports/`: quality report, manual source review, and calibration output.
- `tests/test_banking.py`: data integrity, provenance, retrieval, and CLI checks.
- `tests/test_interface.py`: Streamlit integration checks for results, history, reset,
  settings, session isolation, example questions, and missing-data recovery.

## Output compliance

Every generated answer and exact/semantic cache candidate now passes a synchronous
LangGraph release gate. Local format, sensitive-data, and contact checks run first.
A separate Gemini reviewer checks grounding and banking policy with structured JSON.
The application decides `allow`, `block`, or `unverified`; missing checks, provider
failures, invalid verdicts, and uncertainty never release the draft. No automatic
repair or human approval queue is enabled.

Install the pinned dependencies in `requirements.txt`. The generator retains its
existing model configuration. Choose the independent reviewer before launch:

```bash
export COMPLIANCE_REVIEW_MODEL=gemini-3.8-flash
# Alternatives: gemini-3.5-flash-lite or legacy gemini-2.5-flash if accessible.
python -m streamlit run app.py
```

The reviewer uses the configured Gemini credential. Review is an additional API call
and may incur charges, including on cache hits. Invalid reviewer configuration
withholds generated answers. Retrieval-only mode remains offline; source previews
remove recognized sensitive content and demo contacts but do not claim semantic
approval. Cached entries from the previous policy scope are not reused. Cache
savings displayed in the UI describe avoided generation only; review usage and
latency are reported separately, and no dollar savings are claimed without prices.

The first release prohibits all output phone numbers, email addresses, and URLs
for this fictional demo. Regex/checksum detectors are deliberately limited and may
miss identifiers or overblock legitimate examples. The reviewer is fallible and
may share errors with the generator. “Passed configured checks” is not certification
of correctness or legal compliance. Use synthetic data.

### Replacing components

Implement `OutputValidator.validate(ValidationRequest) -> CheckResult` or
`ComplianceReviewer.review(ReviewRequest) -> ReviewResult`, and provide a versioned
`ComponentDescriptor`. Contracts live in `compliance_contracts.py`.

```python
from banking_chatbot.compliance_config import build_compliance_service

service = build_compliance_service(
    reviewer=my_reviewer,
    validators=(my_format_validator, my_sensitive_validator, my_contact_validator),
)
# Pass compliance_service=service to chat(), or call service.evaluate(candidate, context).
```

Alternatively register trusted validator factories with `validator_factories` and
select them with `validator_ids`; `reviewer_factory` supplies provider construction.
The graph and UI contain no concrete reviewer imports. Changing a model setting,
component version, policy, prompt, or schema changes the compliance fingerprint
included in cache scope. Explicit required checks prevent replacing a comprehensive
reviewer with an adapter that covers only some rules. Test-only adapters require an
explicit `allow_test=True` composition option and are never selected from environment
configuration. Update descriptor versions whenever implementation behavior changes.

Adapters return one explicit outcome per declared rule. The service rejects missing
or duplicate findings, unknown source IDs, inconsistent claim metadata, and a
passing grounding verdict with unsupported claims. A failure wins over passes;
an unavailable required check yields `unverified`. Private claims and drafts remain
in transient local objects, not graph state, returned turns, or routine audit output.
Only previously approved answers are included in model history.

### Evaluation

```bash
# Offline: validate fixture contracts and split separation. Makes no API calls.
python -m banking_chatbot.compliance_evaluation
python -m unittest discover -s tests -v

# Explicit live synthetic evaluation: uses the configured Gemini key and may cost money.
COMPLIANCE_REVIEW_MODEL=gemini-3.8-flash python -m banking_chatbot.compliance_evaluation \
  --live --split development --limit 10 --output reports/compliance-flash.json
COMPLIANCE_REVIEW_MODEL=gemini-3.5-flash-lite python -m banking_chatbot.compliance_evaluation \
  --live --split development --limit 10 --output reports/compliance-lite.json
```

`data/compliance/seed.jsonl` contains 24 draft-labeled synthetic scenarios, split by
scenario family. These are seed fixtures, not the planned independently labeled
240-example benchmark. Expand and adjudicate labels with the team before model
selection or a production release. Reports distinguish uncertain verdicts from
blocks and report compliant withholding as well as violation recall, false blocks,
unsafe releases, per-rule results, tokens, and latency. Do not tune on held-out cases.
Actual model availability and review quality must be established through live tests;
unit tests mock network calls and cannot establish semantic accuracy.

### Release 2: bounded repair

Set `COMPLIANCE_REPAIR_MODE=once` to enable one regeneration attempt for
`RESPONSE_FORMAT`, `SUPPORTED_FACTS`, or `FICTIONAL_CONTEXT` failures. The default
is `off` while live evaluation is pending. Unknown modes disable generated-answer
release with a configuration error. This can add one generation and another review
call; the existing generator may retry a transient HTTP failure once.

The repair adapter receives the question, sanitized evidence, trusted policy, and
rule IDs. It never receives the rejected draft or reviewer prose. The generated
candidate runs through all local and semantic checks again. A second failure,
exception, incomplete repair, or uncertain verdict returns the fixed fallback.
Sensitive-data, contacts, claimed actions, recommendations, and content-safety
failures are not repairable. Payload limits and local sensitive-data screening
apply before repair, even when the first pass stopped at a local format check.

Implement `OutputRepairer.repair(RepairRequest) -> RepairResult` to replace the
repair model and inject it using `repairer=` in the service factory. Its descriptor
uses role ID `repairer`. The default adapter reuses the existing Gemini generator
and its current default model. Repair configuration participates in cache identity.
A cache candidate requiring repair is evicted; the failed original is never served.
Audit results include attempt count and initial rule IDs. Usage aggregates successful
review/repair calls; unknown or failed-call billing remains unavailable. Regenerated
answers do not inherit token counts or latency measurements from the original draft.

This release does not establish improved semantic accuracy. Compare repair enabled
and disabled on independently labeled cases before enabling it for broader use.

### Release 3: independent labeling and paired comparisons

Create separate blank annotation files for two people. Each person reviews the
synthetic cases and fills `decision` with `allow` or `block`; leave it `null` if
undecided. Templates omit the draft expected labels. The dataset remains the source
for the question, answer, and evidence. Use local reviewer identifiers rather than
personal contact information.

```bash
python -m banking_chatbot.compliance_labels --reviewer reviewer-a --output /tmp/labels-a.json
python -m banking_chatbot.compliance_labels --reviewer reviewer-b --output /tmp/labels-b.json
```

The adjudicator requires distinct declared reviewer IDs, the exact dataset digest,
complete case-ID coverage, and agreement. Disagreement or missing labels stays
unresolved; draft labels are never silently promoted to independent labels. This
is an offline team workflow, not authenticated identity verification or an application
human-approval queue. Templates refuse to overwrite existing annotation files.

Preview the workload without reading credentials or making network calls:

```bash
python -m banking_chatbot.compliance_benchmark --repair both --repeats 2 \
  --labels /tmp/labels-a.json /tmp/labels-b.json \
  --output /tmp/comparison-plan.json
```

Add `--live` to execute (may incur charges). `--models` selects reviewer candidates;
`--repair off|once|both` selects explicit variants independent of the application's
environment setting. The same cases are paired across variants with alternating
model order. `--limit` caps distinct cases and `--repeats` caps observations per case.
Dry-run reports include a conservative maximum HTTP call count, including repair
and the existing generator's transient-error retry. No automatic model deployment
or configuration change follows a benchmark.

For a release-quality assessment, use independently agreed held-out labels and an
expanded dataset, for example `--split held_out --limit 120 --max-p95-ms 10000` (choose
the latency budget before tuning). The bundled 12 held-out seed cases are insufficient.
Reports include per-rule metrics, repeated-decision inconsistency, observed token
usage, configuration fingerprints, and explicit readiness failures. Repeated runs do
not increase the number of distinct test cases. All policy rules must be represented.
Production approval stays false: policy sign-off and measured monetary cost remain
separate requirements. Token data is not a dollar-cost estimate.

Repaired outputs require new independent labels. Their original draft labels are
excluded from correctness metrics and cannot support repair-release approval.
Unverified outcomes remain separate from detected violations; withholding every
answer cannot pass the quality gate. Reports contain case IDs and metrics, not raw
candidate text, credentials, or evidence excerpts.
