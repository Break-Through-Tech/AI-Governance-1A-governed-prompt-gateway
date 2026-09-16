# Local banking chatbot: steps 1 and 2

## Scope and data flow

```text
Bundled Kaggle CSV -> validate / normalize / deduplicate -> knowledge.jsonl
                                                       -> quality reports

User question -> question-only TF-IDF -> cosine ranking -> threshold -> up to 3 QA pairs
                                                                  -> no-match fallback
```

The foundation runs locally without a GPU, model download, API key, or network call.
It does not yet include an LLM, conversation memory, a chat interface, or gateway
policies. A similarity score describes word overlap, not answer confidence.

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

- `banking_chatbot/data.py`: preparation, validation, normalization, IDs, reports.
- `banking_chatbot/retrieval.py`: reusable retrieval component.
- `banking_chatbot/evaluation.py`: development-set threshold sweep and metrics.
- `banking_chatbot/__main__.py`: command-line interface.
- `data/banking/`: original source, cleaned knowledge, and development cases.
- `reports/`: quality report, manual source review, and calibration output.
- `tests/test_banking.py`: data integrity, provenance, retrieval, and CLI checks.
