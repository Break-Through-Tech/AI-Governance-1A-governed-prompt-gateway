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
