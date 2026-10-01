# Banking chatbot dataset

- Author: Manoj A S (`manojajj` on Kaggle).
- Source: https://www.kaggle.com/datasets/manojajj/banking-chatbot/data
- Download: https://www.kaggle.com/api/v1/datasets/download/manojajj/banking-chatbot
- Retrieved: September 16, 2026; data card showed version 1.
- License listed by the publisher: CC0 / Public Domain,
  https://creativecommons.org/publicdomain/zero/1.0/.
- Publisher describes synthetic educational/practice question-answer data with no
  sensitive, personal, or classified information about individuals or banks.
- Raw file: `raw/Dataset_Banking_chatbot.csv`, preserved byte for byte from the ZIP.
- Raw SHA-256: `e97f0767b8abed835974ad62ac4650d1517359f534b31a4909f1849ebe303b17`.
- Actual schema: `Query`, `Response`; 145 data records; Windows-1252 encoding.

`processed/knowledge.jsonl` is a reproducible UTF-8 derivative: 144 unique pairs,
stable content-based IDs, all original source row references, and review flags.
Whitespace and typography are normalized; substantive answers are not rewritten.
Rows 16 and 97 are the same question/answer pair and share one cleaned record.
Different answers to identical questions are retained for review. These records
contain no intent labels and must not be treated as Banking77 data.

Row references are 1-based CSV data records, excluding the header. Reordering the
source changes row references, but unchanged normalized question/answer content
keeps its ID. Editing either the question or answer intentionally changes the ID.

`retrieval_dev.json` contains 20 manually authored development cases referencing
this exact source snapshot. It is used for threshold selection, not as an independent
test set. Do not add these prompts to the retrieval index or present their scores
as held-out evaluation. Create separate cases before making general quality claims.

Treat all bank-specific statements, service availability, contact information, and
procedures as fictional demo content. See the manual source review under `reports/`.
