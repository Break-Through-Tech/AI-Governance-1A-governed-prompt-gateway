# Semantic cache demo

Use the **Governed Prompt Gateway** on port 8502 for this walkthrough. It runs single-turn requests and can generate a placeholder answer locally. The banking FAQ chat on port 8501 has different cache requirements.

## Run the demo

From the project directory, start the gateway with the project's Python interpreter. This also works after moving the checkout.

```sh
.venv/bin/python -m streamlit run gateway_app.py --server.port 8502
```

Open [the local gateway](http://127.0.0.1:8502). In **Single request**, choose **semantic** for **Response cache**, turn off **Call Gemini for real** and **Input guardrails**, and leave **Intent override** on the classifier. Submit these prompts in order. The expected results assume a fresh cache and the default settings.

| Step | Submit this prompt | What to show in stage 2 |
| --- | --- | --- |
| 1 | How do I reset my PIN? | **MISS**. The gateway generates an answer, checks it and stores it if approved. |
| 2 | What is the process to reset my personal identification number? | **HIT · semantic**, similarity 1.000. The two wordings have the same normalized features. The placeholder quotes the first prompt because it is the stored answer. |
| 3 | How do I reset my PIN? | **HIT · exact**. The original wording takes the exact lookup path first. |
| 4 | How do I activate my debit card? | **MISS**. This starts a separate FAQ entry. |
| 5 | What are the steps for activating a debit card? | **HIT · semantic**. The generation record shows the cache as provider and zero answer-generation tokens. |
| 6 | How do I not activate my debit card? | **MISS**. The negation requires exact wording for reuse. |
| 7 | What is my account balance? | **BYPASS**, classified as personalized. Repeating it also bypasses. |
| 8 | What are your branch opening hours? | **MISS**. The first question about the general branch schedule seeds the cache. |
| 9 | When are bank branches open? | **HIT · semantic**, similarity 1.000. The response is the approved answer from step 8, with zero answer-generation tokens. |
| 10 | When are bank branches open on Sunday? | **MISS**. The day is part of the question. A general schedule cannot answer this more specific request from cache. |

On a hit, the final status is `cached`. Intent, routing, fresh answer generation and output compliance are skipped. The input stage still runs, including its Gemini call if you enabled input guardrails.

The SQLite cache persists across page refreshes and process restarts. If step 1 hits during rehearsal, an earlier run already seeded it. For a cold demo, stop gateway processes that use the project, move `var/gateway_cache.sqlite3` to a backup location if it exists, then restart. The audit log is independent. The CLI defaults to caching off; use `.venv/bin/python -m gateway "How do I reset my PIN?" --cache semantic --no-log` to exercise the same cache from the same directory.

## Supported cases and boundaries

| Case | Expected behavior |
| --- | --- |
| Banking terms and common English question frames | “What is a wire transfer?” followed by “Could you explain a wire transfer?” hits. “How do I update my phone number?” followed by “Please show me how to change my telephone number.” also hits. |
| Opening hours | The branch-hours pair also hits in reverse order with a cold cache. The same pattern matches “When is the support desk open?” with “What are the operating hours of your support desks?” A named location such as Boston stays separate. |
| Fees | “How much does a wire transfer cost?” followed by “What is the fee for a wire transfer?” hits. |
| Duration | “How long does a card replacement take?” followed by “What is the processing time for a card replacement?” hits. |
| Changed action or product | After seeding debit-card activation, asking how to **deactivate** that debit card or activate a **credit** card misses. The existing vocabulary treats unqualified PIN change and reset questions as equivalent. Adding “forgotten” to the reset question produces a miss in the tested pair. |
| Negation and explicit numbers | Detected negations, digits and supported currency or percent symbols require an exact match. This applies in both directions. A 12-month CD question cannot supply a semantic answer for a 24-month CD question. An eligible exact repeat can still hit. |
| Policy bypass | A balance inquiry, “Transfer $500 to my savings”, a current exchange-rate request, and input with a masked PIN bypass lookup and storage. The eligibility rules use patterns and cover a limited set of phrasings. |
| Competing answers | The best eligible candidate must score at least 0.90 and lead the runner-up by at least 0.05 with the defaults. A close tie generates a fresh answer. |
| Cache modes | `exact` serves eligible repeats. `semantic` tries exact first, then semantic. `semantic-shadow` reports a qualifying semantic candidate as a miss and generates a fresh answer; exact hits can still be served. `off` bypasses. |
| Provider, version and lifetime | Live and simulated answers use separate scopes. Semantic entries from the previous embedder require fresh seeding. Default expiry is 24 hours and capacity is 1,000 entries. |

### Known limitations

- **Indirect questions can still miss.** With only the general branch schedule cached, “How late can I visit a bank branch?” misses. The matcher recognizes common English forms for hours, fees and duration. When it recognizes one, the other question must have the same type and normalized subject, including any day, location or product qualifiers. Other questions use word and character similarity. Multilingual matching has no validated coverage.
- **Routing controls run after lookup.** A hit skips intent classification and routing, including a manual intent override. Use cache mode `off` when demonstrating a forced tier. The gateway partitions entries by provider and configured tiers, and uses one pre-routing model scope.
- **The banking FAQ chat uses a different database.** In `app.py`, retrieval-only requests bypass answer caching. AI answer reuse requires the same recent history, ordered FAQ sources and content, model, policy, and other scope settings. A follow-up in the same conversation can legitimately miss. To compare two first-turn prompts, reset chat between them and use `CACHE_MODE=semantic`; different retrieved rows can still prevent a hit.
- **Changing modes can change the available entries.** Exact-only entries have no embedding version. Semantic and shadow modes share a versioned scope. A rehearsal in shadow mode can therefore affect the candidates available during the live walkthrough.

## What caused the missed and incorrect hits

In isolated runs of the original gateway, “What is the process to reset my PIN?” scored **0.665** against “How do I reset my PIN?”. Spelling out “personal identification number” scored **0.336**. Meanwhile, adding “not” to the debit-card activation question scored **0.979** and incorrectly served the cached answer. The word features used a broad English stop list, while the character features retained the surrounding question wording. The gateway also used SQLite's defaults even when the operator set the semantic threshold or margin.

The revised embedder normalizes common question frames, banking abbreviations and selected synonyms before scoring. Negated and numeric requests have an exact-only path. The gateway reads `SEMANTIC_CACHE_THRESHOLD` and `SEMANTIC_CACHE_MARGIN` when constructing a semantic cache. Restart the app after changing these values. The walkthrough uses the original defaults.

The earlier matcher scored the branch-hours pair at **0.274**. The current matcher extracts the type of information requested and the subject from common question forms. Both demo questions yield `hours` and `branch`; a Sunday-specific question yields `hours` and `branch on sunday`. The patterns apply across subjects and run offline. The score of 1.000 means the normalized features match. It does not establish whether a particular branch is open. The embedding version changed, so seed the first request again after upgrading.

## Other demo features

| Feature | Local demo behavior and limits |
| --- | --- |
| Generation and cost | The simulated provider returns a labeled placeholder and estimated usage. Its content is unsuitable for demonstrating banking advice. Live Gemini generation and the optional input policy check require an API key and were not exercised in this rehearsal. |
| Intent and routing | In the supplied virtual environment, the intent adapter reports `unknown` because `litellm` is missing. The rule-based router still selects a tier and produces estimates. A manual intent override works on a fresh request. |
| Output compliance | Fresh gateway answers run the local format, sensitive-data and contact checks before storage. The non-retrieval gateway lists rules requiring the semantic reviewer under `unreviewed`. |
| Workload tab | The bundled simulated workload exercises routing and cost reporting. It runs without the single-request tab's cache. Use **Single request** for the cache demonstration. |
