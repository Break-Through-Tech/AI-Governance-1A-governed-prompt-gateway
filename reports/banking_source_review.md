# Manual review of the synthetic source

Reviewed the 145 records in the bundled Kaggle CSV on September 16, 2026.
This review applies to SHA-256
`e97f0767b8abed835974ad62ac4650d1517359f534b31a4909f1849ebe303b17`.
It checks internal wording and demo suitability; it does not verify banking policy.
Source row numbers exclude the CSV header.

| Finding | Source rows | Treatment |
| --- | --- | --- |
| Repeated credit-card activation question and answer | 16, 97 | Merge the exact pair and retain both source references. |
| Same password-reset question, different answer wording | 7, 127 | Retain both: one describes email reset; the other also offers customer support. No clear direct contradiction. |
| Same investment-services question, different answer wording | 41, 128 | Retain both; both claim the fictional bank offers investment services. |
| Same inactivity-fee question, different answer wording | 84, 139 | Retain both; row 139 adds "for details". No substantive conflict. |
| Demo telephone and email details | 6, 46 | Preserve and flag; do not treat these as real support destinations. |
| Fictional persona, bank identity, and unverified scale claims | 144, 145 | Retain as source material; "Alice", "XYZ Bank", and branches in five cities are not verified facts. |
| Account closure instructions differ in detail | 10, 124 | One mentions online closure, the other support or a branch. Not necessarily contradictory, but neither is authoritative. |
| PIN changes, joint holders, travel notifications, and product availability are bank-specific | 24, 40, 43, 64, 94 | Retain for a fictional demo; do not generalize to real banks. |
| Rates, fees, withdrawal limits, and hours refer elsewhere | 9, 14, 23, 26, 62 | The dataset does not supply current numeric values. Retrieval cannot answer requests for exact values. |

Several other questions paraphrase the same topic (alerts, statements, account
settings, and car loans). They remain separate reference records because exact-pair
deduplication is deliberately conservative. The automatic flags highlight contact
details, first-person bank claims, and repeated questions; they are not an exhaustive
contradiction detector. Rerun manual review if source content changes.
