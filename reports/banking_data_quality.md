# Banking dataset quality report

Source: `Dataset_Banking_chatbot.csv`; decoded as `cp1252`.

- Input records: 145
- Cleaned records: 144
- Empty records removed: 0
- Duplicate pairs removed: 1
- Records with normalized typography/spacing: 1
- Questions with multiple answer variants: 3

## Questions needing review

- What should I do if I forget my online banking password? (2 answer variants retained)
- Do you offer investment services? (2 answer variants retained)
- What is your policy on account inactivity fees? (2 answer variants retained)

## Contact details

- Source rows [46]: Do not click any links or provide personal information. Forward the email to our fraud department at fraud@example.com for investigation.
- Source rows [6]: Report a lost or stolen debit card immediately by calling our customer service at 1-800-123-4567 or through our mobile app.

## Limits

- Synthetic educational content; not verified policies of a real bank.
- Different answers to the same question are retained and flagged, not declared contradictions.
- Heuristic review flags are not a complete factual or contradiction check.
- Source row numbers count CSV data records starting at 1, excluding the header.

See the JSON report for all flagged record IDs and the source checksum.
