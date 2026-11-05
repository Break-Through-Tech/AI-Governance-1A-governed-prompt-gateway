"""What the customer is shown, derived from the decision record.

Every run ends in one of four ways, and each gets a reply in plain words:

* answered or cached: the answer itself;
* blocked: which gate stopped it and why;
* error: that the model could not answer.

The text is stored in ``DecisionRecord.response`` so the audit log holds what
the customer actually saw, not only the internal decisions behind it.
"""

from __future__ import annotations

from .schema import DecisionRecord


def customer_response(record: DecisionRecord) -> str:
    status = record.final_status
    if status in ("answered", "cached") and record.generation is not None:
        return record.generation.answer

    if status == "blocked":
        guardrails, compliance = record.input_guardrails, record.output_compliance
        if guardrails is not None and guardrails.status == "block":
            # The question never reached a model. Give the category, which Team A's reason carries.
            why = guardrails.reason.strip().rstrip(".") or "it did not pass the input safety check"
            return f"Your question was blocked before it reached the model: {why}."
        if compliance is not None and compliance.message:
            return compliance.message
        return "The answer was blocked by the output compliance check."

    if status == "error":
        detail = record.generation.message if record.generation is not None else ""
        return ("Sorry, the model could not answer right now. " + detail).strip()

    return ""
