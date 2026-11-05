"""Stage 6 adapter: Joohye's output-compliance checks behind ``check_compliance``.

Her gate (``banking_chatbot.compliance*``) has two layers:

* three local validators (format, sensitive data, contact details): plain
  Python, no network, deterministic;
* a semantic reviewer: a second Gemini call that judges the remaining rules
  (unsupported facts, account-action claims, personal recommendations, ...)
  against the retrieved FAQ references.

The gateway runs non-RAG, so there are no references, and her reviewer refuses
to run without them. This adapter therefore enforces the local layer only and
says so in the record: ``unreviewed`` lists every policy rule that still needs
the reviewer. Nothing in her package is edited, and her policy file
(``policies/banking_demo_v1.json``) stays the single source of the rules.

Any failure to run a check withholds the answer. A compliance gate that lets
answers through when it breaks is not a gate.
"""

from __future__ import annotations

from functools import lru_cache

from banking_chatbot.compliance_config import VALIDATOR_FACTORIES, configured_components, load_policy
from banking_chatbot.compliance_contracts import Candidate, ComplianceContext, ValidationRequest

from .schema import ComplianceDecision, GenerationResult

LOCAL_VALIDATORS = ("format", "sensitive", "contact")
WITHHELD = "The answer was blocked because the compliance check could not run."

# Customer-facing wording for the rules the local validators can fail. Any other rule falls
# back to the description in her policy file.
CUSTOMER_REASONS = {
    "SENSITIVE_DATA": "it contained sensitive information, such as a card or account number",
    "CONTACT_POLICY": "it contained a phone number, email address or web link, which this demo does not give out",
    "RESPONSE_FORMAT": "it was longer than the {max_words}-word limit",
    "OUTPUT_COMPLETE": "it was empty or cut off before it finished",
}


def _blocked_message(rules, policy) -> str:
    reasons = [
        CUSTOMER_REASONS.get(rule.id, "it broke this rule: " + rule.description.rstrip(".")).format(max_words=policy.max_words)
        for rule in rules
    ]
    return "The answer was blocked because " + ", and ".join(reasons) + "."


@lru_cache(maxsize=1)
def _setup():
    """Load her policy and build her validators once per process."""
    return load_policy(), configured_components(LOCAL_VALIDATORS, VALIDATOR_FACTORIES)


def policy_version() -> str:
    """Used by the cache adapter: answers approved under one policy are not reused under another."""
    return _setup()[0].version


def check(query: str, generation: GenerationResult) -> ComplianceDecision:
    """Run her local validators on a generated answer and translate the findings into our record."""
    try:
        policy, validators = _setup()
    except Exception:
        return ComplianceDecision(status="fail", flags=["CHECK_UNAVAILABLE"], message=WITHHELD,
                                  reason="the compliance policy could not be loaded, so the answer is withheld")

    ran = [validator.descriptor.component_id for validator in validators]
    # A rule is fully checked only if every component it requires actually ran.
    unreviewed = [rule.id for rule in policy.rules if not set(rule.required_checks) <= set(ran)]
    common = dict(checks=ran, unreviewed=unreviewed, policy_version=policy.version)

    try:
        request = ValidationRequest(
            candidate=Candidate(text=generation.answer, finish_reason=generation.finish_reason or ""),
            context=ComplianceContext(request_id="gateway", question=query, references=()),
            policy=policy,
        )
        failed: set[str] = set()
        for validator in validators:
            failed.update(f.rule_id for f in validator.validate(request).findings if f.outcome != "pass")
    except Exception:
        return ComplianceDecision(status="fail", flags=["CHECK_UNAVAILABLE"], message=WITHHELD,
                                  reason="a compliance check could not run, so the answer is withheld", **common)

    if failed:
        rules = [rule for rule in policy.rules if rule.id in failed]  # keep the policy's own order
        return ComplianceDecision(
            status="fail",
            flags=[rule.id for rule in rules],
            message=_blocked_message(rules, policy),
            reason="failed: " + "; ".join(f"{rule.id} ({rule.description})" for rule in rules),
            **common,
        )
    return ComplianceDecision(
        status="pass",
        reason=(f"local checks passed ({', '.join(ran)}). {len(unreviewed)} rules were not reviewed: "
                "they need the semantic reviewer, which requires retrieved references"),
        **common,
    )
