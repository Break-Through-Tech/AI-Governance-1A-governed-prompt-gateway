"""Test-only adapters; production composition explicitly rejects these."""
from banking_chatbot.compliance_contracts import ComponentDescriptor, ReviewResult, Finding, Usage
from banking_chatbot.compliance_reviewer import RULES
from banking_chatbot.compliance_config import build_compliance_service


class FakeReviewer:
    descriptor = ComponentDescriptor(component_id='reviewer', version='test-1', rule_ids=RULES, test_only=True)
    def __init__(self, outcome='pass'):
        self.outcome, self.calls = outcome, []

    def review(self, request):
        self.calls.append(request)
        return ReviewResult(component_id='reviewer', findings=tuple(
            Finding(component_id='reviewer', rule_id=r, outcome=self.outcome) for r in RULES),
            claims=(), no_factual_claims=True, usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15))


def service(reviewer=None):
    return build_compliance_service(reviewer=reviewer or FakeReviewer(), allow_test=True)
