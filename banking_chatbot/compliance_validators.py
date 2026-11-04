"""Local detectors. Findings never rewrite or release an answer."""
import re
from .compliance_contracts import ComponentDescriptor, CheckResult, Finding

SSN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
SECRET = re.compile(r'\b(?:AIza[\w-]{25,}|sk-[\w-]{16,}|(?:api[_ -]?key|password|secret)\s*[:=]\s*\S+)', re.I)
ACCOUNT = re.compile(r'\b(?:account|routing)\s*(?:number|no\.?|#)?\s*[:=]?\s*\d{6,}', re.I)
CARD = re.compile(r'(?<!\d)(?:\d[ -]?){13,19}(?!\d)')
EMAIL = re.compile(r'\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b')
PHONE = re.compile(r'(?<!\w)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b')
URL = re.compile(r'https?://\S+|\bwww\.\S+', re.I)


def sensitive(text):
    if SSN.search(text) or SECRET.search(text) or ACCOUNT.search(text):
        return True
    for match in CARD.finditer(text):
        digits = [int(c) for c in match.group() if c.isdigit()]
        total = sum((d * 2 - 9 if d * 2 > 9 else d * 2) if i % 2 else d
                    for i, d in enumerate(reversed(digits)))
        if 13 <= len(digits) <= 19 and total % 10 == 0:
            return True
    return False


def safe_preview(text):
    if sensitive(text):
        return '[Sensitive reference content withheld]'
    for pattern in (EMAIL, PHONE, URL):
        text = pattern.sub('[Demo contact removed]', text)
    return text


def result(descriptor, outcomes):
    return CheckResult(component_id=descriptor.component_id, findings=tuple(
        Finding(component_id=descriptor.component_id, rule_id=rule,
                outcome='fail' if failed else 'pass') for rule, failed in outcomes.items()))


class FormatValidator:
    descriptor = ComponentDescriptor(component_id='format', version='1',
                                     rule_ids=('OUTPUT_COMPLETE', 'RESPONSE_FORMAT'))
    def validate(self, request):
        c = request.candidate
        return result(self.descriptor, {
            'OUTPUT_COMPLETE': not c.text.strip() or c.finish_reason != 'STOP',
            'RESPONSE_FORMAT': len(c.text.split()) > request.policy.max_words,
        })


class SensitiveDataValidator:
    descriptor = ComponentDescriptor(component_id='sensitive', version='1', rule_ids=('SENSITIVE_DATA',))
    def validate(self, request):
        return result(self.descriptor, {'SENSITIVE_DATA': sensitive(request.candidate.text)})


class ContactPolicyValidator:
    descriptor = ComponentDescriptor(component_id='contact', version='1', rule_ids=('CONTACT_POLICY',))
    def validate(self, request):
        return result(self.descriptor, {'CONTACT_POLICY': any(
            p.search(request.candidate.text) for p in (EMAIL, PHONE, URL))})
