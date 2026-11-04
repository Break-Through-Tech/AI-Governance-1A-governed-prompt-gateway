"""Provider-neutral policy decisions and the application's release service."""
from time import perf_counter
import hashlib
import json
from .compliance_contracts import (
    Candidate, ComplianceContext, ComplianceResult, ValidationRequest, ReviewRequest,
    CheckResult, ReviewResult, Finding, Usage, RepairRequest, RepairResult,
)
from .compliance_config import validate_configuration
from .compliance_validators import sensitive, safe_preview


def decide(policy, results):
    outcomes = {(r.component_id, f.rule_id): f.outcome for r in results for f in r.findings}
    required = [(c, rule.id) for rule in policy.rules for c in rule.required_checks]
    failures = sorted({rule for c, rule in required if outcomes.get((c, rule)) == 'fail'})
    if failures:
        return 'block', tuple(failures)
    missing = sorted({rule for c, rule in required if outcomes.get((c, rule)) != 'pass'})
    return ('unverified', tuple(missing)) if missing else ('allow', ())


def checked_result(component, raw, context, review=False):
    cls = ReviewResult if review else CheckResult
    value = cls.model_validate(raw.model_dump())
    descriptor = component.descriptor
    findings = value.findings
    if (value.component_id != descriptor.component_id or
        len(findings) != len(descriptor.rule_ids) or
        {f.rule_id for f in findings} != set(descriptor.rule_ids) or
        any(f.component_id != descriptor.component_id for f in findings)):
        raise ValueError('Invalid check coverage')
    source_ids = {r.id for r in context.references}
    if any(not set(f.source_ids) <= source_ids for f in findings):
        raise ValueError('Unknown evidence')
    if review:
        if value.no_factual_claims == bool(value.claims):
            raise ValueError('Inconsistent claim declaration')
        for claim in value.claims:
            if not set(claim.source_ids) <= source_ids or (claim.assessment == 'supported' and not claim.source_ids):
                raise ValueError('Invalid claim evidence')
        grounding = next((f for f in findings if f.rule_id == 'SUPPORTED_FACTS'), None)
        if grounding and grounding.outcome == 'pass' and any(c.assessment != 'supported' for c in value.claims):
            raise ValueError('Contradictory grounding verdict')
    # Never let private claim text/source excerpts enter graph state or audit.
    return CheckResult(component_id=value.component_id, findings=value.findings,
                       usage=value.usage, elapsed_ms=value.elapsed_ms)


def error_result(component):
    d = component.descriptor
    return CheckResult(component_id=d.component_id, findings=tuple(
        Finding(component_id=d.component_id, rule_id=r, outcome='error') for r in d.rule_ids))


class OutputComplianceService:
    def __init__(self, policy, validators, reviewer, **options):
        self.repairer = options.get('repairer')
        self.options = options
        self.fingerprint = validate_configuration(policy, validators, reviewer, **options)
        self.policy, self.validators, self.reviewer = policy, validators, reviewer

    def evaluate(self, candidate, context):
        started = perf_counter()
        attempts, initial_reasons, usage_results = 0, (), []
        try:
            candidate = Candidate.model_validate(candidate.model_dump())
            context = ComplianceContext.model_validate(context.model_dump())
            if validate_configuration(self.policy, self.validators, self.reviewer, **self.options) != self.fingerprint:
                raise ValueError('Configuration changed')
            from .compliance_graph import run_gate
            results = run_gate(self, candidate, context)
            decision, reasons = decide(self.policy, results)
            usage_results = list(results)
            initial_reasons = reasons
            from .compliance_repairer import REPAIRABLE
            if (self.repairer and decision == 'block' and set(reasons) <= set(REPAIRABLE)
                and not any(f.outcome in ('uncertain', 'error') for r in results for f in r.findings)):
                safe_context = self.review_context(candidate, context)
                attempts = 1
                repaired = RepairResult.model_validate(self.repairer.repair(RepairRequest(
                    context=safe_context, policy=self.policy, rule_ids=reasons)).model_dump())
                usage_results.append(CheckResult(component_id='repairer', findings=(), usage=repaired.usage))
                candidate = Candidate(text=repaired.text, finish_reason=repaired.finish_reason, origin=candidate.origin)
                results = run_gate(self, candidate, context)
                usage_results.extend(results)
                decision, reasons = decide(self.policy, results)
        except Exception:
            results, decision, reasons = [], 'unverified', ('CHECK_UNAVAILABLE',)
        counts = {}
        for field in ('input_tokens', 'output_tokens', 'total_tokens'):
            values = [getattr(r.usage, field) for r in usage_results if r.component_id in (self.reviewer.descriptor.component_id, "repairer")]
            counts[field] = sum(values) if values and all(v is not None for v in values) else None
        return ComplianceResult(final_answer=candidate.text if decision == 'allow' else self.policy.fallback,
            decision=decision, reason_codes=reasons, configuration_fingerprint=self.fingerprint,
            request_id=context.request_id, origin=candidate.origin, usage=Usage(**counts),
            policy_version=self.policy.version, repair_attempts=attempts, initial_reason_codes=initial_reasons,
            components=tuple(c.descriptor for c in (*self.validators, self.reviewer, *((self.repairer,) if self.repairer else ()))),
            evidence_digest=hashlib.sha256(json.dumps([r.model_dump() for r in context.references], sort_keys=True).encode()).hexdigest(),
            elapsed_ms=(perf_counter()-started)*1000)

    def local_checks(self, candidate, context):
        request = ValidationRequest(candidate=candidate, context=context, policy=self.policy)
        results = []
        for validator in self.validators:
            try:
                results.append(checked_result(validator, validator.validate(request), context))
            except Exception:
                results.append(error_result(validator))
        return results

    def review_context(self, candidate, context):
        texts = [context.question, *context.history, candidate.text,
                 *(r.question + '\n' + r.answer for r in context.references)]
        if (not context.references or len({r.id for r in context.references}) != len(context.references)
            or sum(map(len, texts)) > 24000 or any(sensitive(t) for t in texts)):
            raise ValueError('Payload cannot be sent')
        return context.model_copy(update={
            'question': safe_preview(context.question),
            'history': tuple(safe_preview(t) for t in context.history),
            'references': tuple(r.model_copy(update={'question': safe_preview(r.question),
                'answer': safe_preview(r.answer)}) for r in context.references),
        })

    def semantic_check(self, candidate, context):
        try:
            safe_context = self.review_context(candidate, context)
            request = ReviewRequest(candidate=candidate, context=safe_context, policy=self.policy)
            return checked_result(self.reviewer, self.reviewer.review(request), safe_context, review=True)
        except Exception:
            return error_result(self.reviewer)
