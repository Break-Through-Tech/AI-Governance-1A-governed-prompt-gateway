"""Trusted adapter composition, capability checks, and approval identity."""
import hashlib
import json
import os
from pathlib import Path
from .compliance_contracts import ComponentDescriptor, Policy
from .compliance_validators import FormatValidator, SensitiveDataValidator, ContactPolicyValidator

VALIDATOR_FACTORIES = {
    'format': FormatValidator,
    'sensitive': SensitiveDataValidator,
    'contact': ContactPolicyValidator,
}


def configured_components(validator_ids, factories):
    """Resolve trusted application registrations, never arbitrary import paths."""
    if len(set(validator_ids)) != len(validator_ids):
        raise ValueError('Duplicate validator selection')
    if any(name not in factories for name in validator_ids):
        raise ValueError('Unknown validator selection')
    return tuple(factories[name]() for name in validator_ids)


def load_policy():
    return Policy.model_validate_json((Path(__file__).parent / 'policies/banking_demo_v1.json').read_text())


def validate_configuration(policy, validators, reviewer, *, allow_test=False, allow_external=True, repairer=None):
    if reviewer is None or not validators:
        raise ValueError('Missing mandatory components')
    if repairer is not None and repairer.descriptor.component_id != 'repairer':
        raise ValueError('Repair adapter must use the repairer role ID')
    components = (*validators, reviewer, *((repairer,) if repairer else ()))
    descriptors = [ComponentDescriptor.model_validate(c.descriptor.model_dump()) for c in components]
    ids = [d.component_id for d in descriptors]
    if len(ids) != len(set(ids)) or not validators or reviewer is None:
        raise ValueError('Invalid component registration')
    rules = {r.id: r for r in policy.rules}
    if len(rules) != len(policy.rules) or not rules:
        raise ValueError('Invalid policy rules')
    for d in descriptors:
        if d.contract_version != '1' or (d.test_only and not allow_test):
            raise ValueError('Incompatible component')
        if d.execution == 'external' and not allow_external:
            raise ValueError('External processing disabled')
        if len(d.rule_ids) != len(set(d.rule_ids)) or not set(d.rule_ids) <= rules.keys():
            raise ValueError('Invalid component rule coverage')
    coverage = {d.component_id: set(d.rule_ids) for d in descriptors}
    for rule in policy.rules:
        if not rule.required_checks or len(set(rule.required_checks)) != len(rule.required_checks):
            raise ValueError('Missing or duplicate required checks')
        for check in rule.required_checks:
            if rule.id not in coverage.get(check, set()):
                raise ValueError('Missing mandatory check')
    payload = {'policy': policy.model_dump(), 'components': [d.model_dump() for d in descriptors]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_compliance_service(*, api_key='', reviewer=None, validators=None, policy=None,
                             allow_test=False, allow_external=True,
                             validator_ids=('format', 'sensitive', 'contact'),
                             validator_factories=None, reviewer_factory=None, repairer=None, repair_mode=None):
    from .compliance import OutputComplianceService
    if reviewer is None:
        from .compliance_reviewer import GeminiReviewer
        factory = reviewer_factory or GeminiReviewer
        reviewer = factory(api_key, model=os.environ.get('COMPLIANCE_REVIEW_MODEL', 'gemini-3.8-flash'))
    if validators is None:
        validators = configured_components(validator_ids, validator_factories or VALIDATOR_FACTORIES)
    mode = repair_mode if repair_mode is not None else os.environ.get('COMPLIANCE_REPAIR_MODE', 'off')
    if mode not in ('off', 'once'):
        raise ValueError('Invalid repair mode')
    if repairer is None and mode == 'once':
        from .compliance_repairer import GeminiRepairer
        repairer = GeminiRepairer(api_key)
    return OutputComplianceService(policy or load_policy(), tuple(validators), reviewer,
                                   allow_test=allow_test, allow_external=allow_external, repairer=repairer)
