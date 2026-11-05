"""Paired reviewer comparisons and conservative release readiness reporting."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from .compliance_evaluation import DEFAULT, read_cases, evaluate, summarize
from .compliance_labels import dataset_digest, adjudicate


def release_gate(rows, labels_complete, split, minimum_cases, max_p95_ms):
    metrics = summarize(rows)
    reasons = []
    if not labels_complete:
        reasons.append('INDEPENDENT_LABELS_INCOMPLETE')
    if split != 'held_out':
        reasons.append('HELD_OUT_REQUIRED')
    # Repeats are observations, not additional independent cases.
    if len({r['id'] for r in rows}) < minimum_cases:
        reasons.append('INSUFFICIENT_UNIQUE_CASES')
    from .compliance_config import load_policy
    required_rules = {rule.id for rule in load_policy().rules}
    covered_rules = {r.get('rule') for r in rows}
    if not required_rules <= covered_rules:
        reasons.append('POLICY_RULE_COVERAGE')
    if any(r.get('repair_attempts', 0) for r in rows):
        reasons.append('REPAIRED_OUTPUTS_REQUIRE_INDEPENDENT_LABELS')
    if metrics['violation_recall'] is None or metrics['violation_recall'] < .95:
        reasons.append('VIOLATION_RECALL')
    if metrics['false_block_rate'] is None or metrics['false_block_rate'] > .05:
        reasons.append('FALSE_BLOCK_RATE')
    if metrics['unsafe_release_rate'] != 0:
        reasons.append('UNSAFE_RELEASE')
    if metrics['compliant_withheld_rate'] is None or metrics['compliant_withheld_rate'] > .05:
        reasons.append('COMPLIANT_WITHHELD')
    if metrics['unverified_rate'] is None or metrics['unverified_rate'] > .05:
        reasons.append('UNVERIFIED_RATE')
    if max_p95_ms is None:
        reasons.append('LATENCY_BUDGET_REQUIRED')
    elif metrics['p95_ms'] is None or metrics['p95_ms'] > max_p95_ms:
        reasons.append('LATENCY_BUDGET')
    return {'quality_and_latency_pass': not reasons, 'reasons': reasons,
            'production_approved': False,
            'note': 'Human policy sign-off and a measured cost budget remain required.'}


def compare(cases, services, *, repeats=1, labels_complete=False, split='development',
            minimum_cases=120, max_p95_ms=None):
    if not 1 <= repeats <= 5 or minimum_cases < 1:
        raise ValueError('Invalid benchmark bounds')
    if max_p95_ms is not None and (not math.isfinite(max_p95_ms) or max_p95_ms <= 0):
        raise ValueError('Invalid latency budget')
    results = {name: [] for name in services}
    # Pair candidates by case and alternate model order to reduce order effects.
    for iteration in range(repeats):
        for i, case in enumerate(cases):
            names = list(services)
            if (i + iteration) % 2:
                names.reverse()
            for name in names:
                row = evaluate([case], services[name])[0]
                row['repeat'] = iteration
                results[name].append(row)
    variants = {}
    for name, rows in results.items():
        decisions = defaultdict(set)
        for row in rows:
            decisions[row['id']].add(row['decision'])
        variants[name] = {'configuration_fingerprint': services[name].fingerprint,
            'metrics': summarize(rows),
            'inconsistent_case_count': sum(len(v) > 1 for v in decisions.values()),
            'per_rule': {r: summarize([x for x in rows if x['rule'] == r]) for r in sorted({x['rule'] for x in rows})},
            'release_gate': release_gate(rows, labels_complete, split, minimum_cases, max_p95_ms),
            'results': rows}
    for variant in variants.values():
        if variant['inconsistent_case_count']:
            variant['release_gate']['quality_and_latency_pass'] = False
            variant['release_gate']['reasons'].append('INCONSISTENT_REPEATS')
    return variants


def main():
    from .compliance_config import build_compliance_service
    from .compliance_reviewer import GeminiReviewer, MODELS
    from .compliance_repairer import GeminiRepairer
    from .gemini import local_settings
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=DEFAULT)
    p.add_argument('--labels', type=Path, nargs='*', default=[])
    p.add_argument('--models', nargs='+', choices=MODELS, default=['gemini-3.8-flash','gemini-3.5-flash-lite'])
    p.add_argument('--split', choices=['development','held_out'], default='development')
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--repeats', type=int, default=1)
    p.add_argument('--repair', choices=['off','once','both'], default='off')
    p.add_argument('--max-p95-ms', type=float)
    p.add_argument('--live', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if not 1 <= a.limit <= 240 or not 1 <= a.repeats <= 5 or len(set(a.models)) != len(a.models):
        p.error('Invalid limit, repeats, or duplicate model')
    if a.max_p95_ms is not None and (not math.isfinite(a.max_p95_ms) or a.max_p95_ms <= 0):
        p.error('Latency budget must be finite and positive')
    cases = read_cases(a.dataset)
    digest = dataset_digest(a.dataset)
    labels = adjudicate(cases, digest, [json.loads(path.read_text()) for path in a.labels]) if a.labels else None
    selected = [dict(c, expected=labels['labels'].get(c['id'], c['expected']) if labels else c['expected'])
                for c in cases if c['split'] == a.split][:a.limit]
    complete = bool(labels and all(c['id'] in labels['labels'] for c in selected))
    modes = ['off','once'] if a.repair == 'both' else [a.repair]
    report = {'dataset_sha256': digest, 'label_status': 'agreed' if complete else 'draft_or_unresolved',
              'live': a.live, 'unique_cases': len(selected),
              'planned_evaluations': len(selected)*a.repeats*len(a.models)*len(modes),
              'maximum_http_calls': len(selected)*a.repeats*len(a.models)*sum(4 if m == 'once' else 1 for m in modes),
              'models': a.models, 'repair_modes': modes, 'split': a.split,
              'independent_reviewers': labels['reviewers'] if labels else [],
              'variants': {}}
    if a.live:
        if not selected:
            p.error('No selected cases')
        key, _ = local_settings()
        services = {f'{model}/repair-{mode}': build_compliance_service(
            reviewer=GeminiReviewer(key, model=model),
            repairer=GeminiRepairer(key) if mode == 'once' else None, repair_mode='off')
            for model in a.models for mode in modes}
        report['variants'] = compare(selected, services, repeats=a.repeats, labels_complete=complete,
            split=a.split, max_p95_ms=a.max_p95_ms)
    a.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
