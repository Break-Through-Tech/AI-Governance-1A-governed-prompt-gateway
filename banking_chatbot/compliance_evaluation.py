"""Offline fixture validation or explicitly requested live synthetic evaluation.

No credentials or answer content are written to reports. Labels in the seed set
are draft author labels, not independently human-adjudicated release evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from .compliance_config import build_compliance_service
from .compliance_contracts import Candidate, ComplianceContext, Reference
from .gemini import local_settings

DEFAULT = Path(__file__).resolve().parent.parent / 'data/compliance/seed.jsonl'


def read_cases(path):
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids, families = set(), {}
    for case in cases:
        if case['id'] in ids or case['expected'] not in ('allow','block'):
            raise ValueError('Duplicate case or invalid label')
        ids.add(case['id'])
        split = case['split']
        if split not in ('development','held_out') or families.setdefault(case['family'],split) != split:
            raise ValueError('Scenario family crosses splits')
        Candidate(text=case['answer'], finish_reason=case.get('finish_reason','STOP'))
        Reference.model_validate(case['reference'])
    return cases


def evaluate(cases, service):
    rows=[]
    for c in cases:
        result=service.evaluate(Candidate(text=c['answer'],finish_reason=c.get('finish_reason','STOP')),
            ComplianceContext(request_id=c['id'],question=c['question'],
                references=(Reference.model_validate(c['reference']),)))
        rows.append({'id':c['id'],'expected':c['expected'],'decision':result.decision,
            'split':c['split'],'rule':c['rule'],'elapsed_ms':result.elapsed_ms,
            'usage':result.usage.model_dump(),'reason_codes':result.reason_codes,
            'repair_attempts':result.repair_attempts})
    return rows


def summarize(rows):
    # Original labels cannot judge a newly regenerated answer.
    original_rows = rows
    rows = [r for r in rows if not r.get('repair_attempts', 0)]
    bad=[r for r in rows if r['expected']=='block']
    good=[r for r in rows if r['expected']=='allow']
    blocked=[r for r in rows if r['decision']=='block']
    ratio=lambda n,d: n/d if d else None
    timings=sorted(r['elapsed_ms'] for r in original_rows)
    return {'cases':len(original_rows), 'scored_original_cases':len(rows),
        'repaired_cases_needing_labels':len(original_rows)-len(rows),
        'observed_tokens':{field: sum(r['usage'][field] for r in original_rows)
            if original_rows and all(r.get('usage', {}).get(field) is not None for r in original_rows) else None
            for field in ('input_tokens','output_tokens','total_tokens')},
        'violation_recall':ratio(sum(r['decision']=='block' for r in bad),len(bad)),
        'block_precision':ratio(sum(r['expected']=='block' for r in blocked),len(blocked)),
        'false_block_rate':ratio(sum(r['decision']=='block' for r in good),len(good)),
        'compliant_withheld_rate':ratio(sum(r['decision']!='allow' for r in good),len(good)),
        'unsafe_release_rate':ratio(sum(r['decision']=='allow' for r in bad),len(bad)),
        'unverified_rate':ratio(sum(r['decision']=='unverified' for r in rows),len(rows)),
        'p50_ms':median(timings) if timings else None,
        'p95_ms':timings[max(0, (95*len(timings)+99)//100-1)] if timings else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=DEFAULT)
    p.add_argument('--live',action='store_true',help='Send synthetic cases to the configured reviewer API (may incur charges).')
    p.add_argument('--limit',type=int,default=10)
    p.add_argument('--split',choices=('development','held_out'),default='development')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    cases=read_cases(args.dataset)
    report={'dataset_sha256':hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            'case_count':len(cases),'label_status':'draft; independent human review required',
            'live':args.live}
    if args.live:
        if not 1 <= args.limit <= 240:
            p.error('--limit must be between 1 and 240')
        key,_=local_settings()
        gate=build_compliance_service(api_key=key)
        rows=evaluate([c for c in cases if c['split']==args.split][:args.limit],gate)
        report.update(configuration_fingerprint=gate.fingerprint,metrics=summarize(rows),results=rows,
            per_rule={rule:summarize([r for r in rows if r['rule']==rule]) for rule in {r['rule'] for r in rows}})
    output=json.dumps(report,indent=2)+'\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output,end='')


if __name__=='__main__': main()
