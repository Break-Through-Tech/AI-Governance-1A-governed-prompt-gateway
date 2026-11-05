"""Offline independent labeling: identity is a team assertion, not authentication."""
import argparse
import hashlib
import json
from pathlib import Path


def dataset_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def template(cases, digest, reviewer):
    if not reviewer.strip():
        raise ValueError('A reviewer identifier is required')
    return {'dataset_sha256': digest, 'reviewer': reviewer.strip(),
            'labels': [{'id': c['id'], 'decision': None} for c in cases]}


def adjudicate(cases, digest, documents):
    """Require agreement from at least two distinct declared reviewers per case."""
    if len(documents) < 2:
        raise ValueError('Two independent label files are required')
    identities, votes = set(), {}
    ids = {c['id'] for c in cases}
    for doc in documents:
        reviewer = doc['reviewer'].strip()
        if not reviewer or reviewer.casefold() in identities or doc['dataset_sha256'] != digest:
            raise ValueError('Duplicate reviewer or mismatched dataset')
        identities.add(reviewer.casefold())
        entries = doc['labels']
        if len(entries) != len(ids) or {e['id'] for e in entries} != ids:
            raise ValueError('Label IDs must exactly match the dataset')
        for entry in entries:
            decision = entry['decision']
            if decision not in ('allow', 'block', None):
                raise ValueError('Invalid label decision')
            votes.setdefault(entry['id'], []).append(decision)
    labels, unresolved = {}, []
    for case_id, decisions in votes.items():
        if None in decisions or len(set(decisions)) != 1:
            unresolved.append(case_id)
        else:
            labels[case_id] = decisions[0]
    return {'dataset_sha256': digest, 'reviewers': sorted(identities),
            'labels': labels, 'unresolved': sorted(unresolved),
            'complete': not unresolved}


def main():
    from .compliance_evaluation import DEFAULT, read_cases
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DEFAULT)
    parser.add_argument('--reviewer', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = template(read_cases(args.dataset), dataset_digest(args.dataset), args.reviewer)
    # Never overwrite an existing person's labels.
    with args.output.open('x') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
