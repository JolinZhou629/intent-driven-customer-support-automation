"""Prepare and validate agent-assisted discovery review packets.

The default mode creates a machine-review proposal from cluster evidence using
transparent rules. It is a placeholder until an LLM provider is configured;
every proposal is marked ``human_review_required`` and must not freeze a catalog.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

DECISIONS = {'keep', 'merge', 'split', 'noise-subgroup', 'unclear', 'exclude-data-quality'}


def _terms(text):
    return set(re.findall(r"[a-z]{3,}", text.casefold()))


def proposal(cluster):
    examples = [x['instruction'] for x in cluster['representatives'] + cluster['boundary_examples']]
    terms = Counter(t for text in examples for t in _terms(text))
    top = [t for t, _ in terms.most_common(5)]
    # Conservative: noise and mixed/weak evidence are never auto-accepted.
    if cluster['noise']:
        decision = 'noise-subgroup'
        confidence = 'low'
    elif len(top) < 2 or cluster['count'] < 40:
        decision = 'unclear'
        confidence = 'low'
    else:
        decision = 'keep'
        confidence = 'low'
    return {
        'source_cluster_id': cluster['cluster_id'],
        'reviewer': 'machine-proposal-v1',
        'decision': decision,
        'candidate_intent_id': None,
        'candidate_name': ' '.join(top[:3]) if decision == 'keep' else '',
        'domain': '',
        'definition': '',
        'include_when': [],
        'exclude_when': [],
        'merge_with_clusters': [],
        'split_proposals': [],
        'representative_ticket_ids': [x['ticket_id'] for x in cluster['representatives']],
        'boundary_ticket_ids': [x['ticket_id'] for x in cluster['boundary_examples']],
        'noise_ticket_ids': [x['ticket_id'] for x in cluster['representatives']] if cluster['noise'] else [],
        'reason': 'Machine proposal from cluster size, keyword frequency, and supplied representatives; semantic and operational review required.',
        'confidence': confidence,
        'follow_up': 'Human adjudication required before catalog drafting.',
        'human_review_required': True,
    }


def validate(item, source):
    if item['decision'] not in DECISIONS or not item['human_review_required']:
        raise ValueError('invalid or unsafe review decision')
    available = {x['ticket_id'] for x in source['representatives'] + source['boundary_examples']}
    ids = set(item['representative_ticket_ids'] + item['boundary_ticket_ids'])
    if not ids <= available:
        raise ValueError('review evidence ID is outside its source cluster')
    if item['decision'] == 'keep' and not item['candidate_name']:
        raise ValueError('keep proposal requires a provisional name')


def run(source_path: Path, output: Path, packet: Path | None = None):
    source = json.loads(source_path.read_text())
    proposals = []
    for cluster in source:
        item = proposal(cluster)
        validate(item, cluster)
        proposals.append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proposals, indent=2, ensure_ascii=False) + '\n')
    if packet:
        packet.parent.mkdir(parents=True, exist_ok=True)
        # Packet contains only evidence needed for review, never labels/responses.
        packet.write_text(json.dumps([
            {'source_cluster_id': c['cluster_id'], 'count': c['count'], 'noise': c['noise'],
             'keywords': c['keywords'], 'representatives': c['representatives'],
             'boundary_examples': c['boundary_examples']}
            for c in source
        ], indent=2, ensure_ascii=False) + '\n')
    return {'clusters': len(source), 'proposals': len(proposals),
            'human_review_required': len(proposals), 'output': str(output)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('reports/discovery-step4/cluster-review-full-min30.json'))
    p.add_argument('--output', type=Path, default=Path('reviews/agent-review-v1.json'))
    p.add_argument('--packet', type=Path, default=Path('reviews/agent-review-packet-v1.json'))
    a = p.parse_args()
    print(json.dumps(run(a.source, a.output, a.packet), indent=2))
