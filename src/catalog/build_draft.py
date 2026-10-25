"""Build a draft catalog from reviewed cluster decisions.

This creates a reviewable draft only. It does not create teacher labels or
claim that generated definitions are human-approved.
"""
import argparse
import json
import re
from pathlib import Path
from ..contracts import Catalog


def slug(text):
    text = re.sub(r'[^a-z0-9]+', '-', text.casefold()).strip('-')
    return text or 'unnamed-intent'


def build(review_path, source_path, output):
    review = json.loads(review_path.read_text())
    source = {x['cluster_id']: x for x in json.loads(source_path.read_text())}
    by_id = {x['source_cluster_id']: x for x in review}
    merge_target = {}
    for item in review:
        if item['decision'] == 'merge':
            targets = item.get('merge_with_clusters', [])
            if len(targets) != 1:
                raise ValueError(f"cluster {item['source_cluster_id']} needs one merge target")
            merge_target[item['source_cluster_id']] = targets[0]
    # Resolve chains and reject cycles.
    def root(cid):
        seen = set()
        while cid in merge_target:
            if cid in seen: raise ValueError(f'merge cycle at cluster {cid}')
            seen.add(cid); cid = merge_target[cid]
        return cid
    entries = {}
    evidence = {}
    for item in review:
        if item['decision'] not in {'keep', 'merge'}: continue
        cid = root(item['source_cluster_id'])
        canonical = by_id.get(cid)
        if not canonical or canonical['decision'] not in {'keep', 'merge'}:
            raise ValueError(f'cluster {item["source_cluster_id"]} resolves to non-catalog cluster {cid}')
        # Prefer reviewed canonical fields, then reviewed source fields.
        chosen = canonical if canonical.get('candidate_name') else item
        name = chosen.get('candidate_name') or source[cid].get('candidate_name') or f'cluster {cid}'
        entry_id = 'bitext.' + slug(name)
        if entry_id in entries and entries[entry_id]['source_cluster_id'] != cid:
            entry_id += '-' + str(cid)
        definition = chosen.get('definition') or f'Provisional intent for requests about {name}.'
        include = chosen.get('include_when') or [f'The request is primarily about {name}.']
        exclude = chosen.get('exclude_when') or ['Requests whose primary goal is a different support action.']
        entries[entry_id] = {
            'id': entry_id, 'domain': chosen.get('domain') or 'unassigned',
            'name': name, 'definition': definition,
            'include_when': include, 'exclude_when': exclude,
            'examples': [x['instruction'] for x in source[cid]['representatives'][:3]],
            'source_cluster_id': cid,
        }
        evidence.setdefault(entry_id, set()).update([item['source_cluster_id'], cid])
    # Remove internal provenance before schema validation, but retain it beside the catalog.
    entries_list = list(entries.values())
    catalog_data = {'version': 'bitext-v0.1-draft', 'entries': [
        {k:v for k,v in x.items() if k != 'source_cluster_id'} for x in entries_list
    ]}
    Catalog.model_validate(catalog_data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(catalog_data, indent=2, ensure_ascii=False) + '\n')
    provenance = {'catalog_version': catalog_data['version'], 'status': 'draft',
                  'human_review_required': True, 'entries': len(entries_list),
                  'excluded_noise_clusters': [x['source_cluster_id'] for x in review if x['decision']=='noise-subgroup'],
                  'excluded_unclear_clusters': [x['source_cluster_id'] for x in review if x['decision']=='unclear'],
                  'entry_source_clusters': {k: sorted(v) for k,v in evidence.items()},
                  'review_file': str(review_path), 'source_file': str(source_path),
                  'definition_status': 'reviewed fields preserved; missing fields are provisional generated text'}
    output.with_name(output.stem + '-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,default=Path('reviews/agent-review-v2.json'))
    p.add_argument('--source',type=Path,default=Path('reports/discovery-step4/cluster-review-full-min30.json'))
    p.add_argument('--output',type=Path,default=Path('config/catalogs/bitext-v0.1-draft.json'))
    a=p.parse_args();print(json.dumps(build(a.review,a.source,a.output),indent=2))
