"""Trusted label-independent grouping and frozen partition publication."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import re
import yaml
from .data_prep import digest, encode, normalize_for_grouping
from .contracts import Ticket

PARTS = ('train', 'dev_selection', 'dev_calibration', 'test')


def tokens(text):
    # Preserve placeholder type, strip punctuation only for duplicate grouping.
    return tuple(re.findall(r'\w+', normalize_for_grouping(text)))


def typo(a, b):
    if min(len(a), len(b)) < 5 or abs(len(a)-len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x,y in zip(a,b)) <= 1
    short,long = sorted((a,b),key=len)
    return any(long[:i]+long[i+1:] == short for i in range(len(long)))


def group_tickets(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        parent[max(a,b)] = min(a,b)
    exact, signatures = {}, defaultdict(list)
    edges = 0
    for i,row in enumerate(rows):
        ts=tokens(row['instruction'])
        if ts in exact:
            union(i,exact[ts]);continue
        exact[ts]=i
        for pos,word in enumerate(ts):
            if len(word)<5:continue
            signature=(pos,ts[:pos],ts[pos+1:])
            for other,otherword in signatures[signature]:
                if typo(word,otherword):union(i,other);edges+=1
            signatures[signature].append((i,word))
    members=defaultdict(list)
    for i,row in enumerate(rows):members[find(i)].append(row['ticket_id'])
    result={}
    for ids in members.values():
        group='group-'+digest(encode(sorted(ids)))
        for ticket_id in ids:result[ticket_id]=group
    return result,edges


def assign(groups, seed, ratios):
    sizes=Counter(groups.values());total=sum(sizes.values())
    targets={k:total*v for k,v in ratios.items()};counts=Counter();allocation={}
    # Largest groups first avoids large final overshoots; seed resolves ties.
    order=sorted(sizes,key=lambda g:(-sizes[g],digest(f'{seed}:{g}'.encode())))
    for group in order:
        part=max(PARTS,key=lambda p:targets[p]-counts[p])
        allocation[group]=part;counts[part]+=sizes[group]
    return {t:allocation[g] for t,g in groups.items()}


def build(prepared, output, config_path):
    config=yaml.safe_load(config_path.read_text());d=config['dataset']
    manifest=json.loads((prepared/'manifest.json').read_text())
    def read(name):
        raw=(prepared/name).read_bytes()
        if digest(raw)!=manifest['outputs'][name]:raise ValueError('prepared checksum mismatch')
        return [json.loads(x) for x in raw.splitlines()]
    rows=read('tickets.jsonl')
    for r in rows:Ticket.model_validate(r)
    groups,edges=group_tickets(rows)
    f=d['development_selection_fraction'];r=d['split_ratios']
    ratios=dict(zip(PARTS,[r['train'],r['development']*f,r['development']*(1-f),r['test']]))
    if any(v<=0 for v in ratios.values()) or abs(sum(ratios.values())-1)>1e-9:
        raise ValueError('invalid split ratios')
    assignments=assign(groups,config['seed'],ratios)
    # Reference files are read only AFTER label-independent assignments exist.
    files={}
    for part in PARTS:
        prefix='sealed/test' if part=='test' else 'blind/'+part
        files[prefix+'/tickets.jsonl']=b''.join(encode(x) for x in rows if assignments[x['ticket_id']]==part)
    for name,dest in [('private/true_labels.jsonl','gold_hidden'),('private/responses.jsonl','responses')]:
        values=read(name)
        if {x['ticket_id'] for x in values}!=set(assignments) or len(values)!=len(assignments):
            raise ValueError('private join mismatch')
        for part in PARTS:
            files[f'{dest}/{part}.jsonl']=b''.join(encode(x) for x in values if assignments[x['ticket_id']]==part)
    files['private/groups.jsonl']=b''.join(encode({'ticket_id':t,'group_id':groups[t],'partition':assignments[t]}) for t in sorted(groups))
    report={'rows':len(rows),'groups':len(set(groups.values())), 'largest_group':max(Counter(groups.values()).values()),
            'typo_edges':edges,'counts':dict(Counter(assignments.values())),
            'seed':config['seed'],'target_ratios':ratios,
            'grouping':'NFKC/casefold/word-token identity; same token context with one-character typo in token of length >=5; transitive components',
            'cross_partition_group_overlap':0,
            'limitations':'No semantic paraphrase detection. Transitive typo components may overmerge; token normalization may remove meaningful punctuation. No label-based stratification.'}
    seen=defaultdict(set)
    for t,g in groups.items():seen[g].add(assignments[t])
    if any(len(v)>1 for v in seen.values()):raise ValueError('group leakage')
    files['audit.json']=encode(report)
    manifest_out={'version':'splits-v1','prepared_manifest_sha256':digest((prepared/'manifest.json').read_bytes()),
                  'config_sha256':digest(config_path.read_bytes()),'implementation_sha256':digest(Path(__file__).read_bytes()),
                  'outputs':{k:digest(v) for k,v in files.items()}}
    files['manifest.json']=encode(manifest_out)
    if output.exists():
        if {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}!=set(files) or any((output/k).read_bytes()!=v for k,v in files.items()):
            raise ValueError('existing split differs; do not overwrite frozen splits')
    else:
        output.mkdir(parents=True)
        for name,raw in files.items():
            p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,default=Path('data/prepared/bitext-v1'))
    p.add_argument('--output',type=Path,default=Path('data/splits/bitext-v1'))
    p.add_argument('--config',type=Path,default=Path('config/experiments/classifier-v1.yaml'))
    a=p.parse_args();print(json.dumps(build(a.prepared,a.output,a.config),indent=2))
