"""Generate a semantic, domain.action catalog draft from an explicit merge map."""
import argparse, json
from pathlib import Path
import yaml
from ..contracts import Catalog

def build(source, merge_map, output):
    data=json.loads(source.read_text()); groups=yaml.safe_load(merge_map.read_text())['groups']
    by={e['id']:e for e in data['entries']}; assigned={}; entries=[]; provenance=[]
    for canonical, members in groups.items():
        missing=[m for m in members if m not in by]
        if missing: raise ValueError(f'{canonical}: missing source IDs {missing}')
        if canonical in assigned: raise ValueError(f'duplicate canonical {canonical}')
        chosen=by[members[0]]; examples=[]
        for member in members:
            assigned[member]=canonical
            for example in by[member].get('examples',[]):
                if example not in examples: examples.append(example)
        domain,action=canonical.split('.',1)
        entries.append({'id':canonical,'domain':domain,'name':action.replace('_',' ').title(),
            'definition':f'User request to {action.replace("_"," ")}.',
            'include_when':[f'The user’s primary goal is {action.replace("_"," ")}.' if False else f'The user asks about {action.replace("_"," ")}.'],
            'exclude_when':['Requests whose primary goal requires a different action or tool.'],
            'examples':examples[:5]})
        provenance.append({'intent_id':canonical,'source_ids':members,'source_examples':len(examples)})
    for entry in data['entries']:
        if entry['id'] in assigned: continue
        entries.append({**entry})
        provenance.append({'intent_id':entry['id'],'source_ids':[entry['id']],'source_examples':len(entry.get('examples',[]))})
    result={'version':'bitext-v0.3-draft','entries':entries};Catalog.model_validate(result)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    output.with_name(output.stem+'-provenance.json').write_text(json.dumps({'status':'draft','human_review_required':True,'source_catalog':str(source),'merge_map':str(merge_map),'entries':len(entries),'provenance':provenance},indent=2)+'\n')
    return len(entries)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('config/catalogs/bitext-v0.2-id-clean-draft.json'));p.add_argument('--merge-map',type=Path,default=Path('config/catalogs/intent-merge-v0.3.yaml'));p.add_argument('--output',type=Path,default=Path('config/catalogs/bitext-v0.3-draft.json'));a=p.parse_args();print('entries',build(a.source,a.merge_map,a.output))
