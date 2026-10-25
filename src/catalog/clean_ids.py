"""Remove numeric collision suffixes from a catalog while preserving traceability."""
import argparse, json, re
from pathlib import Path

def clean(source: Path, output: Path):
    data=json.loads(source.read_text()); used={}; mapping=[]; entries=[]
    for index,entry in enumerate(data['entries']):
        old=entry['id']
        base=re.sub(r'\.\d+$','',old)
        # Numeric suffixes are implementation collisions. If the semantic base
        # still collides, use a human-readable variant, never another number.
        new=base
        if new in used:
            letter=chr(ord('a') + used[base])
            used[base]+=1
            new=f'{base}.variant_{letter}'
        else:
            used[base]=1
        item={**entry,'id':new}
        entries.append(item);mapping.append({'index':index,'old_id':old,'new_id':new,'reason':'numeric suffix removed; semantic collision variant retained' if new!=base else 'numeric suffix removed or unchanged'})
    result={'version':'bitext-v0.2-id-clean-draft','entries':entries}
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    output.with_name(output.stem+'-mapping.json').write_text(json.dumps({'source':str(source),'mapping':mapping,'numeric_ids_remaining':sum(bool(re.search(r'\.\d+$',e['id'])) for e in entries)},indent=2)+'\n')
    return len(entries),len({e['id'] for e in entries}),sum(bool(re.search(r'\.\d+$',e['id'])) for e in entries)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('config/catalogs/bitext-v0.2-draft.json'));p.add_argument('--output',type=Path,default=Path('config/catalogs/bitext-v0.2-id-clean-draft.json'));a=p.parse_args();print(clean(a.source,a.output))
