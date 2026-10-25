"""Create a deterministic, intent-balanced sample from local uncertain predictions."""
import argparse, json, hashlib
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,default=Path('data/annotations/local-v1/teacher-review-queue.jsonl')); p.add_argument('--output',type=Path,default=Path('data/annotations/local-v1/teacher-review-sample-500.jsonl')); p.add_argument('--size',type=int,default=500); a=p.parse_args()
    rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]
    rows.sort(key=lambda r: hashlib.sha256(r['ticket_id'].encode()).hexdigest())
    groups={}
    for row in rows: groups.setdefault(row['suggested_intent_id'],[]).append(row)
    selected=[]; per=max(1,a.size//max(1,len(groups)))
    for intent in sorted(groups): selected.extend(groups[intent][:per])
    if len(selected)<a.size:
        used={r['ticket_id'] for r in selected}; selected.extend(r for r in rows if r['ticket_id'] not in used)
        selected=selected[:a.size]
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in selected))
    print(json.dumps({'input':len(rows),'sample':len(selected),'intents':len({r['suggested_intent_id'] for r in selected}),'per_intent_floor':per},indent=2))
if __name__=='__main__': main()
