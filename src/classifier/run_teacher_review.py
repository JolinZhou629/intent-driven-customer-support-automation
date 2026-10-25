"""Run the teacher only on a selected local-uncertainty review sample."""
import argparse,json
from pathlib import Path
from .openai_teacher import OpenAITeacher

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,default=Path('data/annotations/local-v1/teacher-review-sample-500.jsonl')); p.add_argument('--output',type=Path,default=Path('data/annotations/teacher-v2/review-500.jsonl')); p.add_argument('--cache',type=Path,default=Path('artifacts/teacher-cache-v2')); p.add_argument('--model',default='gpt-5-mini'); a=p.parse_args()
    rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()]; a.output.parent.mkdir(parents=True,exist_ok=True); done=set()
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            try:
                existing=json.loads(line)
                if existing.get('status') in {'resolved','abstained'}:
                    done.add(existing['ticket_id'])
            except Exception: pass
    teacher=OpenAITeacher(Path('config/catalogs/bitext-v1.json'),a.cache,a.model); counts={'resolved':0,'abstained':0,'failed':0,'skipped':0,'cached':0}
    with a.output.open('a') as out:
        for i,row in enumerate(rows,1):
            if row['ticket_id'] in done: counts['skipped']+=1; continue
            try:
                result,cached=teacher.classify(row['ticket_id'],row['instruction']); result.update({'local_suggested_intent_id':row['suggested_intent_id'],'local_confidence':row['confidence'],'local_margin':row['margin'],'routing_outcome':'unknown.require_more_info' if result['status']=='abstained' else None}); out.write(json.dumps(result,sort_keys=True)+'\n'); out.flush(); counts[result['status']]+=1; counts['cached']+=int(cached); print(f'{i}/{len(rows)} {result["status"]}',flush=True)
            except Exception as exc:
                counts['failed']+=1; out.write(json.dumps({'ticket_id':row['ticket_id'],'status':'failed','error_type':type(exc).__name__})+'\n'); out.flush(); print(f'{i}/{len(rows)} failed {type(exc).__name__}',flush=True)
    print(json.dumps(counts,indent=2))
if __name__=='__main__': main()
