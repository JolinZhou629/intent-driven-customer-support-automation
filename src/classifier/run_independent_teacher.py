"""Label independent development samples with the frozen teacher contract."""
import argparse,json
from pathlib import Path
from .openai_teacher import OpenAITeacher

def main():
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['dev_selection','dev_calibration'],required=True);p.add_argument('--model',default='gpt-5-mini');a=p.parse_args();root=Path('.');inp=root/f'data/annotations/independent-review-v1/{a.split}.jsonl';out=root/f'data/annotations/independent-review-v1/{a.split}-teacher-v2.jsonl';cache=root/'artifacts/teacher-cache-v3';rows=[json.loads(x) for x in inp.read_text().splitlines()];done=set()
 if out.exists():
  for line in out.read_text().splitlines():
   try:
    r=json.loads(line)
    if r.get('status') in {'resolved','abstained'}:done.add(r['ticket_id'])
   except Exception:pass
 teacher=OpenAITeacher(root/'config/catalogs/bitext-v1.json',cache,a.model);out.parent.mkdir(parents=True,exist_ok=True);counts={'resolved':0,'abstained':0,'failed':0,'skipped':0,'cached':0}
 with out.open('a') as f:
  for i,row in enumerate(rows,1):
   if row['ticket_id'] in done:counts['skipped']+=1;continue
   try:
    result,cached=teacher.classify(row['ticket_id'],row['instruction']);result.update({'source_split':a.split,'review_type':'teacher_review','routing_outcome':'unknown.require_more_info' if result['status']=='abstained' else None});f.write(json.dumps(result,sort_keys=True)+'\n');f.flush();counts[result['status']]+=1;counts['cached']+=int(cached);print(f'{i}/{len(rows)} {result["status"]}',flush=True)
   except Exception as exc:counts['failed']+=1;print(f'{i}/{len(rows)} failed {type(exc).__name__}',flush=True)
 print(json.dumps(counts,indent=2))
if __name__=='__main__':main()
