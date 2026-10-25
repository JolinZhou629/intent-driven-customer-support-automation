"""Run the 100-ticket Step 6 teacher pilot on blind training tickets only."""
import argparse,json,os,time
from pathlib import Path
from ..data_access import load_tickets
from .openai_teacher import OpenAITeacher

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=100);p.add_argument('--model',default='gpt-5-mini');p.add_argument('--cache',type=Path,default=Path('artifacts/teacher-cache-v1'));p.add_argument('--output',type=Path,default=Path('reports/teacher-pilot-v1.jsonl'));a=p.parse_args()
 tickets=load_tickets(Path('data/splits/bitext-v1'),'training','train')[:a.limit];teacher=OpenAITeacher(Path('config/catalogs/bitext-v1.json'),a.cache,a.model);a.output.parent.mkdir(parents=True,exist_ok=True);counts={'resolved':0,'abstained':0,'cached':0,'failed':0}
 with a.output.open('w') as f:
  for n,t in enumerate(tickets,1):
   try:
    result,cached=teacher.classify(t.ticket_id,t.instruction)
    result['routing_outcome'] = 'unknown.require_more_info' if result['status'] == 'abstained' else None
    counts[result['status']]+=1;counts['cached']+=cached;f.write(json.dumps(result)+'\n');f.flush();print(f'{n}/{len(tickets)} {result["status"]}',flush=True)
   except Exception as exc:
    counts['failed']+=1;f.write(json.dumps({'ticket_id':t.ticket_id,'status':'failed','error_type':type(exc).__name__})+'\n');f.flush();print(f'{n}/{len(tickets)} failed {type(exc).__name__}',flush=True)
 print(json.dumps({'model':a.model,'tickets':len(tickets),**counts},indent=2))
