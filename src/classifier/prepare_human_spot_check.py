"""Prepare a small human spot-check set from difficult independent examples."""
import json,joblib
from pathlib import Path
from ..data_access import load_tickets

def main():
 root=Path('.'); all_tickets={t.ticket_id:t for t in load_tickets(root/'data/splits/bitext-v1','selection','dev_selection')+load_tickets(root/'data/splits/bitext-v1','calibration','dev_calibration')}; rows=[]
 for split in ('dev_selection','dev_calibration'):
  for line in (root/f'data/annotations/independent-review-v1/{split}-teacher-v2.jsonl').read_text().splitlines():
   r=json.loads(line);r['source_split']=split;rows.append(r)
 model=joblib.load(root/'artifacts/students-v2/tfidf_linear_svm.joblib'); resolved=[r for r in rows if r['status']=='resolved']; pred=model.predict([all_tickets[r['ticket_id']].instruction for r in resolved]); disagreements={r['ticket_id'] for r,p in zip(resolved,pred) if r['intent_id']!=p}; model_labels={r['ticket_id']:p for r,p in zip(resolved,pred)}
 catalog={e['id']:e for e in json.loads((root/'config/catalogs/bitext-v1.json').read_text())['entries']}
 selected=[]
 for r in rows:
  if r['status']=='abstained' or r['ticket_id'] in disagreements:selected.append(r)
 selected=selected[:50];out=root/'data/annotations/independent-review-v1/human-spot-check-v1.jsonl'; records=[]
 for r in selected:
  model_id=model_labels.get(r['ticket_id']); teacher_id=r.get('intent_id'); candidate_ids=[x for x in (teacher_id,model_id) if x]
  records.append({'ticket_id':r['ticket_id'],'instruction':all_tickets[r['ticket_id']].instruction,'source_split':r['source_split'],'teacher_status':r['status'],'teacher_intent_id':teacher_id,'teacher_confidence':r.get('confidence'),'teacher_abstention_cause':r.get('abstention_cause'),'teacher_reason':r.get('reason'),'model_intent_id':model_id,'candidate_intents':[{'id':i,'name':catalog[i]['name'],'domain':catalog[i]['domain'],'definition':catalog[i]['definition'],'include_when':catalog[i]['include_when'],'exclude_when':catalog[i]['exclude_when']} for i in dict.fromkeys(candidate_ids)],'review_status':'pending','human_intent_id':None,'human_decision':None,'human_abstention_cause':None,'human_notes':None})
 out.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in records));print(json.dumps({'selected':len(selected),'teacher_abstentions':sum(r['status']=='abstained' for r in selected),'model_disagreements':sum(r['ticket_id'] in disagreements for r in selected)},indent=2))
if __name__=='__main__':main()
