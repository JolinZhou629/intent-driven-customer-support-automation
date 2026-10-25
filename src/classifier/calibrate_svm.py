"""Calibrate SVM acceptance thresholds on the reserved calibration set."""
import json,joblib
from pathlib import Path
from sklearn.metrics import accuracy_score
from ..data_access import load_tickets

def main():
 root=Path('.'); rows=[json.loads(x) for x in (root/'data/annotations/independent-review-v1/dev_calibration-teacher-v2.jsonl').read_text().splitlines()]; tickets={t.ticket_id:t for t in load_tickets(root/'data/splits/bitext-v1','calibration','dev_calibration')}; resolved=[r for r in rows if r['status']=='resolved']; x=[tickets[r['ticket_id']].instruction for r in resolved]; y=[r['intent_id'] for r in resolved]; m=joblib.load(root/'artifacts/students-v2/tfidf_linear_svm.joblib'); scores=m.decision_function(x); pred=m.predict(x); margins=sorted((sorted(s,reverse=True)[0]-sorted(s,reverse=True)[1]) for s in scores)
 grid=[]
 for threshold in [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1.0]:
  accepted=[i for i,s in enumerate(scores) if (sorted(s,reverse=True)[0]-sorted(s,reverse=True)[1])>=threshold];correct=sum(pred[i]==y[i] for i in accepted);grid.append({'margin_threshold':threshold,'accepted':len(accepted),'coverage':len(accepted)/len(y),'accepted_accuracy':correct/len(accepted) if accepted else 0.0})
 eligible=[r for r in grid if r['accepted_accuracy']>=.95 and r['coverage']>=.5];selected=max(eligible,key=lambda r:r['coverage']) if eligible else max(grid,key=lambda r:r['accepted_accuracy'])
 out={'calibration_resolved':len(y),'teacher_abstentions':sum(r['status']=='abstained' for r in rows),'selected':selected,'grid':grid,'policy':'accept only when SVM top-two decision margin meets threshold; otherwise clarify or review'}; (root/'reports/svm-calibration-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(selected,indent=2))
if __name__=='__main__':main()
