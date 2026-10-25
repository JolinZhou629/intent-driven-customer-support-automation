"""Evaluate students after excluding all duplicate groups of the review sample."""
import json,joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,f1_score
from ..data_access import load_tickets

def main():
 root=Path('.'); tickets={t.ticket_id:t for t in load_tickets(root/'data/splits/bitext-v1','training','train')}
 groups={};
 for line in (root/'data/splits/bitext-v1/private/groups.jsonl').read_text().splitlines():
  r=json.loads(line); groups[r['ticket_id']]=r['group_id']
 local=[json.loads(x) for x in (root/'data/annotations/local-v1/train.jsonl').read_text().splitlines()]
 review=[json.loads(x) for x in (root/'data/annotations/teacher-v2/review-500-final.jsonl').read_text().splitlines()]
 eval_ids={r['ticket_id'] for r in review}; eval_groups={groups[x] for x in eval_ids}; labels={r['ticket_id']:r['intent_id'] for r in review if r['status']=='resolved'}
 train=[]
 for r in local:
  if r['ticket_id'] in eval_ids or groups.get(r['ticket_id']) in eval_groups: continue
  if r['status']=='resolved' and r['confidence']>=.75 and r['margin']>=.15: train.append((tickets[r['ticket_id']].instruction,r['intent_id']))
 train.extend((tickets[tid].instruction,label) for tid,label in labels.items())
 test=[(tickets[r['ticket_id']].instruction,r['intent_id']) for r in review if r['status']=='resolved'];x,y=zip(*train);xt,yt=zip(*test)
 models={'tfidf_logreg':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),('clf',LogisticRegression(max_iter=500,C=1.0,class_weight='balanced'))]),'tfidf_linear_svm':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),('clf',LinearSVC(C=1.0,class_weight='balanced'))])};report={'train_size':len(train),'excluded_groups':len(eval_groups),'reviewed_test_size':len(test),'models':{}}
 for n,m in models.items():
  m.fit(x,y);p=m.predict(xt);report['models'][n]={'accuracy':accuracy_score(yt,p),'macro_f1':f1_score(yt,p,average='macro')}
 (root/'reports/group-disjoint-evaluation-v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
