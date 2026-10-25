"""Train students on teacher labels plus conservative local pseudo-labels."""
import json, joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from ..data_access import load_tickets

def main():
 root=Path('.'); tickets={t.ticket_id:t for t in load_tickets(root/'data/splits/bitext-v1','training','train')}
 local=[json.loads(x) for x in (root/'data/annotations/local-v1/train.jsonl').read_text().splitlines()]
 review=[json.loads(x) for x in (root/'data/annotations/teacher-v2/review-500-final.jsonl').read_text().splitlines()]
 eval_ids={r['ticket_id'] for r in review}; labels={r['ticket_id']:r['intent_id'] for r in review if r['status']=='resolved'}
 train=[]
 for r in local:
  if r['ticket_id'] in eval_ids: continue
  if r['status']=='resolved' and r['confidence']>=.75 and r['margin']>=.15: train.append((tickets[r['ticket_id']].instruction,r['intent_id'],'local_conservative'))
 for tid,label in labels.items(): train.append((tickets[tid].instruction,label,'teacher_reviewed'))
 test=[(tickets[r['ticket_id']].instruction,r['intent_id']) for r in review if r['status']=='resolved']
 x,y,_=zip(*train);xt,yt=zip(*test);models={'tfidf_logreg':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),('clf',LogisticRegression(max_iter=500,C=1.0,class_weight='balanced'))]),'tfidf_linear_svm':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),('clf',LinearSVC(C=1.0,class_weight='balanced'))])}
 out=root/'artifacts/students-v2';out.mkdir(parents=True,exist_ok=True);report={'train_size':len(train),'teacher_train':len(labels),'local_conservative_train':len(train)-len(labels),'reviewed_test_size':len(test),'models':{}}
 for name,m in models.items():
  m.fit(x,y);pred=m.predict(xt);joblib.dump(m,out/(name+'.joblib'));report['models'][name]={'accuracy':accuracy_score(yt,pred),'macro_f1':f1_score(yt,pred,average='macro'),'weighted_f1':f1_score(yt,pred,average='weighted')}
 (root/'reports/student-baselines-v2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
