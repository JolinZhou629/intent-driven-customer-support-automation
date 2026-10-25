"""Train TF-IDF logistic-regression and linear-SVM student baselines."""
import json
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from ..data_access import load_tickets

def main() -> None:
    root=Path('.'); local=[json.loads(x) for x in (root/'data/annotations/local-v1/train.jsonl').read_text().splitlines()]
    reviewed=[json.loads(x) for x in (root/'data/annotations/teacher-v2/review-500-final.jsonl').read_text().splitlines()]
    reviewed_ids={r['ticket_id'] for r in reviewed}; tickets={t.ticket_id:t for t in load_tickets(root/'data/splits/bitext-v1','training','train')}
    train=[(tickets[r['ticket_id']].instruction,r['intent_id']) for r in local if r['status']=='resolved' and r['ticket_id'] not in reviewed_ids]
    test=[(tickets[r['ticket_id']].instruction,r['intent_id']) for r in reviewed if r['status']=='resolved']
    models={'tfidf_logreg':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=1)),('clf',LogisticRegression(max_iter=500,C=1.0,class_weight='balanced'))]),'tfidf_linear_svm':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,min_df=1)),('clf',LinearSVC(C=1.0,class_weight='balanced'))])}
    out=root/'artifacts/students-v1';out.mkdir(parents=True,exist_ok=True); report={'train_size':len(train),'reviewed_test_size':len(test),'models':{}}
    x,y=zip(*train); xt,yt=zip(*test)
    for name,model in models.items():
        model.fit(x,y); pred=model.predict(xt); joblib.dump(model,out/(name+'.joblib')); report['models'][name]={'accuracy':accuracy_score(yt,pred),'macro_f1':f1_score(yt,pred,average='macro'),'weighted_f1':f1_score(yt,pred,average='weighted')}
    (root/'reports/student-baselines-v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__': main()
