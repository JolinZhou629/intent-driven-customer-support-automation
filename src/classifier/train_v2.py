"""Map v1 annotations through the approved merge map and train v2 students."""
import json, joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from ..data_access import load_tickets

MAP = {'refund.eligibility':'refund.policy', 'refund.money_back_guarantee':'refund.policy', 'account.update_user_information':'account.update', 'order.add_item':'order.modify', 'order.remove_item':'order.modify', 'order.swap_item':'order.modify', 'payment.cancellation_charge':'account.early_exit_fee'}
def map_id(value):
    return MAP.get(value, value) if value else value

def main():
    root=Path('.'); tickets={t.ticket_id:t for t in load_tickets(root/'data/splits/bitext-v1','training','train')}
    local=[json.loads(x) for x in (root/'data/annotations/local-v1/train.jsonl').read_text().splitlines()]
    review=[json.loads(x) for x in (root/'data/annotations/teacher-v2/review-500-final.jsonl').read_text().splitlines()]
    eval_ids={r['ticket_id'] for r in review}; labels={r['ticket_id']:map_id(r['intent_id']) for r in review if r['status']=='resolved'}; train=[]
    for r in local:
        if r['ticket_id'] not in eval_ids and r['status']=='resolved' and r['confidence']>=.75 and r['margin']>=.15: train.append((tickets[r['ticket_id']].instruction,map_id(r['intent_id'])))
    train.extend((tickets[k].instruction,v) for k,v in labels.items()); test=[(tickets[r['ticket_id']].instruction,map_id(r['intent_id'])) for r in review if r['status']=='resolved']; x,y=zip(*train); xt,yt=zip(*test)
    models={'tfidf_logreg':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),('clf',LogisticRegression(max_iter=500,C=1.0,class_weight='balanced'))]),'tfidf_linear_svm':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True)),('clf',LinearSVC(C=1.0,class_weight='balanced'))])}; out=root/'artifacts/students-v2-catalog'; out.mkdir(parents=True,exist_ok=True); report={'catalog_version':'bitext-v2','train_size':len(train),'test_size':len(test),'models':{}}
    for n,m in models.items():
        m.fit(x,y); pred=m.predict(xt); joblib.dump(m,out/(n+'.joblib')); report['models'][n]={'accuracy':accuracy_score(yt,pred),'macro_f1':f1_score(yt,pred,average='macro')}
    (root/'reports/student-v2-catalog.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
