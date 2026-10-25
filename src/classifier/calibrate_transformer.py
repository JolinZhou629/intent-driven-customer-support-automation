"""Compare confidence thresholds and accepted coverage on calibration data."""
import argparse, json
from pathlib import Path
import joblib, torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train_transformer import TicketDataset, load

def table(y, pred, score, thresholds):
    out=[]
    for t in thresholds:
        keep=[i for i,s in enumerate(score) if s>=t]; cov=len(keep)/len(y)
        out.append({"threshold":t,"accepted":len(keep),"coverage":cov,"accepted_accuracy":accuracy_score([y[i] for i in keep],[pred[i] for i in keep]) if keep else None,"all_ticket_accuracy":sum(pred[i]==y[i] for i in keep)/len(y)})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default='data/transformer-v1/dev_calibration.jsonl'); ap.add_argument('--model',default='artifacts/classifier-transformer-v1'); ap.add_argument('--svm',default='artifacts/students-v2-catalog/tfidf_linear_svm.joblib'); ap.add_argument('--output',default='reports/transformer-calibration-v1.json'); args=ap.parse_args()
    rows=load(args.data); y=[r['label'] for r in rows]; lm={x['intent_id']:x['label'] for x in load('data/transformer-v1/catalog-labels.json')}
    svm=joblib.load(args.svm); raw=svm.decision_function([r['text'] for r in rows]); sp=[lm[x] for x in svm.predict([r['text'] for r in rows])]; sm=[float(sorted(x,reverse=True)[0]-sorted(x,reverse=True)[1]) for x in raw]
    tok=AutoTokenizer.from_pretrained(args.model); model=AutoModelForSequenceClassification.from_pretrained(args.model); device=torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'); model.to(device).eval(); dl=DataLoader(TicketDataset(rows,tok,128),batch_size=8); tp=[]; tc=[]
    with torch.no_grad():
        for b in dl:
            b.pop('labels'); probs=torch.softmax(model(**{k:v.to(device) for k,v in b.items()}).logits,-1); vals,idx=probs.max(-1); tp+=idx.cpu().tolist(); tc+=vals.cpu().tolist()
    result={"data":args.data,"rows":len(rows),"svm": {"raw_margin_threshold_reference":0.2,"thresholds":table(y,sp,sm,[0.0,0.1,0.2,0.3,0.5,0.8,1.0])},"distilbert":{"score":"max_softmax_not_calibrated_probability","thresholds":table(y,tp,tc,[0.5,0.7,0.8,0.9,0.95,0.97,0.99])}}
    Path(args.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
