"""Compute multiclass one-vs-rest ROC-AUC for all student models."""
import json
import numpy as np
from pathlib import Path
import joblib, torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train_transformer import TicketDataset, load

def main():
    label_map={x['intent_id']:x['label'] for x in load('data/transformer-v1/catalog-labels.json')}; n=len(label_map)
    out={}
    tok=AutoTokenizer.from_pretrained('artifacts/classifier-transformer-v1'); model=AutoModelForSequenceClassification.from_pretrained('artifacts/classifier-transformer-v1'); model.eval()
    for split in ('dev_selection','dev_calibration'):
        rows=load(f'data/transformer-v1/{split}.jsonl'); y=[r['label'] for r in rows]; texts=[r['text'] for r in rows]
        onehot=[[int(v==j) for j in range(n)] for v in y]
        scores={}
        logreg=joblib.load('artifacts/students-v2-catalog/tfidf_logreg.joblib'); lp=logreg.predict_proba(texts); scores['logistic_regression']=np.column_stack([lp[:,list(logreg.classes_).index(next(k for k,v in label_map.items() if v==j))] if j in label_map.values() and next(k for k,v in label_map.items() if v==j) in logreg.classes_ else np.zeros(len(rows)) for j in range(n)])
        svm=joblib.load('artifacts/students-v2-catalog/tfidf_linear_svm.joblib'); raw=svm.decision_function(texts); scores['linear_svm']=np.column_stack([raw[:,list(svm.classes_).index(next(k for k,v in label_map.items() if v==j))] if next(k for k,v in label_map.items() if v==j) in svm.classes_ else np.zeros(len(rows)) for j in range(n)])
        pred=[]
        with torch.no_grad():
            for b in DataLoader(TicketDataset(rows,tok,128),batch_size=8):
                b.pop('labels'); pred.append(torch.softmax(model(**b).logits,-1).numpy())
        scores['distilbert']=np.concatenate(pred)
        present=sorted(set(y)); present_y=[[row[j] for j in present] for row in onehot]
        out[split]={k:float(roc_auc_score(present_y,v[:,present],multi_class='ovr',average='macro')) for k,v in scores.items()}
        out[split]['classes_present']=len(present)
    Path('reports/model-auc-v1.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
