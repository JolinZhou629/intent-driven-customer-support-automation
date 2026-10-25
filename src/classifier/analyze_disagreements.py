"""List SVM/DistilBERT disagreements with intent names and ticket text."""
import json, joblib, torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train_transformer import TicketDataset, load

def main():
    rows=load('data/transformer-v1/dev_calibration.jsonl'); names={x['label']:x['intent_id'] for x in load('data/transformer-v1/catalog-labels.json')}
    svm=joblib.load('artifacts/students-v2-catalog/tfidf_linear_svm.joblib'); sp=svm.predict([r['text'] for r in rows]).tolist()
    tok=AutoTokenizer.from_pretrained('artifacts/classifier-transformer-v1'); model=AutoModelForSequenceClassification.from_pretrained('artifacts/classifier-transformer-v1'); model.eval(); pred=[]
    with torch.no_grad():
      for b in DataLoader(TicketDataset(rows,tok,128),batch_size=8):
        b.pop('labels'); pred += model(**{k:v for k,v in b.items()}).logits.argmax(-1).tolist()
    out=[]
    for r,s,p in zip(rows,sp,pred):
      if names[p]!=s: out.append({'ticket_id':r['ticket_id'],'text':r['text'],'gold_intent':names[r['label']],'svm_intent':s,'distilbert_intent':names[p]})
    Path('reports/transformer-svm-disagreements-v1.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
