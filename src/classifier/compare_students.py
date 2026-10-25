"""Compare SVM and DistilBERT on identical labelled JSONL inputs."""
import argparse, json, time
from pathlib import Path
import joblib, torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train_transformer import TicketDataset, load

def transformer_predict(rows, model_dir, max_length, batch_size):
    tok=AutoTokenizer.from_pretrained(model_dir); model=AutoModelForSequenceClassification.from_pretrained(model_dir)
    device=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    dl=DataLoader(TicketDataset(rows,tok,max_length),batch_size=batch_size); pred=[]; conf=[]; latency=[]
    with torch.no_grad():
        for b in dl:
            labels=b.pop("labels"); t=time.perf_counter(); out=model(**{k:v.to(device) for k,v in b.items()}); latency.append(time.perf_counter()-t)
            probs=torch.softmax(out.logits,-1); vals,idx=probs.max(-1); pred += idx.cpu().tolist(); conf += vals.cpu().tolist()
    return pred,conf,sum(latency)/max(1,len(rows)),str(device)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default='artifacts/classifier-transformer-v1'); ap.add_argument('--svm',default='artifacts/students-v2-catalog/tfidf_linear_svm.joblib'); ap.add_argument('--output',default='reports/student-comparison-v1.json'); args=ap.parse_args()
    label_map={x['intent_id']:x['label'] for x in load('data/transformer-v1/catalog-labels.json')}
    results={}
    for name in ('dev_selection','dev_calibration'):
        rows=load(f'data/transformer-v1/{name}.jsonl'); y=[r['label'] for r in rows]
        svm=joblib.load(args.svm); t=time.perf_counter(); sp=[label_map[x] for x in svm.predict([r['text'] for r in rows]).tolist()]; svm_time=(time.perf_counter()-t)/len(rows)
        tp,conf,tr_time,device=transformer_predict(rows,args.model,128,8)
        results[name]={"rows":len(rows),"svm":{"accuracy":accuracy_score(y,sp),"macro_f1":f1_score(y,sp,average='macro',zero_division=0),"avg_latency_seconds":svm_time},"distilbert":{"accuracy":accuracy_score(y,tp),"macro_f1":f1_score(y,tp,average='macro',zero_division=0),"avg_latency_seconds":tr_time,"device":device,"mean_max_softmax":sum(conf)/len(conf)},"disagreements":sum(a!=b for a,b in zip(sp,tp)),"svm_report":classification_report(y,sp,output_dict=True,zero_division=0),"distilbert_report":classification_report(y,tp,output_dict=True,zero_division=0)}
    Path(args.output).write_text(json.dumps(results,indent=2)); print(json.dumps({k:{x:v for x,v in d.items() if x not in ('svm_report','distilbert_report')} for k,d in results.items()},indent=2))
if __name__=='__main__': main()
