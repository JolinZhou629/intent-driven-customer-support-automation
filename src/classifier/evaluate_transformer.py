"""Evaluate a saved transformer bundle on prepared labelled JSONL data."""
import argparse, json
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train_transformer import TicketDataset, load

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model", default="artifacts/classifier-transformer-v1"); ap.add_argument("--data", default="data/transformer-v1/dev_calibration.jsonl"); ap.add_argument("--max-length", type=int, default=128); ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--output"); args=ap.parse_args()
    tok=AutoTokenizer.from_pretrained(args.model); model=AutoModelForSequenceClassification.from_pretrained(args.model)
    device=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"); model.to(device).eval()
    rows=load(args.data); loader=DataLoader(TicketDataset(rows,tok,args.max_length),batch_size=args.batch_size); ys=[]; ps=[]
    with torch.no_grad():
        for b in loader:
            labels=b.pop("labels"); out=model(**{k:v.to(device) for k,v in b.items()}); ys += labels.tolist(); ps += out.logits.argmax(-1).cpu().tolist()
    result={"model":args.model,"data":args.data,"rows":len(rows),"device":str(device),"accuracy":accuracy_score(ys,ps),"macro_f1":f1_score(ys,ps,average="macro",zero_division=0),"report":classification_report(ys,ps,output_dict=True,zero_division=0)}
    print(json.dumps(result,indent=2));
    if args.output: Path(args.output).write_text(json.dumps(result,indent=2))
if __name__ == "__main__": main()
