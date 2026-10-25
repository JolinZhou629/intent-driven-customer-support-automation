"""Fine-tune DistilBERT on the prepared bitext-v2 JSONL labels."""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

def load(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

class TicketDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows; self.tok = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = self.tok(r["text"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in x.items()} | {"labels": torch.tensor(r["label"], dtype=torch.long)}

def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]; loss=0.0
    with torch.no_grad():
        for b in loader:
            b={k:v.to(device) for k,v in b.items()}; out=model(**b); loss += float(out.loss)
            ys.extend(b["labels"].cpu().tolist()); ps.extend(out.logits.argmax(-1).cpu().tolist())
    return {"loss": loss/max(1,len(loader)), "accuracy": accuracy_score(ys, ps), "macro_f1": f1_score(ys, ps, average="macro", zero_division=0)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config", default="configs/transformer-training.yaml"); ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--max-length", type=int, default=128); ap.add_argument("--output", default="artifacts/classifier-transformer-v1"); args=ap.parse_args()
    seed=42; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    root=Path("."); train=load(root/"data/transformer-v1/train.jsonl"); dev=load(root/"data/transformer-v1/dev_selection.jsonl")
    labels=load(root/"data/transformer-v1/catalog-labels.json"); n=len(labels)
    model_name="distilbert-base-uncased"; tokenizer=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n, id2label={str(x["label"]):x["intent_id"] for x in labels}, label2id={x["intent_id"]:x["label"] for x in labels})
    device=torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    tr=DataLoader(TicketDataset(train,tokenizer,args.max_length), batch_size=args.batch_size, shuffle=True); dv=DataLoader(TicketDataset(dev,tokenizer,args.max_length), batch_size=args.batch_size)
    opt=torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01); steps=len(tr)*args.epochs; sch=get_linear_schedule_with_warmup(opt, int(.1*steps), steps)
    best=None; history=[]; start=time.time()
    for epoch in range(args.epochs):
        model.train(); running=0.0
        for b in tr:
            b={k:v.to(device) for k,v in b.items()}; opt.zero_grad(); out=model(**b); out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step(); running += float(out.loss)
        metrics=evaluate(model,dv,device); metrics.update(epoch=epoch+1, train_loss=running/len(tr)); history.append(metrics); print(json.dumps(metrics), flush=True)
        if best is None or metrics["macro_f1"] > best["macro_f1"]: best=metrics; model.save_pretrained(args.output); tokenizer.save_pretrained(args.output)
    out=Path(args.output); out.mkdir(parents=True, exist_ok=True); (out/"training_manifest.json").write_text(json.dumps({"model_name":model_name,"catalog":"bitext-v2","train_rows":len(train),"dev_rows":len(dev),"labels":n,"device":str(device),"epochs":args.epochs,"batch_size":args.batch_size,"max_length":args.max_length,"learning_rate":2e-5,"weight_decay":.01,"seed":seed,"best":best,"history":history,"elapsed_seconds":time.time()-start},indent=2))
    print(json.dumps({"output":args.output,"best":best,"device":str(device)}, indent=2))
if __name__ == "__main__": main()
