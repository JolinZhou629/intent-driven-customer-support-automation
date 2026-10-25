"""Create multiclass micro-average ROC curves as a dependency-free SVG."""
import json, joblib, numpy as np, torch
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from train_transformer import TicketDataset, load

def main():
    rows=load('data/transformer-v1/dev_calibration.jsonl'); n=len(load('data/transformer-v1/catalog-labels.json')); y=np.array([r['label'] for r in rows]); one=np.eye(n)[y]; texts=[r['text'] for r in rows]
    label_ids={x['intent_id']:x['label'] for x in load('data/transformer-v1/catalog-labels.json')}
    scores={}
    for key,path in [('Logistic Regression','artifacts/students-v2-catalog/tfidf_logreg.joblib'),('Linear SVM','artifacts/students-v2-catalog/tfidf_linear_svm.joblib')]:
      m=joblib.load(path); raw=m.predict_proba(texts) if key.startswith('Logistic') else m.decision_function(texts); cols=np.zeros((len(rows),n))
      for col,cls in enumerate(m.classes_): cols[:,label_ids[cls]]=raw[:,col]
      scores[key]=cols
    tok=AutoTokenizer.from_pretrained('artifacts/classifier-transformer-v1'); model=AutoModelForSequenceClassification.from_pretrained('artifacts/classifier-transformer-v1'); model.eval(); chunks=[]
    with torch.no_grad():
      for b in DataLoader(TicketDataset(rows,tok,128),batch_size=8): b.pop('labels'); chunks.append(torch.softmax(model(**b).logits,-1).numpy())
    scores['DistilBERT']=np.concatenate(chunks)
    curves={};
    for name,s in scores.items():
      fpr,tpr,_=roc_curve(one.ravel(),s.ravel()); curves[name]={'fpr':fpr.tolist(),'tpr':tpr.tolist(),'auc':float(auc(fpr,tpr))}
    Path('reports/model-roc-curves-v1.json').write_text(json.dumps(curves,indent=2))
    W,H=900,600; left,top,plotw,ploth=90,60,700,440; colors={'Logistic Regression':'#2878b5','Linear SVM':'#e1812c','DistilBERT':'#3a923a'}; svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="white"/><text x="450" y="30" text-anchor="middle" font-size="22" font-weight="bold">ROC Curve Comparison (multiclass micro-average)</text>']
    for i in range(6):
      ypix=top+ploth*i/5; xpix=left+plotw*i/5; val=1-i/5; svg += [f'<line x1="{left}" y1="{ypix}" x2="{left+plotw}" y2="{ypix}" stroke="#ddd"/>',f'<line x1="{xpix}" y1="{top}" x2="{xpix}" y2="{top+ploth}" stroke="#ddd"/>',f'<text x="{left-10}" y="{ypix+4}" text-anchor="end" font-size="11">{val:.1f}</text>',f'<text x="{xpix}" y="{top+ploth+20}" text-anchor="middle" font-size="11">{1-val:.1f}</text>']
    svg.append(f'<line x1="{left}" y1="{top+ploth}" x2="{left+plotw}" y2="{top}" stroke="#222" stroke-dasharray="8,6"/>')
    for name,c in curves.items():
      pts=[]
      for x,yv in zip(c['fpr'],c['tpr']): pts.append(f'{left+x*plotw:.1f},{top+(1-yv)*ploth:.1f}')
      # Downsample long curves for a compact SVG while preserving shape.
      pts=pts[::max(1,len(pts)//350)]; svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[name]}" stroke-width="2.5"/>')
    svg += [f'<text x="440" y="570" text-anchor="middle" font-size="14">False Positive Rate</text>',f'<text x="18" y="300" text-anchor="middle" font-size="14" transform="rotate(-90 18 300)">True Positive Rate</text>']
    yy=110
    for name,c in curves.items(): svg += [f'<line x1="650" y1="{yy}" x2="680" y2="{yy}" stroke="{colors[name]}" stroke-width="3"/>',f'<text x="690" y="{yy+4}" font-size="13">{name} (AUC = {c["auc"]:.3f})</text>']; yy+=28
    svg += ['<line x1="650" y1="205" x2="680" y2="205" stroke="#222" stroke-dasharray="8,6"/><text x="690" y="209" font-size="13">Random (AUC = 0.500)</text>','<text x="450" y="595" text-anchor="middle" font-size="11">Calibration subset: 180 tickets; 33 intent outputs; micro-average over one-vs-rest labels</text>','</svg>']
    Path('reports/model-roc-curves-v1.svg').write_text('\n'.join(svg)); print(json.dumps({k:round(v['auc'],4) for k,v in curves.items()},indent=2))
if __name__=='__main__': main()
