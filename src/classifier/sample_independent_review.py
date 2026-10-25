"""Create disjoint development/calibration review manifests from reserved splits."""
import argparse,json,hashlib
from pathlib import Path
from ..data_access import load_tickets

def sample(rows,n):
 rows=sorted(rows,key=lambda r:hashlib.sha256(r.ticket_id.encode()).hexdigest())
 return rows[:n]
def main():
 p=argparse.ArgumentParser();p.add_argument('--per-split',type=int,default=200);p.add_argument('--out',type=Path,default=Path('data/annotations/independent-review-v1'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 for split,role in (('dev_selection','selection'),('dev_calibration','calibration')):
  rows=load_tickets(Path('data/splits/bitext-v1'),role,split);chosen=sample(rows,a.per_split)
  (a.out/f'{split}.jsonl').write_text(''.join(json.dumps({'ticket_id':t.ticket_id,'instruction':t.instruction,'review_status':'pending','source_split':split},sort_keys=True)+'\n' for t in chosen))
  print(split,len(chosen))
 manifest={'version':'independent-review-v1','purpose':'human review for model selection and confidence calibration','files':{s:f'{s}.jsonl' for s in ('dev_selection','dev_calibration')},'disjoint_from_training':True,'per_split':a.per_split}
 (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
