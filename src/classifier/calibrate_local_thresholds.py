"""Choose local embedding acceptance thresholds from teacher-reviewed cases."""
import argparse, json
from pathlib import Path
from collections import defaultdict

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--review',type=Path,default=Path('data/annotations/teacher-v2/review-500-final.jsonl')); p.add_argument('--output',type=Path,default=Path('reports/local-threshold-calibration-v1.json')); a=p.parse_args()
    rows=[json.loads(x) for x in a.review.read_text().splitlines()]
    grid=[]
    for threshold in [0.50,0.55,0.60,0.62,0.65,0.70,0.75,0.80]:
        for margin in [0.00,0.03,0.05,0.06,0.08,0.10,0.15]:
            accepted=[r for r in rows if r['local_confidence']>=threshold and r['local_margin']>=margin]
            correct=[r for r in accepted if r['status']=='resolved' and r['local_suggested_intent_id']==r['intent_id']]
            grid.append({'threshold':threshold,'margin':margin,'accepted':len(accepted),'coverage':len(accepted)/len(rows),'agreement':len(correct)/len(accepted) if accepted else 0.0,'abstention_rate':1-len(accepted)/len(rows)})
    eligible=[r for r in grid if r['agreement']>=0.95 and r['coverage']>=0.20]
    selected=max(eligible,key=lambda r:(r['coverage'],-r['threshold'],-r['margin'])) if eligible else max(grid,key=lambda r:r['agreement'])
    result={'sample_size':len(rows),'resolved_teacher':sum(r['status']=='resolved' for r in rows),'abstained_teacher':sum(r['status']=='abstained' for r in rows),'selected':selected,'grid':grid,'note':'Agreement is measured against teacher-resolved labels; abstentions are not treated as local-label errors.'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(selected,indent=2))
if __name__=='__main__': main()
