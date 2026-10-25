"""Restricted child verifies input access without printing ticket contents."""
import argparse
import json
from pathlib import Path
from .data_access import load_tickets

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role');p.add_argument('--partition');p.add_argument('--output');p.add_argument('--check-denials',action='store_true');a=p.parse_args()
    root=Path('data/splits/bitext-v1')
    rows=load_tickets(root,a.role,a.partition)
    result={'role':a.role,'partition':a.partition,'rows_read':len(rows),'denied_reads':[]}
    if a.check_denials:
        denied=[Path('data/raw/bitext/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'),
                Path('data/prepared/bitext-v1/tickets.jsonl'),root/'gold_hidden/train.jsonl',root/'responses/train.jsonl',root/'sealed/test/tickets.jsonl']
        denied += [root/f'blind/{part}/tickets.jsonl' for part in ['train','dev_selection','dev_calibration'] if part!=a.partition]
        for path in denied:
            try:
                with path.open('rb') as f:f.read(1)
            except PermissionError:result['denied_reads'].append(str(path))
            else:raise RuntimeError('forbidden file was readable: '+str(path))
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
