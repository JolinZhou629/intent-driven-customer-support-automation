import json
from pathlib import Path
import pytest
from src.splits import group_tickets,assign
from src.data_access import load_tickets
from src.data_prep import digest,encode


def test_grouping_keeps_typos_and_duplicates_together():
    rows=[{'ticket_id':str(i),'instruction':t} for i,t in enumerate([
        'Please cancel order!', 'please cancel order', 'please cancell order',
        'please do not cancel order', 'please track order'])]
    groups,_=group_tickets(rows)
    assert groups['0']==groups['1']==groups['2']
    assert groups['3']!=groups['0'] and groups['4']!=groups['0']
    ratios={'train':.7,'dev_selection':.075,'dev_calibration':.075,'test':.15}
    a=assign(groups,42,ratios)
    assert a['0']==a['1']==a['2']
    assert a==assign(dict(reversed(list(groups.items()))),42,ratios)


def fixture(root):
    p=root/'blind/train/tickets.jsonl';p.parent.mkdir(parents=True)
    raw=encode({'ticket_id':'x','instruction':'help'})
    p.write_bytes(raw)
    (root/'manifest.json').write_text(json.dumps({'outputs':{'blind/train/tickets.jsonl':digest(raw)}}))
    return p


@pytest.mark.parametrize('role,part',[('discovery','test'),('training','dev_selection'),('selection','train'),('calibration','dev_selection'),('unknown','train'),('discovery','../gold_hidden')])
def test_roles_fail_closed(tmp_path,role,part):
    with pytest.raises(PermissionError):load_tickets(tmp_path,role,part)


def test_checksum_detects_mutation(tmp_path):
    p=fixture(tmp_path);assert len(load_tickets(tmp_path,'discovery','train'))==1
    p.write_text('changed')
    with pytest.raises(ValueError):load_tickets(tmp_path,'discovery','train')
