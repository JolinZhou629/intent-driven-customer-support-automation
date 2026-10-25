import csv
import hashlib
import json
from pathlib import Path
import pytest
from src.data_prep import COLUMNS, normalize_for_grouping, prepare


def source(tmp_path, rows):
    p = tmp_path/'input.csv'
    with p.open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=COLUMNS);writer.writeheader();writer.writerows(rows)
    m=tmp_path/'source.json'
    m.write_text(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':len(rows)}))
    return p,m


def examples():
    return [dict(flags='x',instruction='  Where is {{Order Number}}? ',category='private',intent='hidden',response='Check {{Order Number}}.'),
            dict(flags='y',instruction='  Where is {{Order Number}}? ',category='private',intent='hidden',response='Another response.')]


def test_separation_reproducibility_and_original_text(tmp_path):
    rows=examples();p,m=source(tmp_path,rows)
    out=tmp_path/'result';report=prepare(p,m,out)
    tickets=[json.loads(s) for s in (out/'tickets.jsonl').read_text().splitlines()]
    assert len({r['ticket_id'] for r in tickets})==2
    assert all(set(r)=={'ticket_id','instruction'} for r in tickets)
    assert all(r['instruction']==rows[0]['instruction'] for r in tickets)
    assert report['placeholder_audit']['instruction']['occurrences']==2
    assert 'intent_counts' not in report
    assert prepare(p,m,out)==report
    p,m=source(tmp_path,list(reversed(rows)))
    prepare(p,m,tmp_path/'reordered')
    for name in ['tickets.jsonl','metadata.jsonl','private/true_labels.jsonl','private/responses.jsonl']:
        assert (out/name).read_bytes()==(tmp_path/'reordered'/name).read_bytes()


def test_corrupted_source_rejected(tmp_path):
    p,m=source(tmp_path,examples());p.write_text('corrupt')
    with pytest.raises(ValueError,match='checksum'): prepare(p,m,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_missing_field_rejected(tmp_path):
    rows=examples();rows[0]['intent']=''
    p,m=source(tmp_path,rows)
    with pytest.raises(ValueError,match='missing'):prepare(p,m,tmp_path/'out')


def test_changed_output_not_overwritten(tmp_path):
    p,m=source(tmp_path,examples());out=tmp_path/'out';prepare(p,m,out)
    (out/'tickets.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='already exists'):prepare(p,m,out)
    assert (out/'tickets.jsonl').read_text()=='changed'


def test_group_normalization_preserves_punctuation():
    assert normalize_for_grouping('  ＯＲＤ-42\t Please! ')== 'ord-42 please!'
