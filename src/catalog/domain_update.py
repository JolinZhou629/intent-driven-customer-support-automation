"""Apply the first domain/ID pass to the reviewed catalog draft.

This is an explicit, reviewable mapping pass. It preserves source cluster IDs,
does not use hidden labels, and remains a draft until human approval.
"""
import argparse, json, re
from pathlib import Path
from ..contracts import Catalog

DOMAIN_RULES = [
    ('account', ['account', 'profile', 'password', 'pin', 'early exit', 'exit early', 'subscription']),
    ('order', ['order', 'purchase', 'swap']),
    ('delivery', ['delivery', 'shipping', 'shipment', 'address', 'arrive', 'eta', 'city']),
    ('refund', ['refund', 'reimbursement', 'restitution', 'compensation', 'reimburse']),
    ('payment', ['payment', 'currency', 'charge', 'withdrawal']),
    ('invoice', ['invoice', 'bill']),
    ('security', ['password', 'pwd', 'pin', 'login', 'sign']),
    ('complaint', ['complaint', 'claim', 'reclamation']),
    ('support', ['support', 'agent', 'operator', 'feedback']),
]

def slug(s):
    return re.sub(r'[^a-z0-9]+', '_', s.casefold()).strip('_')

def domain_for(text):
    text=text.casefold()
    for domain, words in DOMAIN_RULES:
        if any(w in text for w in words): return domain
    return 'general'

def action_for(name, definition):
    text=f'{name} {definition}'.casefold()
    for key, action in [('early exit','early_exit_fee'),('exit early','early_exit_fee'),
        ('cancel','cancel'),('track','track'),('status','status'),('refund policy','policy'),
        ('reimbursement policy','policy'),('policy','policy'),('password','credentials'),
        ('pin','credentials'),('invoice','retrieve'),('bill','retrieve'),('address','address'),
        ('delivery','availability'),('shipping','availability'),('city','availability'),
        ('eta','estimate'),('arrive','estimate'),('feedback','feedback'),('complaint','complaint'),
        ('agent','human_support'),('operator','human_support'),('swap','swap_item'),
        ('open account','create'),('new account','create'),('update account','update'),
        ('modify account','update'),('delete account','delete'),('payment','payment_info'),
        ('refund','request'),('reimbursement','request'),('restitution','request')]:
        if key in text:return action
    return slug(name).split('_')[0] or 'request'

def build(source, provenance, output):
    catalog=json.loads(source.read_text()); prov=json.loads(provenance.read_text())
    merged={3: 'early_exit_fee', 5: 'early_exit_fee', 6: 'early_exit_fee'}
    entries=[]; used={}; mapping=[]
    for entry, source_ids in zip(catalog['entries'], [prov['entry_source_clusters'][k] for k in prov['entry_source_clusters']]):
        # provenance insertion order matches catalog generation; use IDs instead
        # where available to keep the mapping explicit.
        cid=source_ids[0]
        text=f"{entry['name']} {entry['definition']}"
        if cid in merged:
            domain='account'; action=merged[cid]; name='Check early-exit fee'; definition='The user wants to view or understand the fee charged for ending an account or service early.'
            include=['The user asks to see, check, or understand an early-exit fee or charge.']
            exclude=['The user asks to cancel an account without asking about the fee.', 'The user disputes an already-applied charge.', 'The user asks about a refund after cancellation.']
        else:
            domain=entry.get('domain') if entry.get('domain') not in (None,'','unassigned') else domain_for(text)
            action=action_for(entry['name'],entry['definition']);name=entry['name'];definition=entry['definition'];include=entry['include_when'];exclude=entry['exclude_when']
        intent_id=f'{domain}.{action}'
        if intent_id in used:
            intent_id=f'{intent_id}.{cid}'
        used[intent_id]=cid
        new={**entry,'id':intent_id,'domain':domain,'name':name,'definition':definition,'include_when':include,'exclude_when':exclude}
        if cid in merged:
            if any(e['id']=='account.early_exit_fee' for e in entries):
                mapping.append({'source_cluster_ids':source_ids,'action':'merged_into_existing','intent_id':'account.early_exit_fee'});continue
        entries.append(new);mapping.append({'source_cluster_ids':source_ids,'action':'kept_or_renamed','intent_id':intent_id})
    data={'version':'bitext-v0.2-draft','entries':entries}
    Catalog.model_validate(data)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    report={'status':'draft','human_review_required':True,'source_catalog':str(source),'entries':len(entries),'early_exit_clusters_merged':[3,5,6], 'mapping':mapping,'domain_rules':'explicit keyword heuristic; review required before freeze'}
    output.with_name(output.stem+'-provenance.json').write_text(json.dumps(report,indent=2)+'\n');return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('config/catalogs/bitext-v0.1-draft.json'));p.add_argument('--provenance',type=Path,default=Path('config/catalogs/bitext-v0.1-draft-provenance.json'));p.add_argument('--output',type=Path,default=Path('config/catalogs/bitext-v0.2-draft.json'));a=p.parse_args();print(json.dumps(build(a.source,a.provenance,a.output),indent=2))
