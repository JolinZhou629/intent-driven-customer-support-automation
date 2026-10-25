"""Create bitext-v2 by applying the approved semantic intent merges."""
import json
from pathlib import Path

MERGES={'refund.policy':['refund.policy','refund.eligibility','refund.money_back_guarantee'],'account.update':['account.update','account.update_user_information'],'order.modify':['order.modify','order.add_item','order.remove_item','order.swap_item'],'account.early_exit_fee':['account.early_exit_fee','payment.cancellation_charge']}
def main():
 src=Path('config/catalogs/bitext-v1.json');out=Path('config/catalogs/bitext-v2.json');data=json.loads(src.read_text());entries={e['id']:e for e in data['entries']};used=set();result=[]
 for target, members in MERGES.items():
  base=dict(entries[target]);base['id']=target;base['examples']=[];base['definition']={'refund.policy':'User asks about refund rules, eligibility, or money-back conditions.','account.update':'User asks to modify information on an existing account or user profile.','order.modify':'User asks to add, remove, swap, or otherwise change an existing order.','account.early_exit_fee':'User asks about a charge or fee associated with cancelling or ending an account or service early.'}[target];base['name']={'refund.policy':'Refund policy and eligibility','account.update':'Update account information','order.modify':'Modify order','account.early_exit_fee':'Early-exit and cancellation fee'}[target]
  for m in members:
   used.add(m)
   for ex in entries[m]['examples']:
    if ex not in base['examples']:base['examples'].append(ex)
  base['examples']=base['examples'][:10];result.append(base)
 for e in data['entries']:
  if e['id'] not in used:result.append(e)
 result={'version':'bitext-v2','status':'frozen','entries':result};out.write_text(json.dumps(result,indent=2)+'\n');Path('config/catalogs/bitext-v2-provenance.json').write_text(json.dumps({'source':'config/catalogs/bitext-v1.json','merges':MERGES,'entries':len(result['entries'])},indent=2)+'\n');print(len(result['entries']))
if __name__=='__main__':main()
