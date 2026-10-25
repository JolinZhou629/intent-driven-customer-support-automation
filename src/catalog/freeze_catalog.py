"""Apply final reviewed names/definitions and freeze the catalog version."""
import argparse,json
from pathlib import Path
from ..contracts import Catalog

REVIEWED={
 'invoice.retrieve':('invoice','Retrieve invoice','User wants to locate an invoice in the account.','invoice lookup'),
 'invoice.download':('invoice','Download invoice','User wants to download or obtain an invoice document.','invoice download'),
 'security.recover_password':('security','Recover password','User cannot access an account and needs to recover or reset a password or PIN.','password recovery'),
 'support.contact_human':('support','Contact human support','User asks to speak with a human support agent or operator.','human support'),
 'support.feedback':('support','Submit feedback','User wants to submit feedback about a service or experience.','feedback'),
 'refund.processing_time':('refund','Refund processing time','User asks how long a refund or reimbursement will take.','refund timing'),
 'refund.status':('refund','Refund status','User asks about the status of a specific refund or reimbursement.','refund status'),
 'refund.request':('refund','Request refund','User asks to receive money back for a purchase or transaction.','refund request'),
 'refund.policy':('refund','Refund policy','User asks about the rules or policy governing refunds.','refund policy'),
 'refund.eligibility':('refund','Refund eligibility','User asks when or in which cases a refund can be requested.','refund eligibility'),
 'delivery.availability':('delivery','Delivery availability','User asks whether orders can be delivered to a country or city.','delivery availability'),
 'delivery.address_change':('delivery','Change delivery address','User asks to change the address for an existing delivery.','delivery address'),
 'delivery.what':('delivery','Shipping methods','User asks which shipping or delivery methods are available.','shipping methods'),
 'order.status':('order','Order status','User asks for the current status of an order or purchase.','order status'),
 'order.delivery_estimate':('order','Delivery estimate','User asks when an order or purchase is expected to arrive.','delivery estimate'),
 'order.track':('order','Track order','User asks how to locate or track an existing order or shipment.','order tracking'),
 'order.add_item':('order','Add item to order','User asks to add an item to an existing order.','add item'),
 'order.cancel':('order','Cancel order','User asks to cancel an existing order or purchase.','cancel order'),
 'order.modify':('order','Modify order','User asks to change an existing order or purchase.','modify order'),
 'order.swap_item':('order','Swap order item','User asks to exchange or swap an item in an order.','swap item'),
 'order.order':('order','Remove order item','User asks to remove an item from an existing order.','remove item'),
 'account.delete':('account','Delete account','User asks to close or delete an account.','delete account'),
 'account.create':('account','Create account','User asks to open or create a new account.','create account'),
 'account.update':('account','Update account','User asks to modify information on an existing account.','update account'),
 'account.change_plan':('account','Change account plan','User asks to change the account plan or tier.','change plan'),
 'account.early_exit_fee':('account','Check early-exit fee','User asks to view or understand the fee for ending an account or service early.','early exit fee'),
 'account.key':('security','Recover account access key','User asks to recover an account access key or profile key.','access key'),
 'account.freemium':('account','Use account plan','User asks how to use or understand an account plan.','account plan'),
 'account.the':('account','Use account profile','User asks how to use an account profile.','account profile'),
 'account.standard':('account','Change to standard plan','User asks to change to the standard account plan.','standard plan'),
 'account.pro':('account','Change to pro plan','User asks to change to the pro account plan.','pro plan'),
 'account.premium':('account','Change to premium plan','User asks to change to the premium account plan.','premium plan'),
 'account.platinum':('account','Change to platinum plan','User asks to change to the platinum account plan.','platinum plan'),
 'payment.cancel':('payment','Cancellation charge','User asks about a charge associated with cancelling an account or service.','cancellation charge'),
 'payment.withdrawal_charge':('payment','Withdrawal charge','User asks about a fee charged for withdrawing money.','withdrawal charge'),
 'payment.methods':('payment','Payment methods','User asks which payment methods or options are accepted.','payment methods'),
 'payment.payment_info':('payment','Payment problem','User reports a problem or needs help with a payment.','payment issue'),
 'general.newsletter':('subscription','Unsubscribe newsletter','User asks to unsubscribe from a company newsletter.','newsletter unsubscribe'),
 'general.what':('support','Support hours','User asks when customer support can be contacted.','support hours'),
 'general.rebate':('refund','Request rebate','User asks to receive a rebate or money back.','rebate request'),
 'general.back':('refund','Money-back guarantee','User asks about a money-back guarantee.','money-back guarantee'),
 'general.assistance':('order','Purchase products','User asks for help purchasing one or more products.','purchase products'),
 'general.articles':('order','Purchase products','User asks for help purchasing several products.','purchase products'),
 'general.registration':('account','Registration problem','User reports a problem registering an account.','registration problem'),
 'security.with':('account','Registration problem','User reports a problem signing up or registering.','registration problem'),
 'general.user':('account','Update user information','User asks to modify their user or account information.','update user information'),
 'complaint.against':('complaint','File complaint','User wants to file a complaint against the company.','complaint'),
 'complaint.reclamation':('complaint','File complaint','User wants to make a reclamation or complaint against the company.','complaint'),
}
RENAMES={
 'general.newsletter':'subscription.unsubscribe','payment.cancel':'payment.cancellation_charge',
 'payment.withdrawal_charge':'payment.withdrawal_fee','account.key':'security.recover_access_key',
 'general.what':'support.hours','payment.payment_info':'payment.issue','general.rebate':'refund.request',
 'general.back':'refund.money_back_guarantee','complaint.against':'complaint.file','complaint.reclamation':'complaint.file',
 'general.assistance':'order.purchase_products','general.articles':'order.purchase_products',
 'security.with':'account.registration_problem','general.registration':'account.registration_problem',
 'account.freemium':'account.use_plan','general.user':'account.update_user_information',
 'account.the':'account.use_profile','order.order':'order.remove_item','delivery.what':'delivery.methods'
}

def freeze(source,output):
 data=json.loads(source.read_text()); merged={}; provenance=[]
 for e in data['entries']:
  if e['id'] not in REVIEWED: raise ValueError('unreviewed catalog entry: '+e['id'])
  domain,name,definition,action=REVIEWED[e['id']]
  # Keep distinct source examples while consolidating exact canonical IDs.
  target=RENAMES.get(e['id'], next((k for k,v in REVIEWED.items() if v==(domain,name,definition,action)),e['id']))
  if target not in merged:
   merged[target]={'id':target,'domain':domain,'name':name,'definition':definition,
    'include_when':[f'The user asks about {action}.'],
    'exclude_when':['Requests whose primary goal is a different support action.'],
    'examples':[]}
  for ex in e.get('examples',[]):
   if ex not in merged[target]['examples']: merged[target]['examples'].append(ex)
  provenance.append({'source_id':e['id'],'frozen_id':target})
 for e in merged.values(): e['examples']=e['examples'][:5]
 result={'version':'bitext-v1','status':'frozen','entries':list(merged.values())}
 Catalog.model_validate({'version':result['version'],'entries':result['entries']})
 ids=[e['id'] for e in result['entries']]
 if len(ids)!=len(set(ids)):raise ValueError('duplicate frozen IDs')
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
 output.with_name(output.stem+'-provenance.json').write_text(json.dumps({'source':str(source),'status':'frozen','source_mapping':provenance,'entries':len(ids)},indent=2)+'\n')
 return len(ids)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('config/catalogs/bitext-v0.3-draft.json'));p.add_argument('--output',type=Path,default=Path('config/catalogs/bitext-v1.json'));a=p.parse_args();print('frozen entries',freeze(a.source,a.output))
