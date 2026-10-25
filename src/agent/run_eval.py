"""Run the deterministic agent vertical-slice fixture suite."""
import json
from pathlib import Path
from .service import SupportAgentService
from ..contracts import AgentHandoff

def main():
    service=SupportAgentService(); rows=[json.loads(x) for x in Path('eval/agent-vertical-slice-v1.jsonl').read_text().splitlines()]; results=[]
    for row in rows:
        h=AgentHandoff(session_id='eval-'+row['id'],ticket_id='ticket-'+row['id'],intent_id=row['intent_id'],domain_id=row['domain_id'],decision=row['decision'],confidence=.9 if row['decision']=='handle' else .1,margin=.3 if row['decision']=='handle' else None,reason_code='accepted_intent' if row['decision']=='handle' else 'unknown_request',catalog_version='bitext-v2',model_version='classifier-v2',routing_policy_version='routing-v1')
        response=service.handle(h,row['id'],row['entities']);results.append({'id':row['id'],'expected':row['expected_status'],'actual':response.status,'passed':response.status==row['expected_status']})
    out={'scenarios':len(results),'passed':sum(r['passed'] for r in results),'results':results};Path('reports/agent-vertical-slice-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
