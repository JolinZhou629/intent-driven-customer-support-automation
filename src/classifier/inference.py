"""Reloadable inference adapter for the packaged classifier-v2 bundle."""
import json, joblib
from pathlib import Path
from ..contracts import AgentHandoff

class ClassifierBundle:
    def __init__(self, root: Path):
        self.root=root; self.manifest=json.loads((root/'manifest.json').read_text()); self.model=joblib.load(root/'model.joblib'); catalog=json.loads((root/'catalog.json').read_text()); self.catalog=catalog; self.entries={e['id']:e for e in catalog['entries']}; calibration=json.loads((root/'calibration.json').read_text()); self.margin_threshold=calibration['selected']['margin_threshold']
    def predict(self, session_id: str, ticket_id: str, text: str) -> AgentHandoff:
        intent=self.model.predict([text])[0]; scores=self.model.decision_function([text])[0]; ordered=sorted(scores,reverse=True); margin=float(ordered[0]-ordered[1]) if len(ordered)>1 else 1.0; accepted=margin>=self.margin_threshold; entry=self.entries[intent]
        return AgentHandoff(session_id=session_id,ticket_id=ticket_id,intent_id=intent if accepted else None,domain_id=entry['domain'] if accepted else None,decision='handle' if accepted else 'clarify',confidence=0.0,margin=margin,reason_code='accepted_intent' if accepted else 'unknown_request',missing_information=[] if accepted else ['request_context'],clarification_question=None if accepted else 'Could you provide more details about what you need help with?',catalog_version=self.manifest['catalog_version'],model_version=self.manifest['bundle_version'],routing_policy_version=self.manifest['routing_policy_version'])
