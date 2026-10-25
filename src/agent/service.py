"""Application-facing deep-agent service for the first vertical slice."""
from pathlib import Path
from ..contracts import AgentHandoff, AgentResponse
from .runtime import AgentRuntime
from .skills import SkillRegistry
from .state_store import MemoryStateStore
from .tools import ToolRegistry
from ..classifier.inference import ClassifierBundle
import re

class SupportAgentService:
    def __init__(self, skill_dir: Path = Path('config/skills'), classifier_dir: Path = Path('artifacts/classifier-v2')):
        self.runtime=AgentRuntime(MemoryStateStore(),SkillRegistry.from_directory(skill_dir),ToolRegistry())
        self.classifier=ClassifierBundle(classifier_dir)

    def handle(self, handoff: AgentHandoff, ticket_text: str, entities: dict[str,str] | None = None) -> AgentResponse:
        return self.runtime.start(handoff,ticket_text,entities)

    def classify_and_handle(self, session_id: str, ticket_id: str, ticket_text: str, entities: dict[str,str] | None = None) -> AgentResponse:
        handoff=self.classifier.predict(session_id,ticket_id,ticket_text)
        extracted=self._extract_entities(ticket_text); extracted.update(entities or {})
        return self.runtime.start(handoff,ticket_text,extracted)

    def continue_workflow(self, session_id: str, reply: str) -> AgentResponse:
        state=self.runtime.state_store.get(session_id)
        if state is None: raise ValueError('unknown session')
        entities=dict(state.collected_entities); entities.update(self._extract_entities(reply))
        state.collected_entities=entities; self.runtime.state_store.put(state)
        if state.intent_id:
            handoff=self.classifier.predict(session_id,state.ticket_id,state.ticket_text)
            handoff.intent_id=state.intent_id; handoff.domain_id=self._domain_for(state.intent_id); handoff.decision='handle'; handoff.reason_code='accepted_intent'; handoff.missing_information=[]; handoff.clarification_question=None
        else:
            handoff=self.classifier.predict(session_id,state.ticket_id,f'{state.ticket_text} {reply}')
        return self.runtime.resume(handoff,state)

    @staticmethod
    def _extract_entities(text: str) -> dict[str,str]:
        entities={}
        for key,pattern in [('order_number',r'\bORD[- ]?\d+\b'),('refund_reference',r'\bREF[- ]?\d+\b'),('invoice_reference',r'\bINV[- ]?\d+\b')]:
            match=re.search(pattern,text,re.I)
            if match: entities[key]=match.group(0).upper().replace(' ','-')
        return entities

    @staticmethod
    def _domain_for(intent_id: str | None) -> str | None:
        return {'order.status':'order','refund.status':'refund','refund.request':'refund','invoice.retrieve':'invoice','invoice.download':'invoice','delivery.address_change':'delivery'}.get(intent_id)
