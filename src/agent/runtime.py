"""Deterministic first-pass agent routing and execution loop."""
from dataclasses import dataclass
from ..contracts import AgentHandoff, AgentResponse, SessionState
from .skills import SkillRegistry, SkillDefinition
from .state_store import StateStore
from .tools import ToolRegistry, ToolResult

@dataclass
class RuntimeLimits:
    max_clarifications: int = 3
    max_tool_calls: int = 5
    max_retries: int = 2

class AgentRuntime:
    def __init__(self, state_store: StateStore, skills: SkillRegistry, tools: ToolRegistry, limits: RuntimeLimits | None = None):
        self.state_store=state_store; self.skills=skills; self.tools=tools; self.limits=limits or RuntimeLimits()

    def start(self, handoff: AgentHandoff, ticket_text: str, entities: dict[str, str] | None = None) -> AgentResponse:
        existing=self.state_store.get(handoff.session_id)
        if existing: return self.resume(handoff, existing)
        state=SessionState(session_id=handoff.session_id,ticket_id=handoff.ticket_id,ticket_text=ticket_text,catalog_version=handoff.catalog_version,status='new',intent_id=handoff.intent_id,collected_entities=entities or {})
        self.state_store.put(state); return self._route(handoff,state)

    def resume(self, handoff: AgentHandoff, state: SessionState) -> AgentResponse:
        return self._route(handoff,state)

    def _route(self, handoff: AgentHandoff, state: SessionState) -> AgentResponse:
        if handoff.decision == 'escalate': return self._escalate(state,'classifier_escalation')
        if handoff.decision == 'clarify': return self._clarify(state, handoff.missing_information or ['request_context'])
        skill=self.skills.for_intent(handoff.intent_id or '')
        if skill is None: return self._escalate(state,'no_skill_for_intent')
        state.skill_id=skill.id; state.status='running'; self.state_store.put(state)
        missing=[e for e in skill.required_entities if e not in state.collected_entities]
        if missing: return self._clarify(state,missing)
        return self._execute(skill,state)

    def _clarify(self,state:SessionState,missing:list[str]) -> AgentResponse:
        if len(state.clarification_history)>=self.limits.max_clarifications: return self._escalate(state,'clarification_limit_reached')
        state.status='waiting_for_user';state.required_entities=list(dict.fromkeys(missing));question='Please provide: '+', '.join(missing)+'.';state.clarification_history.append(question);self.state_store.put(state)
        return AgentResponse(session_id=state.session_id,ticket_id=state.ticket_id,status='waiting_for_user',message=question,intent_id=state.intent_id,skill_id=state.skill_id,missing_information=missing)

    def _execute(self,skill:SkillDefinition,state:SessionState)->AgentResponse:
        calls=0
        evidence=[]
        for step in skill.steps:
            if not step.tool: continue
            calls+=1
            if calls>self.limits.max_tool_calls:return self._escalate(state,'tool_call_limit_reached')
            kwargs={k:v for k,v in state.collected_entities.items() if k in {'order_number','refund_reference','invoice_reference','new_address'}}
            result=self.tools.call(step.tool,**kwargs);state.tool_results.append(result.model_dump());self.state_store.put(state)
            if result.status=='success':
                evidence.append(result.evidence_id)
                continue
            if result.status=='invalid_input':return self._clarify(state,[result.error_code or 'required_information'])
            return self._escalate(state,result.error_code or 'tool_failure')
        if evidence:
            state.status='completed';self.state_store.put(state);return AgentResponse(session_id=state.session_id,ticket_id=state.ticket_id,status='completed',message=self._success_message(skill.id,state.tool_results),intent_id=state.intent_id,skill_id=skill.id,evidence_ids=evidence,tool_call_count=calls)
        return self._escalate(state,'skill_has_no_executable_tool')

    def _escalate(self,state:SessionState,reason:str)->AgentResponse:
        state.status='escalated';self.state_store.put(state);messages={'no_skill_for_intent':'I could not find a supported workflow for this request.','refund_not_eligible':'This order is not eligible for a simulated refund under the current policy.','order_not_found':'I could not find that order in the simulated support database.','tool_failure':'The simulated support lookup failed, so this request needs human support.'};return AgentResponse(session_id=state.session_id,ticket_id=state.ticket_id,status='escalated',message=messages.get(reason,'This request requires human support.'),intent_id=state.intent_id,skill_id=state.skill_id,escalation_reason=reason)

    @staticmethod
    def _success_message(skill_id:str, results:list[dict[str,object]])->str:
        data=results[-1].get('data',{}) if results else {}
        if skill_id=='order_status': return f"Order {data.get('order_number','')} is {data.get('order_status','unknown')}. Estimated delivery: {data.get('estimated_delivery','not available')}."
        if skill_id=='refund_status': return f"Refund {data.get('refund_reference','')} is currently {data.get('refund_status','unknown')}."
        if skill_id=='invoice_retrieval': return f"Invoice {data.get('invoice_reference','')} is available. Simulated path: {data.get('download_path','not available')}."
        if skill_id=='refund_request': return f"Your simulated refund request {data.get('refund_reference','')} was submitted for order {data.get('order_number','')}."
        if skill_id=='delivery_address_change': return f"The delivery address for order {data.get('order_number','')} passed the simulated eligibility check."
        return f"Workflow {skill_id} completed with confirmed tool evidence."
