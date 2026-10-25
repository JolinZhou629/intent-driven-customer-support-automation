from src.agent.runtime import AgentRuntime
from src.agent.skills import SkillRegistry
from src.agent.state_store import MemoryStateStore
from src.agent.tools import ToolRegistry
from src.contracts import AgentHandoff

def handoff(decision='handle',intent='order.status'):
    return AgentHandoff(session_id='session-1',ticket_id='ticket-1',intent_id=intent,domain_id='order',decision=decision,confidence=.9,margin=.3,reason_code='accepted_intent',catalog_version='bitext-v2',model_version='classifier-v2',routing_policy_version='routing-v1')

def test_runtime_clarifies_missing_entity():
    r=AgentRuntime(MemoryStateStore(),SkillRegistry.from_directory(__import__('pathlib').Path('config/skills')),ToolRegistry()).start(handoff(),'Where is my order?')
    assert r.status=='waiting_for_user' and 'order_number' in r.missing_information

def test_runtime_completes_with_state_entity():
    store=MemoryStateStore();runtime=AgentRuntime(store,SkillRegistry.from_directory(__import__('pathlib').Path('config/skills')),ToolRegistry()); runtime.start(handoff(),'Where is my order?'); state=store.get('session-1');state.collected_entities={'order_number':'ORD-1001'};store.put(state);r=runtime.resume(handoff(),state)
    assert r.status=='completed' and r.evidence_ids

def test_runtime_escalates_unknown_skill():
    h=handoff(intent='support.feedback');h.domain_id='support';r=AgentRuntime(MemoryStateStore(),SkillRegistry.from_directory(__import__('pathlib').Path('config/skills')),ToolRegistry()).start(h,'feedback')
    assert r.status=='escalated'
