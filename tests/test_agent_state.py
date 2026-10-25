from pathlib import Path
import pytest
from src.agent.state_store import JsonStateStore, MemoryStateStore
from src.contracts import SessionState

def state(session='s-1'):
    return SessionState(session_id=session, ticket_id='t-1', ticket_text='Where is my order?', catalog_version='bitext-v2', status='waiting_for_user', intent_id='order.status', skill_id='order_status', required_entities=['order_number'], clarification_history=['Please provide your order number.'])

@pytest.mark.parametrize('store', [MemoryStateStore(), JsonStateStore(Path('/tmp/agent-state-test'))])
def test_state_round_trip(store):
    saved=store.put(state()); loaded=store.get('s-1')
    assert saved.session_id == loaded.session_id
    assert loaded.status == 'waiting_for_user'
    assert loaded.required_entities == ['order_number']

def test_json_store_rejects_path_traversal(tmp_path):
    store=JsonStateStore(tmp_path)
    with pytest.raises(ValueError): store.get('../escape')

def test_memory_store_returns_copy():
    store=MemoryStateStore(); original=store.put(state()); loaded=store.get('s-1'); loaded.status='completed'
    assert store.get('s-1').status == 'waiting_for_user'
