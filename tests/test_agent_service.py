from src.agent.service import SupportAgentService
from src.contracts import AgentHandoff

def make(decision='handle',intent='order.status',domain='order'):
    return AgentHandoff(session_id='svc-'+decision+'-'+str(intent),ticket_id='ticket-svc',intent_id=intent,domain_id=domain,decision=decision,confidence=.9,margin=.3,reason_code='accepted_intent',catalog_version='bitext-v2',model_version='classifier-v2',routing_policy_version='routing-v1')

def test_vertical_slice_success():
    r=SupportAgentService().handle(make(), 'Where is my order?', {'order_number':'ORD-1001'})
    assert r.status=='completed' and r.tool_call_count==1

def test_vertical_slice_clarification():
    service=SupportAgentService();r=service.handle(make(), 'Where is my order?')
    assert r.status=='waiting_for_user' and 'order_number' in r.missing_information
    assert service.continue_workflow('svc-handle-order.status','ORD-1001').status=='completed'

def test_vertical_slice_escalation():
    h=make('escalate',None,None);h.reason_code='unknown_request';r=SupportAgentService().handle(h,'Something unsupported')
    assert r.status=='escalated'

def test_refund_and_invoice_skills():
    service=SupportAgentService()
    refund=make(intent='refund.status',domain='refund');refund.session_id='refund-1'
    invoice=make(intent='invoice.retrieve',domain='invoice');invoice.session_id='invoice-1'
    assert service.handle(refund,'What is my refund status?',{'refund_reference':'REF-1001'}).status=='completed'
    assert service.handle(invoice,'Find my invoice',{'invoice_reference':'INV-1001'}).status=='completed'

def test_delivery_address_skill():
    service=SupportAgentService();h=make(intent='delivery.address_change',domain='delivery');h.session_id='address-1'
    assert service.handle(h,'Change my delivery address',{'order_number':'ORD-1002','new_address':'10 Example Road'}).status=='completed'
    h.session_id='address-2';assert service.handle(h,'Change my delivery address',{'order_number':'ORD-1001','new_address':'10 Example Road'}).status=='escalated'
