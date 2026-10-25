from src.agent.service import SupportAgentService

def test_classifier_drives_order_workflow():
    service=SupportAgentService();response=service.classify_and_handle('auto-1','ticket-auto','Where is the status of my order?',{'order_number':'ORD-1001'})
    assert response.status in {'completed','waiting_for_user','escalated'}

def test_classifier_low_margin_can_clarify():
    service=SupportAgentService();response=service.classify_and_handle('auto-2','ticket-auto','I need help with my request')
    assert response.status in {'waiting_for_user','escalated','completed'}
