from src.agent.tools import ToolRegistry

def test_tool_registry_success_and_not_found():
    tools=ToolRegistry()
    result=tools.call('lookup_order', order_number='ORD-1001')
    assert result.status == 'success' and result.data['order_status'] == 'shipped'
    assert tools.call('lookup_order', order_number='ORD-9999').status == 'not_found'

def test_tool_registry_enforces_allowlist():
    assert ToolRegistry().call('delete_customer', customer_id='x').status == 'not_allowed'

def test_address_change_precondition():
    tools=ToolRegistry()
    assert tools.call('validate_delivery_address', order_number='ORD-1001', new_address='new').status == 'not_allowed'
    assert tools.call('validate_delivery_address', order_number='ORD-1002', new_address='new').status == 'success'
