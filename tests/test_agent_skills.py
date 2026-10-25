import json
from pathlib import Path
import pytest
from src.agent.skills import SkillDefinition, SkillRegistry

def test_skill_registry_loads_all_initial_skills():
    registry=SkillRegistry.from_directory(Path('config/skills'))
    assert registry.for_intent('order.status').id == 'order_status'
    assert registry.for_intent('refund.status').id == 'refund_status'
    assert registry.for_intent('invoice.retrieve').id == 'invoice_retrieval'
    assert registry.for_intent('delivery.address_change').id == 'delivery_address_change'

def test_skill_rejects_tool_outside_allowlist():
    skill=SkillDefinition.model_validate({"id":"bad","version":"v1","intents":["x"],"objective":"x","steps":[{"id":"s","instruction":"x","tool":"not_allowed"}],"success_conditions":["done"]})
    with pytest.raises(ValueError): skill.validate_tool_steps()
