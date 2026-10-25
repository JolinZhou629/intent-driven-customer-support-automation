"""Versioned skill contracts and registry for intent-driven workflows."""
import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

class SkillStep(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    id: Text
    instruction: Text
    tool: str | None = None

class SkillDefinition(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    id: Text
    version: Text
    intents: list[Text] = Field(min_length=1)
    objective: Text
    required_entities: list[Text] = Field(default_factory=list)
    allowed_tools: list[Text] = Field(default_factory=list)
    clarification_questions: dict[str, Text] = Field(default_factory=dict)
    preconditions: list[Text] = Field(default_factory=list)
    steps: list[SkillStep] = Field(min_length=1)
    success_conditions: list[Text] = Field(min_length=1)
    escalation_conditions: list[Text] = Field(default_factory=list)

    def supports(self, intent_id: str) -> bool:
        return intent_id in self.intents

    def validate_tool_steps(self) -> 'SkillDefinition':
        unknown = {s.tool for s in self.steps if s.tool and s.tool not in self.allowed_tools}
        if unknown:
            raise ValueError(f'skill steps use tools outside allowlist: {sorted(unknown)}')
        return self

class SkillRegistry:
    def __init__(self, skills: list[SkillDefinition]):
        self.skills = [s.validate_tool_steps() for s in skills]
        self._by_intent = {intent: skill for skill in self.skills for intent in skill.intents}
        if len(self._by_intent) != sum(len(s.intents) for s in self.skills):
            raise ValueError('an intent is assigned to multiple skills')

    @classmethod
    def from_directory(cls, root: Path) -> 'SkillRegistry':
        skills = [SkillDefinition.model_validate_json(p.read_text()) for p in sorted(root.glob('*.json'))]
        if not skills:
            raise ValueError(f'no skill definitions found in {root}')
        return cls(skills)

    def for_intent(self, intent_id: str) -> SkillDefinition | None:
        return self._by_intent.get(intent_id)
