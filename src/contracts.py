"""Strict boundary contracts for blind tickets, labels, and agent handoffs."""
from typing import Annotated, Literal
from pathlib import PurePosixPath
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Digest = Annotated[str, StringConstraints(pattern=r'^[0-9a-f]{64}$')]


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class Ticket(Contract):
    ticket_id: Text
    # Preserve original text, but reject empty/whitespace-only tickets.
    instruction: Annotated[str, StringConstraints(min_length=1)]

    @model_validator(mode='after')
    def nonempty(self):
        if not self.instruction.strip():
            raise ValueError('instruction must contain text')
        return self


class CatalogEntry(Contract):
    id: Text
    domain: Text
    name: Text
    definition: Text
    include_when: Annotated[list[Text], Field(min_length=1)]
    exclude_when: Annotated[list[Text], Field(min_length=1)]
    examples: Annotated[list[Text], Field(min_length=1)]


class Catalog(Contract):
    version: Text
    entries: Annotated[list[CatalogEntry], Field(min_length=1)]

    @model_validator(mode='after')
    def unique_ids(self):
        ids = [e.id for e in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate catalog IDs')
        return self


class Annotation(Contract):
    ticket_id: Text
    catalog_version: Text
    intent_id: Text | None
    status: Literal['resolved', 'abstained', 'unresolved']
    source: Literal['teacher', 'human']
    annotator_id: Text
    annotation_version: Text
    model_version: Text | None = None
    prompt_version: Text | None = None

    @model_validator(mode='after')
    def provenance(self):
        if (self.status == 'resolved') != (self.intent_id is not None):
            raise ValueError('only resolved annotations have a training intent')
        if self.source == 'teacher' and not (self.model_version and self.prompt_version):
            raise ValueError('teacher annotations require model and prompt versions')
        return self


class Candidate(Contract):
    intent_id: Text
    score: Probability


class ClassifierOutput(Contract):
    ticket_id: Text
    intent_id: Text | None
    domain_id: Text | None
    confidence: Probability
    candidates: list[Candidate]
    decision: Literal['handle', 'clarify', 'escalate']
    reason_code: Text
    missing_information: list[Text] = Field(default_factory=list)
    clarification_question: str | None = None
    catalog_version: Text
    model_version: Text
    policy_version: Text

    @model_validator(mode='after')
    def consistent(self):
        if self.decision == 'handle' and self.intent_id is None:
            raise ValueError('handle requires an intent')
        if (self.intent_id is None) != (self.domain_id is None):
            raise ValueError('intent and domain must both be present or absent')
        ids = [c.intent_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate candidate IDs')
        if self.intent_id is not None and self.intent_id not in ids:
            raise ValueError('predicted intent must be a candidate')
        if self.decision == 'clarify' and not (self.missing_information or self.clarification_question):
            raise ValueError('clarification requires missing information or a question')
        if self.decision == 'clarify' and self.intent_id is not None:
            raise ValueError('clarification must not select a final intent')
        return self

    def validate_catalog(self, catalog: Catalog):
        if self.catalog_version != catalog.version:
            raise ValueError('incompatible catalog version')
        entries = {e.id: e for e in catalog.entries}
        if any(c.intent_id not in entries for c in self.candidates):
            raise ValueError('unknown candidate intent')
        if self.intent_id and entries[self.intent_id].domain != self.domain_id:
            raise ValueError('intent/domain mismatch')
        return self


class AgentHandoff(Contract):
    """Versioned classifier result accepted by the agent runtime."""
    session_id: Text
    ticket_id: Text
    intent_id: Text | None
    domain_id: Text | None
    decision: Literal['handle', 'clarify', 'escalate']
    confidence: Probability
    margin: Probability | None = None
    reason_code: Text
    missing_information: list[Text] = Field(default_factory=list)
    clarification_question: str | None = None
    catalog_version: Text
    model_version: Text
    routing_policy_version: Text

    @model_validator(mode='after')
    def consistent(self):
        if self.decision == 'handle' and self.intent_id is None:
            raise ValueError('handled handoff requires an intent')
        if self.decision == 'clarify' and self.intent_id is not None:
            raise ValueError('clarification handoff must not select an intent')
        if self.decision == 'clarify' and not (self.missing_information or self.clarification_question):
            raise ValueError('clarification handoff requires a question or missing information')
        if (self.intent_id is None) != (self.domain_id is None):
            raise ValueError('intent and domain must both be present or absent')
        return self


class AgentResponse(Contract):
    """Structured result returned by the agent runtime."""
    session_id: Text
    ticket_id: Text
    status: Literal['completed', 'waiting_for_user', 'escalated', 'failed']
    message: Annotated[str, StringConstraints(min_length=1)]
    intent_id: Text | None = None
    skill_id: Text | None = None
    missing_information: list[Text] = Field(default_factory=list)
    escalation_reason: Text | None = None
    evidence_ids: list[Text] = Field(default_factory=list)
    tool_call_count: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode='after')
    def consistent(self):
        if self.status == 'waiting_for_user' and not self.missing_information:
            raise ValueError('waiting_for_user requires missing information')
        if self.status == 'escalated' and self.escalation_reason is None:
            raise ValueError('escalated response requires a reason')
        return self


class SessionState(Contract):
    """Persisted state required to resume a multi-turn workflow."""
    session_id: Text
    ticket_id: Text
    ticket_text: Annotated[str, StringConstraints(min_length=1)]
    catalog_version: Text
    status: Literal['new', 'running', 'waiting_for_user', 'completed', 'escalated', 'failed']
    intent_id: Text | None = None
    skill_id: Text | None = None
    required_entities: list[Text] = Field(default_factory=list)
    collected_entities: dict[str, str] = Field(default_factory=dict)
    clarification_history: list[Text] = Field(default_factory=list)
    tool_results: list[dict[str, object]] = Field(default_factory=list)
    audit_event_ids: list[Text] = Field(default_factory=list)


class AuditEvent(Contract):
    """Append-only event for routing, tool, and workflow observability."""
    event_id: Text
    session_id: Text
    event_type: Literal['classifier', 'routing', 'skill_selected', 'clarification', 'tool_call', 'tool_result', 'response', 'escalation', 'error']
    timestamp: Text
    details: dict[str, object] = Field(default_factory=dict)

class ArtifactFile(Contract):
    path: Text
    sha256: Digest

    @model_validator(mode='after')
    def relative_path(self):
        p = PurePosixPath(self.path)
        if p.is_absolute() or '..' in p.parts or '\\' in self.path or self.path == '.':
            raise ValueError('artifact path must be a safe relative file path')
        return self


class ArtifactManifest(Contract):
    schema_version: Literal['1'] = '1'
    model_version: Text
    catalog_version: Text
    policy_version: Text
    preprocessing_version: Text
    calibration_version: Text
    run_id: Text
    dataset_manifest_sha256: Digest
    config_sha256: Digest
    label_ids: Annotated[list[Text], Field(min_length=1)]
    files: Annotated[list[ArtifactFile], Field(min_length=1)]

    @model_validator(mode='after')
    def unique_members(self):
        if len(self.label_ids) != len(set(self.label_ids)):
            raise ValueError('duplicate labels')
        paths = [f.path for f in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError('duplicate artifact paths')
        return self
