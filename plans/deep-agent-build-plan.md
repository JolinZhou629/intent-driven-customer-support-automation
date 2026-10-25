---
title: Deep Agent Build Plan
type: plan
status: active
updated: 2026-09-26
tags: [deep-agent, skills, tools, routing, evaluation]
---

# Deep Agent Build Plan

## 1. Purpose and scope

Build a controlled, intent-aware support agent that consumes the packaged
`bitext-v2` classifier result, selects a versioned skill, gathers missing
information, calls only permitted tools, and produces a grounded response.

The first implementation uses project-created knowledge and simulated tools.
It does not connect to company systems, production endpoints, customer data, or
irreversible external actions.

The classifier answers **what the request is about**. The deep agent determines
**how to handle the request** within routing, skill, and tool policies.

## 2. Target architecture

```text
Customer request
    → session load or creation
    → input validation
    → bitext-v2 classifier
    → deterministic routing policy
        → clarify: ask and save state
        → escalate: record reason and hand off
        → handle: select skill
    → check required information and permissions
    → call permitted tool
    → validate structured result
    → retry, clarify, or escalate when needed
    → grounded response
    → persist state and audit event
```

Core components:

- **Classifier bundle:** `artifacts/classifier-v2/`.
- **Routing policy:** margin threshold, supported-intent rules, and clarification
  outcome `unknown.require_more_info`.
- **Skills:** versioned workflow instructions selected by intent.
- **Tools:** typed, allowlisted functions over fixture data or approved project
  knowledge.
- **Session state:** entities, workflow progress, questions, tool results, and
  outcome.
- **Audit log:** classifier result, routing decision, skill, tool calls, evidence,
  retries, latency, and cost.

## 3. Phase 1 — Define contracts

### Tasks

1. Define the classifier-to-agent input contract.
2. Define the agent response contract.
3. Define session-state and audit-event schemas.
4. Define version compatibility checks for the classifier bundle and catalog.
5. Define error types for invalid input, low confidence, missing information,
   tool failure, unsupported request, and escalation.

### Classifier handoff

The handoff should contain:

```json
{
  "ticket_id": "...",
  "intent_id": "order.status",
  "domain_id": "order",
  "decision": "handle",
  "confidence": 0.91,
  "margin": 0.32,
  "catalog_version": "bitext-v2",
  "model_version": "classifier-v2",
  "routing_policy_version": "routing-v1"
}
```

### Acceptance criteria

- Invalid or incompatible classifier outputs are rejected.
- Clarification outputs contain no final intent.
- Every agent run has a session ID and audit ID.
- All contracts have automated validation tests.

## 4. Phase 2 — Implement the session state

### State fields

```json
{
  "session_id": "...",
  "ticket_text": "...",
  "classifier_output": {},
  "selected_skill": "order_status",
  "required_entities": ["order_number"],
  "collected_entities": {},
  "clarification_history": [],
  "tool_results": [],
  "status": "waiting_for_user",
  "final_outcome": null,
  "audit_events": []
}
```

### Tasks

1. Create a state store interface with an in-memory implementation for tests.
2. Add save, reload, and resume behavior.
3. Ensure state is updated after every clarification and tool call.
4. Add limits for clarification turns, tool calls, retries, and execution time.

### Acceptance criteria

- A clarification response can resume the same workflow.
- State survives a process restart in the selected persistence implementation.
- Repeated requests do not duplicate completed actions.

## 5. Phase 3 — Define skill contracts

Each skill is a versioned workflow definition containing:

- Skill ID and version.
- Supported intent IDs.
- Workflow objective.
- Required entities.
- Clarification questions.
- Allowed tools.
- Preconditions.
- Workflow steps.
- Success conditions.
- Escalation conditions.
- Response requirements.

Example:

```yaml
id: order_status
version: skill-v1
intents: [order.status]
required_entities: [order_number]
allowed_tools: [lookup_order]
preconditions:
  - order_number_is_present
steps:
  - validate_order_number
  - lookup_order
  - verify_lookup_result
success_conditions:
  - order_record_found
escalate_when:
  - order_not_found
  - tool_failure_after_retry
```

## 6. Phase 4 — Build the first skills

Implement the following vertical-slice skills first:

### `order_status`

- Required entity: order number.
- Tool: `lookup_order`.
- Handles existing order status and missing order number.

### `refund_status`

- Required entity: refund or transaction reference where available.
- Tool: `lookup_refund`.
- Distinguishes refund status from refund policy, request, and processing time.

### `invoice_retrieval`

- Required entity: invoice number or account reference.
- Tool: `retrieve_invoice`.
- Returns invoice metadata or a safe retrieval instruction.

### `delivery_address_change`

- Required entity: order number and new address.
- Tool: `validate_delivery_address`.
- Requires a precondition check before any simulated update.

### Acceptance criteria

- Each skill has a schema-validated definition.
- Each skill has at least one success, clarification, and escalation scenario.
- Skills cannot call tools outside their allowlist.

## 7. Phase 5 — Build typed simulated tools

Initial tools:

```text
lookup_order
lookup_refund
retrieve_invoice
validate_delivery_address
```

Every tool must define:

- Input schema.
- Output schema.
- Read or write classification.
- Permission requirement.
- Preconditions.
- Retry policy.
- Failure states.
- Audit fields.

Example result:

```json
{
  "status": "success",
  "order_number": "ORD-1001",
  "order_status": "shipped",
  "estimated_delivery": "2026-10-03",
  "source": "fixture.order_store"
}
```

The agent must not claim an action or fact that is absent from a validated tool
result.

## 8. Phase 6 — Implement routing and execution

### Routing rules

```text
Accepted supported intent
    → select matching skill

Low classifier margin
    → ask clarification or escalate

unknown.require_more_info
    → ask a targeted question

Missing skill-required entity
    → ask a skill-specific question

Unsupported intent or policy boundary
    → escalate
```

### Execution loop

1. Load or create session state.
2. Validate the request and classifier bundle version.
3. Apply the routing policy.
4. Select the skill from the intent catalog.
5. Check required entities and permissions.
6. Ask one clarification question when needed.
7. Prepare and execute an allowlisted tool call.
8. Validate the structured tool result.
9. Retry within the configured limit or escalate.
10. Generate a grounded response.
11. Persist state and audit events.

### Guardrails

- Maximum three clarification turns.
- Maximum two retries per tool call.
- Maximum five tool calls per workflow.
- No tool call when routing decision is `clarify` or `escalate`.
- No external side effects in the school project.
- Tool output and ticket text are treated as data, not instructions.

## 9. Phase 7 — Add the agent runtime

Start with a simple application-controlled execution loop. Add a framework such
as LangGraph or a Deep Agents harness only where it provides clear value for
state, planning, or tool orchestration.

The first runtime should support:

- One request at a time.
- Explicit skill selection.
- Typed tool calls.
- Persistent session state.
- Structured audit events.
- Deterministic stop conditions.

Autonomous sub-agents are deferred until the single-agent vertical slice shows a
need for delegation.

## 10. Phase 8 — Evaluation scenarios

Create fixture scenarios for each skill:

- Valid request with all required information.
- Missing required information.
- Ambiguous classifier result.
- Unsupported request.
- Unknown record.
- Invalid tool input.
- Tool timeout or failure.
- Successful clarification and resume.
- Retry-limit escalation.

Run each scenario in two modes:

1. Correct intent supplied directly to isolate agent behavior.
2. Actual v2 classifier output to measure end-to-end behavior.

Measure:

- Correct skill selection.
- Correct tool selection and arguments.
- Clarification quality.
- Grounding in tool results.
- Workflow completion.
- Escalation correctness.
- Tool calls and retries.
- Latency, tokens, and estimated cost.

## 11. Phase 9 — Package and release

Create a deep-agent handoff bundle containing:

```text
classifier/
skills/
tools/
agent/
eval/
manifest.json
```

Freeze the classifier bundle, catalog, skills, tool schemas, routing policy,
state schema, and evaluation configuration together. Add reload tests and
version compatibility checks.

## 12. Implementation order and checkpoints

| Checkpoint | Deliverable | Completion condition |
|---|---|---|
| A | Contracts and state | Schemas validate and state can resume |
| B | `order_status` skill/tool | Success, clarification, and escalation tests pass |
| C | Four initial skills | Each skill has tool and failure scenarios |
| D | Runtime and routing | End-to-end fixture execution works |
| E | Evaluation | Component and end-to-end metrics are reported |
| F | Release bundle | Reloadable, versioned handoff package is verified |

## 13. Risks and controls

- **Classifier error:** evaluate with and without classifier output.
- **Tool hallucination:** require typed, validated tool results.
- **Workflow loops:** enforce turn, retry, tool, and time limits.
- **Over-broad permissions:** allowlist tools per skill.
- **Catalog changes:** version catalog and regenerate affected labels.
- **Teacher-label bias:** report teacher agreement separately from human spot checks.
- **Uncontrolled cost:** record tokens, calls, retries, and estimated cost per run.
