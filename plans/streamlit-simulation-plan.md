---
title: Streamlit Ticketing Simulation Plan
type: plan
status: active
updated: 2026-09-26
tags: [streamlit, demo, deep-agent, classifier]
---

# Streamlit Ticketing Simulation Plan

## Purpose

Build a local Streamlit application that demonstrates the complete school-project
workflow:

```text
Ticket submitted
  → classifier predicts intent
  → routing policy decides handle, clarify, or escalate
  → deep agent selects a skill
  → skill checks required information
  → permitted simulated tool runs
  → grounded guidance is generated
  → ticket reply and audit trail are displayed
```

The application is a simulation. It uses the packaged `classifier-v2` bundle,
project-created skills, fixture data, and simulated tools. It does not connect to
company systems or perform external actions.

## User interface design

### 1. New ticket panel

Inputs:

- Ticket text.
- Optional ticket ID.
- Optional session ID for resuming a conversation.
- Optional structured entities for demonstration and testing.

Action:

- `Submit ticket` starts or resumes the workflow.

### 2. Classifier panel

Display:

- Predicted intent.
- Domain.
- SVM margin.
- Routing decision.
- Catalog version.
- Model version.

Possible decisions:

- `handle`.
- `clarify`.
- `escalate`.

### 3. Agent execution panel

Display:

- Selected skill and version.
- Required entities.
- Collected entities.
- Current workflow status.
- Tool calls and structured results.
- Retry count.

### 4. Reply panel

Display one of:

- Grounded support guidance.
- A clarification question.
- An escalation message.
- A tool or workflow failure message.

### 5. Audit panel

Display:

- Session ID and ticket ID.
- Classifier event.
- Routing event.
- Skill selection event.
- Tool-call events.
- Evidence IDs.
- Final outcome.
- Latency and estimated cost.

## Application architecture

```text
Streamlit UI
    ↓
Ticket/session controller
    ↓
ClassifierBundle
    ↓
AgentRuntime
    ├── SkillRegistry
    ├── ToolRegistry
    ├── StateStore
    └── Audit logger
    ↓
Structured response
```

The Streamlit layer should call the existing service rather than duplicate
classifier, routing, skill, or tool logic.

## Session behavior

Use `st.session_state` for the active browser session and the existing state
store for workflow state.

When the agent asks for clarification:

1. Display the question.
2. Keep the session and ticket IDs unchanged.
3. Accept the user's follow-up response.
4. Extract or accept the newly supplied entity.
5. Resume the saved workflow.

The UI must clearly distinguish a new ticket from a resumed ticket.

## Initial demonstration scenarios

Implement these scenarios first:

1. Order status with a valid order number.
2. Order status with a missing order number, followed by clarification.
3. Refund status with a valid refund reference.
4. Invoice retrieval with a valid invoice reference.
5. Delivery address change for an eligible order.
6. Delivery address change for an already shipped order.
7. Unsupported request routed to escalation.
8. Low-margin prediction routed to clarification.

## Response generation

The first version should use deterministic response templates populated from
validated tool results. This makes the demonstration reproducible and prevents
the UI from claiming facts that are not present in fixture data.

Example:

```text
Your order {order_number} is {order_status}.
The estimated delivery date is {estimated_delivery}.
```

An LLM response-generation layer can be added later behind a provider interface,
but it must receive only the selected skill, validated evidence, and approved
response instructions.

## Safety and simulation boundaries

- Load only the packaged classifier and project-created configuration.
- Use only allowlisted simulated tools.
- Do not expose credentials, internal endpoints, or company data.
- Do not create real tickets or send external messages.
- Require tool-confirmed results before showing completion.
- Show escalation when a tool rejects an action.
- Keep fixture data clearly labeled as simulated.

## Implementation phases

### Phase A — Streamlit shell

- Create `src/streamlit_app.py`.
- Add page title and ticket input.
- Add session-state initialization.
- Add submit and reset controls.

### Phase B — Service integration

- Connect the UI to `SupportAgentService`.
- Render classifier and routing results.
- Render selected skill and required entities.
- Render structured agent responses.

### Phase C — Clarification loop

- Display missing information.
- Accept follow-up responses.
- Update collected entities.
- Resume the same session.

### Phase D — Audit and evidence display

- Add event collection to the runtime.
- Render tool calls and evidence IDs.
- Display final status and escalation reason.

### Phase E — Demonstration and evaluation

- Add scenario selector.
- Add fixture data reset.
- Run the eight initial scenarios.
- Record completion, clarification, escalation, latency, and tool-call metrics.

## Acceptance criteria

- A user can submit a ticket from the browser.
- The packaged v2 classifier runs without manual handoff construction.
- The UI displays intent, margin, routing decision, skill, and tool result.
- A missing order number produces a clarification question.
- A follow-up order number resumes the same workflow.
- A successful tool result produces grounded guidance.
- A rejected action produces escalation.
- No real external action is performed.
- The application can reload the classifier bundle and skill definitions.
- Streamlit behavior is covered by service-level tests and scenario checks.

## Suggested project files

```text
src/streamlit_app.py
src/agent/service.py
src/agent/runtime.py
src/agent/state_store.py
src/agent/skills.py
src/agent/tools.py
eval/agent-vertical-slice-v1.jsonl
reports/streamlit-demo-v1.json
```

## Metrics to record

- Tickets submitted.
- Completed workflows.
- Clarification rate.
- Escalation rate.
- Tool success and failure rate.
- Average tool calls per ticket.
- Average response latency.
- Estimated model and tool cost.
- Successful resume rate after clarification.
