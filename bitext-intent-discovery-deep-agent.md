---
title: RFC - Bitext Classifier Training and Intent-Driven Deep Agent
date: 2026-08-25
updated: 2026-09-05
type: rfc
status: draft
tags: [rfc, sandbox, bitext, intent-discovery, classifier, deep-agent, evaluation]
---

# RFC: Bitext Classifier Training and Intent-Driven Deep Agent

## Summary

A standalone project with two independently testable parts:

1. **Classifier model training:** discover an intent catalog from unlabeled
   support tickets, obtain teacher labels and human adjudication, and train a
   student classifier. Compare it with a separately trained supervised Bitext
   baseline using a controlled evaluation protocol.
2. **Intent-driven deep agent:** consume the classifier result, select an
   approved runbook, gather missing information, use permitted tools, and
   produce an evidence-backed outcome or escalation.

The classifier predicts what the user needs. The agent decides how to handle
that need within explicit policy and tool permissions. A confident intent is
not sufficient authorization to act and does not imply that required entities
or business preconditions are satisfied.

The initial application foundation is the archived Bitext support project at
`~/llm-wiki/automated-tech-support-ticketing`. The discovery/catalog/teacher/
review methodology is inspired by the separate D365/Kinabalu/AISRE design;
those capabilities are new implementation work here.

## Foundation and scope

The archived Bitext implementation contains:

- A DistilBERT classifier trained on 27 existing intent labels.
- An inference confidence threshold of `0.8` that routes to clarification.
- A CSV-backed KB selecting the first response for each intent.
- A Gemini reply generator, FastAPI `/chat` endpoint, and Streamlit UI.
- A notebook with an 80/20 stratified train/validation split.

It does not implement blind intent discovery, teacher/student distillation,
versioned catalogs, human adjudication, or a tool-using deep agent. The local
archive's CSV is a Git LFS pointer; actual data and model availability must be
verified during setup. Existing checkpoint scores are not clean baselines
unless their training data and evaluation provenance can be established.

Reference evidence:

- [Source pipeline](../../../llm-wiki/automated-tech-support-ticketing/wiki/sources/src-pipeline.md)
- [Training and model artifacts](../../../llm-wiki/automated-tech-support-ticketing/wiki/sources/training-data-and-model-artifacts.md)
- [Application and deployment](../../../llm-wiki/automated-tech-support-ticketing/wiki/sources/app-and-deployment.md)

## Goals and non-goals

Goals:

- Measure how well blind discovery recovers a useful taxonomy and supports
  classification on unseen tickets.
- Measure whether intent-driven tools and planning improve task completion
  compared with the original KB-plus-generator approach.
- Make failures attributable to classification, routing, retrieval, tool use,
  or generation through separate component and end-to-end evaluations.
- Preserve reproducibility through frozen data, catalogs, models, policies,
  KB versions, and scenario fixtures.

Non-goals:

- No changes to the production D365/Kinabalu/AISRE system or use of its data.
- No real external account, payment, refund, or order integrations. Reads and
  state-changing actions operate only on a resettable simulated backend.
- No claim that results on synthetic English Bitext tickets establish
  production performance.

## Dataset and evaluation boundaries

Source: [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).
The published dataset is approximately 26,872 rows with `flags`, `instruction`,
`category`, `intent`, and `response`, under CDLA-Sharing 1.0. The current viewer
reports 11 category values and 27 intent values; preparation must verify counts
from a pinned revision rather than assume the earlier RFC's 10 categories.

### Split before discovery

Phase 0 creates immutable train/development/test partitions before any catalog
work. Use a reproducible, label-independent grouped split, initially targeting
70/15/15. Group exact duplicates and conservatively detected near-duplicate
instruction variants where feasible. Record the grouping method and audit
cross-split similarity; grouping does not guarantee removal of all template
leakage. Do not silently revise partitions after seeing evaluation results.

- **Train:** embeddings, clustering, catalog drafting, teacher training labels,
  human training adjudication, student fitting, and candidate KB material.
- **Development:** blind evaluation and controlled catalog revisions, model
  selection, confidence calibration, and routing thresholds. No fitting UMAP
  or clustering on development/test data in the primary experiment.
- **Test:** instructions, responses, and labels unavailable to development
  processes until the corresponding final evaluation.

Fit discovery transformations on training instructions only. Classify unseen
instructions against the frozen catalog. An optional all-corpus discovery run
must be reported separately as transductive taxonomy recovery, not as evidence
of unseen-ticket generalization.

### Hidden labels and isolated comparison branch

```text
data/
  manifest.json                 # revision, hashes, grouping, split IDs, seed
  blind/train/tickets.jsonl     # opaque ticket_id, instruction
  blind/dev/tickets.jsonl
  sealed/test/tickets.jsonl
  kb_source/train/responses.jsonl
  sealed/test/responses.jsonl
  gold_hidden/true_labels.jsonl # ticket_id, category, intent
  metadata/flags.jsonl          # analysis metadata, not a default model input
```

The trusted preparation process separates labels, responses, and split data.
Shuffle rows and assign opaque IDs so source ordering and IDs do not encode
intent blocks. Discovery, catalog, teacher, review, and student-training jobs
receive only their allowed inputs; a directory naming convention alone is not
an isolation mechanism. Use explicit dataset loaders and input manifests, and
verify that blind jobs cannot access sealed/gold artifacts.

The supervised comparison branch may read training reference labels, but must
run separately and publish no labels, taxonomy, checkpoints, or findings to the
blind branch before its catalog, models, and policies are frozen. Human access
to the public taxonomy cannot be ruled out; record that limitation rather than
claim perfect experimenter blinding.

## Part 1: Classifier model training

### Blind discovery and catalog construction

1. Embed training instructions, initially with
   `sentence-transformers/all-MiniLM-L6-v2`.
2. Compare clustering with and without UMAP; fit UMAP on training data only.
3. Run HDBSCAN across a bounded parameter sweep without fixing the number of
   clusters to the known Bitext intent count.
4. Inspect noise, cluster stability across seeds, representative utterances,
   and boundary cases. Silhouette is supporting evidence, not the sole
   selection criterion.
5. Draft catalog entries with `{id, domain, name, definition, include_when,
   exclude_when, examples}` using training representatives and LLM assistance.
6. Merge or split clusters through documented human review. Store catalog data
   versions and a changelog; use one reusable builder rather than a separate
   Python program for each version.

HDBSCAN noise is an analysis pool, not automatically a semantic unknown class.
Review it for missed intents, ambiguous tickets, and clustering artifacts.

Use a bounded discovery budget, initially at most three revision rounds.
Declare development acceptance criteria and their numerical thresholds before
running the sweep. If criteria are unmet, report the outcome or start a newly
versioned experiment rather than iterate against test results.

### Teacher, review, and student

- Use a local-first annotation pipeline. MiniLM embeddings and lightweight
  similarity or classifier models assign high-confidence tickets without an
  API request. Store the score and the margin between the first- and
  second-best intents.
- Send only low-confidence, ambiguous, unsupported, or high-risk tickets to
  the teacher model. Teacher classification returns a discovered intent or
  abstention using a schema-validated response. Store teacher model, prompt,
  catalog version, rationale/evidence where available, and provenance.
- Keep the teacher as a selective reviewer rather than labeling every ticket.
  This reduces API cost and latency while focusing review on the examples most
  likely to affect boundary quality.
- Review uncertain cases, noise, boundaries, and disagreement cases for
  improvement. These adjudicated examples may enter training.
- Maintain a separate, held-out human evaluation pool that cannot enter
  training. Annotators use the discovered catalog without reference labels.
- Treat an initial 100-ticket review as a pilot, not sufficient per-intent
  validation. Expand based on discovered catalog coverage and uncertainty;
  report sample counts and uncertainty for measured results.
- Train TF-IDF/logistic regression first, then DistilBERT if justified by
  development results. Distinguish pseudo-labels from human-adjudicated labels
  in training records and report their contribution.
- Share preprocessing between training and inference. Preserve original text
  for entity extraction; punctuation removal must not corrupt identifiers.
- Calibrate confidence and select routing thresholds on development data.
  Evaluate unknown inputs explicitly rather than assume softmax confidence
  detects out-of-domain tickets.
- Report local versus teacher-reviewed coverage, API requests, tokens, latency,
  estimated cost per ticket, and the quality difference between accepted local
  labels and teacher-reviewed labels.

### Supervised comparison

Train fresh supervised TF-IDF/logistic regression and, if used in the blind
branch, DistilBERT models on the same training partition with Bitext labels.
Use matching preprocessing and comparable tuning budgets. This comparison
measures the effect of the labeling/catalog approach; it is not a theoretical
upper bound. Do not use the archived checkpoint for primary held-out scores
without proving that it has not seen evaluation examples.

### Classifier artifact and inference contract

Publish a compatible bundle containing the classifier, tokenizer/vectorizer,
label mapping, catalog, calibration parameters, routing policy, and a manifest
with versions and hashes. Derive the number of labels from the catalog rather
than hard-code 27. Reject incompatible bundles at startup.

Example handoff to Part 2:

```json
{
  "ticket_id": "ticket-opaque-id",
  "intent_id": "order.track",
  "domain_id": "order",
  "confidence": 0.93,
  "candidates": [
    {"intent_id": "order.track", "score": 0.93},
    {"intent_id": "order.delivery_issue", "score": 0.05}
  ],
  "decision": "handle",
  "reason_code": "accepted_intent",
  "catalog_version": "bitext-v1",
  "model_version": "student-v1",
  "policy_version": "routing-v1"
}
```

`decision` is produced by the routing policy and is one of `handle`, `clarify`,
or `escalate`; `intent_id` may be null on abstention. The IDs above are examples,
not an imposed taxonomy. The agent also receives original ticket text and
session context. It must not silently change the predicted intent: conflicting
evidence triggers a logged reclassification, clarification, or escalation.

## Part 2: Intent-driven deep agent

### Responsibilities and execution flow

```text
Ticket + classifier result + session context
  → Validate artifact versions and routing decision
  → Select approved intent runbook and domain skill
  → Retrieve relevant KB evidence
  → Check required entities and action preconditions
      → Ask for missing information and resume on the next turn
      → Escalate unsupported or conflicting cases
      → Execute permitted simulated tools
  → Verify tool outcome
  → Draft evidence-backed reply
  → Return outcome and execution trace
```

The first implementation uses an explicit state graph. Add planner/todo and
sub-agent behavior where multi-step scenarios demonstrate a benefit. Domain
skills scope runbooks, prompts, and tools; shared action logic should not be
duplicated into a separate agent for every intent.

State includes original text, classifier result, extracted entities, pending
clarification, selected runbook, retrieved evidence, tool outputs, and action
history. Persist session state so clarification can actually resume a task.
Treat ticket text, retrieved content, and tool strings as data; tool permissions
are enforced by code, not instructions embedded in those inputs.

### Runbooks and knowledge base

Build candidate templates from training responses grouped by discovered intent.
Deduplicate and review them against the simulated backend's policies. Responses
are source material, not authoritative business rules, and test responses must
not enter the KB.

Each approved runbook declares:

- Compatible catalog and intent IDs, version, and source provenance.
- Required entities, allowed tools, and preconditions.
- Ordered steps or permitted branches, clarification and escalation rules.
- Whether an action needs explicit user confirmation.
- Success conditions and reply evidence requirements.

`search_kb(intent_id, query)` retrieves within the compatible approved KB.
A missing or contradictory runbook produces a structured fallback rather than
a generic string that the generator treats as a valid procedure.

### Tools and simulated actions

- `entity_extractor`: extracts concrete values from original ticket text and
  context. `{{Order Number}}` is an unresolved slot, not a usable identifier.
- Read tools: order, account, payment, and refund status.
- Simulated action tools: for example, cancel an eligible order or submit an
  eligible refund request, with explicit preconditions and result schemas.
- Tool runner: validates arguments and permissions, enforces call/time budgets,
  applies bounded retries, and uses idempotency keys for state-changing calls.

Use versioned scenario fixtures with a shared state store. All tools read and
update the same simulated world; independently hashed statuses must not create
contradictory order/payment/refund states. Reset fixture state between tests.
A successful action requires a tool-confirmed result; generated text cannot
establish that an order was cancelled or a refund was submitted.

### Agent output and provider boundary

Return `status` (`answered`, `action_completed`, `clarification_needed`,
`escalated`, or `failed`), reply, intent/catalog versions, evidence references,
action results, and a trace ID. Keep sensitive/internal trace details out of the
user-facing reply. Record tool calls, routing reasons, latency, and token/cost
usage for debugging and comparison.

Use a provider adapter for hosted and local models. Keep graph state and tool
contracts independent of provider SDKs. Begin with one provider; benchmark
additional providers after the workflow and evaluation fixtures are stable.

## Evaluation

### Part 1: Classification and taxonomy

Before unlocking reference labels, measure cluster noise/stability, teacher
schema failures and abstentions, human agreement, and student performance on
held-out human-adjudicated labels. Report accepted-ticket accuracy alongside
coverage, clarification, and escalation rates so abstention cannot conceal
errors. Include independent unknown, ambiguous, and multi-intent stress cases.

After freezing the blind pipeline:

1. Unlock development reference labels only inside an isolated evaluator to
   derive a discovered-to-reference label mapping; do not revise the pipeline.
2. Freeze the mapping, then score the untouched test partition.
3. Report homogeneity, completeness, V-measure, adjusted mutual information,
   and a contingency matrix for catalog merge/split analysis.
4. Report mapped student accuracy and macro-F1. Declare the mapping policy
   (initially many-to-one majority mapping on development), unmapped labels,
   and treatment of abstentions. Report catalog metrics alongside mapped
   scores because many-to-one mapping can hide excessive splitting.
5. Compare with the isolated supervised baseline on the same test tickets.

Score abstentions as misses in all-ticket accuracy and separately report
conditional performance on accepted predictions. Do not fit the label mapping
on test labels. Report class support and uncertainty, including both category
and intent recovery where useful.

The first gold-label evaluation is report-only. Any discovery revision informed
by its findings is a new experiment; the previously inspected test set is no
longer an untouched final benchmark.

### Part 2: Agent behavior

Use concrete fixture tickets with instantiated entities and expected outcomes,
including missing IDs, nonexistent records, ineligible actions, tool failures,
conflicting evidence, unknown intents, and requests that try to bypass policy.
Bitext responses alone cannot serve as ground truth for tool execution.

Compare on the same fixtures:

| System | Question answered |
|---|---|
| Original intent KB + generator | What does the inherited application achieve? |
| Fixed workflow + approved KB + tools | What do retrieval and tool execution add? |
| Deep agent planner + skills + tools | Does planning improve difficult cases enough to justify its cost? |

Measure task completion, correct tool selection/arguments, precondition and
confirmation compliance, state consistency, unsupported factual claims,
clarification/escalation correctness, latency, tool calls, and model cost.
Use deterministic assertions for tool/state outcomes; optional LLM judging is
supplementary for reply quality.

Run agent evaluation both with a correct intent supplied by the fixture and
with actual classifier output. This separates agent failures from upstream
classification failures. Freeze agent configuration before its final fixture
suite is evaluated.

## Reuse plan

| Existing component | Treatment in this project |
|---|---|
| FastAPI `/chat` and Streamlit | Adapt application shell and add session/outcome handling |
| DistilBERT classifier interface | Rework around versioned bundles and discovered label counts |
| Training notebook | Extract reproducible preparation, training, and evaluation commands |
| Hard-coded confidence gate | Replace with a calibrated, versioned routing policy |
| First-response KB lookup | Replace with approved runbooks and query-aware retrieval |
| Gemini generator | Adapt behind provider interface with evidence and tool results |
| Archived trained checkpoint | Optional historical comparison, subject to provenance |
| D365 discovery/catalog/teacher/review patterns | Methodology reference; new code unless separately verified available |

## Repository layout

```text
config/
  experiments/
  catalogs/
  routing/
  skills.yaml
data/
  manifest.json
  blind/
  sealed/
  gold_hidden/
  kb_source/
  metadata/
artifacts/
  classifier/
  kb/
src/
  contracts.py
  data_prep.py
  discovery/           # embedding, clustering, representatives
  catalog/             # reusable builder, schema validation, changelog
  classifier/          # teacher, train, calibrate, infer, artifact manifest
  review/              # review UI, sampling, adjudication
  policy/              # route, permissions, action preconditions
  kb/                  # candidate generation, approval, retrieval
  tools/               # entities, shared mock state, reads, actions
  agent/               # provider adapter, graph, skills, session state
  eval/                # isolated gold scoring, classifier and agent eval
app/
  main.py
  streamlit_app.py
fixtures/
  development/
  sealed_test/
tests/
README.md
```

## Implementation milestones

The executable work sequence for Part 1 is documented in the
[step-by-step classifier training plan](plans/classifier-training-plan.md).

| Milestone | Deliverable and exit evidence |
|---|---|
| 0: Reproducible foundation | Obtain actual pinned data; inspect artifact availability; freeze grouped splits and input boundaries; record configuration |
| 1: Application vertical slice | Order-status fixture with a supplied ID succeeds; a missing ID clarifies and resumes; unknown intent and tool failure produce defined outcomes; a fixture classifier result may be used initially |
| 2: Classifier training | Blind catalog, teacher/review pipeline, TF-IDF student, isolated supervised comparison, versioned inference bundle; wire real predictions into the vertical slice |
| 3: Classifier evaluation | Freeze blind artifacts; run isolated mapping and final scoring; publish metrics and error analysis without a test-driven discovery loop |
| 4: Intent actions | Approved runbooks, shared mock state, one simulated state-changing action, precondition/confirmation/idempotency checks |
| 5: Deep-agent comparison | Add planning where needed; compare original-style generation, fixed workflow, and planner using component and end-to-end evaluation |

Choose concrete development acceptance thresholds, experiment budgets, and
review allocation in Milestone 0 before inspecting results. Do not require a
particular number of discovered intents or catalog versions as a success
criterion.

## Dependencies and remaining decisions

Initial classification stack: `scikit-learn`, `sentence-transformers`,
`umap-learn`, and `hdbscan`; add `transformers`/`torch` for the DistilBERT
comparison. Application stack: FastAPI and Streamlit. Agent orchestration may
use LangGraph and, when justified, `deepagents`, plus the selected provider
SDK. Pin package versions when implementation begins.

Remaining decisions are the initial provider and budget, review capacity,
quantitative development acceptance thresholds, and the first state-changing
mock action. These do not change the two-part architecture or evaluation
boundaries defined above.
