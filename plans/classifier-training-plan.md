---
title: Step-by-Step Classifier Training Plan
type: implementation-plan
status: in-progress
updated: 2026-09-06
tags: [classifier, training, intent-discovery, evaluation]
---

# Step-by-Step Classifier Training Plan

This plan implements Part 1 of the [design](../bitext-intent-discovery-deep-agent.md).
It ends with a versioned classifier bundle ready for the deep agent. Training
has not started. Step 1 foundation modules are implemented; later-stage
commands and modules below remain implementation work.

## Approach

Method choices, course-note corrections, and expected experiment outputs are
recorded in [Classifier Methods and Experiment Rationale](classifier-methods-and-experiments.md).

Build the blind discovery branch first. Train a small TF-IDF/logistic regression
student before attempting DistilBERT. Freeze the blind branch before running
the supervised reference-label comparison, reducing accidental information
transfer between the two experiments. Use the same frozen data partitions for
both branches.

No deep-agent development is required to complete this plan, apart from
validating the classifier handoff schema.

## Initial experiment settings

These are proposed starting settings, not measured results. Record them in
`config/experiments/classifier-v1.yaml` before the first experiment. Changes
after inspecting development results must be logged with their rationale.

| Setting | Initial choice |
|---|---|
| Seed | 42 for partitioning; 42, 43, 44 for clustering stability |
| Partitions | Grouped, label-independent 70% train / 15% development / 15% test |
| Embeddings | Pinned `sentence-transformers/all-MiniLM-L6-v2` revision |
| Catalog revision budget | At most three rounds |
| First student | Word/character TF-IDF + logistic regression |
| Lightweight comparisons | Same TF-IDF + linear SVM; frozen sentence embeddings + logistic regression |
| Review feedback budget | One initial disagreement-driven training review/retraining round |
| Transformer candidate | `distilbert-base-uncased`, only after baseline review |
| Teacher | One selected provider/model, pinned where possible; record exact model ID |
| Teacher pilot | 100 training tickets before full batch |
| Training review | Initial 100-ticket pilot, then targeted expansion |
| Human development evaluation | Target 20 examples per discovered intent, plus ambiguity/unknown cases; report shortages |
| Deployment decision | Development target: at least 95% accepted-ticket accuracy at 80% coverage |
| Unknown stress target | At least 90% of curated out-of-domain cases rejected from `handle` |

The routing targets are provisional experiment objectives, not production
service guarantees. Report uncertainty and all-ticket performance. If no
threshold meets both accuracy and coverage, record the failure; do not lower
thresholds silently. Report unknown-stress results separately from Bitext
coverage. Provider choice and batch budget are settled before paid labeling.

## Step 1 — Set up a reproducible training workspace

- [x] Create `src/`, `config/experiments/`, `tests/`, `reports/`, and artifact/data directories from the design.
- [x] Inspect Python, CPU, memory, and available GPU/MPS support; record the environment.
- [x] Create an isolated environment and pinned dependency specification.
- [x] Implement shared schemas for tickets, catalog entries, annotations, classifier output, and artifact manifests.
- [x] Configure exclusion of datasets, credentials, and large checkpoints from source control.
- [x] Establish run IDs, structured logs, and configuration snapshots.

**Deliverables:** environment specification, initial experiment configuration,
shared contracts, and data-access policy.

**Done when:** imports and schema checks pass; a sample classifier result
validates; the setup can be reproduced from the recorded instructions.

## Step 2 — Obtain and validate the real dataset

- [x] Obtain the actual Bitext data from a pinned source revision. Do not parse the archive's Git LFS pointer as a CSV.
- [x] Record source, license, revision, file checksum, schema, and row count.
- [x] Validate required fields and record missing/empty instructions, duplicates, and placeholder usage.
- [x] Keep original instruction text intact; define a separate normalization used for duplicate grouping.
- [x] Have trusted preparation validate reference-label counts privately; do not expose taxonomy details to discovery jobs.

**Deliverables:** immutable source snapshot and preparation manifest. Reports
visible to the blind branch exclude reference labels and their distributions.

**Done when:** the real dataset is readable, required fields are validated, and
every row has a reproducible opaque ID independent of source ordering.

## Step 3 — Freeze partitions and enforce data boundaries

- [x] Group exact duplicates and conservative near-duplicate variants without intent/category labels.
- [x] Freeze approximately 70/15/15 group partitions and record actual sizes.
- [x] Reserve disjoint development subsets for model selection and final confidence calibration; use group boundaries.
- [x] Separate train responses for later KB work, flags for audit metadata, and all reference labels for isolated evaluation.
- [x] Seal test instructions and responses; no test inspection during discovery or training.
- [x] Implement allowlisted loaders and run blind jobs in an input-restricted environment with only permitted artifacts available.
- [x] Verify that blind jobs cannot read gold labels, responses, or sealed test files.

**Deliverables:** split/group manifests, blind ticket files, sealed artifacts,
and leakage audit.

**Done when:** IDs/groups do not overlap across partitions, input-boundary
checks pass, and split files have immutable hashes. Document residual template
similarity instead of claiming that grouping eliminates every leakage risk.

## Step 4 — Discover candidate intents from training tickets (complete with deferred sweep)

- [x] Embed training instructions and cache vectors with model/configuration hashes.
- [x] Compare no reduction with UMAP at 15 and 30 dimensions, fitted only on training inputs. Deferred as an optional sensitivity experiment after the accepted draft; the full no-reduction baseline is retained.
- [x] Run full-corpus HDBSCAN baseline with `min_cluster_size=30`, `min_samples=10`; record metrics. Variants 60/120 are deferred optional sensitivity runs.
- [x] Inspect noise fraction, cluster sizes, representatives, and boundary utterances; full baseline review artifacts are saved.
- [x] Name and consolidate candidate clusters using machine proposals plus human review; the accepted domain/action draft is `bitext-v0.3-draft`.
- [x] Review the noise pool and keep it outside the catalog; do not automatically label it as an intent.

**Deliverables:** cached embeddings, cluster assignments, baseline report,
review artifacts, merge map, and catalog `bitext-v0.3-draft`.

**Done when:** all training tickets have an analysis assignment or noise status,
each catalog candidate has supporting examples, and no choice used gold labels
or the known target number of intents. The grid is a starting experiment, not a
claim that these parameters will yield the best taxonomy.

## Step 4 review gate

Use the [discovery review worksheet](../reviews/discovery-review-v1.md) before drafting the catalog. The machine proposal and evidence-only packet are available under `reviews/`, but every proposal requires human adjudication. Human review should cover large, small, boundary-heavy, and noise groups.

## Step 5 — Review and freeze the catalog (complete)

- [x] Write stable IDs, domains, definitions, inclusion/exclusion rules, and examples.
- [x] Examine overlaps, broad clusters, and ambiguous cases with a reviewer.
- [x] Use the reviewed merge map to consolidate duplicate intents.
- [x] Validate schema, unique IDs, and explicit abstention behavior.
- [x] Freeze catalog `v1` before generating the final training-label batch.

**Deliverables:** validated catalog, changelog, and unresolved ambiguity report.

**Done when:** every intent has a usable definition, overlaps have a documented
routing rule, and remaining problems are recorded. Do not force the discovered
catalog to contain 27 intents. If a later change alters label semantics, version
the catalog and regenerate affected annotations before retraining.

## Step 6 — Pilot teacher classification and human review

- [x] Label 100 training tickets covering clusters, noise, and boundary cases against the frozen catalog.
- [x] Require valid intent IDs or abstention; log invalid outputs and bounded retry results.
- [x] Store source ticket IDs, teacher/model/prompt/catalog versions, and annotation provenance.
- [x] Have a human review the pilot without reference labels; adjudicate disagreements.
- [x] Estimate full-batch cost from actual pilot token usage and record the batch budget.
- [x] Fix schema/prompt defects before the full batch; retain the pilot history.

**Deliverables:** pilot annotations, review decisions, error report, and batch
cost estimate.

**Done when:** no invalid labels can enter the training set, failures have
explicit states, and the review establishes whether labeling is usable. Human
adjudication must be performed by a human; additional LLM checking is recorded
as model review, not human ground truth.

## Step 7 — Build training labels and independent evaluation labels

- [ ] Generate local-first labels for the training partition using cached
  embeddings and confidence/margin thresholds.
- [ ] Send only uncertain, ambiguous, unsupported, or high-risk tickets to the
  teacher in resumable, cached batches; retain the interrupted full-teacher run
  as historical evidence.
- [ ] Expand training review around ambiguity, rare discovered intents, and teacher failures.
- [ ] Keep teacher-only and human-adjudicated annotations distinguishable.
- [ ] Quarantine unresolved labels; preserve counts instead of silently dropping difficult examples.
- [ ] Human-label disjoint development selection/calibration samples against the catalog.
- [ ] Build separate development stress examples for unknown, ambiguous, and multi-intent requests.
- [ ] Define the held-out test human-review protocol now; obtain those annotations only after the blind pipeline freezes.

**Deliverables:** versioned training annotations, independent development gold,
stress cases, and sampling/coverage report.

**Done when:** evaluation IDs never appear in training annotations, every label
has provenance, and unsupported or undersampled intents are explicitly listed.
Without independent human labels, report teacher agreement only; do not claim
measured human-label accuracy or validated calibration.

The default annotation policy is local-first. The API teacher is a selective
reviewer for difficult cases, not the default annotator for every ticket. The
labeling report must include local coverage, teacher-review coverage, abstention
rate, API requests, tokens, latency, and estimated cost.

## Step 8 — Train and compare lightweight students

- [ ] Fit word and character TF-IDF features only on training instructions.
- [ ] Train logistic regression using resolved discovered labels; do not create an unknown class merely from clustering noise.
- [ ] Use a small predefined regularization search, initially `C = 0.1, 1, 10`, evaluated on human development-selection labels.
- [ ] Compare word-only versus word/character features with a bounded minimum-document-frequency and regularization experiment; fit transforms only on training inputs or within training folds.
- [ ] Train a linear SVM on the same TF-IDF features and a logistic regression classifier on frozen original sentence embeddings, using the same annotation version.
- [ ] Report macro-F1, per-intent support/errors, all-ticket accuracy, inference latency, model size, and memory usage.
- [ ] Inspect disagreements with the teacher and document pseudo-label versus adjudicated-label contributions.
- [ ] Produce a ranked training review queue, preferably using group-aware out-of-fold student predictions; majority agreement does not replace human adjudication.
- [ ] Complete one bounded human training-review round and compare retrained candidates on the same revised labels. Development examples remain outside training.
- [ ] Save vectorizer, model, catalog mapping, preprocessing configuration, and run manifest.

**Deliverables:** initial `student-tfidf-v1` bundle, SVM and frozen-embedding
comparison bundles, feature/model comparison report, and versioned review queue
with adjudication outcomes.

**Done when:** the saved bundle reloads and reproduces predictions, input
preprocessing is shared with inference, and remaining failure cases are known.
This is the first usable trained model.

## Step 9 — Train DistilBERT if the comparison is worthwhile

- [ ] Use the same training labels and partitions as the lightweight student.
- [ ] Derive output label count from the frozen catalog.
- [ ] Measure training-token lengths before choosing the maximum sequence length; report truncation rate.
- [ ] Start from pretrained weights with AdamW and the pretrained dropout settings; record weight decay and optimizer configuration.
- [ ] Start with learning rate `2e-5`, effective batch size 16, and at most three epochs; adapt physical batch size to available memory.
- [ ] Select checkpoints using development-selection macro-F1, never test scores.
- [ ] Compare classification quality, latency, memory, and training cost with the lightweight candidates.

**Deliverables:** transformer bundle and side-by-side development comparison,
or a documented decision to retain a lightweight candidate.

**Done when:** select the model based on recorded evidence. DistilBERT is not a
mandatory upgrade. If this step is skipped, do not imply transformer training
was completed.

## Step 10 — Calibrate routing and freeze the blind pipeline

- [ ] Fit confidence calibration for the selected candidate using only the reserved human calibration subset, disjoint from classifier fitting and architecture selection. Raw SVM margins are not probabilities.
- [ ] Choose routing thresholds on that subset and development stress cases.
- [ ] Report accuracy versus coverage and unknown rejection, with uncertainty and sample sizes.
- [ ] Define `handle`, `clarify`, and `escalate` reasons, including null intent for abstention.
- [ ] Freeze classifier, catalog, preprocessing, calibration, routing, and all run manifests.

**Deliverables:** release-candidate classifier bundle and routing report.

**Done when:** the bundle passes compatibility checks and either meets the
predeclared development objectives or is clearly marked experimental with its
unmet objectives. No final-test result has influenced model selection.

## Step 11 — Run isolated supervised comparison and final scoring

- [ ] After blind freeze, train isolated supervised counterparts for the model families evaluated in the blind branch, using reference training labels, identical partitions, and comparable tuning budgets.
- [ ] Keep comparison artifacts separate; never feed them back into the frozen blind branch.
- [ ] Derive a discovered-to-reference majority mapping from development reference labels in the evaluator, then freeze it.
- [ ] Generate final predictions on sealed test tickets once configuration is frozen.
- [ ] Score taxonomy homogeneity/completeness/V-measure/AMI and mapped accuracy/macro-F1; show merge/split contingency tables.
- [ ] Count abstentions as misses for all-ticket accuracy and separately report accepted-ticket performance and coverage.
- [ ] Report unmapped intents and class support; never fit mappings using test labels.
- [ ] Complete independent blind human test annotation as specified in Step 7 for discovered-catalog evaluation.
- [ ] Compare with the supervised branch and publish limitations, uncertainty, and errors.

**Deliverables:** final classification report, reproducible scoring inputs,
and an explicit record of when test/gold data were unlocked.

**Done when:** every score identifies its labels, partition, denominator, and
model version. Further model changes informed by these findings require a new
experiment; this test set cannot be described as untouched again.

## Step 12 — Package the handoff to the deep agent

- [ ] Export model, tokenizer/vectorizer, label mapping, catalog, calibration, routing policy, and manifest as one compatible bundle.
- [ ] Implement inference returning the design's classifier contract.
- [ ] Verify normal prediction, abstention, empty/invalid input handling, bundle reload, and incompatible-version rejection.
- [ ] Add sample handoffs for a recognized intent, an ambiguous query, and an unknown query.
- [ ] Document inference invocation and measured runtime requirements.

**Deliverables:** reloadable classifier release, inference entry point, and
handoff examples ready for Part 2.

**Done when:** the agent can consume a versioned result without depending on
training code or reading reference labels. Action permissions remain the
agent/policy layer's responsibility.

## Execution order and current status

`1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → optional 9 → 10 → 11 → 12`

Dataset acquisition and source verification in Step 2 completed on 2026-09-06; see
`data/manifests/bitext-source.json`. Step 1 completed on 2026-09-06; see [verification report](../reports/step-1-foundation.md).
Step 2 completed on 2026-09-06; see [data preparation report](../reports/step-2-data-preparation.md).
Step 3 completed on 2026-09-06; see [split/access report](../reports/step-3-splits-and-access.md).
Step 4 completed on 2026-09-12 with the accepted `bitext-v0.3-draft`; see [discovery status](../reports/step-4-discovery.md) and the catalog provenance. Step 5 completed on 2026-09-12 with frozen catalog `bitext-v1`; see the catalog provenance. The UMAP and alternative HDBSCAN grids remain deferred sensitivity experiments. The next task is Step 6: teacher pilot labeling against the frozen catalog. Package installation
uses the environment's normal permission flow.
Human catalog review and adjudication are explicit collaboration points;
independent engineering work can continue while review is pending.
