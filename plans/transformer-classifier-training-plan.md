---
title: DistilBERT Transformer Classifier Training Plan
type: plan
status: planned
updated: 2026-09-27
tags: [classifier, transformer, distilbert, bitext-v2, evaluation]
---

# DistilBERT Transformer Classifier Training Plan

## 1. Purpose

Train a transformer-based intent classifier using the frozen `bitext-v2`
catalog, then compare it fairly with the current TF-IDF classifiers:

- Logistic regression baseline.
- Linear SVM prototype model.
- DistilBERT transformer candidate.

The transformer will be selected only if it improves intent quality enough to
justify its additional training time, memory use, and inference latency.

## 2. Scope and data boundary

Use the existing prepared project data and do not rebuild the catalog during
transformer training.

Required inputs:

- Frozen `bitext-v2` catalog and label mapping.
- Reviewed training labels.
- Existing grouped train/development/calibration/evaluation partitions.
- The same text preprocessing policy used by the baseline models.
- The current SVM evaluation report for comparison.

The final evaluation set must remain unchanged and must not be used for model
selection, hyperparameter tuning, or threshold tuning.

## 3. Target architecture

```text
Ticket text
    → DistilBERT tokenizer
    → DistilBERT encoder
    → classification head
    → intent logits
    → calibrated confidence and abstention policy
    → classifier-to-agent handoff
```

Initial model: `distilbert-base-uncased` with a classification head whose output
size is derived from the frozen `bitext-v2` catalog.

## 4. Training phases

### Phase 1 — Environment and reproducibility

1. Confirm the project Python environment and install compatible versions of
   PyTorch, Transformers, Datasets, scikit-learn, and evaluation utilities.
2. Record Python, PyTorch, Transformers, and hardware versions.
3. Add a fixed random seed for Python, NumPy, PyTorch, and data shuffling.
4. Detect available CPU, Apple Silicon, CUDA, or other accelerator support.
5. Create a run manifest containing model name, catalog version, data hashes,
   seed, hyperparameters, and hardware information.

**Expected output:** reproducible environment report and transformer run
configuration.

### Phase 2 — Prepare transformer datasets

1. Load only resolved training labels from the approved annotation artifact.
2. Map every intent ID to the frozen `bitext-v2` integer label index.
3. Reject unknown, duplicate, malformed, or unmapped labels.
4. Keep development, calibration, and final evaluation examples separate.
5. Measure instruction length distribution and truncation risk.
6. Tokenize with the DistilBERT tokenizer.
7. Start with a maximum sequence length of 128; increase only if the measured
   truncation rate justifies the memory cost.
8. Preserve ticket IDs for error analysis without exposing hidden labels to the
   training loop.

**Expected outputs:** tokenized datasets, label mapping, length report, and
input validation report.

### Phase 3 — Establish the comparison baseline

Before fine-tuning, record the current SVM results using the same partitions and
label version:

- Accuracy.
- Macro-F1.
- Per-intent precision and recall.
- Confusion matrix.
- Accepted coverage and abstention rate.
- CPU inference latency and model size.

This prevents the transformer from being evaluated against an outdated or
different baseline.

### Phase 4 — Fine-tune DistilBERT

Start with a small, bounded configuration:

```yaml
model_name: distilbert-base-uncased
max_length: 128
learning_rate: 2.0e-5
weight_decay: 0.01
effective_batch_size: 16
epochs: 3
warmup_ratio: 0.1
evaluation_strategy: epoch
save_strategy: epoch
early_stopping_metric: eval_macro_f1
```

Implementation requirements:

1. Use AdamW through the Transformers training stack.
2. Select checkpoints using development macro-F1 only.
3. Use class-weighting only if the training distribution and development errors
   demonstrate a meaningful imbalance problem; document any change.
4. Record training loss, development loss, accuracy, macro-F1, learning rate,
   epoch time, and peak memory.
5. Stop early when development macro-F1 does not improve.
6. Save the tokenizer, model configuration, label mapping, and run manifest.

**Expected output:** reloadable `distilbert-bitext-v2-candidate` checkpoint and
training history.

### Phase 5 — Bounded hyperparameter comparison

Run only a small predefined comparison to control time and compute cost:

- Learning rate: `1e-5`, `2e-5`.
- Maximum length: `128`, `256` only if truncation requires it.
- Epochs: `2`, `3`.

Keep the seed, partitions, label mapping, and evaluation rules consistent. Do
not tune against the final test set.

**Expected output:** comparison table with quality, latency, memory, and cost
for each candidate.

### Phase 6 — Confidence calibration and abstention

1. Generate logits on the reserved calibration subset.
2. Convert logits to confidence scores using a documented calibration method.
3. Select `handle`, `clarify`, and `escalate` thresholds using calibration data
   and predefined development stress cases.
4. Report accuracy-versus-coverage curves.
5. Test ambiguous, incomplete, paraphrased, and unsupported requests.
6. Preserve the explicit `unknown.require_more_info` outcome.

Raw softmax confidence must not be described as a calibrated probability unless
calibration evidence supports that claim.

### Phase 7 — Fair final comparison

Compare the selected DistilBERT candidate with the frozen SVM and logistic
regression models using the same evaluation inputs:

| Measure | Logistic regression | Linear SVM | DistilBERT |
|---|---:|---:|---:|
| Accuracy | record | record | record |
| Macro-F1 | record | record | record |
| Accepted coverage | record | record | record |
| Abstention quality | record | record | record |
| Median latency | record | record | record |
| Model size | record | record | record |
| Peak memory | record | record | record |
| Training cost/time | record | record | record |

Also inspect per-intent gains and regressions, especially the intents that were
merged or previously confused in the v2 catalog.

### Phase 8 — Select and package the model

Use development evidence to choose one release candidate:

- Select DistilBERT if its quality improvement is meaningful and operational
  costs remain acceptable.
- Retain SVM if DistilBERT provides only a small improvement or has materially
  worse latency, memory, or cost.

Package the selected candidate with:

- Model and tokenizer or vectorizer.
- Catalog and label mapping.
- Preprocessing configuration.
- Calibration parameters.
- Routing policy.
- Training and evaluation manifest.
- Model card describing data, limitations, and known errors.

The bundle must satisfy the existing classifier-to-agent contract and reload
without importing training code.

## 5. Evaluation and acceptance criteria

The transformer experiment is complete when:

- The checkpoint reloads and produces deterministic predictions under the
  recorded configuration.
- No final-test result influenced model selection or threshold tuning.
- Accuracy, macro-F1, per-intent results, coverage, and abstention are reported.
- Latency, model size, memory, and training cost are measured.
- At least one error analysis compares DistilBERT and SVM disagreements.
- The selected model is justified using quality and operational evidence.
- A new version is created if labels, catalog, preprocessing, or thresholds
  change.

## 6. Risks and controls

- **Overfitting:** use grouped splits, early stopping, and a separate final set.
- **Data leakage:** preserve duplicate checks and immutable partition manifests.
- **Uncalibrated confidence:** calibrate on a reserved subset.
- **High compute cost:** use a bounded search and start with DistilBERT.
- **Small quality gain:** retain the SVM if it offers the better quality-cost
  trade-off.
- **Label noise:** track teacher abstentions and review disagreement cases.

## 7. Implementation deliverables

1. `configs/transformer-training.yaml`
2. `src/classifier/train_transformer.py`
3. `src/classifier/evaluate_transformer.py`
4. `reports/transformer-training-report.md`
5. `reports/transformer-vs-svm-comparison.csv`
6. `artifacts/classifier-transformer-v1/`
7. Updated classifier inference adapter and model card

## 8. Execution order

```text
Environment
  → dataset preparation
  → SVM baseline capture
  → DistilBERT fine-tuning
  → bounded comparison
  → calibration and abstention
  → fair evaluation
  → model selection
  → package and integrate with deep agent
```
