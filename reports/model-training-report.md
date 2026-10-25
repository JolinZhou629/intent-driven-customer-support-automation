---
title: Intent Classifier Model Training Report
type: experiment-report
status: candidate-selection
updated: 2026-09-27
tags: [classifier, teacher-student, svm, logistic-regression, distilbert]
---

# Intent Classifier Model Training Report

## 1. Executive summary

This project trained and compared three student classifiers for the reviewed
`bitext-v2` intent catalog:

1. TF-IDF with logistic regression.
2. TF-IDF with linear SVM.
3. Fine-tuned DistilBERT.

The teacher model was used selectively to review uncertain examples and improve
the training labels. It was not used as the runtime classifier.

DistilBERT produced the highest quality on the current same-input comparison,
while linear SVM was substantially faster and smaller. The recommended current
decision is to keep SVM as the efficient prototype model and retain DistilBERT
as the higher-quality candidate until targeted ambiguous examples and final
threshold validation are completed.

## 2. Training objective

The classifier must identify the user's support intent and return a result that
the deep agent can safely use for routing. The model output includes:

- Intent ID and domain.
- Confidence or decision margin.
- Model and catalog versions.
- An explicit clarification/abstention path for uncertain requests.

The classifier does not perform external actions. Skills, policies, and tools in
the deep-agent layer control what happens after classification.

## 3. Data and catalog

- Source: public Bitext customer-support dataset.
- Prepared training examples: 7,510.
- Development-selection examples: 184.
- Development-calibration examples: 180.
- Final catalog: `bitext-v2` with 33 reviewed intents.
- Data preparation: cleaning, duplicate checks, grouped partitions, catalog
  review, and documented intent merges.

The evaluation examples are reviewed labels used for academic model comparison.
They are not independent production ground truth.

## 4. Teacher–student training process

```text
Public tickets
    → clean text and detect duplicates
    → grouped train/dev/calibration partitions
    → embeddings and candidate intent clusters
    → reviewed bitext-v2 catalog
    → local-first high-confidence labels
    → teacher review of uncertain cases
    → resolved training labels
    → three student classifiers
    → calibration and same-input comparison
    → final model selection
```

### Teacher role

The teacher model, OpenAI `gpt-5-mini`, reviewed selected uncertain, ambiguous,
or low-margin examples using a schema-constrained output. Teacher results stored
the predicted intent, status, and provenance. Abstentions were retained instead
of being forced into an intent.

### Student role

The student models learned from the resolved labels and provide fast,
repeatable runtime predictions. This reduces API cost and latency and makes the
classifier independent of the teacher service during ticket handling.

## 5. Student model 1 — TF-IDF logistic regression

### Method

- Word and character TF-IDF features.
- Multiclass logistic regression.
- Efficient CPU inference and small model artifact.

### Role

Logistic regression is the classical probabilistic baseline. It provides a
transparent reference for measuring whether SVM or DistilBERT adds value.

## 6. Student model 2 — TF-IDF linear SVM

### Method

- The same TF-IDF representation as logistic regression.
- Linear SVM decision boundaries for multiclass classification.
- Decision margin used as an uncertainty signal.

### Role

SVM is the current efficient prototype candidate. It has very low inference
latency and a small artifact, which is useful for a local or high-throughput
ticketing service.

## 7. Student model 3 — DistilBERT

### Method

- Base model: `distilbert-base-uncased`.
- Fine-tuned classification head with 33 output labels.
- Maximum sequence length: 128.
- Learning rate: `2e-5`.
- Batch size: 8.
- Epochs: 3.
- Weight decay: `0.01`.
- Seed: 42.
- Training device: Apple MPS.

### Role

DistilBERT is the semantic transformer candidate. It can model context and word
order more directly than TF-IDF, which may help with overlapping or paraphrased
intents.

## 8. Same-input model comparison

Visual comparison: [student model comparison graph](model-comparison-v1.svg).

All three models were evaluated on the same prepared development and calibration
inputs after mapping their intent IDs to the same `bitext-v2` label indices.

| Split | Logistic regression | Linear SVM | DistilBERT |
|---|---:|---:|---:|
| Development accuracy | 90.76% | 93.48% | **94.57%** |
| Development macro-F1 | 0.885 | 0.893 | **0.926** |
| Calibration accuracy | 93.33% | 95.56% | **96.67%** |
| Calibration macro-F1 | 0.898 | 0.919 | **0.933** |

### Operational comparison

| Model | Approximate artifact size | Mean inference time | Main strength |
|---|---:|---:|---|
| Logistic regression | small TF-IDF artifact | tens of microseconds | Simple baseline |
| Linear SVM | small TF-IDF artifact | 13–33 microseconds | Fast and efficient |
| DistilBERT | approximately 256 MB | 16.9–17.2 milliseconds | Best current quality |

The exact latency depends on batch size and hardware. Peak memory and full
production cost measurement remain future work.

## ROC-AUC comparison

Visual comparison: [multiclass ROC-AUC graph](model-auc-v1.svg).

ROC curve figure: [micro-average ROC curve comparison](model-roc-curves-v1.svg).
The curve uses the same style as a binary ROC chart, with each intent treated
as one-vs-rest and all intent decisions flattened into a micro-average.

Because each small development split contains only 30 of the 33 intents, the
macro one-vs-rest ROC-AUC is reported over the 30 classes present in that split.
This avoids undefined AUC values for classes with no positive examples.

| Split | Logistic regression | Linear SVM | DistilBERT |
|---|---:|---:|---:|
| Development | 0.9863 | 0.9843 | 0.9799 |
| Calibration | 0.9975 | **0.9989** | 0.9958 |

ROC-AUC measures ranking quality across thresholds and does not replace the
accuracy, macro-F1, coverage, or abstention measures used for routing. The
high AUC values alongside lower classification accuracy show why multiple
metrics are needed: a model can rank the correct class well while still making
errors at its selected decision threshold.

## 9. Error analysis

The SVM and DistilBERT predictions disagreed on seven calibration examples.
Important patterns were:

- SVM confused account creation, deletion, and plan-change requests in several
  examples; DistilBERT resolved those examples correctly.
- DistilBERT confused one short contact request with delivery-address change.
- Both models struggled with the short request `check status of refund`, with
  SVM choosing order status and DistilBERT choosing refund policy.
- One short account-update request was classified by DistilBERT as plan change.

These errors indicate that better boundary examples and clarification handling
may provide more benefit than changing hyperparameters alone.

## 10. Selection decision

### Current recommendation: linear SVM for the prototype

SVM is recommended as the current operational prototype because:

- Its quality is close to DistilBERT on the current reviewed samples.
- Its inference latency is dramatically lower.
- Its model artifact is much smaller.
- It is easier to run locally and integrate into the Streamlit demonstration.
- Its decision margin supports selective clarification.

### DistilBERT remains an important candidate

DistilBERT should remain packaged and documented because it currently provides
the best measured accuracy and macro-F1. It may become the final model if:

- Targeted ambiguous examples improve its difficult intent boundaries.
- Its quality advantage remains after final independent validation.
- The project accepts its additional latency and model-size cost.

The decision is therefore based on the quality-versus-cost trade-off, rather
than assuming that the largest model is automatically the best model.

## 11. Next model-improvement plan

1. Add reviewed examples for refund status versus refund policy, account update
   versus plan change, and other short ambiguous requests.
2. Improve catalog inclusion and exclusion rules for those boundaries.
3. Retrain DistilBERT under a new model version.
4. Run a bounded parameter sweep:
   - Learning rates `1e-5` and `2e-5`.
   - Maximum lengths `128` and `256`.
   - Epochs `2` and `3`.
5. Re-run all three models on the same frozen inputs.
6. Calibrate confidence and validate coverage/abstention on a disjoint set.
7. Measure peak memory, latency, and estimated cost.
8. Select the final model and package it behind the classifier-to-agent contract.

## 12. Limitations

- Current evaluation labels are reviewed labels, not independent production
  ground truth.
- The calibration subsets are small and contain uneven intent support.
- The DistilBERT max-softmax sweep is not yet a calibrated probability model.
- The public dataset is synthetic and English-only.
- No production ticketing or operational system is connected.
