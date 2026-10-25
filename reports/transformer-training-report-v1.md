---
title: DistilBERT Transformer Training Report v1
type: experiment-report
status: candidate
updated: 2026-09-27
---

# DistilBERT training report

## Run configuration

- Model: `distilbert-base-uncased`
- Catalog: `bitext-v2` with 33 labels
- Training rows: 7,510
- Development rows: 184 (`dev_selection`)
- Calibration rows: 180 prepared separately
- Maximum length: 128 tokens
- Batch size: 8
- Epochs: 3
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Seed: 42
- Training device: Apple MPS
- Training time: approximately 17.9 minutes

## Training results

| Epoch | Development accuracy | Development macro-F1 |
|---:|---:|---:|
| 1 | 0.8641 | 0.8368 |
| 2 | 0.9348 | 0.9097 |
| 3 | 0.9457 | 0.9263 |

The epoch-3 checkpoint is saved in
`artifacts/classifier-transformer-v1/`.

## Calibration-subset result

On the separate 180-row calibration subset, the saved checkpoint achieved:

- Accuracy: 0.9667
- Macro-F1: 0.9334

This result is for model comparison and future confidence calibration. It was
not used to change the model weights.

## Comparison notes

The existing v2 SVM report used a different 447-row teacher-reviewed evaluation
sample and therefore must not be compared directly with the 184-row DistilBERT
development result. A fair model comparison still requires running SVM and
DistilBERT on the same frozen input set and then measuring latency, memory, and
coverage under comparable abstention policies.

## Fair same-input comparison

Both models were run on the same prepared inputs:

| Split | Model | Accuracy | Macro-F1 | Mean inference time | Disagreements |
|---|---|---:|---:|---:|---:|
| Development (184) | Linear SVM | 0.9348 | 0.8930 | 0.000033 s | — |
| Development (184) | DistilBERT | 0.9457 | 0.9263 | 0.01718 s | 7 |
| Calibration (180) | Linear SVM | 0.9556 | 0.9187 | 0.000013 s | — |
| Calibration (180) | DistilBERT | 0.9667 | 0.9334 | 0.01694 s | 7 |

The transformer improves the current quality metrics by approximately 1.1
percentage points of accuracy on both splits and improves macro-F1 by about
0.015. The SVM is approximately three orders of magnitude faster for this
small-batch CPU comparison and its model artifacts are much smaller (about
2.5 MB versus about 256 MB for DistilBERT).

## Current decision

DistilBERT is a successful trained candidate and shows a modest quality gain on
the same inputs. The SVM remains the efficiency winner. A final choice still
requires confidence calibration, coverage/abstention measurement, and a
decision about whether the quality gain justifies the latency and model-size
cost.

## Confidence and coverage sweep

On the 180-row calibration subset, the raw DistilBERT max-softmax score gave:

- Threshold `0.50`: 100% coverage, 96.67% accepted accuracy.
- Threshold `0.90`: 97.22% coverage, 96.57% accepted accuracy.
- Threshold `0.95`: 95.00% coverage, 97.08% accepted accuracy.
- Threshold `0.99`: 73.89% coverage, 98.50% accepted accuracy.

These scores are a threshold sweep, not calibrated probabilities. The current
candidate threshold for further testing is `0.95`, because it removes a small
number of uncertain cases while retaining high coverage. It must be validated
on a disjoint set before being frozen.

The existing SVM margin threshold `0.20` accepted 97.22% of the same subset at
96.57% accepted accuracy. DistilBERT therefore offers a similar selective
coverage profile with a modest quality advantage, but at substantially higher
latency and model size.

## Next actions

1. **Completed:** generated SVM and DistilBERT predictions on identical
   development and calibration inputs.
2. **In progress:** calibrate DistilBERT confidence and define final
   handle/clarify thresholds.
3. **Partially completed:** measured latency and model size; complete peak-memory
   and end-to-end cost measurements.
4. Inspect DistilBERT/SVM disagreements and difficult v2 intents.
5. Select and package the final candidate only after the fair comparison.

## Disagreement and resource analysis

The seven calibration disagreements identify useful boundary cases:

- SVM confused account creation/deletion/change-plan requests; DistilBERT
  resolved those three examples correctly.
- DistilBERT confused one short contact request with delivery-address change.
- Both models struggled with the short request `check status of refund`, with
  SVM choosing order status and DistilBERT choosing refund policy instead of
  refund status.
- One account-update request was classified by DistilBERT as change plan.

The examples show that short, underspecified tickets remain the main risk. They
should be included in future targeted review and abstention testing.

Artifact sizes are approximately 2.5 MB for the SVM bundle and 256 MB for the
DistilBERT bundle. Mean inference time in the same-input benchmark was about
0.000013–0.000033 seconds for SVM and 0.0169–0.0172 seconds for DistilBERT.
Peak memory was not measured as a separate process yet; the artifact and
latency differences already show a substantial operational cost for the
transformer.
