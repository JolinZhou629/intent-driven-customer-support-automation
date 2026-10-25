# Intent-Driven Customer Support Automation with Deep Agent

This project is a local prototype for automated customer-support ticket handling. It combines a machine-learning intent classifier with a controlled, stateful deep-agent workflow.

The classifier identifies the customer’s intent, domain, and confidence or margin. The agent uses that result to select an approved skill, retrieve project-owned knowledge, call a simulated typed tool, validate the result, and prepare a grounded response. Unclear requests are sent to clarification or escalation instead of being forced into an unreliable intent.

All data and integrations in this repository are public, synthetic, or created for the project. The system does not connect to production ticketing, payment, delivery, or customer-account systems.

## What is included

- A reviewed `bitext-v2` catalog with 33 support intents.
- Three student classifiers: Logistic Regression, Linear SVM, and DistilBERT.
- Selective GPT-5-mini review for uncertain catalog examples during training.
- Confidence-aware `handle`, `clarify`, and `escalate` routing.
- Versioned agent skills, typed simulated tools, fixture data, and project-owned knowledge-base articles.
- Session persistence for clarification and workflow resume.
- A Streamlit interface showing classifier results, agent execution, responses, evidence, and audit details.
- Automated tests and vertical-slice workflow evaluation.

## System flow

```text
Support ticket
  -> preprocessing
  -> intent classifier
  -> confidence and margin policy
  -> handle, clarify, or escalate
  -> versioned agent skill
  -> approved knowledge or simulated tool
  -> validation and evidence
  -> grounded response
  -> saved state and audit event
```

The classifier answers what the request is about. The agent decides how the system may handle it.

## Dataset

The project uses the public [Bitext customer-support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). The source contains approximately 27,000 synthetic customer-support examples. After cleaning and duplicate or near-duplicate removal, the working corpus contains approximately 18,000 tickets.

The project does not include company ticket data, customer information, internal code, credentials, private prompts, proprietary knowledge-base content, or production endpoints.

## Model training

The reviewed catalog was frozen before student training. The resolved labels were used to train all three models.

### Classical students

- 7,510 resolved training tickets.
- 33 `bitext-v2` labels.
- Word and character TF-IDF with unigrams and bigrams.
- `sublinear_tf=True`.
- Balanced class weights.
- `C=1.0` regularization.

### DistilBERT student

- Base model: `distilbert-base-uncased`.
- Maximum sequence length: 128 tokens.
- Learning rate: `2e-5`.
- Batch size: 8.
- Epochs: 3.
- Weight decay: `0.01`.
- Seed: 42.
- Training device: Apple MPS.

GPT-5-mini was used selectively to review uncertain examples against the catalog. It is a training-time review tool, not the production classifier.

## Evaluation results

All three models were evaluated on the same prepared development and calibration inputs.

| Split | Logistic Regression | Linear SVM | DistilBERT |
|---|---:|---:|---:|
| Development accuracy | 90.76% | 93.48% | **94.57%** |
| Development macro-F1 | 0.885 | 0.893 | **0.926** |
| Calibration accuracy | 93.33% | 95.56% | **96.67%** |
| Calibration macro-F1 | 0.898 | 0.919 | **0.933** |

DistilBERT produced the strongest classification quality. Linear SVM is the current prototype runtime choice because its package is approximately 2.5 MB and its measured inference latency is approximately 13–33 microseconds per ticket. DistilBERT is approximately 256 MB and measured approximately 16.9–17.2 milliseconds per ticket.

Functional verification also passed:

- 48 of 48 automated tests passed.
- 9 of 9 vertical-slice scenarios passed.
- Scenarios covered accepted workflows, clarification, resume, escalation, and low-margin routing.

These are local prototype results. Agent latency, peak memory, token usage, and operating cost have not been measured as production metrics because the integrations are simulated.

## Run locally

The tested development environment is macOS ARM64 with Python 3.14.6.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m src.environment
.venv/bin/python -m src.verify_setup
.venv/bin/python -m pytest -q
```

## Start the Streamlit demonstration

```bash
.venv/bin/streamlit run src/streamlit_app.py
```

Open `http://localhost:8501` in a browser. Suggested demo tickets:

```text
Where is my order ORD-1001?
Please retrieve invoice INV-1001.
How can I apply for a refund for order ORD-1002?
I need help with my order.
```

The last example is useful for demonstrating clarification or escalation when the request lacks enough context.

## Repository layout

```text
src/       Python implementation
tests/     automated tests
config/    catalogs, routing, skills, and knowledge base
configs/   training configuration
schemas/   contract schemas
fixtures/  synthetic demonstration data
eval/      vertical-slice evaluation cases
reports/   selected evaluation reports and plots
plans/     finalized project and training plans
docs/      data-access and reproducibility documentation
```

Large model weights, downloaded datasets, generated embeddings, teacher caches, local credentials, and virtual environments are intentionally excluded from source control.

## Limitations and next work

The data is public and synthetic, and the reviewed labels are not independent production ground truth. Entity extraction is currently pattern-based. Future work includes adding examples for short and overlapping intents, recalibrating thresholds on an independent set, improving entity extraction, expanding skills and knowledge articles, and measuring end-to-end agent latency and cost with realistic integrations.

## License and dataset attribution

Review the Bitext dataset license and terms at the source link before redistributing dataset content. This repository contains project code and selected metadata; raw ticket data is not required to run the local demonstration.
