# Training data access policy

Status: Steps 1–3 implement strict schemas, trusted preparation, frozen splits,
role-allowlisted loaders, and a tested macOS restricted child-process probe.
See [Step 3 enforcement and limits](../reports/step-3-splits-and-access.md).
Actual discovery/model jobs must be connected to this runner in their stages;
calling the loader alone does not restrict arbitrary filesystem access.

| Role/stage | Permitted input | Excluded input |
|---|---|---|
| Trusted preparation | Raw dataset and source manifest | No model selection |
| Discovery | Blind training ticket IDs/instructions | Responses, reference labels, development calibration, test |
| Teacher/student fitting | Training tickets, frozen catalog, training annotations | Reference labels, responses, evaluation examples |
| Model selection | Development selection tickets and independent human annotations | Calibration subset and test |
| Calibration | Reserved development calibration examples and frozen model | Test |
| Isolated evaluator | Frozen predictions, reference labels per evaluation protocol | No feedback into frozen blind branch |
| KB building (later) | Training responses, approved catalog/runbooks | Test responses |

Keep raw source read-only by workflow convention. Never write credentials or
raw ticket text into operational logs/configuration snapshots. Experiment
configuration files must contain reproducible settings only; credentials belong
in local environment variables. Strict schemas prevent accidental extra fields
at a validated boundary, but cannot prevent arbitrary filesystem reads.

Step 2 outputs in `data/prepared/bitext-v1/` are trusted preparation artifacts,
not approved blind training inputs. Their `private/` directory contains
reference labels, responses, and audit distributions. The restricted child permits only its assigned partition inside the protected
user-directory tree. Future stages must explicitly extend approved inputs.
