"""Verify imports and a synthetic handoff; does not train or read the dataset."""
import json
from pathlib import Path
import numpy
import scipy
import sklearn
import torch
from .contracts import Ticket, CatalogEntry, Catalog, Annotation, ClassifierOutput, ArtifactManifest
from .runs import create_run, event


def main():
    entry = CatalogEntry(id='example.track', domain='example', name='Track',
                         definition='Illustrative only', include_when=['Track request'],
                         exclude_when=['Cancel request'], examples=['Where is my order?'])
    catalog = Catalog(version='example-v1', entries=[entry])
    result = ClassifierOutput(ticket_id='example-001', intent_id=entry.id,
        domain_id=entry.domain, confidence=0.93, candidates=[{'intent_id': entry.id, 'score': 0.93}],
        decision='handle', reason_code='synthetic_example', catalog_version=catalog.version,
        model_version='example-only', policy_version='example-only').validate_catalog(catalog)
    run = create_run(Path('config/experiments/classifier-v1.yaml'), Path('reports/runs'))
    (run / 'sample-handoff.json').write_text(result.model_dump_json(indent=2) + '\n')
    for model in [Ticket, CatalogEntry, Catalog, Annotation, ClassifierOutput, ArtifactManifest]:
        Path(f'schemas/{model.__name__}.schema.json').write_text(json.dumps(model.model_json_schema(), indent=2) + '\n')
    event(run, 'setup_verified', synthetic_only=True)
    print(f'Setup verified; synthetic handoff and snapshot: {run}')


if __name__ == '__main__':
    main()
