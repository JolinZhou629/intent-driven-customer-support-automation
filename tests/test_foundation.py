from pathlib import Path
import json
import pytest
from pydantic import ValidationError
from src.contracts import Ticket, Annotation, ClassifierOutput, ArtifactFile
from src.runs import create_run


@pytest.mark.parametrize('field', ['intent', 'category', 'response', 'flags'])
def test_blind_ticket_rejects_extra_source_fields(field):
    with pytest.raises(ValidationError):
        Ticket.model_validate({'ticket_id': 'x', 'instruction': 'help', field: 'leak'})


def test_original_text_preserved():
    assert Ticket(ticket_id='x', instruction='  ORD-1042? ').instruction == '  ORD-1042? '


@pytest.mark.parametrize('text', ['', '   '])
def test_blank_ticket_rejected(text):
    with pytest.raises(ValidationError):
        Ticket(ticket_id='x', instruction=text)


def test_teacher_provenance_required():
    with pytest.raises(ValidationError):
        Annotation(ticket_id='x', catalog_version='v1', intent_id='track', status='resolved',
                   source='teacher', annotator_id='teacher', annotation_version='v1')


@pytest.mark.parametrize('confidence', [-0.1, 1.1, float('nan')])
def test_invalid_confidence_rejected(confidence):
    with pytest.raises(ValidationError):
        ClassifierOutput(ticket_id='x', intent_id=None, domain_id=None, confidence=confidence,
                         candidates=[], decision='clarify', reason_code='unknown',
                         catalog_version='v1', model_version='v1', policy_version='v1')


def test_handle_requires_intent():
    with pytest.raises(ValidationError):
        ClassifierOutput(ticket_id='x', intent_id=None, domain_id=None, confidence=0.9,
                         candidates=[], decision='handle', reason_code='unknown',
                         catalog_version='v1', model_version='v1', policy_version='v1')


@pytest.mark.parametrize('path', ['../model', '/tmp/model', 'nested/../../model'])
def test_artifact_path_cannot_escape_bundle(path):
    with pytest.raises(ValidationError):
        ArtifactFile(path=path, sha256='a'*64)


def test_run_snapshots_are_independent(tmp_path):
    config = tmp_path / 'input.yaml'
    config.write_bytes(Path('config/experiments/classifier-v1.yaml').read_bytes())
    original = config.read_bytes()
    first = create_run(config, tmp_path / 'runs')
    second = create_run(config, tmp_path / 'runs')
    config.write_text('changed')
    assert first != second
    assert (first / 'config.yaml').read_bytes() == original
    assert json.loads((first / 'events.jsonl').read_text())['event'] == 'run_created'
