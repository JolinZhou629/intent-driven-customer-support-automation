"""Run directories with immutable input snapshots and structured event logs."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import yaml


def create_run(config_path: Path, report_root: Path) -> Path:
    content = config_path.read_bytes()
    config = yaml.safe_load(content)
    if not isinstance(config, dict) or not config.get('experiment_id'):
        raise ValueError('configuration requires experiment_id')
    ratios = config['dataset']['split_ratios']
    if set(ratios) != {'train', 'development', 'test'} or any(
        not isinstance(v, (int, float)) or not 0 < v < 1 for v in ratios.values()
    ) or abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError('split ratios must be positive and sum to one')
    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:12]
    folder = report_root / run_id
    folder.mkdir(parents=True, exist_ok=False)
    (folder / 'config.yaml').write_bytes(content)
    manifest = {'run_id': run_id, 'created_at': now.isoformat(),
                'config_sha256': hashlib.sha256(content).hexdigest(),
                'experiment_id': config['experiment_id'], 'stage': config['stage']}
    (folder / 'run.json').write_text(json.dumps(manifest, indent=2) + '\n')
    event(folder, 'run_created', stage=config['stage'])
    return folder


def event(folder: Path, name: str, **metrics):
    """Callers must pass operational metadata only, never text or credentials."""
    record = {'timestamp': datetime.now(timezone.utc).isoformat(),
              'event': name, 'metrics': metrics}
    with (folder / 'events.jsonl').open('a') as f:
        f.write(json.dumps(record, allow_nan=False) + '\n')
