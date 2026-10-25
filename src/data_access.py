"""Checksum-verified ticket loader with stage-specific allowlists."""
import json
from pathlib import Path
from .contracts import Ticket
from .data_prep import digest

ALLOWED={'discovery':{'train'},'training':{'train'},'selection':{'dev_selection'},'calibration':{'dev_calibration'}}


def load_tickets(root: Path, role: str, partition: str):
    if partition not in ALLOWED.get(role,set()):raise PermissionError('partition forbidden for role')
    relative=f'blind/{partition}/tickets.jsonl'
    path=root/relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise PermissionError('input escapes split root')
    manifest=json.loads((root/'manifest.json').read_text())
    raw=path.read_bytes()
    if digest(raw)!=manifest['outputs'][relative]:raise ValueError('frozen input checksum mismatch')
    return [Ticket.model_validate(json.loads(line)) for line in raw.splitlines()]
