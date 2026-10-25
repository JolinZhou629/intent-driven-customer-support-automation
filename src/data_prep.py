"""Trusted, offline source validation and separation. Does not create splits."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from .contracts import Ticket

COLUMNS = ['flags', 'instruction', 'category', 'intent', 'response']
PLACEHOLDER = re.compile(r'\{\{\s*([^{}]+?)\s*\}\}')
VERSION = 'preparation-v1'


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_for_grouping(text: str) -> str:
    """Grouping-only key: NFKC, casefold, whitespace collapse; retain punctuation."""
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def encode(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + '\n').encode()


def prepare(source: Path, source_manifest: Path, output: Path) -> dict:
    manifest = json.loads(source_manifest.read_text())
    raw = source.read_bytes()
    if raw.startswith(b'version https://git-lfs'):
        raise ValueError('source is a Git LFS pointer')
    if digest(raw) != manifest['sha256']:
        raise ValueError('source checksum does not match pinned manifest')
    with source.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError('unexpected CSV schema')
        rows = list(reader)
    if len(rows) != manifest['rows'] or not rows:
        raise ValueError('unexpected row count')
    for row in rows:
        if set(row) != set(COLUMNS) or any(not isinstance(row[c], str) or not row[c].strip() for c in COLUMNS):
            raise ValueError('malformed row or missing required field')
    # Stable order even if source rows are permuted. Identical rows remain a
    # multiset: occurrence numbers distinguish copies, never source line numbers.
    rows.sort(key=lambda r: tuple(r[c] for c in ['instruction', 'response', 'flags', 'category', 'intent']))
    occurrences = Counter()
    tickets, labels, responses, metadata = [], [], [], []
    placeholder_counts = {c: Counter() for c in ['instruction', 'response']}
    placeholder_rows = Counter()
    unmatched = Counter()
    conflicts = defaultdict(set)
    for row in rows:
        instruction = row['instruction']
        occurrence = occurrences[instruction]
        occurrences[instruction] += 1
        ticket_id = 'ticket-' + digest(encode([VERSION, instruction, occurrence]))
        ticket = Ticket(ticket_id=ticket_id, instruction=instruction)
        tickets.append(ticket.model_dump())
        labels.append({'ticket_id': ticket_id, 'category': row['category'], 'intent': row['intent']})
        responses.append({'ticket_id': ticket_id, 'response': row['response']})
        metadata.append({'ticket_id': ticket_id, 'flags': row['flags'],
                         'group_key': digest(normalize_for_grouping(instruction).encode())})
        conflicts[instruction].add(row['intent'])
        for column in placeholder_counts:
            matches = PLACEHOLDER.findall(row[column])
            placeholder_counts[column].update(matches)
            placeholder_rows[column] += bool(matches)
            remainder = PLACEHOLDER.sub('', row[column])
            unmatched[column] += '{{' in remainder or '}}' in remainder
    ticket_ids = {t['ticket_id'] for t in tickets}
    if len(ticket_ids) != len(rows):
        raise ValueError('ticket ID collision')
    report = {
        'preparation_version': VERSION, 'source_sha256': digest(raw),
        'rows': len(rows), 'unique_ticket_ids': len(ticket_ids),
        'empty_required_fields': 0,
        'exact_duplicate_instructions': len(rows) - len(occurrences),
        'normalized_duplicate_instructions': len(rows) - len({normalize_for_grouping(r['instruction']) for r in rows}),
        'normalization': 'NFKC + casefold + whitespace collapse; grouping only; original text preserved',
        'id_strategy': 'SHA256 of version, original instruction, within-instruction occurrence; rows canonically ordered before assignment',
        'placeholder_audit': {c: {'rows_with_placeholders': placeholder_rows[c],
            'distinct_slot_names': len(placeholder_counts[c]),
            'occurrences': sum(placeholder_counts[c].values()),
            'rows_with_unmatched_delimiters': unmatched[c]} for c in placeholder_counts},
        'split_status': 'unsplit; trusted preparation only; not authorized for blind training',
        'reference_audit': 'private/reference-audit.json (not included in public report)',
    }
    private_audit = {'category_counts': dict(sorted(Counter(r['category'] for r in rows).items())),
                     'intent_counts': dict(sorted(Counter(r['intent'] for r in rows).items())),
                     'distinct_categories': len({r['category'] for r in rows}),
                     'distinct_intents': len({r['intent'] for r in rows}),
                     'instructions_with_conflicting_reference_intents': sum(len(v)>1 for v in conflicts.values()),
                     'placeholder_slot_counts': {c: dict(sorted(v.items())) for c,v in placeholder_counts.items()}}
    files = {}
    for name, records in [('tickets.jsonl', tickets), ('private/true_labels.jsonl', labels),
                          ('private/responses.jsonl', responses), ('metadata.jsonl', metadata)]:
        files[name] = b''.join(encode(r) for r in sorted(records, key=lambda r:r['ticket_id']))
    files['private/reference-audit.json'] = encode(private_audit)
    files['validation.json'] = encode(report)
    output_manifest = {'preparation_version': VERSION,
                       'source_manifest_sha256': digest(source_manifest.read_bytes()),
                       'source_sha256': digest(raw),
                       'implementation_sha256': digest(Path(__file__).read_bytes()),
                       'outputs': {k: digest(v) for k,v in files.items()}}
    files['manifest.json'] = encode(output_manifest)
    # Refuse partial or different existing results; reruns must be byte-identical.
    if output.exists():
        actual = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
        if actual != set(files) or any((output/k).read_bytes() != v for k,v in files.items()):
            raise ValueError('output already exists with different or partial content; use a new version directory')
        return report
    output.mkdir(parents=True)
    for name, content in files.items():
        path = output/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, default=Path('data/manifests/bitext-source.json'))
    parser.add_argument('--output', type=Path, default=Path('data/prepared/bitext-v1'))
    args = parser.parse_args()
    manifest = json.loads(args.source_manifest.read_text())
    report = prepare(Path(manifest['path']), args.source_manifest, args.output)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
