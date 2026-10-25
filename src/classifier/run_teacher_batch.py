"""Create resumable, provenance-preserving teacher annotations for training tickets."""
import argparse, json
from pathlib import Path
from ..data_access import load_tickets
from .openai_teacher import OpenAITeacher

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--model', default='gpt-5-mini')
    p.add_argument('--cache', type=Path, default=Path('artifacts/teacher-cache-v2'))
    p.add_argument('--output', type=Path, default=Path('data/annotations/teacher-v2/train.jsonl'))
    a = p.parse_args()
    tickets = load_tickets(Path('data/splits/bitext-v1'), 'training', 'train')
    if a.limit is not None:
        tickets = tickets[:a.limit]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if a.output.exists():
        for line in a.output.read_text().splitlines():
            try:
                row = json.loads(line)
                if row.get('status') in {'resolved', 'abstained'}:
                    completed.add(row['ticket_id'])
            except json.JSONDecodeError:
                continue
    teacher = OpenAITeacher(Path('config/catalogs/bitext-v1.json'), a.cache, a.model)
    counts = {'resolved': 0, 'abstained': 0, 'failed': 0, 'cached': 0, 'skipped': 0}
    with a.output.open('a') as f:
        for n, ticket in enumerate(tickets, 1):
            if ticket.ticket_id in completed:
                counts['skipped'] += 1
                continue
            try:
                result, cached = teacher.classify(ticket.ticket_id, ticket.instruction)
                result['routing_outcome'] = 'unknown.require_more_info' if result['status'] == 'abstained' else None
                f.write(json.dumps(result, sort_keys=True) + '\n'); f.flush()
                counts[result['status']] += 1; counts['cached'] += int(cached)
                print(f'{n}/{len(tickets)} {result["status"]}', flush=True)
            except Exception as exc:
                counts['failed'] += 1
                f.write(json.dumps({'ticket_id': ticket.ticket_id, 'status': 'failed', 'error_type': type(exc).__name__}) + '\n'); f.flush()
                print(f'{n}/{len(tickets)} failed {type(exc).__name__}', flush=True)
    print(json.dumps({'model': a.model, 'tickets': len(tickets), **counts}, indent=2))

if __name__ == '__main__':
    main()
