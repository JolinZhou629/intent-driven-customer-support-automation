"""Annotate training tickets locally and emit only uncertain cases for review."""
import argparse, json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from ..data_access import load_tickets

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings', type=Path, default=Path('reports/discovery-step4/embeddings.npy'))
    p.add_argument('--encoder', type=Path, default=Path('artifacts/encoders/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41'))
    p.add_argument('--catalog', type=Path, default=Path('config/catalogs/bitext-v1.json'))
    p.add_argument('--output', type=Path, default=Path('data/annotations/local-v1/train.jsonl'))
    p.add_argument('--review', type=Path, default=Path('data/annotations/local-v1/teacher-review-queue.jsonl'))
    p.add_argument('--threshold', type=float, default=0.62)
    p.add_argument('--margin', type=float, default=0.06)
    a = p.parse_args()
    catalog = json.loads(a.catalog.read_text())['entries']
    model = SentenceTransformer(str(a.encoder), local_files_only=True)
    texts, ids = [], []
    for e in catalog:
        for example in e['examples']:
            texts.append(example); ids.append(e['id'])
    ce = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    prototypes = []
    for intent in [e['id'] for e in catalog]:
        prototypes.append(ce[[i for i, x in enumerate(ids) if x == intent]].mean(axis=0))
    prototypes = np.asarray(prototypes); prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)
    tickets = load_tickets(Path('data/splits/bitext-v1'), 'training', 'train')
    embeddings = np.load(a.embeddings, mmap_mode='r')
    if len(tickets) != len(embeddings):
        raise ValueError(f'ticket/embedding count mismatch: {len(tickets)} != {len(embeddings)}')
    a.output.parent.mkdir(parents=True, exist_ok=True); a.review.parent.mkdir(parents=True, exist_ok=True)
    stats = {'accepted_local': 0, 'teacher_review': 0}
    with a.output.open('w') as out, a.review.open('w') as review:
        for ticket, vector in zip(tickets, embeddings):
            scores = vector @ prototypes.T
            order = np.argsort(-scores); best, second = int(order[0]), int(order[1])
            confidence, margin = float(scores[best]), float(scores[best] - scores[second])
            accepted = confidence >= a.threshold and margin >= a.margin
            row = {'ticket_id': ticket.ticket_id, 'intent_id': catalog[best]['id'] if accepted else None,
                   'status': 'resolved' if accepted else 'review_required', 'source': 'local_embedding',
                   'model_version': 'all-MiniLM-L6-v2', 'confidence': confidence, 'margin': margin,
                   'second_intent_id': catalog[second]['id'], 'threshold': a.threshold, 'margin_threshold': a.margin}
            out.write(json.dumps(row, sort_keys=True) + '\n')
            if not accepted:
                review.write(json.dumps({'ticket_id': ticket.ticket_id, 'instruction': ticket.instruction,
                    'suggested_intent_id': catalog[best]['id'], 'second_intent_id': catalog[second]['id'],
                    'confidence': confidence, 'margin': margin, 'review_reason': 'low_confidence_or_margin'}) + '\n')
            stats['accepted_local' if accepted else 'teacher_review'] += 1
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
