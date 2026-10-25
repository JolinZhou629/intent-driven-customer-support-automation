"""Prepare v2-labelled JSONL files for transformer fine-tuning."""
import json
from pathlib import Path

MAP = {
    "refund.eligibility": "refund.policy",
    "refund.money_back_guarantee": "refund.policy",
    "account.update_user_information": "account.update",
    "order.add_item": "order.modify",
    "order.remove_item": "order.modify",
    "order.swap_item": "order.modify",
    "payment.cancellation_charge": "account.early_exit_fee",
}

def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def main():
    root = Path(".")
    tickets = {r["ticket_id"]: r["instruction"] for r in rows(root / "data/prepared/bitext-v1/tickets.jsonl")}
    catalog = json.loads((root / "artifacts/classifier-v2/catalog.json").read_text())
    labels = {e["id"]: i for i, e in enumerate(catalog["entries"])}
    def mapped(x): return MAP.get(x, x) if x else x
    def write(name, data):
        out = root / "data/transformer-v1" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in data))
    review = rows(root / "data/annotations/teacher-v2/review-500-final.jsonl")
    review_ids = {r["ticket_id"] for r in review}
    local = rows(root / "data/annotations/local-v1/train.jsonl")
    train = []
    for r in local:
        label = mapped(r.get("intent_id"))
        if r.get("status") == "resolved" and r.get("confidence", 0) >= .75 and r.get("margin", 0) >= .15 and r["ticket_id"] not in review_ids and label in labels:
            train.append({"text": tickets[r["ticket_id"]], "label": labels[label], "intent_id": label, "ticket_id": r["ticket_id"]})
    for r in review:
        label = mapped(r.get("intent_id"))
        if r.get("status") == "resolved" and r["ticket_id"] in tickets and label in labels:
            train.append({"text": tickets[r["ticket_id"]], "label": labels[label], "intent_id": label, "ticket_id": r["ticket_id"]})
    write("train.jsonl", train)
    for split in ("dev_selection", "dev_calibration"):
        source = root / f"data/annotations/independent-review-v1/{split}-teacher-v2.jsonl"
        if not source.exists():
            continue
        data = []
        for r in rows(source):
            label = mapped(r.get("intent_id"))
            if r.get("status") == "resolved" and r["ticket_id"] in tickets and label in labels:
                data.append({"text": tickets[r["ticket_id"]], "label": labels[label], "intent_id": label, "ticket_id": r["ticket_id"]})
        write(f"{split}.jsonl", data)
    write("catalog-labels.json", [{"label": i, "intent_id": e["id"]} for i, e in enumerate(catalog["entries"])])
    print(json.dumps({"train": len(train), "labels": len(labels)}, indent=2))

if __name__ == "__main__": main()
