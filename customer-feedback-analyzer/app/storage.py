"""Dataset loading and correction persistence."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "feedback_examples.jsonl"
LABELS = ["POSITIVE", "NEGATIVE", "DELIVERY", "PRODUCT_ISSUE", "SERVICE"]


def load_examples(path: str | Path | None = None) -> list[dict]:
    source = Path(path) if path else DEFAULT_DATASET
    if not source.is_file():
        raise FileNotFoundError(f"Dataset not found: {source}")
    rows = []
    for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
        if not isinstance(item.get("comment"), str) or not item["comment"].strip():
            raise ValueError(f"Line {line_no}: comment must be a non-empty string")
        if item.get("label") not in LABELS:
            raise ValueError(f"Line {line_no}: label must be one of {LABELS}")
        rows.append({"comment": item["comment"].strip(), "label": item["label"]})
    if not rows:
        raise ValueError("Dataset contains no examples")
    return rows


def save_correction(comment: str, predicted: str, corrected: str,
                    path: str | Path = "outputs/corrections.csv") -> None:
    if corrected not in LABELS:
        raise ValueError(f"Unknown category: {corrected}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comment", "predicted", "corrected"])
        if not exists:
            writer.writeheader()
        writer.writerow({"comment": comment, "predicted": predicted, "corrected": corrected})
