"""Evaluate the fine-tuned model and write reproducible result files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from app.core import DEFAULT_DATASET_PATH, DEFAULT_MODEL_DIR, LABELS, load_dataset, load_classifier, classify_comment


def evaluate(dataset_path=DEFAULT_DATASET_PATH, model_dir=DEFAULT_MODEL_DIR, seed=42):
    frame = load_dataset(dataset_path)
    _, eval_frame = train_test_split(
        frame, test_size=0.2, random_state=seed, stratify=frame["label"]
    )
    tokenizer, model, device = load_classifier(model_dir)
    predictions = [classify_comment(text, tokenizer, model, device)["label"] for text in eval_frame["text"]]
    y_true = eval_frame["label"].tolist()
    accuracy = accuracy_score(y_true, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, labels=list(LABELS), average="macro", zero_division=0
    )
    report = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "evaluation_rows": len(eval_frame),
        "labels": list(LABELS),
        "classification_report": classification_report(
            y_true, predictions, labels=list(LABELS), output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=list(LABELS)).tolist(),
    }
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    (report_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame({"text": eval_frame["text"], "actual": y_true, "predicted": predictions}).to_csv(
        report_dir / "predictions.csv", index=False
    )
    pd.DataFrame(report["confusion_matrix"], index=LABELS, columns=LABELS).to_csv(
        report_dir / "confusion_matrix.csv"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "classification_report"}, indent=2))
    return report


if __name__ == "__main__":
    evaluate()
