"""Core utilities and GPT-2 label-token classification."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

LABELS = (
    "PRODUCT_ISSUE",
    "DELIVERY",
    "STAFF_PRAISE",
    "STAFF_COMPLAINT",
    "PRICING",
)
LABEL_TOKENS = tuple(f"<{label}>" for label in LABELS)
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")
DEFAULT_DATASET_PATH = Path(os.getenv("DATASET_PATH", "datasets/feedback.csv"))
DEFAULT_MODEL_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", "artifacts/gpt2-feedback"))
PROMPT_PREFIX = "Customer feedback: "
PROMPT_SUFFIX = "\nCategory:"


def load_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load and validate the labelled feedback CSV."""
    frame = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    frame = frame[["text", "label"]].dropna().copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(str).str.strip().str.upper()
    frame = frame[frame["text"] != ""]
    unknown = sorted(set(frame["label"]) - set(LABELS))
    if unknown:
        raise ValueError(f"Unknown labels in dataset: {unknown}")
    if frame.empty:
        raise ValueError("Dataset contains no usable rows.")
    return frame.reset_index(drop=True)


def make_prompt(text: str) -> str:
    text = str(text).strip()
    if not text:
        raise ValueError("Feedback comment cannot be empty.")
    return f"{PROMPT_PREFIX}{text}{PROMPT_SUFFIX}"


def label_token(label: str) -> str:
    label = str(label).strip().upper()
    if label not in LABELS:
        raise ValueError(f"Unsupported label: {label}")
    return f"<{label}>"


def load_classifier(model_dir: str | Path = DEFAULT_MODEL_DIR, device: str | None = None):
    """Load a locally fine-tuned GPT-2 model. Training must be run first."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_dir}. Run: python -m app.train"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.config.pad_token_id = tokenizer.pad_token_id
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(selected_device)
    model.eval()
    return tokenizer, model, selected_device


def classify_comment(
    text: str,
    tokenizer: Any,
    model: Any,
    device: str = "cpu",
) -> dict[str, Any]:
    """Predict the most likely special label token at the next-token position."""
    import torch

    prompt = make_prompt(text)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    token_ids = [tokenizer.convert_tokens_to_ids(label_token(label)) for label in LABELS]
    label_logits = logits[token_ids]
    probabilities = torch.softmax(label_logits, dim=0)
    best_index = int(torch.argmax(probabilities).item())
    return {
        "label": LABELS[best_index],
        "confidence": float(probabilities[best_index].item()),
        "scores": {
            label: float(probabilities[i].item()) for i, label in enumerate(LABELS)
        },
    }


def save_correction(text: str, predicted: str, corrected: str, path="data/corrections.csv") -> None:
    """Append a human correction locally for later review/retraining."""
    import csv
    from datetime import datetime, timezone

    corrected = corrected.strip().upper()
    predicted = predicted.strip().upper()
    if corrected not in LABELS or predicted not in LABELS:
        raise ValueError("Predicted and corrected labels must be valid categories.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_utc", "text", "predicted", "corrected"])
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "text": text.strip(),
            "predicted": predicted,
            "corrected": corrected,
        })
