"""Inference by generating one of the dedicated category tokens."""
from __future__ import annotations
from pathlib import Path
import re
from .storage import LABELS
from .core import PROMPT_PREFIX, PROMPT_SUFFIX, LABEL_TOKENS

DEFAULT_MODEL_DIR = Path("artifacts/feedback-gpt2")


def load_classifier(model_dir: str | Path = DEFAULT_MODEL_DIR, device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at {path}. Run: python -m app.core train")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to(device)
    model.eval()
    return tokenizer, model, device


def classify_comment(comment: str, tokenizer, model, device: str) -> dict:
    import torch
    if not comment.strip():
        raise ValueError("Please enter a customer comment.")
    prompt = f"{PROMPT_PREFIX}{comment.strip()}{PROMPT_SUFFIX}"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        next_logits = outputs.logits[0, -1, :]
        token_ids = [tokenizer.convert_tokens_to_ids(token) for token in LABEL_TOKENS]
        scores = torch.softmax(next_logits[token_ids], dim=0)
        best = int(torch.argmax(scores).item())
    label = LABELS[best]
    return {"label": label, "confidence": float(scores[best].item()),
            "probabilities": {LABELS[i]: float(scores[i].item()) for i in range(len(LABELS))}}


def keyword_preview(comment: str) -> dict:
    """Transparent non-LLM fallback used only before model training."""
    text = comment.lower()
    rules = {
        "DELIVERY": ["delivery", "shipping", "arrived", "package", "courier", "late"],
        "PRODUCT_ISSUE": ["broken", "defective", "damaged", "size", "quality", "stopped working"],
        "SERVICE": ["staff", "support", "cashier", "checkout", "employee", "rude"],
        "NEGATIVE": ["bad", "poor", "disappointed", "terrible", "awful", "refund"],
        "POSITIVE": ["great", "excellent", "love", "happy", "helpful", "amazing", "perfect"],
    }
    counts = {label: sum(1 for word in words if word in text) for label, words in rules.items()}
    label = max(counts, key=counts.get) if max(counts.values(), default=0) else "NEGATIVE"
    return {"label": label, "confidence": None, "probabilities": {}}
