"""Fine-tune GPT-2 to predict one of the feedback category special tokens."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from app.core import (
    DEFAULT_DATASET_PATH, DEFAULT_MODEL_DIR, DEFAULT_MODEL_NAME,
    LABELS, LABEL_TOKENS, load_dataset, make_prompt, label_token,
)


class FeedbackDataset(torch.utils.data.Dataset):
    def __init__(self, frame, tokenizer, max_length=128):
        self.rows = []
        for row in frame.itertuples(index=False):
            prompt = make_prompt(row.text)
            prompt_ids = tokenizer(
                prompt, add_special_tokens=False, truncation=True,
                max_length=max_length - 2,
            )["input_ids"]
            target_id = tokenizer.convert_tokens_to_ids(label_token(row.label))
            input_ids = prompt_ids + [target_id, tokenizer.eos_token_id]
            labels = [-100] * len(prompt_ids) + [target_id, tokenizer.eos_token_id]
            attention = [1] * len(input_ids)
            pad_len = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
            attention += [0] * pad_len
            self.rows.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train(
    dataset_path=DEFAULT_DATASET_PATH,
    output_dir=DEFAULT_MODEL_DIR,
    model_name=DEFAULT_MODEL_NAME,
    epochs=5,
    seed=42,
):
    set_seed(seed)
    frame = load_dataset(dataset_path)
    train_frame, eval_frame = train_test_split(
        frame, test_size=0.2, random_state=seed, stratify=frame["label"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": list(LABEL_TOKENS)})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    train_data = FeedbackDataset(train_frame, tokenizer)
    eval_data = FeedbackDataset(eval_frame, tokenizer)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        data_seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metrics = trainer.evaluate()
    metrics.update({
        "dataset_rows": int(len(frame)),
        "train_rows": int(len(train_frame)),
        "evaluation_rows": int(len(eval_frame)),
        "labels": list(LABELS),
        "seed": seed,
        "base_model": model_name,
        "epochs": epochs,
    })
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    ns = parser.parse_args()
    train(ns.dataset, ns.output, ns.model, ns.epochs, ns.seed)
