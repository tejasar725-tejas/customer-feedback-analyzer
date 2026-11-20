"""GPT-2 fine-tuning with dedicated category label tokens."""
from __future__ import annotations
import argparse
from pathlib import Path
from .storage import LABELS, load_examples

MODEL_NAME = "gpt2"
LABEL_TOKENS = [f"<{label}>" for label in LABELS]
PROMPT_PREFIX = "Customer comment: "
PROMPT_SUFFIX = "\nCategory:"


def build_training_text(comment: str, label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"Unknown category: {label}")
    return f"{PROMPT_PREFIX}{comment.strip()}{PROMPT_SUFFIX} <{label}>"


def prepare_training_texts(dataset_path: str | Path | None = None) -> list[str]:
    return [build_training_text(row["comment"], row["label"]) for row in load_examples(dataset_path)]


def train_model(dataset_path: str | Path | None = None, model_name: str = MODEL_NAME,
                epochs: int = 5, batch_size: int = 4, max_length: int = 128,
                output_dir: str | Path = "artifacts/feedback-gpt2") -> Path:
    # Lazy imports keep tests and the preview UI usable without downloading ML libraries.
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                               TrainingArguments, DataCollatorForLanguageModeling)

    if epochs < 1 or batch_size < 1 or max_length < 16:
        raise ValueError("epochs/batch_size must be positive and max_length >= 16")
    texts = prepare_training_texts(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": LABEL_TOKENS})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = Dataset.from_dict({"text": texts})
    def tokenize_batch(batch):
        encoded = tokenizer(batch["text"], truncation=True, max_length=max_length,
                            padding="max_length")
        # Train the causal LM to predict the whole formatted sequence, including label token.
        encoded["labels"] = [ids.copy() for ids in encoded["input_ids"]]
        return encoded
    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["text"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=str(Path(output_dir).parent / "training_runs"),
        num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        learning_rate=5e-5, logging_steps=1, save_strategy="no",
        report_to="none", fp16=False, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized,
                      data_collator=collator, tokenizer=tokenizer)
    trainer.train()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 to categorize retail feedback.")
    sub = parser.add_subparsers(dest="command")
    train = sub.add_parser("train", help="Fine-tune GPT-2 with category tokens.")
    train.add_argument("--dataset", default=None)
    train.add_argument("--model-name", default=MODEL_NAME)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--max-length", type=int, default=128)
    train.add_argument("--output-dir", default="artifacts/feedback-gpt2")
    args = parser.parse_args(argv)
    if args.command != "train":
        parser.print_help()
        return 0
    path = train_model(args.dataset, args.model_name, args.epochs,
                       args.batch_size, args.max_length, args.output_dir)
    print(f"Fine-tuned model saved to: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
