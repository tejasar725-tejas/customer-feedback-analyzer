from pathlib import Path

import pandas as pd
import pytest

from app.core import LABELS, LABEL_TOKENS, label_token, load_dataset, make_prompt


def test_dataset_exists_and_has_required_columns():
    frame = load_dataset("datasets/feedback.csv")
    assert {"text", "label"} == set(frame.columns)
    assert len(frame) >= 100
    assert set(frame["label"]) == set(LABELS)


def test_dataset_is_balanced():
    counts = load_dataset("datasets/feedback.csv")["label"].value_counts()
    assert counts.max() - counts.min() <= 1


def test_label_tokens_are_special_token_strings():
    assert len(LABEL_TOKENS) == 5
    assert label_token("pricing") == "<PRICING>"


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        label_token("UNKNOWN")


def test_prompt_format_and_empty_input():
    assert make_prompt("  Late parcel  ") == "Customer feedback: Late parcel\nCategory:"
    with pytest.raises(ValueError):
        make_prompt("   ")


def test_invalid_dataset_label_rejected(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"text": "hello", "label": "OTHER"}]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_dataset(path)
