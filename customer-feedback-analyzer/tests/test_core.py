import pytest
from app.core import build_training_text, prepare_training_texts, LABEL_TOKENS
from app.storage import LABELS, load_examples
from app.llm import keyword_preview


def test_dataset_has_all_categories():
    rows = load_examples()
    assert len(rows) >= 15
    assert set(row["label"] for row in rows) == set(LABELS)


def test_training_text_uses_special_label_token():
    text = build_training_text("Package arrived late", "DELIVERY")
    assert "Customer comment:" in text
    assert "Category:" in text
    assert "<DELIVERY>" in text


def test_invalid_label_rejected():
    with pytest.raises(ValueError):
        build_training_text("A comment", "UNKNOWN")


def test_prepare_training_texts_matches_dataset():
    assert len(prepare_training_texts()) == len(load_examples())
    assert all(any(token in text for token in LABEL_TOKENS) for text in prepare_training_texts())


def test_preview_delivery_classification():
    assert keyword_preview("My delivery arrived late").get("label") == "DELIVERY"
