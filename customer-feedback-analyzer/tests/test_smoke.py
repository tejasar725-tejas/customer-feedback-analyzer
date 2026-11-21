from app.llm import keyword_preview


def test_preview_returns_known_category():
    result = keyword_preview("The staff were helpful and friendly")
    assert result["label"] in {"POSITIVE", "NEGATIVE", "DELIVERY", "PRODUCT_ISSUE", "SERVICE"}
