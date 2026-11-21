"""Streamlit UI for retail customer feedback categorization."""
from __future__ import annotations
import streamlit as st
import pandas as pd

from app.storage import LABELS, load_examples, save_correction
from app.llm import load_classifier, classify_comment, keyword_preview

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="📊", layout="wide")
st.title("Customer Feedback Analyzer")
st.caption("Categorize customer comments and spot recurring retail feedback themes.")

try:
    examples = load_examples()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Categories")
    st.markdown("- **POSITIVE** — praise and satisfaction\n- **NEGATIVE** — dissatisfaction\n- **DELIVERY** — shipping and arrival\n- **PRODUCT_ISSUE** — defects and quality\n- **SERVICE** — staff and support")
    st.divider()
    st.caption("Train the model with:")
    st.code("python -m app.core train --epochs 5", language="powershell")

comment = st.text_area("Enter a customer comment", height=130,
                       placeholder="Example: My order arrived three days late and the box was damaged.")
if st.button("Classify feedback", type="primary", disabled=not comment.strip()):
    try:
        tokenizer, model, device = load_classifier()
        result = classify_comment(comment, tokenizer, model, device)
        mode = "Fine-tuned GPT-2"
    except FileNotFoundError:
        result = keyword_preview(comment)
        mode = "Keyword preview (model not trained)"
    st.session_state.current = {"comment": comment, **result, "mode": mode}
    st.session_state.history.append({"comment": comment, "category": result["label"]})

if "current" in st.session_state:
    current = st.session_state.current
    st.subheader("Classification result")
    st.caption(current["mode"])
    col1, col2 = st.columns(2)
    col1.metric("Predicted category", current["label"])
    if current["confidence"] is not None:
        col2.metric("Label confidence", f"{current['confidence']:.1%}")
    else:
        col2.caption("Confidence is available after GPT-2 training.")
    if current["probabilities"]:
        st.bar_chart(pd.Series(current["probabilities"], name="Probability"))
    st.markdown("**Correct this prediction (optional)**")
    corrected = st.selectbox("Correct category", LABELS,
                             index=LABELS.index(current["label"]), key="corrected_label")
    if st.button("Save correction for future training"):
        save_correction(current["comment"], current["label"], corrected)
        st.success("Correction saved to outputs/corrections.csv. It is not automatically used for training.")

if st.session_state.history:
    st.divider()
    st.subheader("Feedback trend dashboard")
    df = pd.DataFrame(st.session_state.history)
    counts = df["category"].value_counts().reindex(LABELS, fill_value=0)
    st.bar_chart(counts)
    st.dataframe(df, use_container_width=True, hide_index=True)
