"""Streamlit UI for the Customer Feedback Analyzer."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.core import (
    DEFAULT_MODEL_DIR, LABELS, classify_comment, load_classifier, save_correction
)

st.set_page_config(page_title="Customer Feedback Analyzer", page_icon="🛍️", layout="wide")
st.title("Customer Feedback Analyzer")
st.caption("Fine-tuned GPT-2 classifies retail feedback into actionable categories.")

with st.sidebar:
    st.header("About the categories")
    descriptions = {
        "PRODUCT_ISSUE": "Damaged, defective, missing, or incorrect products",
        "DELIVERY": "Late, missing, or mishandled deliveries",
        "STAFF_PRAISE": "Positive feedback about store or support staff",
        "STAFF_COMPLAINT": "Negative feedback about staff interactions",
        "PRICING": "Incorrect prices, unexpected fees, or missing discounts",
    }
    for label, desc in descriptions.items():
        st.markdown(f"**{label.replace('_', ' ').title()}**  \n{desc}")
    st.divider()
    st.caption(f"Model directory: `{DEFAULT_MODEL_DIR}`")

@st.cache_resource(show_spinner="Loading fine-tuned GPT-2...")
def get_model():
    return load_classifier(DEFAULT_MODEL_DIR)

tab_classify, tab_dashboard, tab_about = st.tabs(["Classify feedback", "Feedback dashboard", "Project details"])

with tab_classify:
    comment = st.text_area(
        "Customer comment",
        placeholder="Example: The delivery arrived three days late and the tracking did not update.",
        height=120,
        max_chars=1000,
    )
    if st.button("Classify comment", type="primary", disabled=not comment.strip()):
        try:
            tokenizer, model, device = get_model()
            with st.spinner("Analyzing comment..."):
                result = classify_comment(comment, tokenizer, model, device)
            st.session_state["last_result"] = {"text": comment, **result}
            st.success(f"Predicted category: **{result['label'].replace('_', ' ').title()}**")
            st.metric("Model confidence", f"{result['confidence']:.1%}")
            st.bar_chart(pd.Series(result["scores"], name="Probability"))
        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not classify this comment: {exc}")

    previous = st.session_state.get("last_result")
    if previous:
        st.subheader("Correct this prediction")
        corrected = st.selectbox("Correct category (optional)", LABELS, index=LABELS.index(previous["label"]))
        if st.button("Save correction"):
            save_correction(previous["text"], previous["label"], corrected)
            st.success("Correction saved locally in data/corrections.csv.")

with tab_dashboard:
    st.subheader("Category distribution")
    st.caption("Counts are based on comments classified during this browser session.")
    history = st.session_state.get("history", [])
    if st.button("Add latest prediction to dashboard", disabled=not st.session_state.get("last_result")):
        item = st.session_state["last_result"]
        history.append({"comment": item["text"], "category": item["label"]})
        st.session_state["history"] = history
    if history:
        history_df = pd.DataFrame(history)
        st.bar_chart(history_df["category"].value_counts())
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("Classify a comment, then add it to the dashboard.")

with tab_about:
    st.markdown("""
    **Model:** GPT-2 (`gpt2`) fine-tuned with Hugging Face `Trainer`.

    **Method:** five category labels are registered as special tokenizer tokens. The model learns to predict
    the category token immediately after a feedback prompt. Inference compares the next-token logits for
    the five allowed label tokens and reports a normalized confidence score.

    **Dataset:** `datasets/feedback.csv` contains a balanced synthetic demonstration dataset. It is generated
    for this project and is not real customer data. Do not use its evaluation score as evidence of production
    performance. Replace or augment it with permissioned, representative, anonymized feedback before deployment.

    **Reproducibility:** the training/evaluation split and random seeds are fixed by default (`42`).
    """)
