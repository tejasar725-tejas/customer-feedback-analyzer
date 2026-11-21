# Customer Feedback Analyzer

A Streamlit application that categorizes retail customer comments using a GPT-2 model fine-tuned to generate dedicated category label tokens.

## Categories

- `<POSITIVE>` — praise, satisfaction, compliments
- `<NEGATIVE>` — complaints, dissatisfaction, poor experience
- `<DELIVERY>` — shipping, delivery, late or damaged arrival
- `<PRODUCT_ISSUE>` — product quality, defects, sizing or functionality
- `<SERVICE>` — staff, support, checkout or in-store service

A comment can mention multiple topics, but this prototype predicts one primary category.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train the model

```powershell
python -m app.core train --epochs 5
```

This downloads GPT-2 from Hugging Face, adds the category tokens to the tokenizer, resizes the model embeddings, and fine-tunes it with Hugging Face Trainer. The trained tokenizer and model are saved in `artifacts/feedback-gpt2`.

The dataset is intentionally small and illustrative. Training may take time; a GPU is recommended. GPT-2 is a causal language model, so training examples are formatted as `Customer comment: ... Category: <LABEL>`. At inference, the model generates the label token after the prompt.

## Run the Streamlit app

```powershell
streamlit run app/main.py
```

If a trained model is not present, the app uses a clearly labelled keyword-based preview so you can test the interface. This fallback is not GPT-2 classification.

## Tests

```powershell
pytest -v
```

## Dataset

Edit `data/feedback_examples.jsonl`. Each line contains `comment` and `label`. Labels must be one of the five categories above. Add reviewed examples to improve coverage and balance.

## Evaluation and limitations

Manually review predictions on comments not included in the training set. Track per-category precision/recall and inspect confusion between delivery, product and service complaints. The starter dataset is small and does not establish production accuracy. Do not use the preview classifier as a substitute for the fine-tuned model. Feedback corrections are saved to `outputs/corrections.csv` for later dataset curation; they are not automatically used to retrain the model.
