[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/rt2mmyrf68/badge.svg)](https://ask.thili.ai/b/rt2mmyrf68)
# Customer Feedback Analyzer

A Streamlit application that classifies retail customer feedback using a GPT-2 model fine-tuned to predict special category tokens.

## Categories

| Label token | Meaning |
|---|---|
| `<PRODUCT_ISSUE>` | Damaged, defective, missing, or incorrect product |
| `<DELIVERY>` | Late, missing, or mishandled delivery |
| `<STAFF_PRAISE>` | Positive staff interaction |
| `<STAFF_COMPLAINT>` | Negative staff interaction |
| `<PRICING>` | Incorrect price, unexpected fee, or missing discount |

## Project structure

```text
app/
  core.py          # data validation, prompts, token inference
  train.py         # tokenizer extension and Hugging Face Trainer fine-tuning
  evaluate.py      # held-out evaluation and result exports
  main.py          # Streamlit user interface and dashboard
datasets/
  feedback.csv     # balanced, synthetic example dataset
tests/
  test_core.py     # tests that do not require downloading GPT-2
.github/workflows/
  ci.yml           # GitHub Actions test workflow
reports/           # generated evaluation results
artifacts/         # generated local model (git-ignored)
```

## Setup

Python 3.12 and Git are recommended.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## Train GPT-2

The first training run downloads the public `gpt2` checkpoint from Hugging Face. Internet access is required for that initial download. Training is compute-intensive; a GPU is recommended.

```bash
python -m app.train --epochs 5 --seed 42
```

This step:
1. Loads and validates the labelled CSV.
2. Makes a deterministic, stratified 80/20 train/evaluation split.
3. Adds five special category tokens to the GPT-2 tokenizer.
4. Resizes the GPT-2 token embeddings.
5. Fine-tunes with Hugging Face `Trainer`.
6. Saves model/tokenizer files and training metrics under `artifacts/gpt2-feedback/`.

To use another local Hugging Face causal language model:

```bash
python -m app.train --model gpt2 --epochs 5 --seed 42
```

## Evaluate

After training:

```bash
python -m app.evaluate
```

Evaluation writes `reports/evaluation.json`, `reports/predictions.csv`, and `reports/confusion_matrix.csv`. Use these actual outputs in your project submission; do not manually invent metrics.

## Run the app

```bash
streamlit run app/main.py
```

The app supports single-comment classification, confidence scores, a session dashboard, and locally saved human corrections (`data/corrections.csv`). The trained model must exist before classification is available.

## Tests

The test suite validates the data, labels, prompts, and input checks without downloading GPT-2:

```bash
pytest -q
```

GitHub Actions runs the same test suite on pushes and pull requests. Open the repository's **Actions** tab and confirm the workflow is green.

## Dataset provenance and limitations

`datasets/feedback.csv` is a **synthetic demonstration dataset generated for this project**. It is not scraped, collected from customers, or representative of real retail populations. The model's results on this small synthetic dataset are not evidence of real-world accuracy. For a meaningful evaluation, replace or augment it with permissioned, anonymized, representative customer feedback and document the source, labeling process, and license/consent.

## Model artifacts and GitHub

The fine-tuned GPT-2 checkpoint is generated locally in `artifacts/` and is intentionally excluded from normal Git commits because a full GPT-2 model can be hundreds of megabytes. The code and dataset are versioned; the model can be reproduced by running the training command. If the submission checker requires a downloadable checkpoint, use Git LFS or a model repository and document the exact revision.

## Security

Never commit `.env`, API keys, customer personal information, or private feedback. The `.gitignore` excludes `.env`, local data, and generated model artifacts.
