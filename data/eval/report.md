# TruthGuard claim-verification evaluation

Generated 2026-11-22 00:30:55. Validation pairs: 491, test pairs: 472.
Model selected on validation macro-F1: **MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli**.

Caveats: Only datasets with real evidence text are evaluated (VitaminC, SciFact; FEVER once the Wikipedia dump is added); LIAR is excluded. Each dataset's labels are relative to different things, so read the per-dataset rows, not just the pooled score. Models trained on FEVER or VitaminC are not held-out on those datasets. Pairs use gold evidence, so this measures the verifier alone, not retrieval.

## Validation (all candidate models)

| Model | Accuracy | Macro-F1 | Training data (unverified note) |
|---|---|---|---|
| cross-encoder/nli-deberta-v3-base | 56.6% | 57.1% | SNLI + MultiNLI. DeBERTa-v3. |
| cross-encoder/nli-roberta-base | 52.1% | 52.7% | SNLI + MultiNLI. RoBERTa. |
| MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | 60.1% | 59.8% | MNLI + FEVER-NLI + ANLI. FEVER NUMBERS ARE CONTAMINATED. DeBERTa-v3. |

## Held-out test (selected model only)

#### MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
n=472  accuracy=59.7%  macro-F1=59.1%  macro-P=66.7%  macro-R=64.1%

| Verdict | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTED | 91.2% | 53.4% | 67.3% | 253 |
| REFUTED | 82.7% | 62.3% | 71.1% | 146 |
| NOT_ENOUGH_INFO | 26.2% | 76.7% | 39.0% | 73 |

Confusion matrix (rows = true, columns = predicted: SUPPORTED, REFUTED, NOT_ENOUGH_INFO)

    [135, 9, 109]
    [6, 91, 49]
    [7, 10, 56]

| Dataset | n | Accuracy | Macro-F1 (labels present) |
|---|---|---|---|
| scifact | 172 | 42.4% | 44.7% |
| vitaminc | 300 | 69.7% | 65.9% |