# TruthGuard claim-verification evaluation

Generated 2026-11-22 13:22:06. Validation pairs: 42751, test pairs: 41597.
Model selected on validation macro-F1: **MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli**.

Caveats: Only datasets with real evidence text are evaluated (VitaminC, SciFact; FEVER once the Wikipedia dump is added); LIAR is excluded. Each dataset's labels are relative to different things, so read the per-dataset rows, not just the pooled score. Models trained on FEVER or VitaminC are not held-out on those datasets. Pairs use gold evidence, so this measures the verifier alone, not retrieval.

## Validation (all candidate models)

| Model | Accuracy | Macro-F1 | Training data (unverified note) |
|---|---|---|---|
| MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli | 67.1% | 64.9% | MNLI + FEVER-NLI + ANLI. FEVER NUMBERS ARE CONTAMINATED. DeBERTa-v3. |

## Held-out test (selected model only)

#### MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
n=41597  accuracy=67.5%  macro-F1=65.3%  macro-P=68.2%  macro-R=68.9%

| Verdict | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| SUPPORTED | 87.0% | 65.5% | 74.8% | 20771 |
| REFUTED | 83.7% | 67.4% | 74.7% | 14144 |
| NOT_ENOUGH_INFO | 33.8% | 73.7% | 46.4% | 6682 |

Confusion matrix (rows = true, columns = predicted: SUPPORTED, REFUTED, NOT_ENOUGH_INFO)

    [13615, 723, 6433]
    [1404, 9539, 3201]
    [630, 1130, 4922]

| Dataset | n | Accuracy | Macro-F1 (labels present) |
|---|---|---|---|
| scifact | 172 | 42.4% | 44.7% |
| vitaminc | 41425 | 67.6% | 65.4% |
## SciFact validation: evidence selection experiment

Evaluated 141 SciFact validation records with provided rationale evidence using the same NLI model (`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`).

| Evidence strategy | Accuracy | Macro-F1 |
|---|---:|---:|
| Annotated rationale sentences | 35.46% | 33.29% |
| Full abstract | 56.03% | 40.49% |
| TF-IDF selected abstract sentences | **58.16%** | **41.07%** |

TF-IDF ranked abstract sentences by relevance to each claim and selected sentences within the model's 512-token input limit. It selected an average of 9.81 sentences per record.

**Finding:** TF-IDF selection achieved the best validation result, improving accuracy by 2.13 percentage points over full abstracts and macro-F1 by 0.58 percentage points.

**Limitations:** This is a validation experiment, not an independent test result. The subset contains only 141 records with provided rationale evidence and has no NEI examples. The method has not yet been integrated into the production verifier.