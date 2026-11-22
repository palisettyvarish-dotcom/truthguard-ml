from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, classification_report
from truthguard_ml.evaluate import HFNLIModel, load_pairs, LABELS

models = [
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "cross-encoder/nli-deberta-v3-base",
]

pairs = [
    p for p in load_pairs("data/processed/records.jsonl", "test", 0)
    if p["dataset"] == "scifact"
    and p["evidence_status"] == "provided"
]

print("SciFact rationale records:", len(pairs))

for model_name in models:
    print("\n" + "=" * 75)
    print("MODEL:", model_name)

    model = HFNLIModel(model_name)
    predictions, probabilities = model.predict(
        [(p["evidence"], p["claim"]) for p in pairs]
    )

    actual = [p["label"] for p in pairs]

    print("Accuracy:", round(accuracy_score(actual, predictions), 4))
    print("Macro-F1:", round(f1_score(
        actual, predictions, labels=LABELS, average="macro", zero_division=0
    ), 4))
    print(classification_report(
        actual, predictions, labels=LABELS, zero_division=0
    ))
