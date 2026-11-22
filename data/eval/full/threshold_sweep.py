import csv
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, recall_score

groups = defaultdict(list)

with open("data/eval/full/scifact_predictions.csv", newline="") as f:
    for row in csv.DictReader(f):
        groups[row["claim"]].append(row)

labels = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
actuals = []
scores = []

for claim, rows in groups.items():
    gold = {r["actual"] for r in rows}
    if len(gold) != 1:
        raise ValueError(f"Conflicting labels: {claim}")

    actuals.append(next(iter(gold)))
    scores.append((
        max(float(r["p_supported"]) for r in rows),
        max(float(r["p_refuted"]) for r in rows)
    ))

print("Threshold | Accuracy | Macro-F1 | Sup recall | Ref recall | NEI recall")

for threshold in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    predictions = []

    for support, refute in scores:
        if support >= threshold and refute >= threshold:
            pred = "NOT_ENOUGH_INFO"
        elif support >= threshold:
            pred = "SUPPORTED"
        elif refute >= threshold:
            pred = "REFUTED"
        else:
            pred = "NOT_ENOUGH_INFO"

        predictions.append(pred)

    recalls = recall_score(
        actuals, predictions, labels=labels,
        average=None, zero_division=0
    )

    print(
        f"{threshold:.2f}      | "
        f"{accuracy_score(actuals, predictions):.4f}   | "
        f"{f1_score(actuals, predictions, labels=labels, average='macro', zero_division=0):.4f}   | "
        f"{recalls[0]:.4f}      | {recalls[1]:.4f}      | {recalls[2]:.4f}"
    )
