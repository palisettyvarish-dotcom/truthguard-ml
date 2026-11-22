import csv
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix

groups = defaultdict(list)

with open("data/eval/full/scifact_predictions.csv", newline="") as f:
    for row in csv.DictReader(f):
        groups[row["claim"]].append(row)

actuals, predictions = [], []

for claim, rows in groups.items():
    gold = {r["actual"] for r in rows}
    if len(gold) != 1:
        raise ValueError(f"Conflicting labels: {claim}")

    s = max(float(r["p_supported"]) for r in rows)
    r = max(float(r["p_refuted"]) for r in rows)

    if s >= 0.50 and r >= 0.50:
        pred = "NOT_ENOUGH_INFO"
    elif s >= 0.50:
        pred = "SUPPORTED"
    elif r >= 0.50:
        pred = "REFUTED"
    else:
        pred = "NOT_ENOUGH_INFO"

    actuals.append(next(iter(gold)))
    predictions.append(pred)

labels = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]

print(classification_report(actuals, predictions, labels=labels, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(actuals, predictions, labels=labels))
