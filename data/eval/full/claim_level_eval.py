import csv
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

groups = defaultdict(list)

with open("data/eval/full/scifact_predictions.csv", newline="") as f:
    for row in csv.DictReader(f):
        groups[row["claim"]].append(row)

labels = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
actuals, predictions = [], []

for claim, rows in groups.items():
    gold = {r["actual"] for r in rows}
    if len(gold) != 1:
        raise ValueError(f"Conflicting labels: {claim}")

    support = max(float(r["p_supported"]) for r in rows)
    refute = max(float(r["p_refuted"]) for r in rows)

    if support >= 0.80 and refute >= 0.80:
        pred = "NOT_ENOUGH_INFO"
    elif support >= 0.80:
        pred = "SUPPORTED"
    elif refute >= 0.80:
        pred = "REFUTED"
    else:
        pred = "NOT_ENOUGH_INFO"

    actuals.append(next(iter(gold)))
    predictions.append(pred)

print("Unique claims:", len(groups))
print("Accuracy:", round(accuracy_score(actuals, predictions), 4))
print("Macro-F1:", round(f1_score(actuals, predictions, labels=labels, average="macro", zero_division=0), 4))
print("\n", classification_report(actuals, predictions, labels=labels, zero_division=0))
print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(actuals, predictions, labels=labels))
