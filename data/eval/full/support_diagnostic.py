import csv
from collections import defaultdict
from statistics import mean

groups = defaultdict(list)

with open("data/eval/full/scifact_predictions.csv", newline="") as f:
    for row in csv.DictReader(f):
        groups[row["claim"]].append(row)

correct = []
missed = []

for claim, rows in groups.items():
    if rows[0]["actual"] != "SUPPORTED":
        continue

    score = max(float(r["p_supported"]) for r in rows)
    target = correct if score >= 0.50 else missed
    target.append((score, len(rows)))

for name, values in [("Correctly supported", correct), ("Missed supported", missed)]:
    print("\n" + name)
    print("Claims:", len(values))
    if values:
        print("Mean max support probability:", round(mean(x[0] for x in values), 4))
        print("Mean rationale records:", round(mean(x[1] for x in values), 2))
        print("Highest probability:", round(max(x[0] for x in values), 4))
        print("Lowest probability:", round(min(x[0] for x in values), 4))
