import csv
from collections import defaultdict

groups = defaultdict(list)

with open("data/eval/full/scifact_predictions.csv", newline="") as f:
    for row in csv.DictReader(f):
        groups[row["claim"]].append(row)

missed = []

for claim, rows in groups.items():
    if rows[0]["actual"] != "SUPPORTED":
        continue

    score = max(float(r["p_supported"]) for r in rows)

    if score < 0.50:
        missed.append((score, claim, rows))

for score, claim, rows in sorted(missed)[:10]:
    print("\n" + "=" * 80)
    print("MAX SUPPORT:", round(score, 4))
    print("CLAIM:", claim)

    for row in rows:
        print("\nEVIDENCE:", row["evidence"])
        print("PROBABILITIES:",
              "S", row["p_supported"],
              "R", row["p_refuted"],
              "NEI", row["p_nei"])
