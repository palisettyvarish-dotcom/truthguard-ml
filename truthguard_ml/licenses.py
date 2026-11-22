"""Dataset registry. License text below is a NOTE FROM MEMORY, not a verified fact.

The author had no network access when this was written. Read each license on the
official page before use; the pipeline refuses to download without --accept-licenses.
"""
DATASETS = {
    "fever": {
        "official": "https://fever.ai/dataset/fever.html",
        "paper": "Thorne et al., NAACL 2018",
        "license_note": "Believed CC BY-SA 3.0 family (claims + Wikipedia text). Share-alike obligations may apply to derived data.",
        "label_definition": "Claim vs. Wikipedia (2017 dump). NEI = Wikipedia has insufficient evidence.",
    },
    "vitaminc": {
        "official": "https://github.com/TalSchuster/VitaminC (HF: tals/vitaminc)",
        "paper": "Schuster et al., NAACL 2021",
        "license_note": "Believed CC BY-SA 3.0 (Wikipedia-derived). Verify.",
        "label_definition": "Claim vs. ONE given evidence sentence; contrastive revisions flip labels.",
    },
    "scifact": {
        "official": "https://github.com/allenai/scifact",
        "paper": "Wadden et al., EMNLP 2020",
        "license_note": "Believed CC BY-NC 2.0 (NON-COMMERCIAL). Verify before any commercial use.",
        "label_definition": "Scientific claim vs. abstract rationale. NEI = cited abstract has no rationale.",
    },
    "liar": {
        "official": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
        "paper": "Wang, ACL 2017",
        "license_note": "License unclear (HF card lists none). Derived from PolitiFact; check PolitiFact terms. Treat as research-only until confirmed.",
        "label_definition": "Six PolitiFact truthfulness ratings of a statement. NO evidence text. Not an NLI label.",
    },
}


def license_banner() -> str:
    lines = ["Dataset licenses (from memory, UNVERIFIED - read the official pages):"]
    for k, v in DATASETS.items():
        lines.append(f"  - {k}: {v['license_note']}  [{v['official']}]")
    return "\n".join(lines)
