# TruthGuard ML - stage 1: dataset pipeline

Status: code written; **17 unit/e2e tests pass on small synthetic fixtures only**.
NOT yet run on real FEVER / LIAR / SciFact / VitaminC data (no network in the build sandbox).

## Run
    pip install -r requirements.txt
    python -m truthguard_ml.download --accept-licenses [--skip-wiki]   # read licenses first
    python -m truthguard_ml.pipeline --raw data/raw --out data/processed
    python -m unittest discover -s tests -v

Outputs in data/processed: records.jsonl, build_report.json, dropped_duplicates.jsonl, conflicts.jsonl.

## Decisions you should review
- LIAR has 6 PolitiFact grades and no evidence text. It is kept (provenance, claim prior) but NOT mapped
  to SUPPORTED/REFUTED/NEI by default; "half-true" is never turned into NOT_ENOUGH_INFO.
- Each record stores label_original and label_scope (what the label is relative to). FEVER = vs Wikipedia,
  VitaminC = vs one sentence, SciFact = vs abstract rationale. Report metrics per dataset, not only pooled.
- FEVER evidence text needs the wiki-pages dump. Without it, SUPPORTS/REFUTES rows are `unresolved` and excluded
  from NLI pairs. FEVER NEI claims have no evidence, so they are also excluded; NEI pairs for FEVER must come
  from retrieval later (labelled as retrieved, never invented). SciFact NEI uses the cited abstract.
- SciFact test claims have no labels; a missing `evidence` key is skipped, never read as NEI.
- Leakage: records group by claim, evidence text, and near-duplicate claims (MinHash + Jaccard >= 0.6); whole
  groups go to one split by seeded hash. verify_no_leakage re-checks independently and the build fails on a hit.
  `--group-by-page` is stricter but may create very large groups (see largest_group_fraction in the report).
- Licenses in licenses.py are my unverified recollection. SciFact is believed non-commercial. Check them.
- download.py URLs are from memory and untested.
