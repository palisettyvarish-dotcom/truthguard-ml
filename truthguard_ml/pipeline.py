"""Build the processed dataset: load -> clean -> map labels -> dedup -> group -> split -> verify.

Expected raw layout (see README for how each file is obtained):
  raw/fever/{train,shared_task_dev}.jsonl   [+ raw/fever/wiki-pages/*.jsonl for evidence text]
  raw/vitaminc/{train,dev,test}.jsonl
  raw/scifact/{corpus,claims_train,claims_dev}.jsonl
  raw/liar/{train,valid,test}.tsv
Missing datasets are skipped and reported, never faked.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

from . import licenses
from .dedup import finalize, dedup_exact, claim_label_disagreements
from .loaders import (read_jsonl, parse_fever_row, build_wiki_lookup, fever_needed_pages,
                      parse_vitaminc_row, parse_scifact_row, parse_liar_row, parse_liar_tsv_line)
from .split import build_groups, assign_splits, verify_no_leakage


def load_all(raw: Path, liar_policy="exclude", stats=None):
    stats = stats if stats is not None else Counter()
    records, present = [], {}
    fdir = raw / "fever"
    ffiles = [(f, s) for f, s in (("train.jsonl", "train"), ("shared_task_dev.jsonl", "dev")) if (fdir / f).exists()]
    if ffiles:
        rows = [(s, r) for f, s in ffiles for r in read_jsonl(fdir / f)]
        wiki = None
        if (fdir / "wiki-pages").is_dir():
            wiki = build_wiki_lookup(fdir / "wiki-pages", fever_needed_pages(r for _, r in rows))
        for s, r in rows:
            records += parse_fever_row(r, s, wiki, stats)
        present["fever"] = {"wiki_pages_available": wiki is not None}
    vdir = raw / "vitaminc"
    vfiles = [(f"{s}.jsonl", s) for s in ("train", "dev", "test") if (vdir / f"{s}.jsonl").exists()]
    for f, s in vfiles:
        for r in read_jsonl(vdir / f):
            records += parse_vitaminc_row(r, s, stats)
    if vfiles:
        present["vitaminc"] = {}
    sdir = raw / "scifact"
    if (sdir / "data" / "corpus.jsonl").exists():   # official tarball unpacks into data/
        sdir = sdir / "data"
    if (sdir / "corpus.jsonl").exists():
        corpus = {str(d["doc_id"]): d for d in read_jsonl(sdir / "corpus.jsonl")}
        for s in ("train", "dev"):
            p = sdir / f"claims_{s}.jsonl"
            if p.exists():
                for r in read_jsonl(p):
                    records += parse_scifact_row(r, corpus, s, stats)
        present["scifact"] = {}
    ldir = raw / "liar"
    for fn, s in (("train.tsv", "train"), ("valid.tsv", "valid"), ("test.tsv", "test")):
        if (ldir / fn).exists():
            with open(ldir / fn, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        records += parse_liar_row(parse_liar_tsv_line(line), s, liar_policy, stats=stats)
            present["liar"] = {"policy": liar_policy}
    return records, present, stats


def build(raw: Path, out: Path, liar_policy="exclude", ratios=(0.8, 0.1, 0.1), seed=13,
          near_dup=0.6, group_by_page=False, allow_leakage=False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    loaded, present, skip_stats = load_all(raw, liar_policy)
    recs = finalize(loaded)
    kept, dropped, conflicts = dedup_exact(recs)
    groups = build_groups(kept, near_dup, group_by_page=group_by_page)
    assign_splits(kept, groups, ratios, seed)
    leak = verify_no_leakage(kept, near_dup, check_pages=group_by_page)
    if leak["violation_count"] and not allow_leakage:
        raise RuntimeError(f"Leakage detected: {leak}")
    sizes = Counter(groups)
    elig = [r for r in kept if r.nli_eligible]
    examples, n_disagree = claim_label_disagreements(kept)
    report = {
        "datasets_present": present, "liar_policy": liar_policy, "seed": seed, "ratios": ratios,
        "rows_loaded": len(loaded), "skipped_rows": dict(skip_stats), "after_finalize": len(recs),
        "exact_duplicates_dropped": len(dropped), "conflicting_records_quarantined": len(conflicts),
        "kept": len(kept), "nli_eligible": len(elig),
        "largest_group_fraction": round(max(sizes.values()) / len(kept), 4) if kept else 0,
        "by_split": dict(Counter(r.split for r in kept)),
        "by_dataset_split": {f"{d}/{s}": c for (d, s), c in sorted(Counter((r.dataset, r.split) for r in kept).items())},
        "eligible_by_label": dict(Counter(r.label_common for r in elig)),
        "eligible_by_dataset_label": {f"{d}/{l}": c for (d, l), c in sorted(Counter((r.dataset, r.label_common) for r in elig).items())},
        "evidence_status": dict(Counter(r.evidence_status for r in kept)),
        "unmapped_label_records": sum(r.label_common is None for r in kept),
        "claim_label_disagreements_total": n_disagree, "claim_label_disagreement_examples": examples,
        "leakage_check": leak, "license_notes": {k: v["license_note"] for k, v in licenses.DATASETS.items()},
        "warnings": [],
    }
    if report["largest_group_fraction"] > 0.2:
        report["warnings"].append("One leakage group holds >20% of records; split ratios may be skewed.")
    if not elig:
        report["warnings"].append("No NLI-eligible records (need evidence text). FEVER needs wiki-pages to resolve evidence.")
    with open(out / "records.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    for name, rs in (("dropped_duplicates", dropped), ("conflicts", conflicts)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    (out / "build_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--liar-policy", choices=["exclude", "extremes_opt_in"], default="exclude")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--group-by-page", action="store_true", help="stricter: keep all records of a source page together")
    a = ap.parse_args(argv)
    print(licenses.license_banner(), file=sys.stderr)
    rep = build(a.raw, a.out, a.liar_policy, seed=a.seed, group_by_page=a.group_by_page)
    print(json.dumps({k: rep[k] for k in ("kept", "nli_eligible", "by_split", "warnings")}, indent=2))


if __name__ == "__main__":
    main()
