"""Build and query the knowledge base.

  python -m truthguard_ml.kb_cli build  --records data/processed/records.jsonl --out data/kb --splits train --limit 20000
  python -m truthguard_ml.kb_cli search --kb data/kb "Manchester population 2017"

--splits train   (default) keeps validation/test evidence OUT, so evaluation is honest.
--splits all     for deployment, after evaluation is finished.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .kb import KnowledgeBase, build_passages, iter_jsonl, embedder_from_name, DEFAULT_MODEL


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--records", type=Path, default=Path("data/processed/records.jsonl"))
    b.add_argument("--out", type=Path, default=Path("data/kb"))
    b.add_argument("--splits", nargs="+", default=["train"], choices=["train", "validation", "test", "all"])
    b.add_argument("--limit", type=int, default=None, help="index only N passages (quick trial)")
    b.add_argument("--model", default=DEFAULT_MODEL, help="or hashing-256 for a non-semantic smoke test")
    b.add_argument("--batch-size", type=int, default=64)
    s = sub.add_parser("search")
    s.add_argument("--kb", type=Path, default=Path("data/kb"))
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        passages = build_passages(iter_jsonl(a.records), a.splits, limit=a.limit)
        print(f"{len(passages)} unique evidence passages from splits={a.splits}")
        kb = KnowledgeBase.build(passages, embedder_from_name(a.model), a.batch_size,
                                 meta={"splits": a.splits, "limit": a.limit, "source_records": str(a.records)})
        kb.save(a.out)
        print(f"Saved to {a.out} (dense backend: {kb.dense.backend})")
    else:
        kb = KnowledgeBase.load(a.kb)
        for r in kb.search(a.query, k=a.k):
            print(json.dumps({"text": r["text"][:200], "source": r["source_name"], "url": r["source_url"],
                              "datasets": r["datasets"], "origin": r["origin"], "scores": r["scores"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
