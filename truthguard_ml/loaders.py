"""Row parsers for the official raw formats. They never invent evidence or URLs.

Formats handled:
  FEVER    official JSONL (train.jsonl / shared_task_dev.jsonl) + optional wiki-pages dump
  VitaminC JSONL exported from Hugging Face (claim, evidence, label, page, ...)
  SciFact  official release (corpus.jsonl, claims_{train,dev}.jsonl)
  LIAR     official TSV (14 columns)
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional

from .labels import SCOPES, map_label
from .schema import (Record, EV_PROVIDED, EV_NONE_BY_DEF, EV_UNRESOLVED,
                     EV_CITED_ABSTRACT, EV_NOT_PROVIDED, NEI)
from .text import clean_text, wikipedia_url


def read_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _skip(stats: Optional[Counter], reason: str):
    if stats is not None:
        stats[reason] += 1
    return []


# ----------------------------------------------------------------- FEVER
def build_wiki_lookup(wiki_dir, needed_pages: set) -> dict:
    """Read the FEVER wiki-pages dump, keeping only pages we need (it is huge)."""
    out: dict = {}
    for p in sorted(Path(wiki_dir).glob("*.jsonl")):
        for row in read_jsonl(p):
            if row.get("id") in needed_pages:
                sents = {}
                for ln in (row.get("lines") or "").split("\n"):
                    parts = ln.split("\t")
                    if len(parts) >= 2 and parts[0].isdigit():
                        sents[int(parts[0])] = parts[1]
                out[row["id"]] = sents
    return out


def fever_needed_pages(rows: Iterable[dict]) -> set:
    pages = set()
    for r in rows:
        for ev_set in r.get("evidence") or []:
            for item in ev_set:
                if len(item) >= 3 and item[2]:
                    pages.add(item[2])
    return pages


def parse_fever_row(row: dict, split: str, wiki_lookup: Optional[Mapping] = None,
                    stats: Optional[Counter] = None) -> list[Record]:
    claim = clean_text(row.get("claim"))
    if not claim:
        return _skip(stats, "fever_missing_claim")
    orig = row.get("label")
    if orig is None:
        return _skip(stats, "fever_unlabelled")        # e.g. test set without labels
    common = map_label("fever", orig)
    rid = str(row.get("id"))
    base = dict(dataset="fever", source_split=split, source_record_id=rid, claim=claim,
                label_original=str(orig), label_common=common, label_scope=SCOPES["fever"],
                source_name="Wikipedia (FEVER 2017 dump)")
    if common == NEI:
        return [Record(evidence=None, evidence_status=EV_NONE_BY_DEF, **base)]
    # SUPPORTED / REFUTED: find the first evidence set whose text we can resolve.
    first_ids: list = []
    first_page = None
    for ev_set in row.get("evidence") or []:
        items = [(i[2], i[3]) for i in ev_set if len(i) >= 4 and i[2] is not None and i[3] is not None]
        if not items:
            continue
        if not first_ids:
            first_ids = [f"{p}#{s}" for p, s in items]
            first_page = items[0][0]
        if wiki_lookup is None:
            continue
        texts = []
        for p, s in items:
            t = (wiki_lookup.get(p) or {}).get(int(s))
            if not t:
                texts = []
                break
            texts.append(t)
        if texts:
            ev = clean_text(" ".join(texts))
            return [Record(evidence=ev, evidence_status=EV_PROVIDED, evidence_ids=[f"{p}#{s}" for p, s in items],
                           source_url=wikipedia_url(items[0][0]), page_key=f"wiki:{items[0][0]}", **base)]
    return [Record(evidence=None, evidence_status=EV_UNRESOLVED, evidence_ids=first_ids,
                   source_url=wikipedia_url(first_page), page_key=f"wiki:{first_page}" if first_page else None,
                   **base)]


def group_fever_hf_rows(rows: Iterable[dict]) -> list[dict]:
    """Convert flat HF rows (one row per evidence item) to the official nested format."""
    claims: dict = {}
    for r in rows:
        c = claims.setdefault(r["id"], {"id": r["id"], "label": r.get("label"), "claim": r.get("claim"), "evidence": {}})
        ann = r.get("evidence_annotation_id")
        if r.get("evidence_wiki_url"):
            c["evidence"].setdefault(ann, []).append([ann, r.get("evidence_id"), r["evidence_wiki_url"], r.get("evidence_sentence_id")])
    out = []
    for c in claims.values():
        c["evidence"] = list(c["evidence"].values()) or [[[None, None, None, None]]]
        out.append(c)
    return out


# -------------------------------------------------------------- VitaminC
def parse_vitaminc_row(row: dict, split: str, stats: Optional[Counter] = None) -> list[Record]:
    claim = clean_text(row.get("claim"))
    if not claim:
        return _skip(stats, "vitaminc_missing_claim")
    orig = row.get("label")
    if orig is None:
        return _skip(stats, "vitaminc_unlabelled")
    common = map_label("vitaminc", orig)
    ev = clean_text(row.get("evidence"))
    page = row.get("page")
    if ev:
        status = EV_PROVIDED
    else:
        status = EV_NONE_BY_DEF if common == NEI else EV_UNRESOLVED
    rid = str(row.get("unique_id") or row.get("case_id") or f"{claim[:40]}|{(ev or '')[:40]}")
    meta = {k: row[k] for k in ("case_id", "unique_id", "revision_type") if k in row}
    return [Record(dataset="vitaminc", source_split=split, source_record_id=rid, claim=claim, evidence=ev,
                   evidence_status=status, label_original=str(orig), label_common=common,
                   label_scope=SCOPES["vitaminc"], source_name="Wikipedia (VitaminC revision)",
                   source_url=wikipedia_url(page), page_key=f"wiki:{page.replace(' ', '_')}" if page else None,
                   metadata=meta)]


# --------------------------------------------------------------- SciFact
def parse_scifact_row(row: dict, corpus: Mapping, split: str, stats: Optional[Counter] = None) -> list[Record]:
    claim = clean_text(row.get("claim"))
    if not claim:
        return _skip(stats, "scifact_missing_claim")
    if "evidence" not in row:
        # Test-set claims ship without labels. Absence of the key is NOT "no evidence".
        return _skip(stats, "scifact_unlabelled")
    ev = row["evidence"] or {}
    cid = str(row.get("id"))
    out: list[Record] = []
    common_kw = dict(dataset="scifact", source_split=split, claim=claim, label_scope=SCOPES["scifact"],
                     source_name="SciFact (S2ORC abstract)")

    def doc_of(d):
        return corpus.get(str(d))

    for doc_id, rationales in ev.items():
        doc = doc_of(doc_id)
        for i, rat in enumerate(rationales):
            orig = rat["label"]
            text = None
            if doc:
                try:
                    text = clean_text(" ".join(doc["abstract"][j] for j in rat["sentences"]))
                except (IndexError, KeyError, TypeError):
                    text = None
            out.append(Record(source_record_id=f"{cid}:{doc_id}:{i}", evidence=text,
                              evidence_status=EV_PROVIDED if text else EV_UNRESOLVED,
                              label_original=str(orig), label_common=map_label("scifact", orig),
                              evidence_ids=[f"{doc_id}#" + ",".join(map(str, rat["sentences"]))],
                              page_key=f"scifact:{doc_id}", metadata={"doc_id": str(doc_id), "title": (doc or {}).get("title")},
                              **common_kw))
    if not ev:
        cited = row.get("cited_doc_ids") or []
        for doc_id in cited:
            doc = doc_of(doc_id)
            text = clean_text(" ".join(doc["abstract"])) if doc and doc.get("abstract") else None
            out.append(Record(source_record_id=f"{cid}:{doc_id}:nei", evidence=text,
                              evidence_status=EV_CITED_ABSTRACT if text else EV_UNRESOLVED,
                              label_original="NOINFO (no rationale annotated)", label_common=NEI,
                              evidence_ids=[f"{doc_id}#abstract"], page_key=f"scifact:{doc_id}",
                              metadata={"doc_id": str(doc_id), "title": (doc or {}).get("title")}, **common_kw))
        if not cited:
            out.append(Record(source_record_id=f"{cid}:nodoc", evidence=None, evidence_status=EV_NONE_BY_DEF,
                              label_original="NOINFO (no cited docs)", label_common=NEI, **common_kw))
    return out


# ------------------------------------------------------------------ LIAR
LIAR_COLUMNS = ["id", "label", "statement", "subject", "speaker", "job_title", "state_info",
                "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts",
                "mostly_true_counts", "pants_on_fire_counts", "context"]


def parse_liar_tsv_line(line: str) -> dict:
    parts = line.rstrip("\n").split("\t")
    parts += [""] * (len(LIAR_COLUMNS) - len(parts))
    return dict(zip(LIAR_COLUMNS, parts))


def parse_liar_row(row: dict, split: str, policy: str = "exclude", label_names: Optional[list] = None,
                   stats: Optional[Counter] = None) -> list[Record]:
    claim = clean_text(row.get("statement"))
    if not claim:
        return _skip(stats, "liar_missing_claim")
    orig = row.get("label")
    if orig is None or orig == "":
        return _skip(stats, "liar_unlabelled")
    if isinstance(orig, int):
        if not label_names:
            raise ValueError("integer LIAR label needs label_names (take them from the dataset features)")
        orig = label_names[orig]
    common = map_label("liar", orig, policy)
    meta = {k: row.get(k) for k in ("speaker", "subject", "job_title", "state_info", "party_affiliation", "context") if row.get(k)}
    return [Record(dataset="liar", source_split=split, source_record_id=str(row.get("id")), claim=claim,
                   evidence=None, evidence_status=EV_NOT_PROVIDED, label_original=str(orig),
                   label_common=common, label_scope=SCOPES["liar"], source_name="PolitiFact rating (via LIAR)",
                   source_url=None, metadata=meta)]
