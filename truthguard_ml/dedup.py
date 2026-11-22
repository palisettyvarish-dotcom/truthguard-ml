"""Finalise keys, exact de-duplication across datasets, conflict quarantine."""
from __future__ import annotations
import hashlib
from collections import defaultdict
from typing import Iterable
from .schema import Record
from .text import norm_key

DEFAULT_PRIORITY = ("scifact", "vitaminc", "fever", "liar")


def finalize(records: Iterable[Record]) -> list[Record]:
    out = []
    for r in records:
        r.claim_norm = norm_key(r.claim)
        r.evidence_norm = norm_key(r.evidence) if r.evidence else ""
        h = f"{r.dataset}|{r.source_split}|{r.source_record_id}|{r.claim_norm}|{r.evidence_norm}"
        r.record_id = hashlib.sha1(h.encode()).hexdigest()[:16]
        if r.claim_norm:
            out.append(r)
    return out


def dedup_exact(records: list[Record], priority=DEFAULT_PRIORITY):
    """Same (claim, evidence) = duplicate. Same pair with different mapped labels = conflict:
    all copies are quarantined, never silently resolved. Evidence-less records only
    merge within their own dataset (LIAR 'no evidence' != FEVER 'NEI')."""
    prio = {d: i for i, d in enumerate(priority)}
    buckets = defaultdict(list)
    for r in records:
        buckets[(r.claim_norm, r.evidence_norm, "" if r.evidence_norm else r.dataset)].append(r)
    kept, dropped, conflicts = [], [], []
    for grp in buckets.values():
        labels = {r.label_common for r in grp if r.label_common is not None}
        if len(labels) > 1:
            conflicts.extend(grp)
            continue
        grp.sort(key=lambda r: (prio.get(r.dataset, 99), r.record_id))
        kept.append(grp[0])
        dropped.extend(grp[1:])
    return kept, dropped, conflicts


def claim_label_disagreements(records: list[Record], limit: int = 25):
    """Same claim, different labels with DIFFERENT evidence. Informational, not removed.
    Returns (examples[:limit], total_count)."""
    by_claim = defaultdict(list)
    for r in records:
        if r.label_common:
            by_claim[r.claim_norm].append(r)
    out = []
    for c, grp in by_claim.items():
        labs = {r.label_common for r in grp}
        if len(labs) > 1:
            out.append({"claim": grp[0].claim, "labels": sorted(labs), "datasets": sorted({r.dataset for r in grp})})
    out.sort(key=lambda d: d["claim"])
    return (out if limit is None else out[:limit]), len(out)
