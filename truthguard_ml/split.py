"""Leakage-safe splitting. Records are grouped, and a whole group goes to one split.

Records join a group when they share: the normalised claim, the normalised evidence
text, (optionally) the source page, or a near-duplicate claim (MinHash-LSH, then
exact Jaccard verification). Contrastive pairs (VitaminC) and multi-evidence claims
(FEVER, SciFact) therefore stay together, as do cross-dataset copies.
"""
from __future__ import annotations
import hashlib, zlib
from collections import defaultdict
import numpy as np
from .schema import Record

_P = 4294967311  # prime > 2**32


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def shingles(norm_text: str) -> set:
    w = norm_text.split()
    return set(w) | {f"{a} {b}" for a, b in zip(w, w[1:])}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def near_duplicate_pairs(texts: list[str], threshold: float = 0.6, num_perm: int = 128,
                         bands: int = 32, max_bucket: int = 500, seed: int = 1):
    """Return index pairs (i, j) of texts whose shingle Jaccard >= threshold."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, 2**31, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, 2**31, size=num_perm, dtype=np.uint64)
    rows = num_perm // bands
    sets = [shingles(t) for t in texts]
    buckets = defaultdict(list)
    for i, s in enumerate(sets):
        if not s:
            continue
        hv = np.fromiter((zlib.crc32(x.encode()) for x in s), dtype=np.uint64, count=len(s))
        sig = ((hv[:, None] * a[None, :] + b[None, :]) % np.uint64(_P)).min(axis=0)
        for k in range(bands):
            buckets[(k, sig[k * rows:(k + 1) * rows].tobytes())].append(i)
    seen, pairs = set(), []
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        ids = ids[:max_bucket]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                i, j = ids[x], ids[y]
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                if jaccard(sets[i], sets[j]) >= threshold:
                    pairs.append((i, j))
    return pairs


def build_groups(records: list[Record], near_dup_threshold: float = 0.6,
                 group_by_evidence: bool = True, group_by_page: bool = False) -> list[int]:
    uf = UnionFind(len(records))
    first_claim, first_ev, first_page = {}, {}, {}
    for i, r in enumerate(records):
        uf.union(i, first_claim.setdefault(r.claim_norm, i))
        if group_by_evidence and r.evidence_norm:
            uf.union(i, first_ev.setdefault(r.evidence_norm, i))
        if group_by_page and r.page_key:
            uf.union(i, first_page.setdefault(r.page_key, i))
    if near_dup_threshold:
        uniq = list(first_claim.keys())
        for i, j in near_duplicate_pairs(uniq, near_dup_threshold):
            uf.union(first_claim[uniq[i]], first_claim[uniq[j]])
    return [uf.find(i) for i in range(len(records))]


def assign_splits(records: list[Record], groups: list[int], ratios=(0.8, 0.1, 0.1), seed: int = 13):
    """Deterministic: hash(seed, canonical group key) -> train/validation/test."""
    assert abs(sum(ratios) - 1) < 1e-9
    key_of = {}
    for r, g in zip(records, groups):
        key_of[g] = min(key_of.get(g, r.claim_norm), r.claim_norm)
    cut1, cut2 = ratios[0], ratios[0] + ratios[1]
    for r, g in zip(records, groups):
        u = int(hashlib.sha256(f"{seed}|{key_of[g]}".encode()).hexdigest()[:15], 16) / 16 ** 15
        r.split = "train" if u < cut1 else "validation" if u < cut2 else "test"
    return records


def verify_no_leakage(records: list[Record], near_dup_threshold: float = 0.6, check_pages: bool = False) -> dict:
    """Independent check on the final assignment. Returns counts of violations."""
    def cross(keyfn):
        seen = defaultdict(set)
        for r in records:
            k = keyfn(r)
            if k:
                seen[k].add(r.split)
        return [k for k, v in seen.items() if len(v) > 1]
    rep = {"claim_overlap": cross(lambda r: r.claim_norm), "evidence_overlap": cross(lambda r: r.evidence_norm)}
    if check_pages:
        rep["page_overlap"] = cross(lambda r: r.page_key)
    by_claim = {}
    for r in records:
        by_claim.setdefault(r.claim_norm, r.split)
    uniq = list(by_claim)
    rep["near_duplicate_claim_overlap"] = [(uniq[i], uniq[j]) for i, j in near_duplicate_pairs(uniq, near_dup_threshold)
                                           if by_claim[uniq[i]] != by_claim[uniq[j]]] if near_dup_threshold else []
    total = sum(len(v) for v in rep.values())
    out = {k: v[:10] for k, v in rep.items()}
    out["violation_count"] = total
    return out
