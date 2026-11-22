"""Searchable knowledge base: BM25 (keyword) + dense embeddings (semantic) + RRF fusion.

An entry is a unique EVIDENCE PASSAGE, with provenance (source name/URL, dataset) and the
historical claims it was linked to. Those linked labels are history: they describe an old
claim, NOT the claim being checked, and must never be used as a verdict for a new claim.
Every result carries origin="historical_dataset" so live-web evidence can be told apart.
"""
from __future__ import annotations
import hashlib, json, time
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

from .text import clean_text

TOKEN_PATTERN = r"(?u)\b\w+\b"
ORIGIN = "historical_dataset"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------------------------------------------ passages
def iter_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_passages(records: Iterable[dict], splits: Sequence[str] = ("train",),
                   statuses: Sequence[str] = ("provided",), limit: Optional[int] = None,
                   max_linked: int = 5) -> list[dict]:
    """Group records by identical evidence text. Only real evidence from allowed splits."""
    by_ev: dict[str, dict] = {}
    allow_all = "all" in splits
    for r in records:
        if not r.get("evidence") or r.get("evidence_status") not in statuses:
            continue
        if not allow_all and r.get("split") not in splits:
            continue
        key = r.get("evidence_norm") or ""
        if not key:
            continue
        p = by_ev.get(key)
        if p is None:
            p = by_ev[key] = {
                "passage_id": hashlib.sha1(key.encode()).hexdigest()[:16],
                "text": r["evidence"], "source_name": r.get("source_name"), "source_url": r.get("source_url"),
                "page_key": r.get("page_key"), "evidence_ids": r.get("evidence_ids") or [],
                "datasets": [], "linked_claims": [], "n_linked": 0,
            }
        if r["dataset"] not in p["datasets"]:
            p["datasets"].append(r["dataset"])
        if not p["source_url"] and r.get("source_url"):
            p["source_url"] = r["source_url"]
        p["n_linked"] += 1
        if len(p["linked_claims"]) < max_linked:
            p["linked_claims"].append({"claim": r["claim"], "label": r.get("label_common"),
                                       "dataset": r["dataset"], "record_id": r.get("record_id")})
    out = sorted(by_ev.values(), key=lambda p: p["passage_id"])   # hash order = deterministic pseudo-random
    for p in out:
        p["datasets"].sort()
    return out[:limit] if limit else out


# ---------------------------------------------------------------------- BM25
def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx], kind="stable")]


class BM25Index:
    """Okapi BM25 over a sparse matrix; term weights are precomputed so a query is a column sum."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.vec: Optional[CountVectorizer] = None
        self.W: Optional[sparse.csc_matrix] = None

    def fit(self, docs: Sequence[str]) -> "BM25Index":
        self.vec = CountVectorizer(token_pattern=TOKEN_PATTERN, stop_words="english", dtype=np.float32)
        tf = self.vec.fit_transform(docs).tocoo()
        n = tf.shape[0]
        df = np.bincount(tf.col, minlength=tf.shape[1]).astype(np.float64)
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.bincount(tf.row, weights=tf.data, minlength=n)
        avg = float(dl.mean()) if n and dl.mean() > 0 else 1.0
        denom = tf.data + self.k1 * (1 - self.b + self.b * dl[tf.row] / avg)
        data = (idf[tf.col] * tf.data * (self.k1 + 1) / denom).astype(np.float32)
        self.W = sparse.csc_matrix((data, (tf.row, tf.col)), shape=tf.shape)
        return self

    def scores(self, query: str) -> np.ndarray:
        cols = np.unique(self.vec.transform([query]).indices)
        if len(cols) == 0:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        return np.asarray(self.W[:, cols].sum(axis=1)).ravel()

    def search(self, query: str, k: int):
        s = self.scores(query)
        idx = _topk(s, k)
        idx = idx[s[idx] > 0]
        return idx, s[idx]

    def save(self, d: Path) -> None:
        sparse.save_npz(d / "bm25_weights.npz", self.W)
        vocab = {t: int(i) for t, i in self.vec.vocabulary_.items()}
        (d / "bm25_vocab.json").write_text(json.dumps({"vocab": vocab, "k1": self.k1, "b": self.b}))

    @classmethod
    def load(cls, d: Path) -> "BM25Index":
        meta = json.loads((d / "bm25_vocab.json").read_text())
        o = cls(meta["k1"], meta["b"])
        o.W = sparse.load_npz(d / "bm25_weights.npz").tocsc()
        o.vec = CountVectorizer(token_pattern=TOKEN_PATTERN, stop_words="english", vocabulary=meta["vocab"])
        return o


# --------------------------------------------------------------------- dense
class DenseIndex:
    """Inner-product search over L2-normalised vectors. Uses FAISS if installed, else NumPy."""

    def __init__(self, emb: np.ndarray, use_faiss: Optional[bool] = None):
        self.emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.backend, self._faiss = "numpy", None
        if use_faiss is not False:
            try:
                import faiss
                ix = faiss.IndexFlatIP(self.emb.shape[1])
                ix.add(self.emb)
                self._faiss, self.backend = ix, "faiss"
            except ImportError:
                if use_faiss:
                    raise

    def search(self, qvec: np.ndarray, k: int):
        k = min(k, len(self.emb))
        q = np.asarray(qvec, dtype=np.float32).reshape(1, -1)
        if self._faiss is not None:
            d, i = self._faiss.search(q, k)
            return i[0][i[0] >= 0], d[0][i[0] >= 0]
        s = self.emb @ q[0]
        idx = _topk(s, k)
        return idx, s[idx]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 1000, convert_to_numpy=True).astype(np.float32)


class HashingEmbedder:
    """Lexical stand-in for tests/smoke runs. NOT semantic: never evaluate quality with it."""

    def __init__(self, dim: int = 256):
        self.dim, self.name = dim, f"hashing-{dim}"
        self._v = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2", ngram_range=(1, 2))

    def encode(self, texts: Sequence[str], batch_size: int = 1024) -> np.ndarray:
        out = [self._v.transform(texts[i:i + batch_size]).toarray() for i in range(0, len(texts), batch_size)]
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, self.dim), np.float32)


def embedder_from_name(name: str):
    if name.startswith("hashing-"):
        return HashingEmbedder(int(name.split("-")[1]))
    return SentenceTransformerEmbedder(name)


# -------------------------------------------------------------------- fusion
def rrf(rank_lists: dict[str, list], weights: Optional[dict] = None, k: int = 60) -> dict:
    """Reciprocal Rank Fusion: score(d) = sum_w weight / (k + rank), rank starting at 1."""
    weights = weights or {}
    out: dict = {}
    for name, ids in rank_lists.items():
        w = weights.get(name, 1.0)
        for rank, i in enumerate(ids, start=1):
            out[i] = out.get(i, 0.0) + w / (k + rank)
    return out


# ------------------------------------------------------------ knowledge base
class KnowledgeBase:
    def __init__(self, passages: list[dict], bm25: BM25Index, dense: DenseIndex, embedder, meta: dict):
        self.passages, self.bm25, self.dense, self.embedder, self.meta = passages, bm25, dense, embedder, meta
        self._emb_matrix = dense.emb

    @classmethod
    def build(cls, passages: list[dict], embedder, batch_size: int = 64, meta: Optional[dict] = None):
        if not passages:
            raise ValueError("No passages to index (check --splits and evidence availability).")
        texts = [p["text"] for p in passages]
        bm25 = BM25Index().fit(texts)
        emb = embedder.encode(texts, batch_size=batch_size)
        m = {"created": time.strftime("%F %T"), "n_passages": len(passages), "embedder": embedder.name,
             "dim": int(emb.shape[1]), **(meta or {})}
        return cls(passages, bm25, DenseIndex(emb), embedder, m)

    def save(self, d) -> None:
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        with open(d / "passages.jsonl", "w", encoding="utf-8") as f:
            for p in self.passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        np.save(d / "embeddings.npy", self._emb_matrix)
        self.bm25.save(d)
        (d / "kb_meta.json").write_text(json.dumps(self.meta, indent=2))

    @classmethod
    def load(cls, d, embedder=None):
        d = Path(d)
        meta = json.loads((d / "kb_meta.json").read_text())
        passages = list(iter_jsonl(d / "passages.jsonl"))
        emb = np.load(d / "embeddings.npy")
        assert len(passages) == len(emb) == meta["n_passages"], "KB files are inconsistent"
        return cls(passages, BM25Index.load(d), DenseIndex(emb), embedder or embedder_from_name(meta["embedder"]), meta)

    def search(self, query: str, k: int = 10, k_each: int = 50, w_bm25: float = 1.0,
               w_dense: float = 1.0, rrf_k: int = 60) -> list[dict]:
        q = clean_text(query)
        if not q:
            return []
        b_idx, b_sc = self.bm25.search(q, k_each)
        d_idx, d_sc = self.dense.search(self.embedder.encode([q])[0], k_each)
        b_rank = {int(i): r for r, i in enumerate(b_idx, 1)}
        d_rank = {int(i): r for r, i in enumerate(d_idx, 1)}
        b_score = {int(i): float(s) for i, s in zip(b_idx, b_sc)}
        d_score = {int(i): float(s) for i, s in zip(d_idx, d_sc)}
        fused = rrf({"bm25": [int(i) for i in b_idx], "dense": [int(i) for i in d_idx]},
                    {"bm25": w_bm25, "dense": w_dense}, rrf_k)
        out = []
        for i, s in sorted(fused.items(), key=lambda kv: -kv[1])[:k]:
            p = self.passages[i]
            out.append({**p, "origin": ORIGIN,
                        "scores": {"bm25": b_score.get(i), "dense": d_score.get(i), "rrf": s},
                        "ranks": {"bm25": b_rank.get(i), "dense": d_rank.get(i)}})
        return out
