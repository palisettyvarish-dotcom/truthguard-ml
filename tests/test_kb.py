import tempfile, unittest
from pathlib import Path
import numpy as np

from truthguard_ml.kb import (BM25Index, DenseIndex, HashingEmbedder, KnowledgeBase, build_passages, rrf, ORIGIN)


def rec(claim, ev, split="train", ds="vitaminc", label="SUPPORTED", status="provided", url=None, name="Wikipedia"):
    from truthguard_ml.text import norm_key
    return {"claim": claim, "evidence": ev, "evidence_norm": norm_key(ev), "evidence_status": status, "split": split,
            "dataset": ds, "label_common": label, "source_name": name, "source_url": url, "record_id": claim[:8], "page_key": None,
            "evidence_ids": []}


DOCS = ["Manchester is a city in England with a population of 545,500 in 2017.",
        "The Eiffel Tower is a wrought iron lattice tower in Paris, France.",
        "Aspirin reduces the risk of heart attack in some patients.",
        "Mount Everest is the highest mountain above sea level."]


class BM25Tests(unittest.TestCase):
    def test_ranking_and_roundtrip(self):
        ix = BM25Index().fit(DOCS)
        idx, sc = ix.search("population of Manchester", 3)
        self.assertEqual(idx[0], 0); self.assertTrue(sc[0] > 0)
        idx2, _ = ix.search("tower Paris", 3); self.assertEqual(idx2[0], 1)
        self.assertEqual(len(ix.search("zzzz qqqq", 3)[0]), 0)           # no match -> nothing, not junk
        with tempfile.TemporaryDirectory() as td:
            ix.save(Path(td)); ix2 = BM25Index.load(Path(td))
            np.testing.assert_allclose(ix.scores("heart attack aspirin"), ix2.scores("heart attack aspirin"), rtol=1e-6)

    def test_rare_terms_weigh_more(self):
        ix = BM25Index().fit(["common common common rare", "common common common common", "common words only here"])
        idx, _ = ix.search("common rare", 3); self.assertEqual(idx[0], 0)


class DenseTests(unittest.TestCase):
    def test_nearest(self):
        emb = np.eye(4, dtype=np.float32)
        ix = DenseIndex(emb, use_faiss=False)
        idx, sc = ix.search(np.array([0, 0, 1, 0], dtype=np.float32), 2)
        self.assertEqual(idx[0], 2); self.assertAlmostEqual(float(sc[0]), 1.0, places=5)
        self.assertEqual(ix.backend, "numpy")
        self.assertEqual(len(ix.search(np.ones(4, dtype=np.float32), 99)[0]), 4)   # k > n is safe


class FusionTests(unittest.TestCase):
    def test_rrf(self):
        s = rrf({"a": [1, 2, 3], "b": [3, 1, 4]}, k=60)
        self.assertGreater(s[1], s[2]); self.assertGreater(s[3], s[4])   # in both lists beats in one
        w = rrf({"a": [1], "b": [2]}, {"a": 1.0, "b": 3.0}); self.assertGreater(w[2], w[1])


class PassageTests(unittest.TestCase):
    def test_grouping_splits_status_and_provenance(self):
        recs = [rec("Claim one about Manchester.", DOCS[0], url="https://en.wikipedia.org/wiki/Manchester"),
                rec("Another claim about Manchester.", DOCS[0], label="REFUTED"),
                rec("Test claim.", DOCS[1], split="test"),
                rec("Val claim.", DOCS[2], split="validation"),
                rec("Cited abstract claim.", "Long cited abstract text.", status="cited_abstract_no_rationale", ds="scifact"),
                rec("No url claim.", DOCS[3], url=None)]
        ps = build_passages(recs, ("train",))
        texts = {p["text"] for p in ps}
        self.assertEqual(texts, {DOCS[0], DOCS[3]})                   # test/val/abstract-only excluded
        m = next(p for p in ps if p["text"] == DOCS[0])
        self.assertEqual(m["n_linked"], 2); self.assertEqual(m["source_url"], "https://en.wikipedia.org/wiki/Manchester")
        self.assertIsNone(next(p for p in ps if p["text"] == DOCS[3])["source_url"])   # never invented
        self.assertEqual(len(build_passages(recs, ("all",))), 4)
        self.assertEqual(build_passages(recs, ("train",), limit=1), build_passages(recs, ("train",))[:1])   # deterministic

    def test_missing_evidence_skipped(self):
        self.assertEqual(build_passages([rec("c", None), rec("d", "")], ("train",)), [])


class FakeEmbedder:
    """Controlled vectors so we can test that dense-only hits survive fusion."""
    name = "fake"

    def __init__(self, table, query_vec):
        self.table, self.query_vec = table, query_vec

    def encode(self, texts, batch_size=64):
        return np.array([self.table.get(t, self.query_vec) for t in texts], dtype=np.float32)


class KBTests(unittest.TestCase):
    def setUp(self):
        recs = [rec(f"claim {i}", d, url=f"https://example.org/{i}") for i, d in enumerate(DOCS)]
        self.passages = build_passages(recs, ("train",))

    def test_build_save_load_search(self):
        kb = KnowledgeBase.build(self.passages, HashingEmbedder(256), meta={"splits": ["train"]})
        with tempfile.TemporaryDirectory() as td:
            kb.save(td)
            kb2 = KnowledgeBase.load(td)
            res = kb2.search("What is the population of Manchester?", k=3)
        self.assertEqual(res[0]["text"], DOCS[0])
        self.assertEqual(res[0]["origin"], ORIGIN)
        self.assertIn("rrf", res[0]["scores"]); self.assertTrue(res[0]["source_url"].startswith("https://example.org/"))
        self.assertEqual(kb2.search("   ", k=3), [])

    def test_hybrid_keeps_lexical_only_and_dense_only_hits(self):
        ps = self.passages
        by_text = {p["text"]: i for i, p in enumerate(ps)}
        # dense space: query points at the Everest passage, which shares no words with the query
        table = {p["text"]: np.eye(len(ps))[i] for i, p in enumerate(ps)}
        emb = FakeEmbedder(table, query_vec=np.eye(len(ps))[by_text[DOCS[3]]])
        kb = KnowledgeBase.build(ps, emb)
        res = kb.search("Manchester population", k=2)
        got = {r["text"] for r in res}
        self.assertEqual(got, {DOCS[0], DOCS[3]})                      # lexical hit + semantic hit
        r_lex = next(r for r in res if r["text"] == DOCS[0]); r_den = next(r for r in res if r["text"] == DOCS[3])
        self.assertIsNotNone(r_lex["ranks"]["bm25"]); self.assertIsNone(r_den["ranks"]["bm25"])

    def test_empty_kb_rejected(self):
        with self.assertRaises(ValueError):
            KnowledgeBase.build([], HashingEmbedder())


if __name__ == "__main__":
    unittest.main()
