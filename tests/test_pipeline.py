import json, tempfile, unittest
from collections import Counter
from pathlib import Path

from truthguard_ml.text import clean_text, norm_key, wikipedia_url
from truthguard_ml.labels import map_label, UnknownLabelError
from truthguard_ml.loaders import (parse_fever_row, parse_vitaminc_row, parse_scifact_row, parse_liar_row,
                                   parse_liar_tsv_line, group_fever_hf_rows)
from truthguard_ml.dedup import finalize, dedup_exact
from truthguard_ml.split import build_groups, assign_splits, verify_no_leakage, near_duplicate_pairs
from truthguard_ml.pipeline import build


class TextTests(unittest.TestCase):
    def test_clean_and_keys(self):
        self.assertEqual(clean_text("Paris -LRB- France -RRB- is  big ."), "Paris (France) is big.")
        self.assertIsNone(clean_text("  \u0000 "))
        self.assertEqual(norm_key("The Eiffel Tower, in Paris!"), norm_key("the eiffel tower in paris"))

    def test_wikipedia_url_only_from_title(self):
        self.assertEqual(wikipedia_url("Barack_Obama"), "https://en.wikipedia.org/wiki/Barack_Obama")
        self.assertIsNone(wikipedia_url(None))
        self.assertIn("(", wikipedia_url("Paris_-LRB-France-RRB-"))


class LabelTests(unittest.TestCase):
    def test_maps(self):
        self.assertEqual(map_label("fever", "NOT ENOUGH INFO"), "NOT_ENOUGH_INFO")
        self.assertEqual(map_label("scifact", "CONTRADICT"), "REFUTED")
        self.assertEqual(map_label("vitaminc", "SUPPORTS"), "SUPPORTED")

    def test_liar_never_mapped_by_default(self):
        for l in ("pants-fire", "half-true", "true"):
            self.assertIsNone(map_label("liar", l))
        self.assertEqual(map_label("liar", "true", "extremes_opt_in"), "SUPPORTED")
        self.assertIsNone(map_label("liar", "half-true", "extremes_opt_in"))  # middle never becomes NEI

    def test_unknown_label_raises(self):
        with self.assertRaises(UnknownLabelError):
            map_label("fever", "MAYBE")


class LoaderTests(unittest.TestCase):
    def test_fever_nei_has_no_evidence_and_no_url(self):
        r = parse_fever_row({"id": 1, "label": "NOT ENOUGH INFO", "claim": "X did Y.",
                             "evidence": [[[None, None, None, None]]]}, "train")[0]
        self.assertIsNone(r.evidence); self.assertEqual(r.evidence_status, "none_by_definition")
        self.assertIsNone(r.source_url); self.assertFalse(r.nli_eligible)

    def test_fever_resolves_or_flags_unresolved(self):
        row = {"id": 2, "label": "SUPPORTS", "claim": "Paris is in France.",
               "evidence": [[[9, 9, "Paris", 0]]]}
        ok = parse_fever_row(row, "train", {"Paris": {0: "Paris is the capital of France."}})[0]
        self.assertEqual(ok.evidence_status, "provided"); self.assertTrue(ok.nli_eligible)
        self.assertEqual(ok.source_url, "https://en.wikipedia.org/wiki/Paris")
        un = parse_fever_row(row, "train", None)[0]
        self.assertEqual(un.evidence_status, "unresolved"); self.assertIsNone(un.evidence)
        self.assertFalse(un.nli_eligible)            # no fabricated evidence
        miss = parse_fever_row(row, "train", {"Paris": {5: "other"}})[0]
        self.assertEqual(miss.evidence_status, "unresolved")

    def test_fever_unlabelled_skipped(self):
        st = Counter()
        self.assertEqual(parse_fever_row({"id": 3, "claim": "c"}, "test", stats=st), [])
        self.assertEqual(st["fever_unlabelled"], 1)

    def test_fever_hf_grouping(self):
        rows = [{"id": 1, "label": "SUPPORTS", "claim": "c", "evidence_annotation_id": 5, "evidence_id": 6,
                 "evidence_wiki_url": "Paris", "evidence_sentence_id": 0},
                {"id": 1, "label": "SUPPORTS", "claim": "c", "evidence_annotation_id": 5, "evidence_id": 7,
                 "evidence_wiki_url": "Paris", "evidence_sentence_id": 1}]
        g = group_fever_hf_rows(rows)
        self.assertEqual(len(g), 1); self.assertEqual(len(g[0]["evidence"][0]), 2)

    def test_vitaminc(self):
        r = parse_vitaminc_row({"claim": "A won in 2019.", "evidence": "A won the 2019 final.", "label": "SUPPORTS",
                                "page": "A final", "case_id": "7"}, "train")[0]
        self.assertTrue(r.nli_eligible); self.assertEqual(r.page_key, "wiki:A_final")

    def test_scifact_rationale_nei_and_unlabelled(self):
        corpus = {"1": {"doc_id": 1, "title": "T", "abstract": ["S0.", "S1.", "S2."]}}
        sup = parse_scifact_row({"id": 1, "claim": "c", "evidence": {"1": [{"sentences": [0, 2], "label": "SUPPORT"}]},
                                 "cited_doc_ids": [1]}, corpus, "train")
        self.assertEqual(sup[0].label_common, "SUPPORTED"); self.assertEqual(sup[0].evidence, "S0. S2.")
        nei = parse_scifact_row({"id": 2, "claim": "d", "evidence": {}, "cited_doc_ids": [1]}, corpus, "train")[0]
        self.assertEqual(nei.label_common, "NOT_ENOUGH_INFO"); self.assertEqual(nei.evidence_status, "cited_abstract_no_rationale")
        st = Counter()
        self.assertEqual(parse_scifact_row({"id": 3, "claim": "hidden test claim"}, corpus, "test", st), [])
        self.assertEqual(st["scifact_unlabelled"], 1)   # missing key must NOT become NEI

    def test_liar(self):
        row = parse_liar_tsv_line("1\tpants-fire\tThe moon is cheese.\tspace\tbob\tjob\tTX\tnone\t0\t1\t0\t0\t2\ta speech")
        r = parse_liar_row(row, "train")[0]
        self.assertIsNone(r.label_common); self.assertEqual(r.label_original, "pants-fire")
        self.assertEqual(r.evidence_status, "not_provided"); self.assertFalse(r.nli_eligible)
        r2 = parse_liar_row({"id": "2", "label": 3, "statement": "s"}, "train", label_names=["false", "half-true", "mostly-true", "true", "barely-true", "pants-fire"])[0]
        self.assertEqual(r2.label_original, "true")
        with self.assertRaises(ValueError):
            parse_liar_row({"id": "3", "label": 3, "statement": "s"}, "train")


class DedupSplitTests(unittest.TestCase):
    def _v(self, claim, ev, label, ds="vitaminc", page=None, i=[0]):
        i[0] += 1
        return parse_vitaminc_row({"claim": claim, "evidence": ev, "label": label, "page": page, "unique_id": str(i[0])}, "train")[0] if ds == "vitaminc" else None

    def test_exact_dup_and_conflict(self):
        a = self._v("Cats are mammals.", "Cats are mammals.", "SUPPORTS")
        b = self._v("cats are mammals", "Cats are mammals", "SUPPORTS")
        c = self._v("Dogs bark.", "Dogs bark.", "SUPPORTS")
        d = self._v("Dogs bark.", "Dogs bark!", "REFUTES")
        kept, dropped, conflicts = dedup_exact(finalize([a, b, c, d]))
        self.assertEqual(len(kept), 1); self.assertEqual(len(dropped), 1); self.assertEqual(len(conflicts), 2)

    def test_near_dup_pairs(self):
        t = ["the quick brown fox jumps over the lazy dog today", "the quick brown fox jumps over the lazy dog",
             "completely unrelated sentence about economics and trade"]
        pairs = near_duplicate_pairs([norm_key(x) for x in t], 0.6)
        self.assertIn((0, 1), pairs); self.assertNotIn((0, 2), pairs)

    def test_groups_prevent_leakage(self):
        recs = []
        for n in range(300):   # contrastive pairs share a claim; near-dup variants share wording
            claim = f"Entity{n} scored {n} points in the final match of season {n}."
            recs.append(self._v(claim, f"Entity{n} scored {n} points.", "SUPPORTS"))
            recs.append(self._v(claim, f"Entity{n} scored {n+1} points.", "REFUTES"))
            recs.append(self._v(claim + " Reportedly.", f"Another source about entity{n}.", "NOT ENOUGH INFO"))
        recs = finalize(recs)
        groups = build_groups(recs)
        assign_splits(recs, groups, seed=5)
        rep = verify_no_leakage(recs)
        self.assertEqual(rep["violation_count"], 0, rep)
        self.assertEqual({r.split for r in recs}, {"train", "validation", "test"})
        # deterministic
        again = [r.split for r in recs]
        assign_splits(recs, build_groups(recs), seed=5)
        self.assertEqual(again, [r.split for r in recs])

    def test_leak_detector_catches_manual_leak(self):
        recs = finalize([self._v("Same claim here.", "evidence one", "SUPPORTS"), self._v("Same claim here.", "evidence two", "REFUTES")])
        recs[0].split, recs[1].split = "train", "test"
        self.assertGreater(verify_no_leakage(recs)["violation_count"], 0)


class EndToEnd(unittest.TestCase):
    def test_build(self):
        with tempfile.TemporaryDirectory() as td:
            raw, out = Path(td) / "raw", Path(td) / "out"
            (raw / "vitaminc").mkdir(parents=True); (raw / "scifact").mkdir(); (raw / "liar").mkdir(); (raw / "fever").mkdir()
            vit = [{"claim": f"Team{i} won game {i}.", "evidence": f"Team{i} won game {i} by two.", "label": "SUPPORTS", "page": f"Team {i}", "unique_id": f"v{i}"} for i in range(60)]
            vit += [{"claim": f"Team{i} won game {i}.", "evidence": f"Team{i} lost game {i}.", "label": "REFUTES", "page": f"Team {i}", "unique_id": f"w{i}"} for i in range(60)]
            vit.append(dict(vit[0], unique_id="dup"))                  # exact duplicate
            (raw / "vitaminc" / "train.jsonl").write_text("\n".join(map(json.dumps, vit)))
            corpus = [{"doc_id": i, "title": f"T{i}", "abstract": [f"Drug{i} lowers risk {i}.", "Other."]} for i in range(40)]
            (raw / "scifact" / "corpus.jsonl").write_text("\n".join(map(json.dumps, corpus)))
            claims = [{"id": i, "claim": f"Drug{i} reduces risk {i}.", "evidence": {str(i): [{"sentences": [0], "label": "SUPPORT"}]}, "cited_doc_ids": [i]} for i in range(20)]
            claims += [{"id": 100 + i, "claim": f"Drug{i} cures condition {i}.", "evidence": {}, "cited_doc_ids": [i]} for i in range(20)]
            (raw / "scifact" / "claims_train.jsonl").write_text("\n".join(map(json.dumps, claims)))
            (raw / "liar" / "train.tsv").write_text("\n".join(f"{i}\tfalse\tPolitician claims thing number {i}.\ts\tp\tj\tTX\tn\t0\t0\t0\t0\t0\tc" for i in range(30)))
            fever = [{"id": i, "label": "NOT ENOUGH INFO", "claim": f"Obscure person {i} did something.", "evidence": [[[None, None, None, None]]]} for i in range(10)]
            fever += [{"id": 50 + i, "label": "SUPPORTS", "claim": f"City{i} is in country {i}.", "evidence": [[[1, 1, f"City{i}", 0]]]} for i in range(10)]
            (raw / "fever" / "train.jsonl").write_text("\n".join(map(json.dumps, fever)))
            rep = build(raw, out)
            self.assertEqual(rep["leakage_check"]["violation_count"], 0)
            self.assertEqual(rep["exact_duplicates_dropped"], 1)
            self.assertEqual(rep["unmapped_label_records"], 30)            # all LIAR rows
            self.assertEqual(rep["evidence_status"]["unresolved"], 10)     # FEVER S/R without wiki dump
            self.assertNotIn("liar/SUPPORTED", rep["eligible_by_dataset_label"])
            self.assertFalse(any(k.startswith("fever/") for k in rep["eligible_by_dataset_label"]))
            self.assertEqual(rep["eligible_by_dataset_label"]["scifact/NOT_ENOUGH_INFO"], 20)
            for f in ("records.jsonl", "build_report.json", "conflicts.jsonl", "dropped_duplicates.jsonl"):
                self.assertTrue((out / f).exists())
            rows = [json.loads(l) for l in (out / "records.jsonl").read_text().splitlines()]
            self.assertTrue(all(r["split"] in ("train", "validation", "test") for r in rows))
            self.assertTrue(all("label_original" in r and "dataset" in r for r in rows))


if __name__ == "__main__":
    unittest.main()
