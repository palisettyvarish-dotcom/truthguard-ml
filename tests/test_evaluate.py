import json, tempfile, unittest
from pathlib import Path

from truthguard_ml.evaluate import (nli_label_map, compute_metrics, load_pairs, run_experiment, write_report,
                                    LABELS, CAVEATS)

S, R, N = "SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"


class LabelMapTests(unittest.TestCase):
    def test_any_order_is_read_from_config(self):
        self.assertEqual(nli_label_map({0: "contradiction", 1: "entailment", 2: "neutral"}), {0: R, 1: S, 2: N})
        self.assertEqual(nli_label_map({0: "ENTAILMENT", 1: "NEUTRAL", 2: "CONTRADICTION"}), {0: S, 1: N, 2: R})

    def test_unknown_labels_rejected_not_guessed(self):
        with self.assertRaises(ValueError):
            nli_label_map({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})
        with self.assertRaises(ValueError):
            nli_label_map({0: "entailment", 1: "contradiction"})        # no neutral


class MetricTests(unittest.TestCase):
    def test_hand_computed(self):
        t = [S, S, R, R, N, N]; p = [S, R, R, R, N, S]
        m = compute_metrics(t, p, ["a"] * 6)["overall"]
        self.assertAlmostEqual(m["accuracy"], 4 / 6)
        self.assertAlmostEqual(m["per_class"][S]["precision"], 1 / 2)
        self.assertAlmostEqual(m["per_class"][S]["recall"], 1 / 2)
        self.assertAlmostEqual(m["per_class"][R]["precision"], 2 / 3)
        self.assertAlmostEqual(m["per_class"][N]["recall"], 1 / 2)
        self.assertEqual(m["confusion_matrix"]["matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(m["macro_f1"], (0.5 + 0.8 + 2 / 3) / 3, places=6)

    def test_macro_ignores_absent_classes_and_by_dataset(self):
        t = [S, R, S, R]; p = [S, R, R, R]
        m = compute_metrics(t, p, ["x", "x", "y", "y"])
        self.assertEqual(m["overall"]["macro_over"], [S, R])          # NEI absent: not counted as 0
        self.assertEqual(set(m["by_dataset"]), {"x", "y"})
        self.assertEqual(m["by_dataset"]["x"]["accuracy"], 1.0); self.assertEqual(m["by_dataset"]["y"]["accuracy"], 0.5)


def write_records(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))


def row(i, ds, split, label, status="provided", ev="evidence text"):
    return {"dataset": ds, "split": split, "claim": f"claim {i}", "evidence": ev, "label_common": label,
            "evidence_status": status, "record_id": f"r{i:04d}"}


class LoadPairsTests(unittest.TestCase):
    def test_filters_and_sampling(self):
        rows = [row(i, "vitaminc", "test", S) for i in range(50)]
        rows += [row(100, "vitaminc", "train", S), row(101, "fever", "test", R, status="unresolved", ev=None),
                 row(102, "liar", "test", None, status="not_provided", ev=None), row(103, "scifact", "test", N, status="cited_abstract_no_rationale")]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "r.jsonl"; write_records(p, rows)
            a = load_pairs(p, "test", 10); b = load_pairs(p, "test", 10)
            self.assertEqual(a, b)                                       # deterministic
            self.assertEqual(sum(x["dataset"] == "vitaminc" for x in a), 10)
            self.assertEqual({x["dataset"] for x in a}, {"vitaminc", "scifact"})   # no fever(unresolved), liar, train
            self.assertTrue(all(x["evidence"] for x in a))


class Fake:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def predict(self, pairs):
        self.calls.append(len(pairs))
        return [self.fn(p) for p in pairs], None


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        mk = lambda split: [{"dataset": "d", "claim": c, "evidence": c, "label": l} for c, l in
                            [("a", S), ("b", R), ("c", N), ("d", S), ("e", R), ("f", N)]]
        self.val, self.test = mk("v"), mk("t")

    def test_selection_uses_validation_and_test_only_for_best(self):
        truth = {"a": S, "b": R, "c": N, "d": S, "e": R, "f": N}
        good = Fake(lambda p: truth[p[1]])
        bad = Fake(lambda p: S)
        res = run_experiment({"bad": bad, "good": good}, self.val, self.test)
        self.assertEqual(res["selected_on_validation"], "good")
        self.assertEqual(list(res["test"]), ["good"])
        self.assertEqual(bad.calls, [6])                      # validation only: never saw the test set
        self.assertEqual(good.calls, [6, 6])                  # validation + test
        self.assertEqual(res["test"]["good"]["overall"]["accuracy"], 1.0)

    def test_report_files(self):
        truth = {"a": S, "b": R, "c": N, "d": S, "e": R, "f": N}
        res = run_experiment({"m<1>": Fake(lambda p: truth[p[1]])}, self.val, self.test, {"broken/model": "boom"})
        with tempfile.TemporaryDirectory() as td:
            write_report(res, Path(td), {"generated": "now", "caveats": CAVEATS})
            for f in ("metrics.json", "report.md", "report.html"):
                self.assertTrue((Path(td) / f).exists())
            h = (Path(td) / "report.html").read_text()
            self.assertIn("m&lt;1&gt;", h); self.assertNotIn("m<1>", h)      # escaped
            self.assertIn("Confusion matrix", h); self.assertIn("broken/model", h)
            self.assertIn("100.0%", (Path(td) / "report.md").read_text())

    def test_no_models_is_an_error(self):
        with self.assertRaises(RuntimeError):
            run_experiment({}, self.val, self.test)


if __name__ == "__main__":
    unittest.main()
