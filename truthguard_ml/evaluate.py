"""Pretrained NLI baseline: pick the model on VALIDATION, report on TEST only for that model.

Input is a (premise = evidence, hypothesis = claim) pair, so the model never judges a claim
without evidence. Label order is read from each model's config (never assumed).

Contamination warning: some NLI models were trained on FEVER (and others on VitaminC). Their
scores on those datasets are NOT held-out. KNOWN_TRAINING_DATA records what I remember of each
model card; verify against the card before trusting any number.
"""
from __future__ import annotations
import argparse, html, json, random, time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .schema import SUPPORTED, REFUTED, NEI, USABLE_EVIDENCE

LABELS = (SUPPORTED, REFUTED, NEI)
DEFAULT_MODELS = ["cross-encoder/nli-deberta-v3-base", "cross-encoder/nli-roberta-base",
                  "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"]
KNOWN_TRAINING_DATA = {   # from memory, UNVERIFIED
    "cross-encoder/nli-deberta-v3-base": "SNLI + MultiNLI. DeBERTa-v3.",
    "cross-encoder/nli-roberta-base": "SNLI + MultiNLI. RoBERTa.",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli": "MNLI + FEVER-NLI + ANLI. FEVER NUMBERS ARE CONTAMINATED. DeBERTa-v3.",
}


def nli_label_map(id2label: dict) -> dict:
    out = {}
    for i, name in id2label.items():
        n = str(name).lower()
        if n.startswith("entail"):
            out[int(i)] = SUPPORTED
        elif n.startswith("contradict"):
            out[int(i)] = REFUTED
        elif n.startswith("neutral"):
            out[int(i)] = NEI
        else:
            raise ValueError(f"Cannot interpret NLI label {name!r}; check the model card before using it.")
    if set(out.values()) != set(LABELS):
        raise ValueError(f"Model labels do not cover {LABELS}: {id2label}")
    return out


class HFNLIModel:
    """Hugging Face sequence-classification NLI model. Needs torch + transformers."""

    def __init__(self, name: str, device: str | None = None, max_length: int = 512, batch_size: int = 16):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch, self.name, self.max_length, self.batch_size = torch, name, max_length, batch_size
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
        self.map = nli_label_map(self.model.config.id2label)
        self.order = [next(i for i, l in self.map.items() if l == lab) for lab in LABELS]   # column per LABELS entry

    def predict(self, pairs: Sequence[tuple]):
        """pairs = [(evidence, claim)]. Returns (labels, probs[n,3] in LABELS order)."""
        torch = self.torch
        probs = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                b = pairs[i:i + self.batch_size]
                enc = self.tok([p[0] for p in b], [p[1] for p in b], truncation="only_first",
                               max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                p = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
                probs.append(p[:, self.order])
        probs = np.vstack(probs) if probs else np.zeros((0, 3))
        return [LABELS[j] for j in probs.argmax(axis=1)], probs


# ------------------------------------------------------------------ data
def load_pairs(records_path, split: str, max_per_dataset: int | None = None, seed: int = 13) -> list[dict]:
    by_ds = defaultdict(list)
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("split") == split and r.get("label_common") and r.get("evidence")
                    and r.get("evidence_status") in USABLE_EVIDENCE):
                by_ds[r["dataset"]].append({"dataset": r["dataset"], "claim": r["claim"], "evidence": r["evidence"],
                                            "label": r["label_common"], "record_id": r.get("record_id"),
                                            "evidence_status": r["evidence_status"]})
    out = []
    for ds in sorted(by_ds):
        rows = sorted(by_ds[ds], key=lambda x: x["record_id"] or "")
        random.Random(f"{seed}|{ds}|{split}").shuffle(rows)
        out += rows[:max_per_dataset] if max_per_dataset else rows
    return out


# --------------------------------------------------------------- metrics
def _block(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(LABELS), zero_division=0)
    present = [i for i in range(3) if s[i] > 0]
    return {"n": len(y_true), "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else None,
            "macro_f1": float(np.mean([f[i] for i in present])) if present else None,
            "macro_precision": float(np.mean([p[i] for i in present])) if present else None,
            "macro_recall": float(np.mean([r[i] for i in present])) if present else None,
            "macro_over": [LABELS[i] for i in present],
            "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i in range(3)},
            "confusion_matrix": {"labels": list(LABELS), "rows_are": "true", "matrix": confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()}}


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str], datasets: Sequence[str]) -> dict:
    out = {"overall": _block(list(y_true), list(y_pred)), "by_dataset": {}}
    for d in sorted(set(datasets)):
        idx = [i for i, x in enumerate(datasets) if x == d]
        out["by_dataset"][d] = _block([y_true[i] for i in idx], [y_pred[i] for i in idx])
    return out


def evaluate_model(model, pairs: list[dict]) -> dict:
    labels, _ = model.predict([(p["evidence"], p["claim"]) for p in pairs])
    return compute_metrics([p["label"] for p in pairs], labels, [p["dataset"] for p in pairs])


def run_experiment(models: dict, val_pairs: list[dict], test_pairs: list[dict], errors: dict | None = None) -> dict:
    """Validation for every model; TEST is scored for the validation-selected model only."""
    val = {name: evaluate_model(m, val_pairs) for name, m in models.items()}
    if not val:
        raise RuntimeError("No model could be evaluated.")
    best = max(val, key=lambda n: val[n]["overall"]["macro_f1"] or -1)
    return {"selected_on_validation": best, "validation": val, "test": {best: evaluate_model(models[best], test_pairs)},
            "errors": errors or {}, "n_validation": len(val_pairs), "n_test": len(test_pairs)}


# ---------------------------------------------------------------- report
def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _md_block(title: str, m: dict) -> str:
    o = m["overall"]
    s = [f"#### {title}", f"n={o['n']}  accuracy={_pct(o['accuracy'])}  macro-F1={_pct(o['macro_f1'])}  "
         f"macro-P={_pct(o['macro_precision'])}  macro-R={_pct(o['macro_recall'])}", "",
         "| Verdict | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for lab, c in o["per_class"].items():
        s.append(f"| {lab} | {_pct(c['precision'])} | {_pct(c['recall'])} | {_pct(c['f1'])} | {c['support']} |")
    s += ["", "Confusion matrix (rows = true, columns = predicted: " + ", ".join(LABELS) + ")", ""]
    s += ["    " + str(row) for row in o["confusion_matrix"]["matrix"]]
    s += ["", "| Dataset | n | Accuracy | Macro-F1 (labels present) |", "|---|---|---|---|"]
    for d, b in m["by_dataset"].items():
        s.append(f"| {d} | {b['n']} | {_pct(b['accuracy'])} | {_pct(b['macro_f1'])} |")
    return "\n".join(s)


def render_markdown(res: dict, meta: dict) -> str:
    out = ["# TruthGuard claim-verification evaluation", "",
           f"Generated {meta.get('generated')}. Validation pairs: {res['n_validation']}, test pairs: {res['n_test']}.",
           f"Model selected on validation macro-F1: **{res['selected_on_validation']}**.", "",
           "Caveats: " + " ".join(meta.get("caveats", [])), "", "## Validation (all candidate models)", ""]
    out += ["| Model | Accuracy | Macro-F1 | Training data (unverified note) |", "|---|---|---|---|"]
    for n, m in res["validation"].items():
        out.append(f"| {n} | {_pct(m['overall']['accuracy'])} | {_pct(m['overall']['macro_f1'])} | {KNOWN_TRAINING_DATA.get(n, 'unknown')} |")
    for n, e in res.get("errors", {}).items():
        out.append(f"| {n} | FAILED: {e} | | |")
    out += ["", "## Held-out test (selected model only)", ""]
    for n, m in res["test"].items():
        out.append(_md_block(n, m))
    return "\n".join(out)


def render_html(res: dict, meta: dict) -> str:
    e = html.escape
    def table(rows, head):
        return "<table><tr>" + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr>" + "".join(
            "<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in r) + "</tr>" for r in rows) + "</table>"
    parts = ["<!doctype html><meta charset=utf-8><title>TruthGuard evaluation</title><style>",
             "body{font:14px system-ui;max-width:960px;margin:2rem auto;padding:0 1rem}table{border-collapse:collapse;margin:.5rem 0}",
             "td,th{border:1px solid #ccc;padding:4px 10px;text-align:right}td:first-child,th:first-child{text-align:left}",
             ".cm td{width:90px;text-align:center}.warn{background:#fff4d6;padding:8px 12px;border-radius:6px}</style>",
             "<h1>TruthGuard claim-verification evaluation</h1>",
             f"<p>Generated {e(str(meta.get('generated')))}. Selected on validation: <b>{e(res['selected_on_validation'])}</b>. "
             f"Validation n={res['n_validation']}, test n={res['n_test']}.</p>",
             f"<p class=warn>{e(' '.join(meta.get('caveats', [])))}</p>", "<h2>Validation: all candidates</h2>",
             table([[n, _pct(m['overall']['accuracy']), _pct(m['overall']['macro_f1']), KNOWN_TRAINING_DATA.get(n, "unknown")]
                    for n, m in res["validation"].items()], ["Model", "Accuracy", "Macro-F1", "Training data (unverified)"])]
    for n, e_ in res.get("errors", {}).items():
        parts.append(f"<p>FAILED {e(n)}: {e(e_)}</p>")
    for n, m in res["test"].items():
        o = m["overall"]
        parts += [f"<h2>Held-out test: {e(n)}</h2>",
                  f"<p>n={o['n']} accuracy {_pct(o['accuracy'])}, macro-F1 {_pct(o['macro_f1'])}, macro-P {_pct(o['macro_precision'])}, macro-R {_pct(o['macro_recall'])}</p>",
                  "<h3>By verdict</h3>", table([[l, _pct(c["precision"]), _pct(c["recall"]), _pct(c["f1"]), c["support"]] for l, c in o["per_class"].items()],
                                               ["Verdict", "Precision", "Recall", "F1", "Support"]),
                  "<h3>Confusion matrix (rows true, columns predicted)</h3><table class=cm><tr><th></th>" + "".join(f"<th>{e(l)}</th>" for l in LABELS) + "</tr>"]
        mx = np.array(o["confusion_matrix"]["matrix"]); top = max(int(mx.max()), 1)
        for lab, row in zip(LABELS, mx):
            parts.append(f"<tr><th>{e(lab)}</th>" + "".join(f"<td style='background:rgba(30,100,220,{0.08 + 0.7 * v / top:.2f})'>{int(v)}</td>" for v in row) + "</tr>")
        parts += ["</table><h3>By dataset</h3>", table([[d, b["n"], _pct(b["accuracy"]), _pct(b["macro_f1"])] for d, b in m["by_dataset"].items()],
                                                      ["Dataset", "n", "Accuracy", "Macro-F1 (labels present)"])]
    return "\n".join(parts)


def write_report(res: dict, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"meta": meta, **res}, indent=2))
    (out / "report.md").write_text(render_markdown(res, meta))
    (out / "report.html").write_text(render_html(res, meta))


CAVEATS = [
    "Only datasets with real evidence text are evaluated (VitaminC, SciFact; FEVER once the Wikipedia dump is added); LIAR is excluded.",
    "Each dataset's labels are relative to different things, so read the per-dataset rows, not just the pooled score.",
    "Models trained on FEVER or VitaminC are not held-out on those datasets.",
    "Pairs use gold evidence, so this measures the verifier alone, not retrieval.",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path, default=Path("data/processed/records.jsonl"))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--max-per-dataset", type=int, default=300, help="cap per dataset per split (None-like: 0 = all)")
    ap.add_argument("--out", type=Path, default=Path("data/eval"))
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args(argv)
    cap = a.max_per_dataset or None
    val, test = load_pairs(a.records, "validation", cap), load_pairs(a.records, "test", cap)
    for name, ps in (("validation", val), ("test", test)):
        c = defaultdict(int)
        for p in ps:
            c[(p["dataset"], p["label"])] += 1
        print(name, len(ps), dict(c))
    models, errors = {}, {}
    for n in a.models:
        try:
            print("loading", n, flush=True)
            models[n] = HFNLIModel(n, batch_size=a.batch_size)
        except Exception as ex:   # missing model / unreadable labels: record and continue
            errors[n] = f"{type(ex).__name__}: {ex}"
            print("FAILED", n, errors[n])
    res = run_experiment(models, val, test, errors)
    write_report(res, a.out, {"generated": time.strftime("%F %T"), "caveats": CAVEATS, "max_per_dataset": cap, "models": a.models})
    t = res["test"][res["selected_on_validation"]]["overall"]
    print(f"selected {res['selected_on_validation']}: test accuracy {_pct(t['accuracy'])}, macro-F1 {_pct(t['macro_f1'])}")
    print(f"report: {a.out}/report.html")


if __name__ == "__main__":
    main()
