"""Per-dataset label mapping. Nothing is mapped implicitly; unknown labels raise.

Equivalence decisions (see licenses.DATASETS[*]['label_definition']):
  FEVER    SUPPORTS/REFUTES/NEI  -> claim-level vs Wikipedia.  Mapped, scope tagged.
  VitaminC SUPPORTS/REFUTES/NEI  -> relative to ONE evidence sentence. Mapped, scope tagged.
  SciFact  SUPPORT/CONTRADICT/NEI-> relative to abstract rationale. Mapped, scope tagged.
  LIAR     6 truthfulness grades -> NOT evidence-relative. Excluded by default.
Scopes differ, so the scope is stored and reported per dataset; splits/metrics
are also reported per dataset rather than pooled only.
"""
from __future__ import annotations
from typing import Optional
from .schema import SUPPORTED, REFUTED, NEI


class UnknownLabelError(KeyError):
    pass


_MAPS = {
    "fever": {"SUPPORTS": SUPPORTED, "REFUTES": REFUTED, "NOT ENOUGH INFO": NEI},
    "vitaminc": {"SUPPORTS": SUPPORTED, "REFUTES": REFUTED, "NOT ENOUGH INFO": NEI},
    "scifact": {"SUPPORT": SUPPORTED, "CONTRADICT": REFUTED, "NOINFO": NEI},
}
SCOPES = {
    "fever": "claim_vs_wikipedia_2017_dump",
    "vitaminc": "claim_vs_single_evidence_sentence",
    "scifact": "claim_vs_abstract_rationale",
    "liar": "claim_truthfulness_no_evidence",
}
LIAR_LABELS = ("pants-fire", "false", "barely-true", "half-true", "mostly-true", "true")
# Opt-in only: map extremes, leave the middle unmapped (half-true != NEI).
_LIAR_EXTREMES = {"pants-fire": REFUTED, "false": REFUTED, "true": SUPPORTED}


def map_label(dataset: str, original: str, liar_policy: str = "exclude") -> Optional[str]:
    o = str(original).strip()
    if dataset == "liar":
        if o not in LIAR_LABELS:
            raise UnknownLabelError(f"liar: unknown label {original!r}")
        if liar_policy == "exclude":
            return None
        if liar_policy == "extremes_opt_in":
            return _LIAR_EXTREMES.get(o)
        raise ValueError(f"unknown liar_policy {liar_policy!r}")
    m = _MAPS[dataset]
    key = o.upper().replace("_", " ")
    if dataset == "scifact" and key in ("NOT ENOUGH INFO", "NEI", "NOINFO"):
        key = "NOINFO"
    if key not in m:
        raise UnknownLabelError(f"{dataset}: unknown label {original!r}")
    return m[key]
