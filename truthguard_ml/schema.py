"""Common record schema. Original labels and provenance are always preserved."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

SUPPORTED, REFUTED, NEI = "SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"
COMMON_LABELS = (SUPPORTED, REFUTED, NEI)

# How the evidence field should be interpreted.
EV_PROVIDED = "provided"                      # real evidence text from the dataset
EV_NONE_BY_DEF = "none_by_definition"         # NEI claim, dataset gives no evidence
EV_UNRESOLVED = "unresolved"                  # evidence ids exist, text not available
EV_CITED_ABSTRACT = "cited_abstract_no_rationale"  # SciFact NEI: cited abstract, no rationale
EV_NOT_PROVIDED = "not_provided"              # dataset never ships evidence (LIAR)
USABLE_EVIDENCE = (EV_PROVIDED, EV_CITED_ABSTRACT)


@dataclass
class Record:
    dataset: str
    source_split: str
    source_record_id: str
    claim: str
    evidence: Optional[str]
    evidence_status: str
    label_original: str
    label_common: Optional[str]        # None = no defensible mapping
    label_scope: str                   # what the original label is relative to
    source_name: Optional[str] = None
    source_url: Optional[str] = None   # only set when derivable from a real identifier
    page_key: Optional[str] = None     # e.g. "wiki:Barack_Obama", "scifact:12345"
    evidence_ids: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # filled by finalize()
    claim_norm: str = ""
    evidence_norm: str = ""
    record_id: str = ""
    split: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def nli_eligible(self) -> bool:
        """Usable as a (claim, evidence) -> label training/eval pair."""
        return (self.label_common is not None and bool(self.evidence)
                and self.evidence_status in USABLE_EVIDENCE)
