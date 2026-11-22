"""Heuristic claim extraction. Picks checkable-looking sentences; it does not judge them.

Limits (by design, stated in the API's `limitations`): rule-based, English only, no coreference.
Sentences that depend on earlier context ("He said it rose 5%") are skipped because a claim
must stand alone to be checked against evidence.
"""
from __future__ import annotations
import re
from .text import clean_text, norm_key

_ABBR = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "vs.", "etc.", "U.S.", "U.K.", "Inc.", "Ltd.", "No.", "Gen.", "Sen.", "Rep.", "Gov.", "Lt.", "Col.", "Mt."]
_SPLIT = re.compile(r'(?<=[.!?])["\u201d\']?\s+(?=["\u201c\']?[A-Z0-9])')
_FACT_VERBS = re.compile(r"\b(is|are|was|were|has|have|had|said|says|announced|found|reported|increased|decreased|rose|fell|dropped|killed|won|lost|born|died|signed|passed|approved|banned|reached|totall?ed|averaged|doubled|tripled|cut|raised|launched|released|confirmed|showed|shows|estimated|caused|causes|reduced|reduces|costs?|earned|elected|appointed|founded|built|located|ranks?|ranked)\b", re.I)
_OPINION = re.compile(r"\b(i think|i believe|in my opinion|we believe|should|must|ought to|beautiful|amazing|terrible|awful|best|worst|great|horrible|wonderful|disgrace|outrageous|unbelievable)\b", re.I)
_HEDGE = re.compile(r"\b(may|might|could|perhaps|possibly|allegedly|reportedly|some say|rumou?red)\b", re.I)
_DEPENDENT_START = re.compile(r"^(he|she|they|it|this|that|these|those|his|her|their|its|however|but|and|also|such)\b", re.I)
_NUM = re.compile(r"\d")
_STAT = re.compile(r"(\d+\s?%|\$\s?\d|\u00a3\s?\d|\u20ac\s?\d|\b(19|20)\d{2}\b|\bpercent\b|\bmillion\b|\bbillion\b)", re.I)
_CAP = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+")


def split_sentences(text: str) -> list[str]:
    t = clean_text(text) or ""
    for i, a in enumerate(_ABBR):
        t = t.replace(a, a.replace(".", f"\u2024{i}\u2024"))
    parts = _SPLIT.split(t)
    out = []
    for p in parts:
        for i, a in enumerate(_ABBR):
            p = p.replace(a.replace(".", f"\u2024{i}\u2024"), a)
        p = p.strip()
        if p:
            out.append(p)
    return out


def _category(s: str) -> str:
    if re.search(r'["\u201c].+["\u201d]', s) or re.search(r"\b(said|says|told|claimed|stated)\b", s, re.I):
        return "Attributed statement"
    if _STAT.search(s) or _NUM.search(s):
        return "Numerical / statistical claim"
    return "Factual statement"


def score_sentence(s: str) -> int:
    words = s.split()
    if len(words) < 6 or len(words) > 60 or s.endswith("?"):
        return -99
    score = 0
    if _NUM.search(s):
        score += 2
    if _STAT.search(s):
        score += 1
    caps = len(_CAP.findall(s))
    score += 2 if caps >= 2 else 1 if caps == 1 else 0
    if _FACT_VERBS.search(s):
        score += 2
    if _OPINION.search(s):
        score -= 3
    if _HEDGE.search(s):
        score -= 1
    if _DEPENDENT_START.match(s):
        score -= 3
    return score


def extract_claims(text: str, max_claims: int = 10, min_score: int = 3) -> list[dict]:
    seen, cands = set(), []
    for idx, s in enumerate(split_sentences(text)):
        sc = score_sentence(s)
        key = norm_key(s)
        if sc >= min_score and key and key not in seen:
            seen.add(key)
            cands.append((sc, idx, s))
    top = sorted(cands, key=lambda c: (-c[0], c[1]))[:max_claims]
    top.sort(key=lambda c: c[1])                       # keep article order
    return [{"claim_id": f"c{i + 1}", "text": s, "category": _category(s), "score": sc, "sentence_index": idx}
            for i, (sc, idx, s) in enumerate(top)]
