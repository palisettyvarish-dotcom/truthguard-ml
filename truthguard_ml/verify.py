"""Claim verification: gather evidence (local KB + optional live web) -> NLI per evidence -> decide.

Rules, in plain terms:
  * The verdict comes from comparing the claim with each retrieved passage (evidence is always used).
  * Historical labels attached to KB passages are never used as a verdict.
  * Strong support AND strong contradiction => NOT_ENOUGH_INFO with conflict=True (sources disagree).
  * No strong evidence either way => NOT_ENOUGH_INFO. That is "unknown", never "false".
  * Explanations are templates over real retrieved text; nothing is generated or invented.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence
from urllib.parse import unquote

from .schema import SUPPORTED, REFUTED, NEI
from .text import norm_key
from .web import WebSearchError, WebSearchProvider, filter_independent

HIST, WEB = "historical_dataset", "live_web"
LABELS = (SUPPORTED, REFUTED, NEI)          # probability column order used by every predictor


_STOP_CAPS = {"The", "A", "An", "In", "On", "At", "It", "He", "She", "They", "This", "That", "These", "Those", "Its", "His", "Her",
              "Their", "We", "I", "You", "And", "But", "Or", "If", "As", "By", "For", "From", "Of", "To", "With", "After", "Before",
              "During", "According", "However", "While", "When", "Where", "Since", "Although",
              "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
              "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}


def claim_entities(claim: str) -> set:
    """Capitalised words (a cheap proper-noun proxy), minus common sentence starters, months and weekdays."""
    toks = re.findall(r"\b[A-Z][A-Za-z0-9'\u2019\-]*", claim)
    return {re.sub(r"['\u2019]s$", "", t).lower() for t in toks if t not in _STOP_CAPS and len(t) > 1}


_SUFFIX = {"fc", "inc", "ltd", "jr", "sr", "co", "corp", "llc", "plc"}


def primary_subject(claim: str) -> list:
    """First run of consecutive capitalised words (the claim's likely subject), e.g. ['manchester', 'united'].
    Company/club suffixes (F.C., Inc.) are skipped. Evidence must mention ALL of these words."""
    run = []
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9'\u2019\-\.]*", claim):
        tok = re.sub(r"['\u2019]s$", "", m.group(0).rstrip(".,"))
        cap = tok[:1].isupper() and len(tok) > 1 and tok not in _STOP_CAPS
        if cap and tok.lower().replace(".", "") in _SUFFIX:
            continue
        if cap:
            run.append(tok.lower())
        elif run:
            break
    return run


def mentions_entity(entities: set, text: str) -> bool:
    words = set(re.findall(r"\w+", text.lower()))
    return any(set(re.findall(r"\w+", e)) <= words for e in entities)



def topic_relevant(claim: str, evidence: str) -> bool:
    """Check whether evidence discusses the claim's actual topic."""
    claim_words = set(re.findall(r"\w+", claim.lower()))
    evidence_words = set(re.findall(r"\w+", evidence.lower()))

    stop = {
        "the", "a", "an", "is", "was", "are", "were", "of",
        "in", "on", "at", "to", "for", "by", "with", "and",
        "or", "than", "more", "its", "it", "world", "worlds"
    }

    claim_words -= stop
    claim_words = {w for w in claim_words if len(w) > 2 and not w.isdigit()}

    # Remove the named entity: mentioning a place alone is not topical evidence.
    subject_words = set(primary_subject(claim))
    topic_words = claim_words - subject_words

    # "Largest democracy" requires population or electorate-related evidence.
    if {"largest", "democracy"} <= topic_words:
        return bool(evidence_words & {
            "population", "populations", "voter", "voters",
            "electorate", "electorates", "citizen", "citizens"
        })

    return bool(topic_words & evidence_words)

def _title_from(page_key: Optional[str], url: Optional[str]) -> Optional[str]:
    if page_key and ":" in page_key:
        return unquote(page_key.split(":", 1)[1]).replace("_", " ")
    if url and "wikipedia.org/wiki/" in url:
        return unquote(url.rsplit("/", 1)[-1]).replace("_", " ")
    return None


@dataclass
class VerifierConfig:
    commit_threshold: float = 0.80      # P(entail) or P(contradict) needed to count as strong evidence
    show_threshold: float = 0.50        # shown as supporting / contradicting evidence above this
    local_k: int = 8
    web_k: int = 10
    max_evidence_chars: int = 1500
    snippet_chars: int = 280
    min_sources_to_refute: int = 2      # calling a claim contradicted needs this many distinct sources (false accusations cost more than "unknown")
    require_entity_match: bool = True   # evidence must mention an entity from the claim (text or page title)


@dataclass
class Evidence:
    text: str
    source_name: Optional[str]
    source_url: Optional[str]
    origin: str                         # HIST or WEB
    published: Optional[str] = None
    detail: str = ""
    page_title: Optional[str] = None
    off_topic: bool = False
    p_support: float = 0.0
    p_refute: float = 0.0
    p_neutral: float = 0.0

    def snippet(self, n: int) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= n else t[: n - 1].rstrip() + "\u2026"


@dataclass
class ClaimResult:
    claim: str
    verdict: str
    confidence: Optional[float]
    uncertainty: str                    # "low" | "medium" | "high"
    conflict: bool
    evidence_basis: str                 # historical_dataset | live_web | both | none
    supporting: list = field(default_factory=list)
    contradicting: list = field(default_factory=list)
    neutral_count: int = 0
    checked: int = 0
    explanation: str = ""
    notes: list = field(default_factory=list)


class Verifier:
    def __init__(self, nli, kb=None, web: Optional[WebSearchProvider] = None, config: Optional[VerifierConfig] = None,
                 model_name: str = "unknown-nli"):
        self.nli, self.kb, self.web, self.cfg, self.model_name = nli, kb, web, config or VerifierConfig(), model_name

    # ------------------------------------------------------------ evidence
    def gather(self, claim: str, article_url: Optional[str] = None) -> tuple[list[Evidence], list[str]]:
        items, notes = [], []
        if self.kb is not None:
            for r in self.kb.search(claim, k=self.cfg.local_k):
                ds = ", ".join(r.get("datasets") or [])
                items.append(Evidence(r["text"], r.get("source_name"), r.get("source_url"), HIST,
                                      detail=f"Archived dataset passage ({ds}); date unknown, may be outdated.",
                                      page_title=_title_from(r.get("page_key"), r.get("source_url"))))
        if self.web is not None:
            try:
                for w in filter_independent(self.web.search(claim, self.cfg.web_k), article_url):
                    items.append(Evidence(w.snippet, w.title or w.url, w.url, WEB, w.published,
                                          detail="Live web search result (snippet only), from a different site than the article.",
                                          page_title=w.title))
            except Exception as ex:           # a failed search must not become "no evidence found"
                detail = ex.args[0] if isinstance(ex, WebSearchError) and ex.args else type(ex).__name__
                notes.append(f"Web search failed ({detail}); result is based on local evidence only.")
        return items, notes

    # -------------------------------------------------------------- decide
    def verify(self, claim: str, article_url: Optional[str] = None) -> ClaimResult:
        items, notes = self.gather(claim, article_url)
        if self.web is None:
            notes.append("Live web search is disabled; only archived dataset evidence was checked.")
        if len(claim.split()) > 30:
            notes.append("This claim bundles several facts, so a single verdict on it is less reliable.")
        if not items:
            return ClaimResult(claim, NEI, None, "high", False, "none", checked=0, notes=notes,
                               explanation="No evidence was retrieved for this claim. That does not mean it is false.")
        pairs = [(e.text[: self.cfg.max_evidence_chars], claim) for e in items]
        _, probs = self.nli.predict(pairs)
        subject = primary_subject(claim) if self.cfg.require_entity_match else []
        ents = claim_entities(claim) if self.cfg.require_entity_match else set()
        ignored = 0
        for e, p in zip(items, probs):
            e.p_support, e.p_refute, e.p_neutral = float(p[0]), float(p[1]), float(p[2])
            words = set(re.findall(r"\w+", f"{e.text} {e.page_title or ''}".lower()))
            entity_match = (
                all(w in words for w in subject)
                if subject else
                (not ents or mentions_entity(ents, f"{e.text} {e.page_title or ''}"))
            )
            topic_match = (
                topic_relevant(claim, f"{e.text} {e.page_title or ''}")
                if (subject or ents) else True
            )
            on_topic = entity_match and topic_match
            if not on_topic:
                e.off_topic, e.p_support, e.p_refute, e.p_neutral = True, 0.0, 0.0, 1.0   # about something else: cannot support or contradict
                ignored += 1
        if ignored:
            notes.append(f"{ignored} retrieved passage(s) were about a different subject than the claim and were not used.")
        return self.decide(claim, items, notes)

    def decide(self, claim: str, items: list[Evidence], notes: list[str]) -> ClaimResult:
        c = self.cfg
        strong_s = [e for e in items if e.p_support >= c.commit_threshold]
        strong_r = [e for e in items if e.p_refute >= c.commit_threshold]
        sup = self._unique([e for e in items if e.p_support >= c.show_threshold], lambda e: e.p_support)[:3]
        ref = self._unique([e for e in items if e.p_refute >= c.show_threshold], lambda e: e.p_refute)[:3]
        neutral = sum(1 for e in items if e.p_support < c.show_threshold and e.p_refute < c.show_threshold)
        conflict = bool(strong_s and strong_r)
        weak_refute = False
        if conflict or not (strong_s or strong_r):
            verdict, win = NEI, []
        elif strong_s:
            verdict, win = SUPPORTED, strong_s
        elif len({(e.source_url or e.source_name) for e in strong_r}) >= c.min_sources_to_refute:
            verdict, win = REFUTED, strong_r
        else:
            verdict, win, weak_refute = NEI, [], True
            notes.append("One retrieved passage appears to contradict this claim, but that is not enough sources to call it contradicted.")
        origins = {e.origin for e in (win or items)}
        basis = "both" if len(origins) == 2 else next(iter(origins)) if origins else "none"
        best = max((e.p_support if verdict == SUPPORTED else e.p_refute for e in win), default=None)
        distinct = {(e.source_url or e.source_name) for e in win}
        opposite = (ref if verdict == SUPPORTED else sup) if verdict != NEI else []
        if verdict == NEI or conflict:
            unc = "high"
        elif len(distinct) >= 2 and best >= 0.90 and not opposite:
            unc = "low"
        else:
            unc = "medium" if best >= 0.90 or len(distinct) >= 2 else "high"
        if verdict != NEI and basis == HIST:
            notes.append("This verdict rests only on archived dataset evidence (older Wikipedia/scientific text); it may be outdated.")
            unc = "high" if unc == "medium" else "medium" if unc == "low" else unc
        if verdict != NEI and opposite:
            notes.append("Some retrieved evidence points the other way, so treat this verdict with caution.")
        res = ClaimResult(claim, verdict, best, unc, conflict, basis, sup, ref, neutral, len(items), notes=notes)
        res.explanation = self.explain(res, win, weak_refute)
        return res

    def _unique(self, evs, prob):
        """One entry per (source, visible snippet): near-identical passages are not shown or counted twice."""
        seen, out = set(), []
        for e in sorted(evs, key=lambda e: -prob(e)):
            k = (e.source_url or e.source_name, norm_key(e.snippet(self.cfg.snippet_chars)))
            if k not in seen:
                seen.add(k)
                out.append(e)
        return out

    # ------------------------------------------------------------- explain
    def explain(self, r: ClaimResult, win: list[Evidence], weak_refute: bool = False) -> str:
        n = self.cfg.snippet_chars
        tag = lambda e: "live web" if e.origin == WEB else "archived dataset"
        cite = lambda e: f'{e.source_name or "unnamed source"} ({tag(e)}): "{e.snippet(n)}"'
        if r.conflict:
            return ("Sources disagree. Supporting: " + cite(r.supporting[0]) + " Contradicting: " + cite(r.contradicting[0])
                    + " No verdict is given because the evidence conflicts.")
        if r.verdict == SUPPORTED:
            return f"{len(win)} retrieved passage(s) support this claim. Strongest: " + cite(win[0] if not r.supporting else r.supporting[0])
        if r.verdict == REFUTED:
            return f"{len(win)} retrieved passage(s) contradict this claim. Strongest: " + cite(win[0] if not r.contradicting else r.contradicting[0])
        if weak_refute:
            return ("Possible contradiction, not confirmed. " + cite(r.contradicting[0]) + " Only one source points this way, so no verdict is given. "
                    "This does not mean the claim is false.")
        return (f"None of the {r.checked} retrieved passage(s) clearly supports or contradicts this claim. "
                "This means it could not be verified here; it does not mean the claim is false.")
