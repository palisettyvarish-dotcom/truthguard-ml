"""Framework-free service layer: builds the extension's report and the clean verify-claim response."""
from __future__ import annotations
import re
from collections import Counter
from typing import Optional
from urllib.parse import urlparse

from .claims import extract_claims, split_sentences
from .schema import SUPPORTED, REFUTED, NEI
from .verify import Verifier, ClaimResult, Evidence, HIST, WEB

CLASSIFICATION = {SUPPORTED: "Supported by evidence", REFUTED: "Contradicted by evidence", NEI: "Unverified / insufficient evidence"}
MIXED = "Misleading or missing context"      # used for conflicting evidence
LIMITATIONS = [
    "Automated evidence matching, not a final judgment. It can be wrong.",
    "Claim extraction is rule-based and English-only; sentences that depend on earlier context are skipped.",
    "Archived dataset evidence is older text (e.g. Wikipedia snapshots) and may be outdated.",
    "NOT_ENOUGH_INFO means the claim could not be verified, not that it is false.",
]
ATTRIBUTION_NOTE = ("This reports what someone said. Evidence that it was said is separate from evidence about whether the statement is true; "
                    "the verdict above is about the content of the statement.")
_SENSATIONAL = re.compile(r"\b(shocking|unbelievable|you won't believe|breaking|outrage|bombshell|destroyed|slams|explosive|terrifying|miracle|secret|exposed|insane)\b", re.I)


def _evidence_json(e: Evidence, prob: float, snippet_chars: int = 280) -> dict:
    return {"source_name": f"{e.source_name or 'Unnamed source'} [{'live web' if e.origin == WEB else 'archived dataset'}]",
            "source_url": e.source_url, "snippet": e.snippet(snippet_chars), "published_date": e.published,
            "divergence_notes": e.detail, "origin": e.origin, "nli_probability": round(prob, 4)}


def claim_json(r: ClaimResult, claim_id: str = "c1", category: str = "Factual statement") -> dict:
    cls = MIXED if r.conflict else CLASSIFICATION[r.verdict]
    missing = None
    if r.conflict:
        missing = "Retrieved sources disagree about this claim."
    elif r.notes:
        missing = " ".join(r.notes)
    return {
        "claim_id": claim_id, "claim_text": r.claim, "category": category, "classification": cls, "explanation": r.explanation,
        "missing_context": missing,
        "supporting_evidence": [_evidence_json(e, e.p_support) for e in r.supporting],
        "contradicting_evidence": [_evidence_json(e, e.p_refute) for e in r.contradicting],
        # additive fields (the side panel ignores these)
        "verdict": r.verdict, "confidence": None if r.confidence is None else round(r.confidence, 4), "uncertainty": r.uncertainty,
        "conflicting_evidence": r.conflict, "evidence_basis": r.evidence_basis, "evidence_checked": r.checked,
        "neutral_evidence_count": r.neutral_count,
    }


class FactCheckService:
    def __init__(self, verifier: Verifier, max_claims: int = 8, engine_name: Optional[str] = None):
        self.v, self.max_claims = verifier, max_claims
        self.engine = engine_name or f"TruthGuard Evidence Engine ({verifier.model_name})"

    # ----------------------------------------------------------- endpoints
    def health(self) -> dict:
        kb = self.v.kb
        return {"status": "ok", "gemini_active": False, "default_mode": "Local Evidence Engine", "engine": self.engine,
                "knowledge_base": None if kb is None else {"passages": kb.meta.get("n_passages"), "embedder": kb.meta.get("embedder"),
                                                          "splits": kb.meta.get("splits"), "dense_backend": kb.dense.backend},
                "web_search_enabled": self.v.web is not None, "web_search_provider": getattr(self.v.web, "name", None), "nli_model": self.v.model_name}

    def verify_claim(self, claim: str, context_url: Optional[str] = None) -> dict:
        r = self.v.verify(claim, context_url)
        sources = [e.source_url for e in r.supporting + r.contradicting if e.source_url]
        return {"claim": claim, "verdict": r.verdict, "confidence": None if r.confidence is None else round(r.confidence, 4),
                "uncertainty": r.uncertainty, "conflicting_evidence": r.conflict, "evidence_basis": r.evidence_basis,
                "supporting_evidence": claim_json(r)["supporting_evidence"], "contradicting_evidence": claim_json(r)["contradicting_evidence"],
                "neutral_evidence_count": r.neutral_count, "evidence_checked": r.checked, "explanation": r.explanation,
                "sources": sorted(set(sources)), "notes": r.notes, "engine": self.engine, "limitations": LIMITATIONS}

    def analyze_claim(self, claim_text: str, context_url: Optional[str] = None, article_title: Optional[str] = None) -> dict:
        art = {"url": context_url or "", "title": article_title or "Selected text", "content": claim_text,
               "headline_evaluable": bool(article_title)}
        if len(split_sentences(claim_text)) >= 2:          # a pasted/selected passage: check its claims one by one
            claims = extract_claims(claim_text, self.max_claims)
            if claims:
                return self._report(art, [(c["text"], c["category"], self.v.verify(c["text"], context_url)) for c in claims])
        r = self.v.verify(claim_text, context_url)
        return self._report(art, [(claim_text, "Factual statement", r)])

    def analyze(self, payload: dict) -> dict:
        selected = (payload.get("selected_text") or "").strip()
        body = selected if len(selected) >= 20 else (payload.get("content") or "")
        claims = extract_claims(body, self.max_claims)
        if not claims and selected:                      # an explicit selection is checked as written
            claims = [{"claim_id": "c1", "text": selected[:500], "category": "Factual statement"}]
        results = [(c["text"], c["category"], self.v.verify(c["text"], payload.get("url"))) for c in claims]
        return self._report(payload, results)

    # -------------------------------------------------------------- report
    def _headline(self, title: str, body: str) -> dict:
        n_caps = sum(1 for w in title.split() if len(w) > 3 and w.isupper())
        hits = len(_SENSATIONAL.findall(title)) + title.count("!") + n_caps
        score = min(100, 15 * hits)
        flags = []
        if _SENSATIONAL.search(title):
            flags.append("Emotive wording in headline")
        if title.count("!"):
            flags.append("Exclamation marks in headline")
        if n_caps:
            flags.append("ALL-CAPS words in headline")
        stance, expl = "Consistent", "No contradiction between the headline and the article text was detected."
        if not title or len(body) < 200:
            return {"sensationalism_score": score, "emotional_tone": "Emotive" if score >= 30 else "Neutral",
                    "headline_vs_content_stance": stance, "flags": flags,
                    "explanation": "Headline consistency was NOT evaluated: no headline and article body were available to compare "
                                   "(for example, selected text or a typed claim). Sensationalism is a simple wording heuristic."}
        if body and len(body) > 200:
            try:
                _, p = self.v.nli.predict([(body[:3000], title)])
                if p[0][1] >= 0.8:
                    stance, expl = "Contradicted", "The article text appears to contradict the headline."
                    flags.append("Headline may contradict the article")
                elif p[0][0] < 0.5:
                    expl = "The headline is not clearly supported by the opening of the article text, but is not contradicted either."
                    flags.append("Headline not directly supported by article text")
            except Exception:
                expl = "Headline consistency could not be evaluated."
        return {"sensationalism_score": score, "emotional_tone": "Emotive" if score >= 30 else "Neutral",
                "headline_vs_content_stance": stance, "explanation": expl + " Sensationalism is a simple wording heuristic.", "flags": flags}

    def _report(self, art: dict, results: list) -> dict:
        claims = [claim_json(r, f"c{i + 1}", cat) | {"claim_text": txt} for i, (txt, cat, r) in enumerate(results)]
        for c in claims:
            if c["category"] == "Attributed statement":
                c["missing_context"] = ((c["missing_context"] + " ") if c["missing_context"] else "") + ATTRIBUTION_NOTE
        rs = [r for _, _, r in results]
        n = Counter()
        for r in rs:
            n["conflict" if r.conflict else r.verdict] += 1
        total = len(rs)
        corroborated = sum(1 for r in rs if r.verdict != NEI and r.evidence_basis in (WEB, "both"))
        refuted_firm = sum(1 for r in rs if r.verdict == REFUTED and r.uncertainty != "high")
        if total == 0 or (n[SUPPORTED] == 0 and n[REFUTED] == 0 and n["conflict"] == 0):
            overall = "Insufficient Evidence"
        elif refuted_firm >= 2 and refuted_firm >= n[SUPPORTED]:
            overall = "Significant Misinformation Detected"
        elif n[REFUTED] or n["conflict"] or n[SUPPORTED] / total < 0.6 or corroborated == 0:   # archived evidence alone never earns "Strong"
            overall = "Partially Supported with Context Needed"
        else:
            overall = "Strong Evidence Support"
        flagged = [c["claim_text"] for c, r in zip(claims, rs) if r.verdict == REFUTED or r.conflict]
        hist_only = sum(1 for r in rs if r.verdict != NEI and r.evidence_basis == HIST)
        context = []
        if self.v.web is None:
            context.append("Live web search was off, so evidence comes only from archived datasets.")
        if hist_only:
            context.append(f"{hist_only} verdict(s) rely only on archived dataset evidence that may be outdated.")
        if total == 0:
            context.append("No checkable factual claims were found in the text.")
        url, title = art.get("url") or "", art.get("title") or ""
        text = art.get("content") or ""
        return {
            "article_title": title, "source_url": url, "domain": urlparse(url).netloc or "unknown", "author": art.get("author"),
            "published_date": art.get("published_date"), "engine": self.engine, "is_demo_mode": False,
            "has_warning": bool(flagged), "warning_reason": (f"{len(flagged)} claim(s) are contradicted by retrieved evidence or have conflicting sources. "
                                                             "This is automated evidence matching, not a final judgment.") if flagged else "",
            "warning_affected_claims": flagged,
            "overall_evidence_assessment": overall,
            "assessment_summary": f"Checked {total} claim(s): {n[SUPPORTED]} supported, {n[REFUTED]} contradicted, {n['conflict']} with conflicting evidence, {n[NEI]} unverified.",
            "headline_analysis": self._headline(title if art.get("headline_evaluable", True) else "", text),
            "summary": {"tldr": f"{total} claim(s) checked against retrieved evidence. {n[SUPPORTED]} supported, {n[REFUTED]} contradicted, {n['conflict']} conflicting, {n[NEI]} could not be verified.",
                        "verified_elements": [c["claim_text"] for c in claims if c["verdict"] == SUPPORTED and not c["conflicting_evidence"]][:6],
                        "uncertain_or_disputed": [c["claim_text"] for c in claims if c["verdict"] == NEI][:6],
                        "important_missing_context": context},
            "source_transparency": {"domain_trust_indicator": "Not assessed", "publisher_type": "Not classified",
                                    "has_author_byline": bool(art.get("author")), "author_name": art.get("author"),
                                    "citations_count": len(re.findall(r"https?://", text)),
                                    "has_publication_date": bool(art.get("published_date")), "publication_date": art.get("published_date"),
                                    "corroboration_status": f"{corroborated} of {total} claim(s) corroborated by independent live web sources",
                                    "transparency_notes": ["Source reputation is not rated; a domain alone does not prove truth or falsity.",
                                                           "Link count covers URLs visible in the extracted text only."] + LIMITATIONS},
            "claims": claims, "limitations": LIMITATIONS,
        }
