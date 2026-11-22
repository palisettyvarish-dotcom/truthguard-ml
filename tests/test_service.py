import unittest
import numpy as np

from truthguard_ml.claims import extract_claims, split_sentences
from truthguard_ml.schema import SUPPORTED, REFUTED, NEI
from truthguard_ml.service import FactCheckService, CLASSIFICATION, MIXED
from truthguard_ml.verify import Verifier, VerifierConfig, HIST, WEB
from truthguard_ml.web import WebResult, filter_independent, registrable_domain


class FakeNLI:
    """Keyword-driven: 'SUPPORTS' / 'CONTRADICTS' in the evidence text decides the probabilities."""

    def predict(self, pairs):
        out = []
        for ev, claim in pairs:
            if "SUPPORTS" in ev:
                out.append([0.95, 0.02, 0.03])
            elif "CONTRADICTS" in ev:
                out.append([0.03, 0.95, 0.02])
            elif "WEAKSUP" in ev:
                out.append([0.60, 0.10, 0.30])
            else:
                out.append([0.10, 0.10, 0.80])
        p = np.array(out)
        return [("SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO")[i] for i in p.argmax(1)], p


class FakeKB:
    def __init__(self, passages):
        self.passages = passages
        self.meta = {"n_passages": len(passages), "embedder": "fake", "splits": ["train"]}
        self.dense = type("D", (), {"backend": "numpy"})()

    def search(self, q, k=10, **kw):
        return self.passages[:k]


def P(text, url="https://en.wikipedia.org/wiki/X", name="Wikipedia (VitaminC revision)", ds=("vitaminc",), page_key=None):
    return {"text": text, "source_name": name, "source_url": url, "datasets": list(ds), "origin": HIST, "page_key": page_key}


class FakeWeb:
    name = "fake"

    def __init__(self, results=None, fail=False):
        self.results, self.fail, self.queries = results or [], fail, []

    def search(self, query, k):
        self.queries.append(query)
        if self.fail:
            raise TimeoutError("boom")
        return self.results


def svc(passages=(), web=None, **kw):
    kw.setdefault("require_entity_match", False)      # most tests use generic evidence; subject-gate tests turn it on
    v = Verifier(FakeNLI(), FakeKB(list(passages)) if passages is not None else None, web, VerifierConfig(**kw), "fake-nli")
    return FactCheckService(v)


ARTICLE = ("The city council approved a budget of $4.2 million on March 3, 2024. Dr. Smith said unemployment fell to 4.1% in 2023. "
           "What do you think? It was amazing, I think. He said it rose. The Eiffel Tower is located in Paris, France.")


class ClaimExtraction(unittest.TestCase):
    def test_keeps_checkable_skips_rest(self):
        cs = extract_claims(ARTICLE)
        texts = [c["text"] for c in cs]
        self.assertEqual(len(cs), 3)
        self.assertTrue(all("amazing" not in t and "What do you" not in t and not t.startswith("He ") for t in texts))
        self.assertEqual([c["text"] for c in cs], [t for t in split_sentences(ARTICLE) if t in texts])   # article order
        self.assertEqual(cs[1]["category"], "Attributed statement")

    def test_abbreviations_do_not_split_and_limit(self):
        self.assertEqual(len(split_sentences("Dr. Smith met Mr. Jones in the U.S. last year.")), 1)
        self.assertLessEqual(len(extract_claims(ARTICLE, max_claims=2)), 2)
        self.assertEqual(extract_claims(""), [])


class WebFiltering(unittest.TestCase):
    def test_domains(self):
        self.assertEqual(registrable_domain("https://www.news.bbc.co.uk/a"), "bbc.co.uk")
        self.assertEqual(registrable_domain("https://sub.example.com/x"), "example.com")
        rs = [WebResult("a", "https://www.site.com/1", "s"), WebResult("b", "https://site.com/2", "s"),
              WebResult("c", "https://other.org/3", "s"), WebResult("d", "https://third.net/4", " ")]
        out = filter_independent(rs, "https://news.site.com/article")
        self.assertEqual([r.url for r in out], ["https://other.org/3"])     # own domain + empty snippet dropped



    def test_source_priority(self):
        rs = [
            WebResult("Social", "https://instagram.com/post", "social evidence"),
            WebResult("News", "https://reuters.com/story", "news evidence"),
            WebResult("Government", "https://example.gov.in/report", "official evidence"),
        ]

        out = filter_independent(rs, None)

        self.assertEqual(
            [r.title for r in out],
            ["Government", "News", "Social"],
        )

class Decisions(unittest.TestCase):
    def test_supported_historical_only_is_flagged_outdated_risk(self):
        r = svc([P("Paris hosts the tower. SUPPORTS")]).v.verify("The Eiffel Tower is in Paris.")
        self.assertEqual(r.verdict, SUPPORTED); self.assertEqual(r.evidence_basis, HIST)
        self.assertIn(r.uncertainty, ("medium", "high")); self.assertTrue(any("outdated" in n for n in r.notes))
        self.assertIn("Paris hosts the tower. SUPPORTS", r.explanation)          # quotes real evidence, verbatim

    def test_refuted_needs_two_sources(self):
        r = svc([P("It is in Rome. CONTRADICTS"), P("Also Rome. CONTRADICTS", url="https://b.org")]).v.verify("claim")
        self.assertEqual((r.verdict, r.conflict), (REFUTED, False)); self.assertEqual(len(r.contradicting), 2)
        one = svc([P("It is in Rome. CONTRADICTS")]).v.verify("claim")
        self.assertEqual(one.verdict, NEI)                                      # a single passage cannot call a claim contradicted
        self.assertEqual(len(one.contradicting), 1)                             # but it is still shown for the reader to judge
        self.assertIn("not confirmed", one.explanation); self.assertIn("does not mean the claim is false", one.explanation)
        self.assertEqual(one.uncertainty, "high")

    def test_conflict_is_not_a_verdict(self):
        r = svc([P("one SUPPORTS"), P("two CONTRADICTS", url="https://x.org")]).v.verify("claim")
        self.assertEqual((r.verdict, r.conflict, r.uncertainty), (NEI, True, "high"))
        self.assertIn("disagree", r.explanation)

    def test_unknown_is_not_false(self):
        r = svc([P("unrelated text")]).v.verify("claim")
        self.assertEqual(r.verdict, NEI); self.assertFalse(r.conflict); self.assertIn("does not mean the claim is false", r.explanation)
        self.assertEqual(r.checked, 1)

    def test_weak_evidence_below_threshold_not_committed(self):
        r = svc([P("WEAKSUP only")]).v.verify("claim")            # P(support)=0.60 < 0.80
        self.assertEqual(r.verdict, NEI); self.assertEqual(len(r.supporting), 1)     # shown, but not a verdict

    def test_no_evidence_at_all(self):
        r = svc([]).v.verify("claim")
        self.assertEqual((r.verdict, r.evidence_basis, r.confidence, r.checked), (NEI, "none", None, 0))

    def test_two_independent_live_sources_give_low_uncertainty(self):
        web = FakeWeb([WebResult("A", "https://a.com/1", "report SUPPORTS it"), WebResult("B", "https://b.org/2", "other SUPPORTS it")])
        r = svc([], web).v.verify("claim", "https://article.net/x")
        self.assertEqual((r.verdict, r.evidence_basis, r.uncertainty), (SUPPORTED, WEB, "low"))

    def test_high_threshold_forces_abstain(self):
        r = svc([P("a SUPPORTS"), P("b SUPPORTS", url="https://b.org"), P("c WEAKCON CONTRADICTS", url="https://c.org")],
                commit_threshold=0.99).v.verify("claim")
        self.assertEqual(r.verdict, NEI)                           # nothing reaches 0.99 -> abstain


class SubjectGate(unittest.TestCase):
    CLAIM = "Manchester had a population of more than 540,000 in 2017."

    def test_wrong_city_cannot_create_a_false_conflict(self):
        """Regression: a passage about Virginia Beach was read as contradicting a claim about Manchester."""
        good = P("Manchester is a city with a population of 545,500 as of 2017. SUPPORTS", url="https://en.wikipedia.org/wiki/Manchester", page_key="wiki:Manchester")
        dup = P("Manchester is a city with a population of 545,500 as of 2017. SUPPORTS", url="https://en.wikipedia.org/wiki/Manchester", page_key="wiki:Manchester")
        bad = P("In 2017 the estimated population was 450,435. CONTRADICTS", url="https://en.wikipedia.org/wiki/Virginia_Beach,_Virginia", page_key="wiki:Virginia_Beach,_Virginia")
        r = svc([good, dup, bad], require_entity_match=True).v.verify(self.CLAIM)
        self.assertEqual((r.verdict, r.conflict), (SUPPORTED, False))
        self.assertEqual(r.contradicting, [])
        self.assertEqual(len(r.supporting), 1)                                   # duplicates collapsed
        self.assertTrue(any("different subject" in n for n in r.notes))
        self.assertEqual(r.neutral_count, 1)

    def test_page_title_rescues_on_topic_evidence(self):
        ev = P("In 2017 the estimated population was 545,500. SUPPORTS", url="https://en.wikipedia.org/wiki/Manchester", page_key="wiki:Manchester")
        r = svc([ev], require_entity_match=True).v.verify(self.CLAIM)
        self.assertEqual(r.verdict, SUPPORTED)

    def test_gate_off_for_claims_without_named_entities_and_for_months(self):
        r = svc([P("something SUPPORTS")], require_entity_match=True).v.verify("the sky is blue today because of scattering")
        self.assertEqual(r.verdict, SUPPORTED)                                   # no entity -> gate skipped
        from truthguard_ml.verify import claim_entities
        self.assertEqual(claim_entities("On March 3, 2024 the Council approved it."), {"council"})
        self.assertEqual(claim_entities("Nikolaj Coster-Waldau's films."), {"nikolaj", "coster-waldau"})

    def test_subject_mention_alone_is_not_relevant(self):
        ev = P(
            "Manchester hosted a music festival in 2017. SUPPORTS",
            url="https://example.org/manchester-festival",
            page_key="festival:Manchester",
        )
        r = svc([ev], require_entity_match=True).v.verify(self.CLAIM)
        self.assertEqual(r.verdict, NEI)
        self.assertEqual(r.supporting, [])
        self.assertTrue(any("different subject" in n for n in r.notes))

    def test_unrelated_only_evidence_is_unknown_not_false(self):
        r = svc([P("Paris is lovely. CONTRADICTS", page_key="wiki:Paris")], require_entity_match=True).v.verify(self.CLAIM)
        self.assertEqual((r.verdict, r.conflict), (NEI, False))


class SameFamilyDifferentSubject(unittest.TestCase):
    """Regression from the live panel: a true claim about Manchester United was 'contradicted' by a Manchester City passage."""
    CLAIM = ("Manchester United F.C. is a Premier League football club based at Old Trafford in Greater Manchester, England, "
             "founded in 1878 as Newton Heath LYR F.C. before adopting its current name in 1902")

    def test_primary_subject(self):
        from truthguard_ml.verify import primary_subject
        self.assertEqual(primary_subject(self.CLAIM), ["manchester", "united"])
        self.assertEqual(primary_subject("In 2017, Manchester had a population of 545,500."), ["manchester"])
        self.assertEqual(primary_subject("The Eiffel Tower is in Paris."), ["eiffel", "tower"])
        self.assertEqual(primary_subject("the sky is blue"), [])

    def test_other_club_cannot_refute(self):
        city = P("Manchester City Football Club is a football club based in Manchester, England, that competes in the Premier League. CONTRADICTS",
                 url="https://en.wikipedia.org/wiki/Manchester_City_F.C.", page_key="wiki:Manchester_City_F.C.")
        city2 = P("Manchester City play at the Etihad. CONTRADICTS", url="https://en.wikipedia.org/wiki/Etihad", page_key="wiki:Etihad_Stadium_(Manchester)")
        r = svc([city, city2], require_entity_match=True).v.verify(self.CLAIM)
        self.assertEqual((r.verdict, r.contradicting), (NEI, []))
        self.assertTrue(any("different subject" in n for n in r.notes))
        self.assertTrue(any("bundles several facts" in n for n in r.notes))
        mu = P("Manchester United was founded in 1878 as Newton Heath LYR. SUPPORTS", url="https://en.wikipedia.org/wiki/Manchester_United_F.C.", page_key="wiki:Manchester_United_F.C.")
        r2 = svc([city, mu], require_entity_match=True).v.verify(self.CLAIM)
        self.assertEqual((r2.verdict, r2.conflict), (SUPPORTED, False))

    def test_unwarned_when_only_weak_refutation(self):
        rep = svc([P("It is elsewhere. CONTRADICTS")]).analyze({"url": "u", "title": "t", "content": ARTICLE})
        self.assertFalse(rep["has_warning"])                                    # a lone passage never raises the misleading-content banner
        self.assertTrue(all(c["classification"] == "Unverified / insufficient evidence" for c in rep["claims"]))


class BraveProviderTests(unittest.TestCase):
    """Network is mocked: these check parsing and error handling, not that Brave itself accepts the request."""

    def _provider(self, **kw):
        from truthguard_ml.web import BraveSearchProvider
        return BraveSearchProvider(api_key="SECRET-KEY-123", min_interval=0, **kw)

    def test_parses_results_and_strips_html(self):
        import io, json
        from unittest import mock
        body = json.dumps({"web": {"results": [{"title": "A <b>title</b>", "url": "https://a.com/x", "description": "some <strong>text</strong>", "page_age": "2026-11-01"},
                                               {"title": "no url", "description": "skip"}]}}).encode()

        class Resp(io.BytesIO):
            def __enter__(self): return self
            def __exit__(self, *a): return False
        with mock.patch("truthguard_ml.web.urllib.request.urlopen", return_value=Resp(body)) as m:
            out = self._provider().search("claim text", 5)
        self.assertEqual([(r.title, r.snippet, r.published) for r in out], [("A title", "some text", "2026-11-01")])
        sent = m.call_args[0][0]
        self.assertIn("claim+text", sent.full_url); self.assertNotIn("SECRET-KEY-123", sent.full_url)   # key only in a header
        self.assertEqual(sent.get_header("X-subscription-token"), "SECRET-KEY-123")

    def test_http_errors_are_explained_without_leaking_the_key(self):
        import io, urllib.error
        from unittest import mock
        from truthguard_ml.web import WebSearchError
        err = urllib.error.HTTPError("https://x", 422, "Unprocessable", {}, io.BytesIO(b'{"error":{"code":"SUBSCRIPTION_TOKEN_INVALID"}}'))
        with mock.patch("truthguard_ml.web.urllib.request.urlopen", side_effect=err):
            with self.assertRaises(WebSearchError) as cm:
                self._provider().search("claim", 3)
        msg = str(cm.exception)
        self.assertIn("HTTP 422", msg); self.assertIn("SUBSCRIPTION_TOKEN_INVALID", msg); self.assertIn("not subscribed", msg)
        self.assertNotIn("SECRET-KEY-123", msg)

    def test_rate_limit_retries_once_then_reports(self):
        import io, urllib.error
        from unittest import mock
        from truthguard_ml.web import WebSearchError
        mk = lambda: urllib.error.HTTPError("https://x", 429, "Too Many", {}, io.BytesIO(b"{}"))
        with mock.patch("truthguard_ml.web.urllib.request.urlopen", side_effect=[mk(), mk()]) as m, mock.patch("truthguard_ml.web.time.sleep"):
            with self.assertRaises(WebSearchError) as cm:
                self._provider().search("claim", 3)
        self.assertEqual(m.call_count, 2); self.assertIn("HTTP 429", str(cm.exception))

    def test_requests_are_spaced(self):
        from unittest import mock
        from truthguard_ml.web import BraveSearchProvider
        p = BraveSearchProvider(api_key="k", min_interval=1.0)
        with mock.patch("truthguard_ml.web.time.sleep") as sl, mock.patch.object(BraveSearchProvider, "_get", return_value=b'{"web":{"results":[]}}'):
            p.search("a", 1); p.search("b", 1)
        self.assertTrue(sl.called and sl.call_args[0][0] <= 1.0)

    def test_failure_reason_reaches_the_panel(self):
        from truthguard_ml.web import WebSearchError

        class Boom:
            name = "boom"

            def search(self, q, k):
                raise WebSearchError("HTTP 422 SUBSCRIPTION_TOKEN_INVALID, key invalid or not subscribed to this API")
        r = svc([P("a SUPPORTS")], Boom()).v.verify("claim")
        self.assertEqual(r.verdict, SUPPORTED)
        self.assertTrue(any("HTTP 422" in n and "Web search failed" in n for n in r.notes))


class TavilyProviderTests(unittest.TestCase):
    """Network mocked: checks request shape, parsing, key handling and errors, not that Tavily accepts the request."""

    class Resp:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return self.data

    def test_request_shape_and_parsing(self):
        import json
        from unittest import mock
        from truthguard_ml.web import TavilySearchProvider
        payload = json.dumps({"answer": "GENERATED ANSWER", "results": [{"title": "T", "url": "https://a.com/x", "content": "snippet text", "published_date": "2026-11-20"},
                                                                         {"title": "no url", "content": "skip"}]}).encode()
        with mock.patch("truthguard_ml.web.urllib.request.urlopen", return_value=self.Resp(payload)) as m:
            out = TavilySearchProvider(api_key="TVLY-SECRET").search("The claim sentence.", 5)
        self.assertEqual([(r.title, r.url, r.snippet, r.published) for r in out], [("T", "https://a.com/x", "snippet text", "2026-11-20")])
        req = m.call_args[0][0]
        self.assertEqual(req.full_url, "https://api.tavily.com/search"); self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Authorization"), "Bearer TVLY-SECRET")
        body = json.loads(req.data)
        self.assertEqual(body["query"], "The claim sentence."); self.assertFalse(body["include_answer"])
        self.assertNotIn("TVLY-SECRET", req.full_url + req.data.decode())            # key only in the header
        self.assertNotIn("GENERATED ANSWER", [r.snippet for r in out])               # the generated answer is never used as evidence

    def test_keyless_mode_and_missing_key(self):
        from unittest import mock
        from truthguard_ml.web import TavilySearchProvider
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                TavilySearchProvider()
            with mock.patch("truthguard_ml.web.urllib.request.urlopen", return_value=self.Resp(b'{"results": []}')) as m:
                TavilySearchProvider(keyless=True).search("claim", 3)
        req = m.call_args[0][0]
        self.assertEqual(req.get_header("X-tavily-access-mode"), "keyless"); self.assertIsNone(req.get_header("Authorization"))

    def test_errors_are_explained_without_leaking_the_key(self):
        import io, urllib.error
        from unittest import mock
        from truthguard_ml.web import TavilySearchProvider, WebSearchError
        err = urllib.error.HTTPError("https://x", 401, "Unauthorized", {}, io.BytesIO(b'{"detail":{"error":"Unauthorized: missing or invalid API key."}}'))
        with mock.patch("truthguard_ml.web.urllib.request.urlopen", side_effect=err):
            with self.assertRaises(WebSearchError) as cm:
                TavilySearchProvider(api_key="TVLY-SECRET").search("claim", 3)
        self.assertIn("HTTP 401", str(cm.exception)); self.assertIn("key rejected", str(cm.exception))
        self.assertNotIn("TVLY-SECRET", str(cm.exception))

    def test_provider_selection_from_environment(self):
        from unittest import mock
        from truthguard_ml.web import provider_from_env
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(provider_from_env())
        with mock.patch.dict("os.environ", {"TRUTHGUARD_WEB_SEARCH": "tavily", "TAVILY_API_KEY": "k"}, clear=True):
            self.assertEqual(provider_from_env().name, "tavily")
        with mock.patch.dict("os.environ", {"TRUTHGUARD_WEB_SEARCH": "tavily-keyless"}, clear=True):
            self.assertTrue(provider_from_env().keyless)
        with mock.patch.dict("os.environ", {"TRUTHGUARD_WEB_SEARCH": "tavily"}, clear=True):
            with self.assertRaises(RuntimeError):
                provider_from_env()                                                    # selected but no key: fail loudly, not silently off

    def test_health_names_the_provider(self):
        self.assertEqual(svc([P("a SUPPORTS")], FakeWeb()).health()["web_search_provider"], "fake")
        self.assertIsNone(svc([P("a SUPPORTS")]).health()["web_search_provider"])


class Privacy(unittest.TestCase):
    def test_only_claim_text_reaches_search_and_failures_are_visible(self):
        web = FakeWeb(fail=True)
        s = svc([P("local SUPPORTS")], web)
        rep = s.analyze({"url": "https://private.example/secret?token=abc", "title": "Secret title", "content": ARTICLE, "author": "Jane Doe"})
        self.assertTrue(web.queries)
        sentences = set(split_sentences(ARTICLE))
        self.assertTrue(all(q in sentences for q in web.queries))          # only sentences-as-claims, never url/title/author/body
        self.assertTrue(all("private.example" not in q and "Secret" not in q and "Jane" not in q for q in web.queries))
        c = rep["claims"][0]
        self.assertEqual(c["verdict"], SUPPORTED)                                   # local evidence still used
        self.assertIn("Web search failed", c["missing_context"])                    # failure not disguised as "no evidence"


def check_extension_shape(t: unittest.TestCase, rep: dict):
    """Mirrors what the side panel dereferences; wrong types here would crash its renderer."""
    for k in ("article_title", "source_url", "domain", "engine", "warning_reason", "assessment_summary", "overall_evidence_assessment"):
        t.assertIsInstance(rep[k], str, k)
    t.assertIsInstance(rep["is_demo_mode"], bool); t.assertIsInstance(rep["has_warning"], bool)
    t.assertIsInstance(rep["warning_affected_claims"], list)
    h = rep["headline_analysis"]
    t.assertIsInstance(h["sensationalism_score"], (int, float)); t.assertIsInstance(h["flags"], list)
    t.assertIn(h["headline_vs_content_stance"], ("Consistent", "Exaggerated", "Contradicted"))
    for k in ("tldr",):
        t.assertIsInstance(rep["summary"][k], str)
    for k in ("verified_elements", "uncertain_or_disputed", "important_missing_context"):
        t.assertIsInstance(rep["summary"][k], list)
    st = rep["source_transparency"]
    for k in ("domain_trust_indicator", "publisher_type", "corroboration_status"):
        t.assertIsInstance(st[k], str)
    t.assertIsInstance(st["citations_count"], int); t.assertIsInstance(st["has_author_byline"], bool)
    t.assertIsInstance(st["has_publication_date"], bool); t.assertIsInstance(st["transparency_notes"], list)
    allowed = set(CLASSIFICATION.values()) | {MIXED}
    for c in rep["claims"]:
        t.assertIn(c["classification"], allowed)
        for k in ("claim_id", "claim_text", "category", "explanation"):
            t.assertIsInstance(c[k], str)
        for lst in ("supporting_evidence", "contradicting_evidence"):
            t.assertIsInstance(c[lst], list)
            for e in c[lst]:
                for k in ("source_name", "snippet", "divergence_notes"):
                    t.assertIsInstance(e[k], str)
                t.assertIn("source_url", e)


class ReportTests(unittest.TestCase):
    def test_schema_and_additive_fields(self):
        s = svc([P("Tower in Paris. SUPPORTS"), P("budget CONTRADICTS", url="https://b.org")])
        rep = s.analyze({"url": "https://news.example.com/a", "title": "Council passes budget", "content": ARTICLE})
        check_extension_shape(self, rep)
        self.assertEqual(rep["domain"], "news.example.com")
        self.assertTrue(all("verdict" in c and "uncertainty" in c and "evidence_basis" in c for c in rep["claims"]))

    def test_selected_text_is_what_gets_checked(self):
        rep = svc([P("x SUPPORTS")]).analyze({"url": "u", "title": "t", "content": ARTICLE,
                                                "selected_text": "Paris is the capital city of France and has about 2.1 million people."})
        self.assertEqual(len(rep["claims"]), 1); self.assertIn("Paris", rep["claims"][0]["claim_text"])

    def test_analyze_claim_and_verify_claim(self):
        s = svc([P("a SUPPORTS")])
        rep = s.analyze_claim("The Eiffel Tower is in Paris.", "https://x.com", "T")
        check_extension_shape(self, rep); self.assertEqual(rep["claims"][0]["classification"], "Supported by evidence")
        v = s.verify_claim("The Eiffel Tower is in Paris.")
        for k in ("verdict", "confidence", "uncertainty", "supporting_evidence", "contradicting_evidence", "sources", "explanation", "limitations", "evidence_basis"):
            self.assertIn(k, v)
        self.assertEqual(v["sources"], ["https://en.wikipedia.org/wiki/X"])

    def test_overall_assessment_rules(self):
        none_s = svc([P("irrelevant")]).analyze({"url": "u", "title": "t", "content": ARTICLE})
        self.assertEqual(none_s["overall_evidence_assessment"], "Insufficient Evidence")
        self.assertFalse(none_s["has_warning"])
        ref = svc([P("a CONTRADICTS"), P("b CONTRADICTS", url="https://b.org")]).analyze({"url": "u", "title": "t", "content": ARTICLE})
        self.assertTrue(ref["has_warning"]); self.assertEqual(len(ref["warning_affected_claims"]), 3)
        self.assertIn(ref["overall_evidence_assessment"], ("Significant Misinformation Detected", "Partially Supported with Context Needed"))

    def test_strong_label_needs_live_corroboration(self):
        two = [P("a SUPPORTS"), P("b SUPPORTS", url="https://b.org")]
        hist = svc(two).analyze({"url": "u", "title": "t", "content": ARTICLE})
        self.assertTrue(all(c["verdict"] == SUPPORTED for c in hist["claims"]))
        self.assertEqual(hist["overall_evidence_assessment"], "Partially Supported with Context Needed")   # archived only
        web = FakeWeb([WebResult("A", "https://a.com/1", "SUPPORTS it"), WebResult("B", "https://b.org/2", "SUPPORTS too")])
        live = svc([], web).analyze({"url": "https://own.net/x", "title": "t", "content": ARTICLE})
        self.assertEqual(live["overall_evidence_assessment"], "Strong Evidence Support")
        self.assertEqual(live["has_warning"], False)
        self.assertIn("3 of 3", live["source_transparency"]["corroboration_status"])

    def test_conflict_maps_to_mixed_classification(self):
        rep = svc([P("one SUPPORTS"), P("two CONTRADICTS", url="https://x.org")]).analyze({"url": "u", "title": "t", "content": ARTICLE})
        self.assertTrue(all(c["classification"] == MIXED and c["verdict"] == NEI and c["conflicting_evidence"] for c in rep["claims"]))
        self.assertTrue(rep["has_warning"])

    def test_headline_not_evaluated_without_headline_and_body(self):
        s = svc([P("a SUPPORTS")])
        rep = s.analyze_claim("The Eiffel Tower is in Paris.")
        self.assertIn("NOT evaluated", rep["headline_analysis"]["explanation"])
        self.assertNotIn("No contradiction", rep["headline_analysis"]["explanation"])
        long_body = ARTICLE * 3
        rep2 = s.analyze({"url": "u", "title": "Council passes budget", "content": long_body})
        self.assertNotIn("NOT evaluated", rep2["headline_analysis"]["explanation"])
        check_extension_shape(self, rep)

    def test_pasted_passage_is_split_into_claims(self):
        s = svc([P("a SUPPORTS")])
        multi = s.analyze_claim(ARTICLE)
        self.assertGreaterEqual(len(multi["claims"]), 2)
        self.assertTrue(all(len(c["claim_text"].split()) < 30 for c in multi["claims"]))
        single = s.analyze_claim("The Eiffel Tower is located in Paris, France.")
        self.assertEqual(len(single["claims"]), 1)
        check_extension_shape(self, multi)

    def test_attributed_statements_are_labelled_as_such(self):
        rep = svc([P("a SUPPORTS")]).analyze({"url": "u", "title": "t", "content": ARTICLE})
        att = [c for c in rep["claims"] if c["category"] == "Attributed statement"]
        self.assertTrue(att and all("reports what someone said" in c["missing_context"] for c in att))
        other = [c for c in rep["claims"] if c["category"] != "Attributed statement"]
        self.assertTrue(all("reports what someone said" not in (c["missing_context"] or "") for c in other))

    def test_no_claims_is_not_a_verdict(self):
        rep = svc([P("a SUPPORTS")]).analyze({"url": "u", "title": "t", "content": "Nice day. What?"})
        check_extension_shape(self, rep)
        self.assertEqual((rep["claims"], rep["overall_evidence_assessment"]), ([], "Insufficient Evidence"))
        self.assertTrue(any("No checkable" in x for x in rep["summary"]["important_missing_context"]))

    def test_health_and_headline(self):
        s = svc([P("a SUPPORTS")])
        h = s.health()
        self.assertEqual((h["status"], h["gemini_active"], h["web_search_enabled"]), ("ok", False, False))
        self.assertIn("default_mode", h); self.assertEqual(h["knowledge_base"]["passages"], 1)
        hl = s._headline("SHOCKING!! Council DESTROYED budget", "")
        self.assertGreater(hl["sensationalism_score"], 30); self.assertEqual(hl["emotional_tone"], "Emotive")
        calm = s._headline("Council approves budget", "")
        self.assertEqual(calm["sensationalism_score"], 0)
        con = s._headline("Council rejects budget", "word " * 100 + " CONTRADICTS")
        self.assertEqual(con["headline_vs_content_stance"], "Contradicted")


if __name__ == "__main__":
    unittest.main()
