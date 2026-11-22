"""Live web evidence. Only the CLAIM TEXT is ever sent to a search provider: never the article
body, URL, title or any user identifier. Results from the article's own site are dropped so the
evidence is independent. Domain reputation is NOT assessed here."""
from __future__ import annotations
import json, os, re, time, urllib.error, urllib.parse, urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

_TWO_PART_TLDS = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au", "net.au", "org.au", "co.nz", "co.in", "co.jp", "com.br", "co.za"}


class WebSearchError(RuntimeError):
    """Safe-to-display reason a search failed (status code and a short hint; never the key or the query)."""


_HTTP_HINTS = {400: "request rejected", 401: "key rejected", 403: "key not allowed for this plan", 422: "key invalid or not subscribed to this API",
               429: "rate limit or quota reached"}


def _fetch(req: urllib.request.Request, timeout: float) -> bytes:
    """Open a request and turn failures into WebSearchError with a safe, short reason (never the key or the query)."""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        code = ""
        try:
            body = json.loads(e.read().decode("utf-8", "replace"))
            err = body.get("error") or body.get("detail") or {}
            if isinstance(err, dict):
                err = err.get("code") or err.get("error") or ""
            code = re.sub(r"[^A-Za-z0-9_ \-]", "", str(err))[:80]
        except Exception:
            pass
        hint = _HTTP_HINTS.get(e.code, "")
        raise WebSearchError(f"HTTP {e.code}" + (f" {code}" if code else "") + (f", {hint}" if hint else "")) from None
    except urllib.error.URLError as e:
        raise WebSearchError(f"network error: {re.sub(r'[^A-Za-z0-9_ .:-]', '', str(e.reason))[:80]}") from None
    except TimeoutError:
        raise WebSearchError("timed out") from None


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str
    published: Optional[str] = None


class WebSearchProvider(Protocol):
    name: str

    def search(self, query: str, k: int) -> Sequence[WebResult]: ...


def registrable_domain(url: str) -> str:
    host = (urllib.parse.urlparse(url if "//" in url else "//" + url).hostname or "").lower()
    host = host[4:] if host.startswith("www.") else host
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _TWO_PART_TLDS:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


_TRUSTED_SUFFIXES = (".gov", ".gov.in", ".edu", ".edu.in", ".ac.in")

_TRUSTED_NEWS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "indianexpress.com",
    "thehindu.com",
}

_SOCIAL_DOMAINS = {
    "instagram.com",
    "facebook.com",
    "x.com",
    "twitter.com",
    "reddit.com",
    "tiktok.com",
}


def source_priority(url: str) -> int:
    domain = registrable_domain(url)

    if not domain:
        return 0
    if (
        domain.endswith(".gov.in")
        or domain == "gov.in"
        or domain.endswith(".gov")
        or domain == "gov"
        or domain.endswith(".edu.in")
        or domain == "edu.in"
        or domain.endswith(".ac.in")
        or domain == "ac.in"
        or domain.endswith(".edu")
        or domain == "edu"
    ):
        return 3

    if domain in _TRUSTED_NEWS:
        return 2

    if domain in _SOCIAL_DOMAINS:
        return 0

    return 1


def filter_independent(
    results: Sequence[WebResult],
    article_url: Optional[str],
) -> list[WebResult]:
    """Prioritise source types and retain one result per independent domain."""
    own = registrable_domain(article_url) if article_url else ""
    seen = set()
    out = []

    ranked = sorted(
        results,
        key=lambda r: source_priority(r.url),
        reverse=True,
    )

    for r in ranked:
        d = registrable_domain(r.url)

        if not d or d == own or d in seen or not r.snippet.strip():
            continue

        seen.add(d)
        out.append(r)

    return out


_TAG = re.compile(r"<[^>]+>")


class BraveSearchProvider:
    """Brave Search API (needs BRAVE_API_KEY in the backend environment). Not run against the real service by the author.
    Requests are spaced by BRAVE_MIN_INTERVAL seconds (default 1.0) because some plans allow ~1 request/second."""
    name = "brave"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 10.0, min_interval: Optional[float] = None):
        self.api_key = (api_key or os.environ.get("BRAVE_API_KEY") or "").strip()
        if not self.api_key:
            raise RuntimeError("BRAVE_API_KEY is not set")
        self.timeout = timeout
        self.min_interval = float(os.environ.get("BRAVE_MIN_INTERVAL", "1.0")) if min_interval is None else min_interval
        self._last = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def _get(self, url: str) -> bytes:
        return _fetch(urllib.request.Request(url, headers={"Accept": "application/json", "X-Subscription-Token": self.api_key}), self.timeout)

    def search(self, query: str, k: int = 5) -> list[WebResult]:
        url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": min(max(k, 1), 20)})
        self._throttle()
        try:
            raw = self._get(url)
        except WebSearchError as ex:
            if "HTTP 429" not in str(ex):
                raise
            time.sleep(max(self.min_interval, 1.5))       # one polite retry on a rate limit
            self._throttle()
            raw = self._get(url)
        data = json.loads(raw.decode("utf-8"))
        out = []
        for it in (data.get("web") or {}).get("results", []):
            if it.get("url"):
                out.append(WebResult(_TAG.sub("", it.get("title", "")), it["url"], _TAG.sub("", it.get("description", "")), it.get("page_age")))
        return out


class TavilySearchProvider:
    """Tavily Search REST API (https://api.tavily.com/search). Key from TAVILY_API_KEY in the backend environment.
    keyless=True uses Tavily's documented rate-limited keyless mode (header X-Tavily-Access-Mode: keyless).
    Only the claim text is sent; Tavily's generated 'answer' is never requested or used, only source snippets.
    Written from the public setup guide and memory of the API; not run against the real service by the author."""
    name = "tavily"
    URL = "https://api.tavily.com/search"

    def __init__(self, api_key: Optional[str] = None, keyless: bool = False, timeout: float = 15.0):
        self.api_key = (api_key or os.environ.get("TAVILY_API_KEY") or "").strip()
        self.keyless, self.timeout = keyless, timeout
        if not self.api_key and not keyless:
            raise RuntimeError("TAVILY_API_KEY is not set (or use TRUTHGUARD_WEB_SEARCH=tavily-keyless)")

    def search(self, query: str, k: int = 5) -> list[WebResult]:
        body = json.dumps({"query": query, "max_results": min(max(k, 1), 10), "search_depth": "basic", "include_answer": False}).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        else:
            headers["X-Tavily-Access-Mode"] = "keyless"
        data = json.loads(_fetch(urllib.request.Request(self.URL, data=body, headers=headers, method="POST"), self.timeout).decode("utf-8"))
        return [WebResult(str(it.get("title") or ""), it["url"], str(it.get("content") or ""), it.get("published_date"))
                for it in data.get("results", []) if it.get("url")]


def provider_from_env() -> Optional[WebSearchProvider]:
    """Web search is OFF unless TRUTHGUARD_WEB_SEARCH is brave, tavily or tavily-keyless (keys come from the environment)."""
    mode = os.environ.get("TRUTHGUARD_WEB_SEARCH", "").strip().lower()
    if mode == "brave":
        return BraveSearchProvider()
    if mode == "tavily":
        return TavilySearchProvider()
    if mode == "tavily-keyless":
        return TavilySearchProvider(keyless=True)
    return None
