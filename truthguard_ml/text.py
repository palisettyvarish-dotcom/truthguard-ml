"""Text cleaning (kept readable) and normalisation keys (for dedup/leakage)."""
from __future__ import annotations
import re, unicodedata
from typing import Optional
from urllib.parse import quote

_FEVER_ESC = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
              "-LCB-": "{", "-RCB-": "}", "-COLON-": ":"}
_CTRL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_SP_BEFORE = re.compile(r"\s+([,.;:!?%)\]\}])")
_SP_AFTER = re.compile(r"([(\[\{])\s+")


def clean_text(text) -> Optional[str]:
    """Unescape tokenised markup, NFKC-normalise, collapse whitespace. None if empty."""
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    for k, v in _FEVER_ESC.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    text = _SP_BEFORE.sub(r"\1", text)
    text = _SP_AFTER.sub(r"\1", text)
    return text or None


def norm_key(text) -> str:
    """Aggressive key: lower-case, no punctuation. Used only for comparisons."""
    t = clean_text(text)
    if not t:
        return ""
    return _WS.sub(" ", _PUNCT.sub(" ", t.lower())).strip()


def wikipedia_url(title) -> Optional[str]:
    """URL derived from a real Wikipedia page title; None when no title exists."""
    if not title or not isinstance(title, str):
        return None
    t = title
    for k, v in _FEVER_ESC.items():
        t = t.replace(k, v)
    t = _WS.sub("_", t.strip())
    if not t:
        return None
    return "https://en.wikipedia.org/wiki/" + quote(t, safe="_()',:!*-.~")
