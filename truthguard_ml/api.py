"""FastAPI wrapper. Run:  uvicorn truthguard_ml.api:create_app --factory --port 8001

Routes (all under /api):  GET /health   POST /analyze   POST /analyze-claim   POST /verify-claim   GET /demos
`/analyze`, `/analyze-claim`, `/health` and `/demos` match what the extension's side panel already calls.
To merge into your existing FastAPI app instead:  app.include_router(make_router(service))

Secrets come from the environment only (e.g. BRAVE_API_KEY); nothing is read from requests.
Request bodies are never logged. NOTE: written without FastAPI installed, so this file is untested.
"""
from __future__ import annotations
import os, threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .service import FactCheckService


class AnalyzeRequest(BaseModel):
    url: str = ""
    title: str = ""
    content: str = Field("", max_length=200_000)
    author: Optional[str] = None
    published_date: Optional[str] = None
    selected_text: Optional[str] = Field(None, max_length=50_000)


class AnalyzeClaimRequest(BaseModel):
    claim_text: str = Field(..., min_length=3, max_length=2000)
    context_url: Optional[str] = None
    article_title: Optional[str] = None


class VerifyClaimRequest(BaseModel):
    claim: str = Field(..., min_length=3, max_length=2000)
    context_url: Optional[str] = None


def _dump(m) -> dict:
    return m.model_dump() if hasattr(m, "model_dump") else m.dict()


def make_router(service: FactCheckService) -> APIRouter:
    r, lock = APIRouter(prefix="/api"), threading.Lock()      # one model call at a time

    @r.get("/health")
    def health():
        return service.health()

    @r.post("/analyze")
    def analyze(req: AnalyzeRequest):
        with lock:
            return service.analyze(_dump(req))

    @r.post("/analyze-claim")
    def analyze_claim(req: AnalyzeClaimRequest):
        with lock:
            return service.analyze_claim(req.claim_text, req.context_url, req.article_title)

    @r.post("/verify-claim")
    def verify_claim(req: VerifyClaimRequest):
        with lock:
            return service.verify_claim(req.claim, req.context_url)

    @r.get("/demos")
    def demos():
        return {"demos": []}              # no canned demo scenarios: every result comes from real evidence

    return r


def build_service_from_env() -> FactCheckService:
    from .evaluate import HFNLIModel
    from .kb import KnowledgeBase
    from .verify import Verifier
    from .web import provider_from_env
    kb_dir = Path(os.environ.get("TRUTHGUARD_KB_DIR", "data/kb"))
    kb = KnowledgeBase.load(kb_dir) if (kb_dir / "kb_meta.json").exists() else None
    model = os.environ.get("TRUTHGUARD_NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    return FactCheckService(Verifier(HFNLIModel(model), kb, provider_from_env(), model_name=model))


def create_app(service: Optional[FactCheckService] = None) -> FastAPI:
    app = FastAPI(title="TruthGuard Evidence API", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origin_regex=r"^(chrome-extension://.*|http://(localhost|127\.0\.0\.1)(:\d+)?)$",
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    app.include_router(make_router(service or build_service_from_env()))
    return app
