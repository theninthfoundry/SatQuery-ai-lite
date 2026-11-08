"""
backend/app.py

FastAPI application. `api/analyze.py` re-exports `app` as the Vercel entrypoint.

Every failure path returns the same structured JSON body
(`status`, `ok`, `error.code`, `error.message`) and never a stack trace.
"""

from __future__ import annotations

import os
import re
import uuid
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend import capabilities, config
from backend.geospatial.stac import StacError, compare_stac_items, inspect_stac_item, strip_private
from backend.pipeline import json_safe, run_analysis, run_diagnostics

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ── Request schemas ──────────────────────────────────────────────────────

class AnalyzeOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    grid_size: int = Field(config.CHANGE.grid_size, ge=config.LIMITS.min_grid_size, le=config.LIMITS.max_grid_size)
    base_threshold: int = Field(config.CHANGE.base_threshold, ge=1, le=254)
    thresholds: Optional[List[int]] = Field(None, max_length=config.LIMITS.max_thresholds)


class ClientNote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int = Field(ge=0, le=1)
    resized_from: List[int] = Field(min_length=2, max_length=2)
    resized_to: List[int] = Field(min_length=2, max_length=2)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["single", "cross", "bitemporal"]
    query: str
    images: List[str] = Field(min_length=1, max_length=config.LIMITS.max_images)
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)
    client_notes: List[ClientNote] = Field(default_factory=list, max_length=2)

    @field_validator("query")
    @classmethod
    def _query(cls, v: str) -> str:
        v = _CONTROL.sub(" ", v)
        v = " ".join(v.split())
        if not v:
            raise ValueError("query must not be empty")
        if len(v) > config.LIMITS.max_query_chars:
            raise ValueError(f"query must be at most {config.LIMITS.max_query_chars} characters")
        return v


class DiagnosticsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    images: List[str] = Field(min_length=1, max_length=config.LIMITS.max_images)


class StacInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: List[Dict[str, Any]] = Field(min_length=1, max_length=2)
    purpose: Optional[Literal["change", "fusion"]] = None


# ── Body-size limiting (pure ASGI: also bounds chunked uploads) ──────────

class BodySizeLimit:
    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    if int(value) > self.max_bytes:
                        return await self._reject(send)
                except ValueError:
                    return await self._reject(send)
        seen = 0
        too_big = False

        async def limited_receive():
            nonlocal seen, too_big
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    too_big = True
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        responded = False

        async def guarded_send(message):
            nonlocal responded
            if too_big and not responded:
                responded = True
                await self._reject(send)
                return
            if not responded:
                await send(message)

        await self.app(scope, limited_receive, guarded_send)

    async def _reject(self, send):
        body = (
            b'{"status":"REJECTED_INPUT","ok":false,"error":{"code":"PAYLOAD_TOO_LARGE",'
            b'"message":"Request body exceeds the allowed size."}}'
        )
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


# ── App ──────────────────────────────────────────────────────────────────

def _err(status: str, code: str, message: str, http: int, **extra: Any) -> JSONResponse:
    return JSONResponse(
        status_code=http,
        content=json_safe({"schema_version": config.SCHEMA_VERSION, "status": status, "ok": False,
                           "error": {"code": code, "message": message, **extra}}),
    )


def create_app() -> FastAPI:
    app = FastAPI(title="SatQuery AI Lite API", version=config.ENGINE_VERSION)

    origins = [o.strip() for o in os.environ.get("SATQUERY_CORS_ORIGINS", "").split(",") if o.strip()]
    if origins:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    app.add_middleware(BodySizeLimit, max_bytes=config.LIMITS.max_request_bytes)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        fields = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()[:10]]
        first = (fields[0]["message"] if fields else "invalid request").removeprefix("Value error, ")
        return _err("REJECTED_INPUT", "INVALID_REQUEST", first, 422, fields=fields)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        return _err("ERROR", f"HTTP_{exc.status_code}", str(exc.detail), exc.status_code)

    @app.exception_handler(Exception)
    async def _unexpected(_: Request, exc: Exception):
        return _err("ERROR", "INTERNAL_ERROR", "An unexpected internal error occurred.", 500, request_id=str(uuid.uuid4()))

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "satquery-ai-lite", "engine_version": config.ENGINE_VERSION, "schema_version": config.SCHEMA_VERSION}

    @app.get("/api/capabilities")
    def get_capabilities():
        return capabilities.registry()

    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest):
        http, body = run_analysis(req.model_dump())
        return JSONResponse(status_code=http, content=body)

    @app.post("/api/diagnostics")
    def diagnostics_endpoint(req: DiagnosticsRequest):
        http, body = run_diagnostics(req.model_dump())
        return JSONResponse(status_code=http, content=body)

    @app.post("/api/stac/inspect")
    def stac_inspect(req: StacInspectRequest):
        import json
        if len(json.dumps(req.items)) > config.LIMITS.max_stac_bytes:
            return _err("REJECTED_INPUT", "STAC_TOO_LARGE", "STAC payload exceeds the size limit.", 413)
        try:
            inspected = [inspect_stac_item(i) for i in req.items]
            comparison = None
            if len(inspected) == 2:
                if req.purpose is None:
                    return _err("REJECTED_INPUT", "PURPOSE_REQUIRED", "'purpose' (change|fusion) is required when two items are supplied.", 422)
                comparison = compare_stac_items(inspected[0], inspected[1], req.purpose)
        except StacError as e:
            return _err("REJECTED_INPUT", "INVALID_STAC_ITEM", str(e), 422)
        return JSONResponse(content=json_safe({
            "schema_version": config.SCHEMA_VERSION, "status": "COMPLETED", "ok": True,
            "items": [strip_private(i) for i in inspected], "comparison": comparison,
            "scope_note": "Metadata inspection only. Nothing was fetched; no asset was opened or analysed.",
        }))

    if os.environ.get("SATQUERY_SERVE_STATIC") == "1":
        # Local development convenience: `SATQUERY_SERVE_STATIC=1 uvicorn api.analyze:app`
        from fastapi.staticfiles import StaticFiles

        public = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public")
        app.mount("/", StaticFiles(directory=public, html=True), name="static")

    return app


app = create_app()
