"""
FastAPI server exposing the multi-agent pipeline as a streaming API
and serving the built React frontend from `frontend/dist`.

Endpoints:
    POST /api/ask          — Server-Sent Events stream of pipeline progress
    GET  /api/health       — liveness check
    GET  /*                — serves the React SPA (after `npm run build`)
"""

import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.app.pipeline_service import stream_baseline, stream_multiagent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")


app = FastAPI(title="Financial Research Assistant")

# Dev CORS: Vite dev server runs on :5173 and proxies /api, but allow
# direct cross-origin during development as a safety net.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    mode: str = "multiagent"         # "multiagent" | "baseline"
    retrieval: str = "hybrid"        # "hybrid" | "dense" | "bm25"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/ask")
def ask(req: AskRequest):
    """Stream pipeline events as Server-Sent Events."""
    if not req.question.strip():
        return JSONResponse({"error": "Question is empty"}, status_code=400)

    def event_stream():
        try:
            iterator = (
                stream_baseline(req.question, req.retrieval)
                if req.mode == "baseline"
                else stream_multiagent(req.question, req.retrieval)
            )
            for event in iterator:
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.exception("Pipeline error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # disable nginx/proxy buffering if deployed behind one
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


# Frontend (static SPA)

_FRONTEND_DIST = Path(__file__).parent.parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=_FRONTEND_DIST / "assets"),
        name="assets",
    )

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        # SPA fallback: any non-API path returns index.html
        index = _FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(index)
        return JSONResponse({"error": "Frontend not built"}, status_code=404)
else:
    @app.get("/")
    def root():
        return JSONResponse(
            {
                "message": "Frontend not built.",
                "build_command": "cd frontend && npm install && npm run build",
                "dev_command": "cd frontend && npm run dev",
            }
        )
