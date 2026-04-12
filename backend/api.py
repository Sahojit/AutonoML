"""
FastAPI Backend
───────────────
Endpoints:
  POST /chat       — run full multi-agent pipeline
  POST /memory     — ingest documents into the vector store
  GET  /history    — retrieve conversation history
  GET  /health     — liveness probe
  GET  /status     — system info
  DELETE /session  — remove a session
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

sys.path.insert(0, ".")

from agents.base_agent import configure_logging
from backend.session_store import get_session_store
from config.settings import settings
from core.query_router import RouteType, classify_route, get_router
from core.tracing import get_tracing_status, setup_tracing
from memory.memory_manager import MemoryManager
from memory.vector_store import get_vector_store
from orchestrator.agent_orchestrator import AgentOrchestrator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

configure_logging(settings.LOG_LEVEL)
logger = logging.getLogger("multiagent.api")

# ---------------------------------------------------------------------------
# Session registry (in-process; swap with Redis for multi-worker deployments)
# ---------------------------------------------------------------------------

_orchestrator: Optional[AgentOrchestrator] = None
_thread_pool = ThreadPoolExecutor(max_workers=4)

limiter = Limiter(key_func=get_remote_address)


def _get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


def _get_or_create_memory(session_id: str) -> MemoryManager:
    return get_session_store().get_or_create(session_id)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Multi-Agent AI System v%s starting…", settings.VERSION)
    setup_tracing()
    _get_orchestrator()
    yield
    logger.info("API shutting down — flushing sessions…")
    _thread_pool.shutdown(wait=False)
    store = get_session_store()
    for sid in store.all_ids():
        try:
            mem = store.get(sid)
            if mem:
                mem.save_session_to_long_term()
        except Exception:                  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade Multi-Agent AI System API",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    session_id: Optional[str] = Field(None, description="Reuse an existing session")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "Analyse the Iris dataset and train a Random Forest classifier.",
                "session_id": None,
            }
        }
    }


class ChatResponse(BaseModel):
    final_answer: str
    session_id: str
    goal: str = ""
    plan: Dict[str, Any] = Field(default_factory=dict)
    step_records: List[Dict[str, Any]] = Field(default_factory=list)
    duration_s: Optional[float] = None
    timestamp: float = Field(default_factory=time.time)
    route: str = "pipeline"          # direct | rag | tool | pipeline | cache
    from_cache: bool = False


class MemoryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    session_id: Optional[str] = None


class MemoryResponse(BaseModel):
    message: str
    doc_ids: List[str]


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[Dict[str, Any]]
    count: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
async def health() -> Dict[str, str]:
    return {"status": "ok", "version": settings.VERSION}


@app.get("/status", tags=["System"])
async def status() -> Dict[str, Any]:
    _get_orchestrator()  # ensure warm
    vs = get_vector_store()
    try:
        doc_count = vs.get_collection_count()
    except Exception:                      # noqa: BLE001
        doc_count = -1
    return {
        "status": "running",
        "version": settings.VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "active_sessions": get_session_store().count(),
        "vector_store_docs": doc_count,
        "max_iterations": settings.MAX_AGENT_ITERATIONS,
        "parallel_research": settings.PARALLEL_RESEARCH,
        **get_tracing_status(),
    }


@app.post("/chat", response_model=ChatResponse, tags=["Agent"])
@limiter.limit("10/minute")
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    Execute the full multi-agent pipeline:
    Planner → Research (parallel) → Execution → Evaluation (loop) → Answer.
    """
    session_id = body.session_id or str(uuid.uuid4())
    logger.info("POST /chat | session=%s | query='%s'", session_id, body.query[:80])

    _get_or_create_memory(session_id)

    try:
        orch = _get_orchestrator()
        loop = asyncio.get_event_loop()
        result: Dict[str, Any] = await loop.run_in_executor(
            _thread_pool, orch.run, body.query, session_id
        )
        return ChatResponse(**result)
    except Exception as exc:               # noqa: BLE001
        logger.exception("Error in /chat: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/smart", response_model=ChatResponse, tags=["Agent"])
@limiter.limit("10/minute")
async def chat_smart(request: Request, body: ChatRequest) -> ChatResponse:
    """
    Adaptive routing endpoint (Step 6 — Smart Routing).

    Classifies the query and routes to the cheapest sufficient path:
      direct   -> single LLM call         (~1-3 s)
      rag      -> memory retrieval + LLM  (~2-5 s)
      tool     -> web search + LLM        (~3-8 s)
      pipeline -> full 4-agent pipeline   (~15-60 s)

    Returns the same ChatResponse schema as /chat.
    """
    session_id = body.session_id or str(uuid.uuid4())
    logger.info("POST /chat/smart | session=%s | query='%s'", session_id, body.query[:80])

    memory = _get_or_create_memory(session_id)
    router = get_router()

    try:
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _thread_pool,
            lambda: router.route(body.query, memory, session_id),
        )
        # Fill missing ChatResponse fields with defaults
        result.setdefault("goal",         body.query)
        result.setdefault("plan",         {})
        result.setdefault("step_records", [])
        result.setdefault("route",        "direct")
        result.setdefault("from_cache",   False)
        return ChatResponse(**{k: v for k, v in result.items()
                               if k in ChatResponse.model_fields})
    except Exception as exc:               # noqa: BLE001
        logger.exception("Error in /chat/smart: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream", tags=["Agent"])
@limiter.limit("10/minute")
async def chat_stream(request: Request, body: ChatRequest) -> StreamingResponse:
    """
    Streaming endpoint (Step 8 — Real-time streaming responses).

    Uses the smart router.  Yields Server-Sent Events:
      data: __ROUTE__:<route>\\n\\n      (first chunk — route taken)
      data: __STATUS__:<msg>\\n\\n       (pipeline status updates)
      data: <text chunk>\\n\\n           (answer tokens)

    In Streamlit, consume with requests stream=True + iter_lines().
    In a browser, use EventSource or fetch with ReadableStream.
    """
    session_id = body.session_id or str(uuid.uuid4())
    logger.info("POST /chat/stream | session=%s | query='%s'", session_id, body.query[:80])

    memory = _get_or_create_memory(session_id)
    router = get_router()

    async def _sse_generator():
        # The sync streaming generator runs in a thread; we bridge it here
        # via an asyncio queue so we never block the event loop.
        import queue as _q_mod
        q: _q_mod.Queue = _q_mod.Queue()
        _DONE = object()

        def _run_stream():
            try:
                for chunk in router.route_stream(body.query, memory, session_id):
                    q.put(chunk)
            except Exception as exc:       # noqa: BLE001
                q.put(f"\n[Stream error: {exc}]")
            finally:
                q.put(_DONE)

        import threading
        threading.Thread(target=_run_stream, daemon=True).start()

        loop = asyncio.get_event_loop()
        while True:
            # Poll the sync queue without blocking the event loop
            chunk = await loop.run_in_executor(None, q.get)
            if chunk is _DONE:
                break
            # SSE format
            yield f"data: {chunk}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@app.post("/memory", response_model=MemoryResponse, tags=["Memory"])
async def store_memory(request: MemoryRequest) -> MemoryResponse:
    """Ingest a document into the long-term vector store."""
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("POST /memory | session=%s | len=%d", session_id, len(request.text))
    try:
        mem = _get_or_create_memory(session_id)
        meta = request.metadata or {}
        meta.setdefault("source", "user_upload")
        doc_ids = mem.store_in_long_term([request.text], [meta])
        return MemoryResponse(
            message=f"Ingested successfully ({len(doc_ids)} chunk(s)).",
            doc_ids=doc_ids,
        )
    except Exception as exc:               # noqa: BLE001
        logger.exception("Error in /memory: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/history", response_model=HistoryResponse, tags=["Memory"])
async def get_history(
    session_id: str = Query(..., description="Session ID"),
) -> HistoryResponse:
    mem = get_session_store().get(session_id)
    if mem is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    msgs = mem.get_conversation_history()
    return HistoryResponse(session_id=session_id, messages=msgs, count=len(msgs))


@app.delete("/session/{session_id}", tags=["System"])
async def delete_session(session_id: str) -> Dict[str, str]:
    if not get_session_store().delete(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return {"message": f"Session '{session_id}' deleted."}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "backend.api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
