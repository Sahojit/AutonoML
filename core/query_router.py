"""
QueryRouter - Adaptive Routing Layer
Routes every query to the cheapest sufficient execution path:
  DIRECT   -> single LLM call with history         (~1-3 s)
  RAG      -> ChromaDB retrieval + LLM synthesis   (~2-5 s)
  TOOL     -> web_search + LLM summarisation       (~3-8 s)
  PIPELINE -> full 4-agent pipeline                (~15-60 s)

Route selection:
  1. Cache hit?           -> return cached (TTL 300 s)
  2. Real-time keywords?  -> TOOL
  3. Complex / ML / code? -> PIPELINE
  4. Follow-up query?     -> RAG  (uses conversation history)
  5. Memory context hit?  -> RAG
  6. Default              -> DIRECT
"""
from __future__ import annotations

import hashlib
import logging
import queue as _queue_module
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, Optional

from config.settings import settings
from memory.memory_manager import MemoryManager
from models.llm_loader import get_llm
from tools.tool_registry import tool_registry

logger = logging.getLogger("multiagent.core.router")


# ---------------------------------------------------------------------------
# Route enum
# ---------------------------------------------------------------------------

class RouteType(str, Enum):
    DIRECT   = "direct"
    RAG      = "rag"
    TOOL     = "tool"
    PIPELINE = "pipeline"


# ---------------------------------------------------------------------------
# Keyword classifiers
# ---------------------------------------------------------------------------

_REALTIME_KW = {
    "latest", "today", "current", "now", "news", "live",
    "real-time", "realtime", "recent", "right now", "price",
    "weather", "trending", "breaking", "this week", "this month",
    "stock", "crypto", "exchange rate",
}

_PIPELINE_KW = {
    "train", "analyze", "analyse", "compare", "implement",
    "build", "create", "write code", "execute", "run",
    "dataset", "model", "classifier", "regression", "cluster",
    "predict", "evaluate", "experiment", "machine learning",
    "deep learning", "neural network", "fine-tune", "finetune",
    "xgboost", "random forest", "gradient boosting",
    "sql query", "query the database", "generate a report",
    "auto experiment", "auto-experiment",
}

_FOLLOWUP_KW = {
    "improve it", "fix it", "make it", "update it", "change it",
    "can you also", "what about", "and also", "now do",
    "retry", "try again", "do the same", "as before",
    "in the previous", "from before", "you mentioned",
    "previous", "earlier", "last time", "history",
}

# Short general-knowledge queries that should never hit RAG or pipeline
_SIMPLE_QUESTION_STARTS = (
    "what is", "what are", "who is", "who are", "how does", "how do",
    "explain", "define", "tell me about", "what does", "why is", "why are",
    "when did", "where is", "where are",
)


def _is_simple_query(q: str) -> bool:
    """True for short factual/definitional queries with no real-time, pipeline, or follow-up intent."""
    q = q.strip()
    if not q:
        return True   # empty → trivially simple; direct LLM will handle gracefully
    word_count = len(q.split())
    if word_count > 10:
        return False
    # Real-time, pipeline, or follow-up content disqualifies a "simple" query
    if any(kw in q for kw in _REALTIME_KW):
        return False
    if any(kw in q for kw in _PIPELINE_KW):
        return False
    if any(kw in q for kw in _FOLLOWUP_KW):
        return False
    return any(q.startswith(start) for start in _SIMPLE_QUESTION_STARTS) or word_count <= 5


def classify_route(
    query: str,
    has_memory_context: bool = False,
    has_conversation: bool = False,
) -> RouteType:
    """Return the most appropriate RouteType for query."""
    q     = query.lower().strip()
    words = set(q.split())

    # 1. Short general-knowledge queries always go direct — never block on memory context
    if _is_simple_query(q):
        return RouteType.DIRECT

    # 2. Real-time / live-data queries
    if words & _REALTIME_KW or any(kw in q for kw in _REALTIME_KW):
        return RouteType.TOOL

    # 3. Complex ML / code / data tasks
    if any(kw in q for kw in _PIPELINE_KW):
        return RouteType.PIPELINE

    # 4. Follow-up or memory-reference queries
    if any(kw in q for kw in _FOLLOWUP_KW):
        return RouteType.RAG

    if has_conversation and has_memory_context:
        # Only use RAG if *both* conditions are true — avoids spurious RAG on new sessions
        return RouteType.RAG

    return RouteType.DIRECT


# ---------------------------------------------------------------------------
# TTL result cache  (Step 10 - incremental execution / skip when cached)
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    result: Dict[str, Any]
    created_at: float = field(default_factory=time.time)
    ttl: float = 300.0

    def valid(self) -> bool:
        return (time.time() - self.created_at) < self.ttl


class ResultCache:
    """
    In-memory TTL cache keyed by SHA-256(normalised_query + recent_history).

    Only DIRECT and RAG results are cached.
    PIPELINE results are never cached - they write to memory (side effects).
    """

    def __init__(self, ttl: float = 300.0) -> None:
        self._store: Dict[str, _CacheEntry] = {}
        self._ttl = ttl

    def _key(self, query: str, ctx: str) -> str:
        raw = f"{query.strip().lower()}|{ctx}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, query: str, ctx: str = "") -> Optional[Dict[str, Any]]:
        key   = self._key(query, ctx)
        entry = self._store.get(key)
        if entry and entry.valid():
            logger.debug("[Cache] HIT key=%s", key)
            return entry.result
        if entry:
            del self._store[key]
        return None

    def set(self, query: str, ctx: str, result: Dict[str, Any]) -> None:
        self._store[self._key(query, ctx)] = _CacheEntry(result=result, ttl=self._ttl)

    def clear(self) -> None:
        self._store.clear()

    def evict_expired(self) -> int:
        stale = [k for k, v in self._store.items() if not v.valid()]
        for k in stale:
            del self._store[k]
        return len(stale)

    @property
    def size(self) -> int:
        return len(self._store)


_cache = ResultCache(ttl=300.0)


# ---------------------------------------------------------------------------
# Prompt templates  (Step 9 - full conversational memory injection)
# ---------------------------------------------------------------------------

def _direct_prompt(history: str, query: str) -> str:
    return (
        "You are a helpful AI assistant. Answer the user concisely.\n\n"
        "FORMATTING RULES:\n"
        "- Always wrap code in fenced blocks with a language tag, e.g. ```python ... ```\n"
        "- Never place code inline with prose — use a separate block.\n"
        "- Use ```sql, ```bash, ```javascript etc. where appropriate.\n\n"
        f"Conversation history:\n{history or 'None'}\n\n"
        f"User: {query}\n\nAssistant:"
    )


_FORMATTING_RULES = (
    "FORMATTING RULES:\n"
    "- Always wrap code in fenced blocks with a language tag, e.g. ```python ... ```\n"
    "- Never place code inline with prose — use a separate fenced block.\n"
    "- Use ```sql, ```bash, ```javascript etc. where appropriate.\n\n"
)


def _rag_prompt(context: str, history: str, query: str) -> str:
    return (
        "You are a helpful AI assistant with access to a knowledge base.\n\n"
        + _FORMATTING_RULES
        + f"Relevant context from memory:\n{context}\n\n"
        f"Conversation history:\n{history or 'None'}\n\n"
        f"User: {query}\n\n"
        "Answer using the context. If insufficient, say so.\n\nAssistant:"
    )


def _tool_prompt(tool_output: str, history: str, query: str) -> str:
    return (
        "You are a helpful AI assistant.\n\n"
        + _FORMATTING_RULES
        + f"Real-time information retrieved:\n{tool_output}\n\n"
        f"Conversation history:\n{history or 'None'}\n\n"
        f"User: {query}\n\n"
        "Summarise the retrieved information to answer accurately.\n\nAssistant:"
    )


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

def _llm_call(prompt: str, agent_type: str = "research") -> str:
    llm = get_llm(agent_type)
    try:
        resp = llm.invoke(prompt)
        return resp.content if hasattr(resp, "content") else str(resp)
    except Exception as exc:               # noqa: BLE001
        logger.warning("[Router] LLM call failed: %s", exc)
        return f"I encountered an error: {exc}"


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Central routing controller.

    Public API
    ----------
    route(query, memory, session_id)            -> dict  (non-streaming)
    route_stream(query, memory, session_id)     -> Generator[str, ...]  (streaming)
    """

    def __init__(self) -> None:
        self._orchestrator = None

    def _get_orchestrator(self):
        if self._orchestrator is None:
            from orchestrator.agent_orchestrator import AgentOrchestrator
            self._orchestrator = AgentOrchestrator()
        return self._orchestrator

    # ------------------------------------------------------------------
    # Non-streaming entry point
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        memory: MemoryManager,
        session_id: str,
        force: Optional[RouteType] = None,
    ) -> Dict[str, Any]:
        """
        Route query and return a ChatResponse-compatible dict.

        Step 10 — Check cache first; skip all agents if hit.
        """
        ctx_key = memory.get_formatted_history(last_n=3)

        # Cache check (skip pipeline entirely when hit)
        cached = _cache.get(query, ctx_key)
        if cached:
            logger.info("[Router] session=%s | CACHE HIT", session_id)
            return {**cached, "from_cache": True}

        # Classify
        recent_docs = memory.retrieve_from_long_term(query, k=2)
        has_context = bool(recent_docs)
        has_convo   = bool(memory.get_conversation_history())
        route = force or classify_route(query, has_context, has_convo)

        logger.info(
            "[Router] session=%s | route=%-8s | %s",
            session_id, route.value, query[:80],
        )

        t0     = time.perf_counter()
        result = self._dispatch(route, query, memory, session_id)
        result["route"]      = route.value
        result["duration_s"] = round(time.perf_counter() - t0, 3)
        result["session_id"] = session_id
        result.setdefault("from_cache", False)

        # Cache only cheap routes
        if route in (RouteType.DIRECT, RouteType.RAG):
            _cache.set(query, ctx_key, result)

        return result

    # ------------------------------------------------------------------
    # Streaming entry point  (Step 8)
    # ------------------------------------------------------------------

    def route_stream(
        self,
        query: str,
        memory: MemoryManager,
        session_id: str,
    ) -> Generator[str, None, None]:
        """
        Yield text chunks for real-time streaming.

        DIRECT / RAG / TOOL: true token-by-token streaming from LLM.
        PIPELINE: runs fully in a thread, then streams the final answer.

        Each chunk is a raw string. Callers may wrap in SSE (data: ...\\n\\n).
        """
        ctx_key = memory.get_formatted_history(last_n=3)

        # Serve from cache instantly
        cached = _cache.get(query, ctx_key)
        if cached:
            answer = cached.get("final_answer", "")
            yield "__ROUTE__:cache\n"
            for i in range(0, len(answer), 80):
                yield answer[i:i + 80]
            return

        recent_docs = memory.retrieve_from_long_term(query, k=2)
        has_context = bool(recent_docs)
        has_convo   = bool(memory.get_conversation_history())
        route = classify_route(query, has_context, has_convo)

        logger.info("[Router:stream] session=%s | route=%s", session_id, route.value)
        yield f"__ROUTE__:{route.value}\n"

        if route == RouteType.PIPELINE:
            yield "__STATUS__:Running agent pipeline (this may take 15–60 s)...\n"
            # Run synchronously in calling thread (executor handles this at API layer)
            result = self._run_pipeline(query, session_id)
            answer = result.get("final_answer", "No result returned.")
            for i in range(0, len(answer), 80):
                yield answer[i:i + 80]
                time.sleep(0.015)   # pacing for smooth UX
        else:
            yield from self._stream_llm_chunks(route, query, memory)

    # ------------------------------------------------------------------
    # Route implementations
    # ------------------------------------------------------------------

    def _dispatch(
        self,
        route: RouteType,
        query: str,
        memory: MemoryManager,
        session_id: str,
    ) -> Dict[str, Any]:
        if route == RouteType.DIRECT:
            return self._run_direct(query, memory)
        if route == RouteType.RAG:
            return self._run_rag(query, memory)
        if route == RouteType.TOOL:
            return self._run_tool(query, memory)
        return self._run_pipeline(query, session_id)

    def _run_direct(self, query: str, memory: MemoryManager) -> Dict[str, Any]:
        """Single LLM call with full conversation history injected."""
        history = memory.get_formatted_history(last_n=8)
        answer  = _llm_call(_direct_prompt(history, query), agent_type="research")
        if not answer or not answer.strip():
            answer = "I couldn't generate a confident answer. Please try rephrasing your question."
        memory.add_message("user", query)
        memory.add_message("assistant", answer)
        return {
            "final_answer": answer,
            "goal":         query,
            "plan":         {},
            "step_records": [],
        }

    def _run_rag(self, query: str, memory: MemoryManager) -> Dict[str, Any]:
        """ChromaDB retrieval + LLM synthesis. Falls back to direct if no docs found."""
        docs = memory.retrieve_from_long_term(
            query, k=settings.LONG_TERM_MEMORY_TOP_K
        )
        # Fallback: no memory context → just answer directly
        if not docs:
            logger.info("[Router] RAG found no docs — falling back to DIRECT")
            return self._run_direct(query, memory)

        context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
        history = memory.get_formatted_history(last_n=8)
        answer  = _llm_call(_rag_prompt(context, history, query), agent_type="research")
        if not answer or not answer.strip():
            answer = "I couldn't find a confident answer from memory. Please try rephrasing."
        memory.add_message("user", query)
        memory.add_message("assistant", answer)
        return {
            "final_answer": answer,
            "goal":         query,
            "plan":         {},
            "step_records": [{
                "step_id":    1,
                "step_type":  "research",
                "task":       f"RAG retrieval: {query}",
                "status":     "done",
                "output_full": f"Retrieved {len(docs)} doc(s).\n\n{answer}",
                "duration_s": None,
            }],
        }

    def _run_tool(self, query: str, memory: MemoryManager) -> Dict[str, Any]:
        """Real-time web search + LLM summary. Falls back to direct LLM if search fails."""
        tool_res = tool_registry.execute("web_search", query=query, num_results=5)

        if not tool_res.success:
            logger.warning("[Router] Web search failed (%s) — falling back to DIRECT", tool_res.error)
            result = self._run_direct(query, memory)
            result["step_records"] = [{
                "step_id":    1,
                "step_type":  "execution",
                "task":       f"Web search (failed, used LLM fallback): {query}",
                "status":     "failed",
                "output_full": f"Search error: {tool_res.error}",
                "duration_s": round(tool_res.duration_s, 3),
            }]
            return result

        tool_output = tool_res.format()
        history = memory.get_formatted_history(last_n=8)
        answer  = _llm_call(_tool_prompt(tool_output, history, query), agent_type="research")
        if not answer or not answer.strip():
            answer = "I retrieved search results but couldn't summarise them. Please try again."
        memory.add_message("user", query)
        memory.add_message("assistant", answer)
        return {
            "final_answer": answer,
            "goal":         query,
            "plan":         {},
            "step_records": [{
                "step_id":    1,
                "step_type":  "execution",
                "task":       f"Web search: {query}",
                "status":     "done",
                "output_full": tool_output,
                "duration_s": round(tool_res.duration_s, 3),
            }],
        }

    def _run_pipeline(self, query: str, session_id: str) -> Dict[str, Any]:
        """Full 4-agent orchestrated pipeline."""
        return self._get_orchestrator().run(query, session_id)

    # ------------------------------------------------------------------
    # LLM token streaming  (real-time, runs sync generator in thread)
    # ------------------------------------------------------------------

    def _stream_llm_chunks(
        self, route: RouteType, query: str, memory: MemoryManager
    ) -> Generator[str, None, None]:
        """
        Yield tokens from the LLM's .stream() method.

        The sync LangChain generator runs in a daemon thread; chunks are
        passed to the main thread via a queue so FastAPI's async generator
        can yield them without blocking the event loop.
        """
        history = memory.get_formatted_history(last_n=8)

        if route == RouteType.RAG:
            docs    = memory.retrieve_from_long_term(query, k=settings.LONG_TERM_MEMORY_TOP_K)
            context = (
                "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
                if docs else "No relevant context."
            )
            prompt = _rag_prompt(context, history, query)
        elif route == RouteType.TOOL:
            tool_res    = tool_registry.execute("web_search", query=query, num_results=5)
            tool_output = (
                tool_res.format() if tool_res.success
                else f"Web search unavailable: {tool_res.error}"
            )
            prompt = _tool_prompt(tool_output, history, query)
        else:
            prompt = _direct_prompt(history, query)

        llm = get_llm("research")
        _DONE = object()
        q: _queue_module.Queue = _queue_module.Queue()

        def _producer() -> None:
            try:
                for chunk in llm.stream(prompt):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if text:
                        q.put(text)
            except AttributeError:
                # LLM doesn't support .stream() — fall back
                resp = llm.invoke(prompt)
                text = resp.content if hasattr(resp, "content") else str(resp)
                for i in range(0, len(text), 80):
                    q.put(text[i:i + 80])
            except Exception as exc:           # noqa: BLE001
                q.put(f"\n[Stream error: {exc}]")
            finally:
                q.put(_DONE)

        threading.Thread(target=_producer, daemon=True).start()

        accumulated = []
        while True:
            try:
                chunk = q.get(timeout=120)
            except _queue_module.Empty:
                break
            if chunk is _DONE:
                break
            accumulated.append(chunk)
            yield chunk

        # Store full response in memory
        memory.add_message("user", query)
        memory.add_message("assistant", "".join(accumulated))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_router: Optional[QueryRouter] = None


def get_router() -> QueryRouter:
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router
