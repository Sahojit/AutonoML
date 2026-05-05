"""
Two-tier Memory Manager
───────────────────────
• Short-term  – in-process sliding window (deque) of the current conversation.
• Long-term   – Chroma vector store; persists across restarts.

Research Agent queries long-term memory BEFORE generating a response.
Summaries and full conversations are stored back after each session.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from config.settings import settings
from memory.vector_store import get_vector_store

logger = logging.getLogger("multiagent.memory")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    role: str                     # "user" | "assistant" | "system"
    content: str
    agent_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "agent_name": self.agent_name,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentContext:
    """Mutable snapshot of the current pipeline state."""
    user_query: str
    plan: Optional[str] = None
    retrieved_docs: List[Document] = field(default_factory=list)
    execution_result: Optional[str] = None
    evaluation_feedback: Optional[str] = None
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    Central memory hub shared by all agents within a session.

    Short-term  : deque capped at ``SHORT_TERM_MEMORY_LIMIT`` turns.
    Long-term   : Chroma vector store (persistent across restarts).

    The Research Agent calls ``retrieve_from_long_term()`` before answering
    to pull relevant past context.  Intermediate reasoning and summaries are
    stored via ``store_in_long_term()``.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._short_term: deque[Message] = deque(
            maxlen=settings.SHORT_TERM_MEMORY_LIMIT
        )
        self._vector_store = get_vector_store()
        self._context: Optional[AgentContext] = None
        logger.info("MemoryManager initialised for session=%s", session_id)

    # ------------------------------------------------------------------
    # Short-term
    # ------------------------------------------------------------------

    def add_message(
        self,
        role: str,
        content: str,
        agent_name: Optional[str] = None,
    ) -> None:
        msg = Message(role=role, content=content, agent_name=agent_name)
        self._short_term.append(msg)
        logger.debug(
            "[MemoryManager] session=%s | added %s message (%d chars)",
            self.session_id, role, len(content),
        )

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._short_term]

    def get_formatted_history(self, last_n: int = 10) -> str:
        msgs = list(self._short_term)[-last_n:]
        lines = []
        for m in msgs:
            tag = f"[{m.agent_name}]" if m.agent_name else ""
            lines.append(f"{m.role.upper()}{tag}: {m.content}")
        return "\n".join(lines)

    def clear_short_term(self) -> None:
        self._short_term.clear()
        logger.debug("[MemoryManager] session=%s | short-term cleared", self.session_id)

    # ------------------------------------------------------------------
    # Long-term (vector store)
    # ------------------------------------------------------------------

    def store_in_long_term(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Embed and persist *texts* in Chroma with session metadata."""
        base_meta = metadatas or [{}] * len(texts)
        enriched = []
        for meta in base_meta:
            m = meta.copy()
            m["session_id"] = self.session_id
            m["timestamp"]  = time.time()
            enriched.append(m)
        try:
            ids = self._vector_store.add_documents(texts, enriched)
            logger.debug(
                "[MemoryManager] session=%s | stored %d docs in long-term",
                self.session_id, len(ids),
            )
            return ids
        except Exception as exc:           # noqa: BLE001
            logger.warning(
                "[MemoryManager] long-term store failed: %s", exc
            )
            return []

    def retrieve_from_long_term(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Return top-k semantically relevant documents from Chroma."""
        try:
            return self._vector_store.similarity_search(
                query, k=k or settings.LONG_TERM_MEMORY_TOP_K, filter=filter
            )
        except Exception as exc:           # noqa: BLE001
            logger.warning("[MemoryManager] retrieval failed: %s", exc)
            return []

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        try:
            return self._vector_store.similarity_search_with_score(
                query, k=k or settings.LONG_TERM_MEMORY_TOP_K
            )
        except Exception as exc:           # noqa: BLE001
            logger.warning("[MemoryManager] score-retrieval failed: %s", exc)
            return []

    def get_collection_count(self) -> int:
        try:
            return self._vector_store.get_collection_count()
        except Exception:                  # noqa: BLE001
            return 0

    # ------------------------------------------------------------------
    # Agent context
    # ------------------------------------------------------------------

    def set_context(self, context: AgentContext) -> None:
        self._context = context

    def get_context(self) -> Optional[AgentContext]:
        return self._context

    def update_context(self, **kwargs: Any) -> None:
        if self._context is None:
            # Auto-create a minimal context rather than raising
            self._context = AgentContext(user_query="")
        for key, value in kwargs.items():
            if hasattr(self._context, key):
                setattr(self._context, key, value)
            else:
                self._context.metadata[key] = value

    # ------------------------------------------------------------------
    # Self-reflection memory
    # ------------------------------------------------------------------

    def store_reflection(
        self,
        task: str,
        error: str,
        suggestion: str,
    ) -> None:
        """
        Persist a failure-case reflection so future executions can avoid
        repeating the same mistake.

        Stored with type="reflection" metadata so it can be filtered
        separately from general conversation history.
        """
        text = (
            f"[REFLECTION]\n"
            f"Task: {task}\n"
            f"Error: {error}\n"
            f"Suggestion: {suggestion}"
        )
        self.store_in_long_term(
            texts=[text],
            metadatas=[{
                "type": "reflection",
                "session_id": self.session_id,
            }],
        )
        logger.info(
            "[MemoryManager] session=%s | reflection stored for task: %s",
            self.session_id, task[:80],
        )

    def retrieve_reflections(self, task: str, k: int = 3) -> list[str]:
        """
        Return up to *k* past reflection texts semantically similar to *task*.

        The ExecutionAgent injects these into its prompt as
        "Previous mistakes to avoid" before each run.
        """
        try:
            docs = self._vector_store.similarity_search(
                query=task,
                k=k,
                filter={"type": "reflection"},
            )
            return [d.page_content for d in docs]
        except Exception as exc:           # noqa: BLE001
            # Chroma filter may not be supported on all backends — fall back
            # to unfiltered search and extract only reflection entries
            logger.debug(
                "[MemoryManager] filtered reflection search failed (%s); "
                "falling back to unfiltered.", exc,
            )
            try:
                docs = self._vector_store.similarity_search(query=task, k=k * 2)
                return [
                    d.page_content
                    for d in docs
                    if "[REFLECTION]" in d.page_content
                ][:k]
            except Exception:              # noqa: BLE001
                return []

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def save_session_to_long_term(self) -> None:
        """Persist the full conversation into the vector DB at session end."""
        history = self.get_conversation_history()
        if not history:
            return
        texts = [
            f"{m['role']}: {m['content']}"
            for m in history
            if m.get("role") in ("user", "assistant")
        ]
        if not texts:
            return
        meta = [
            {"type": "conversation", "session_id": self.session_id}
            for _ in texts
        ]
        self.store_in_long_term(texts, meta)
        logger.info(
            "[MemoryManager] session=%s | persisted %d messages to long-term.",
            self.session_id, len(texts),
        )
