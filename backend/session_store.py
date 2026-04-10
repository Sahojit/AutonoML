"""
SessionStore
────────────
Pluggable session registry for MemoryManager instances.

Two backends:
  InMemorySessionStore  — default, single-process dict (no extra deps)
  RedisSessionStore     — multi-worker safe; requires redis-py + a running Redis

The active backend is selected at startup via settings.REDIS_URL:
  - unset / empty  → InMemorySessionStore
  - set             → RedisSessionStore

Usage::

    from backend.session_store import get_session_store
    store = get_session_store()
    mem   = store.get_or_create("session-id")
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from config.settings import settings
from memory.memory_manager import MemoryManager

logger = logging.getLogger("multiagent.session_store")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SessionStore(ABC):
    """Interface shared by all session backends."""

    @abstractmethod
    def get_or_create(self, session_id: str) -> MemoryManager:
        """Return an existing MemoryManager or create a new one."""

    @abstractmethod
    def get(self, session_id: str) -> Optional[MemoryManager]:
        """Return an existing MemoryManager or None."""

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""

    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Return True if the session is registered."""

    @abstractmethod
    def count(self) -> int:
        """Return number of active sessions."""

    @abstractmethod
    def all_ids(self) -> List[str]:
        """Return all active session IDs."""


# ---------------------------------------------------------------------------
# In-memory backend (default)
# ---------------------------------------------------------------------------

class InMemorySessionStore(SessionStore):
    """
    Single-process in-memory store backed by a plain dict.
    Sessions are lost on process restart.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, MemoryManager] = {}
        logger.info("InMemorySessionStore initialised.")

    def get_or_create(self, session_id: str) -> MemoryManager:
        if session_id not in self._sessions:
            self._sessions[session_id] = MemoryManager(session_id=session_id)
            logger.info("New session: %s (total=%d)", session_id, len(self._sessions))
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[MemoryManager]:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("Session deleted: %s", session_id)
            return True
        return False

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def count(self) -> int:
        return len(self._sessions)

    def all_ids(self) -> List[str]:
        return list(self._sessions.keys())


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

class RedisSessionStore(SessionStore):
    """
    Multi-worker session store backed by Redis.

    Stores conversation history (JSON) in Redis with a TTL.
    MemoryManager instances are kept in a local process cache; on cache
    miss the history is rehydrated from Redis.

    Requires:  pip install redis
    Config:    REDIS_URL=redis://localhost:6379/0
               SESSION_TTL_SECONDS=3600
    """

    def __init__(self, redis_url: str, ttl: int = 3600) -> None:
        try:
            import redis as redis_lib
        except ImportError as exc:
            raise ImportError(
                "redis package is required for RedisSessionStore. "
                "Install it with: pip install redis"
            ) from exc

        self._redis = redis_lib.from_url(redis_url, decode_responses=True)
        self._ttl   = ttl
        self._local: Dict[str, MemoryManager] = {}   # process-local cache
        logger.info("RedisSessionStore initialised — url=%s  ttl=%ds", redis_url, ttl)

    # key helpers
    def _history_key(self, sid: str) -> str:
        return f"session:{sid}:history"

    def _meta_key(self, sid: str) -> str:
        return f"session:{sid}:meta"

    def get_or_create(self, session_id: str) -> MemoryManager:
        if session_id in self._local:
            self._redis.expire(self._history_key(session_id), self._ttl)
            return self._local[session_id]

        mem = MemoryManager(session_id=session_id)

        # Rehydrate history from Redis if it exists
        raw = self._redis.get(self._history_key(session_id))
        if raw:
            try:
                messages = json.loads(raw)
                for msg in messages:
                    mem.add_message(msg["role"], msg["content"])
                logger.info("Rehydrated session %s (%d messages)", session_id, len(messages))
            except Exception:                # noqa: BLE001
                logger.warning("Failed to rehydrate session %s", session_id)

        self._local[session_id] = mem
        return mem

    def get(self, session_id: str) -> Optional[MemoryManager]:
        if session_id in self._local:
            return self._local[session_id]
        if self._redis.exists(self._history_key(session_id)):
            return self.get_or_create(session_id)
        return None

    def delete(self, session_id: str) -> bool:
        existed = self._redis.delete(
            self._history_key(session_id),
            self._meta_key(session_id),
        ) > 0
        self._local.pop(session_id, None)
        if existed:
            logger.info("Session deleted from Redis: %s", session_id)
        return existed

    def exists(self, session_id: str) -> bool:
        return (
            session_id in self._local
            or bool(self._redis.exists(self._history_key(session_id)))
        )

    def count(self) -> int:
        keys = self._redis.keys("session:*:history")
        return len(keys)

    def all_ids(self) -> List[str]:
        keys = self._redis.keys("session:*:history")
        return [k.split(":")[1] for k in keys]

    def flush_to_redis(self, session_id: str) -> None:
        """Persist the in-process MemoryManager history to Redis."""
        mem = self._local.get(session_id)
        if not mem:
            return
        try:
            history = mem.get_conversation_history()
            self._redis.set(
                self._history_key(session_id),
                json.dumps(history),
                ex=self._ttl,
            )
        except Exception:                    # noqa: BLE001
            logger.warning("Failed to flush session %s to Redis", session_id)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_store: Optional[SessionStore] = None


def get_session_store() -> SessionStore:
    """Return the singleton SessionStore, creating it on first call."""
    global _store
    if _store is None:
        if settings.REDIS_URL:
            logger.info("Using RedisSessionStore — %s", settings.REDIS_URL)
            _store = RedisSessionStore(
                redis_url=settings.REDIS_URL,
                ttl=settings.SESSION_TTL_SECONDS,
            )
        else:
            _store = InMemorySessionStore()
    return _store
