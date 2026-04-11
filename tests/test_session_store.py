"""
Unit tests for SessionStore — InMemorySessionStore and factory.
Redis backend is tested with a mock; no live Redis required.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from backend.session_store import (
    InMemorySessionStore,
    RedisSessionStore,
    get_session_store,
)
from memory.memory_manager import MemoryManager


# ---------------------------------------------------------------------------
# InMemorySessionStore
# ---------------------------------------------------------------------------

class TestInMemorySessionStore:
    def setup_method(self):
        self.store = InMemorySessionStore()

    def test_get_or_create_returns_memory_manager(self):
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.return_value = MagicMock(spec=MemoryManager)
            mem = self.store.get_or_create("s1")
            assert mem is not None

    def test_get_or_create_same_session_returns_same_instance(self):
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.return_value = MagicMock(spec=MemoryManager)
            m1 = self.store.get_or_create("s1")
            m2 = self.store.get_or_create("s1")
            assert m1 is m2

    def test_get_or_create_different_sessions_different_instances(self):
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.side_effect = [MagicMock(spec=MemoryManager), MagicMock(spec=MemoryManager)]
            m1 = self.store.get_or_create("s1")
            m2 = self.store.get_or_create("s2")
            assert m1 is not m2

    def test_get_returns_none_for_unknown(self):
        assert self.store.get("nonexistent") is None

    def test_get_returns_existing_session(self):
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.return_value = MagicMock(spec=MemoryManager)
            self.store.get_or_create("s1")
            assert self.store.get("s1") is not None

    def test_exists_false_for_unknown(self):
        assert self.store.exists("ghost") is False

    def test_exists_true_after_create(self):
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.return_value = MagicMock(spec=MemoryManager)
            self.store.get_or_create("s1")
            assert self.store.exists("s1") is True

    def test_delete_returns_true_for_existing(self):
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.return_value = MagicMock(spec=MemoryManager)
            self.store.get_or_create("s1")
            assert self.store.delete("s1") is True

    def test_delete_returns_false_for_unknown(self):
        assert self.store.delete("ghost") is False

    def test_delete_removes_session(self):
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.return_value = MagicMock(spec=MemoryManager)
            self.store.get_or_create("s1")
            self.store.delete("s1")
            assert self.store.exists("s1") is False

    def test_count_zero_initially(self):
        assert self.store.count() == 0

    def test_count_increments_on_create(self):
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.side_effect = [MagicMock(spec=MemoryManager), MagicMock(spec=MemoryManager)]
            self.store.get_or_create("s1")
            self.store.get_or_create("s2")
            assert self.store.count() == 2

    def test_count_decrements_on_delete(self):
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.return_value = MagicMock(spec=MemoryManager)
            self.store.get_or_create("s1")
            self.store.delete("s1")
            assert self.store.count() == 0

    def test_all_ids_empty_initially(self):
        assert self.store.all_ids() == []

    def test_all_ids_returns_created_sessions(self):
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.side_effect = [MagicMock(spec=MemoryManager), MagicMock(spec=MemoryManager)]
            self.store.get_or_create("s1")
            self.store.get_or_create("s2")
            assert set(self.store.all_ids()) == {"s1", "s2"}


# ---------------------------------------------------------------------------
# RedisSessionStore (mocked Redis)
# ---------------------------------------------------------------------------

class TestRedisSessionStore:
    def _make_store(self):
        mock_redis_client = MagicMock()
        mock_redis_client.get.return_value = None
        mock_redis_client.exists.return_value = 0
        mock_redis_client.keys.return_value = []
        mock_redis_client.delete.return_value = 1

        mock_redis_module = MagicMock()
        mock_redis_module.from_url.return_value = mock_redis_client

        with patch.dict("sys.modules", {"redis": mock_redis_module}):
            with patch("backend.session_store.MemoryManager") as MockMem:
                MockMem.return_value = MagicMock(spec=MemoryManager)
                MockMem.return_value.get_conversation_history.return_value = []
                MockMem.return_value.add_message = MagicMock()
                store = RedisSessionStore.__new__(RedisSessionStore)
                store._redis = mock_redis_client
                store._ttl = 60
                store._local = {}
                return store, mock_redis_client

    def test_get_or_create_returns_memory_manager(self):
        store, _ = self._make_store()
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.return_value = MagicMock(spec=MemoryManager)
            mem = store.get_or_create("s1")
            assert mem is not None

    def test_get_or_create_reuses_local_cache(self):
        store, _ = self._make_store()
        with patch("backend.session_store.MemoryManager") as MockMem:
            MockMem.return_value = MagicMock(spec=MemoryManager)
            m1 = store.get_or_create("s1")
            m2 = store.get_or_create("s1")
            assert m1 is m2

    def test_rehydrates_history_from_redis(self):
        store, mock_redis = self._make_store()
        history = [{"role": "user", "content": "hello"}]
        mock_redis.get.return_value = json.dumps(history)

        with patch("backend.session_store.MemoryManager") as MockMem:
            mock_mem = MagicMock(spec=MemoryManager)
            MockMem.return_value = mock_mem
            store.get_or_create("s2")
            mock_mem.add_message.assert_called_once_with("user", "hello")

    def test_delete_removes_from_redis(self):
        store, mock_redis = self._make_store()
        mock_redis.delete.return_value = 1
        result = store.delete("s1")
        mock_redis.delete.assert_called()
        assert result is True

    def test_delete_returns_false_when_not_exists(self):
        store, mock_redis = self._make_store()
        mock_redis.delete.return_value = 0
        assert store.delete("ghost") is False

    def test_count_from_redis_keys(self):
        store, mock_redis = self._make_store()
        mock_redis.keys.return_value = ["session:s1:history", "session:s2:history"]
        assert store.count() == 2

    def test_all_ids_parsed_from_keys(self):
        store, mock_redis = self._make_store()
        mock_redis.keys.return_value = ["session:abc:history", "session:xyz:history"]
        ids = store.all_ids()
        assert set(ids) == {"abc", "xyz"}

    def test_flush_to_redis_stores_history(self):
        store, mock_redis = self._make_store()
        with patch("backend.session_store.MemoryManager") as MockMem:
            mock_mem = MagicMock(spec=MemoryManager)
            mock_mem.get_conversation_history.return_value = [{"role": "user", "content": "hi"}]
            MockMem.return_value = mock_mem
            store.get_or_create("s1")
            store.flush_to_redis("s1")
            mock_redis.set.assert_called_once()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestGetSessionStore:
    def test_returns_in_memory_when_no_redis_url(self):
        import backend.session_store as ss
        ss._store = None
        with patch("backend.session_store.settings") as mock_settings:
            mock_settings.REDIS_URL = None
            store = get_session_store()
            assert isinstance(store, InMemorySessionStore)
        ss._store = None

    def test_returns_same_singleton(self):
        import backend.session_store as ss
        ss._store = None
        with patch("backend.session_store.settings") as mock_settings:
            mock_settings.REDIS_URL = None
            s1 = get_session_store()
            s2 = get_session_store()
            assert s1 is s2
        ss._store = None
