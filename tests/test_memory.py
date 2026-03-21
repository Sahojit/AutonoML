"""
Memory system tests — fully mocked vector store.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import pytest
from langchain.docstore.document import Document

from memory.memory_manager import MemoryManager, AgentContext, Message


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def memory(tmp_path):
    """MemoryManager with a mocked vector store."""
    with patch("memory.memory_manager.get_vector_store") as mock_vs_factory:
        mock_vs = MagicMock()
        mock_vs.add_documents.return_value = ["id-1"]
        mock_vs.similarity_search.return_value = [
            Document(page_content="Mock doc content", metadata={"source": "test"})
        ]
        mock_vs_factory.return_value = mock_vs
        yield MemoryManager(session_id="test-session"), mock_vs


# ──────────────────────────────────────────────────────────────────────────────
# Short-term memory
# ──────────────────────────────────────────────────────────────────────────────

class TestShortTermMemory:

    def test_add_and_retrieve_messages(self, memory):
        mem, _ = memory
        mem.add_message("user", "Hello!")
        mem.add_message("assistant", "Hi there!")
        history = mem.get_conversation_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_formatted_history(self, memory):
        mem, _ = memory
        mem.add_message("user", "Query 1")
        mem.add_message("assistant", "Answer 1", agent_name="execution")
        fmt = mem.get_formatted_history()
        assert "USER" in fmt
        assert "ASSISTANT" in fmt
        assert "[execution]" in fmt

    def test_clear_short_term(self, memory):
        mem, _ = memory
        mem.add_message("user", "test")
        mem.clear_short_term()
        assert len(mem.get_conversation_history()) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Long-term memory
# ──────────────────────────────────────────────────────────────────────────────

class TestLongTermMemory:

    def test_store_in_long_term(self, memory):
        mem, mock_vs = memory
        ids = mem.store_in_long_term(["Some knowledge"], [{"type": "doc"}])
        mock_vs.add_documents.assert_called_once()
        assert len(ids) > 0

    def test_retrieve_from_long_term(self, memory):
        mem, mock_vs = memory
        docs = mem.retrieve_from_long_term("test query")
        mock_vs.similarity_search.assert_called_once()
        assert len(docs) == 1
        assert docs[0].page_content == "Mock doc content"


# ──────────────────────────────────────────────────────────────────────────────
# Agent context
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentContext:

    def test_set_and_get_context(self, memory):
        mem, _ = memory
        ctx = AgentContext(user_query="Analyse data")
        mem.set_context(ctx)
        retrieved = mem.get_context()
        assert retrieved is ctx
        assert retrieved.user_query == "Analyse data"

    def test_update_context(self, memory):
        mem, _ = memory
        ctx = AgentContext(user_query="Test")
        mem.set_context(ctx)
        mem.update_context(plan="Step 1: do X")
        assert mem.get_context().plan == "Step 1: do X"

    def test_update_context_unknown_key_goes_to_metadata(self, memory):
        mem, _ = memory
        ctx = AgentContext(user_query="Test")
        mem.set_context(ctx)
        mem.update_context(custom_key="custom_value")
        assert ctx.metadata.get("custom_key") == "custom_value"
