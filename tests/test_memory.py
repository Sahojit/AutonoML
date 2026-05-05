"""
Memory system tests — fully mocked vector store.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

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

    def test_update_context_without_prior_set_auto_creates(self, memory):
        mem, _ = memory
        mem.update_context(plan="auto plan")
        assert mem.get_context() is not None
        assert mem.get_context().plan == "auto plan"

    def test_get_context_returns_none_before_set(self, memory):
        mem, _ = memory
        assert mem.get_context() is None

    def test_context_execution_result_field(self, memory):
        mem, _ = memory
        ctx = AgentContext(user_query="Run model")
        mem.set_context(ctx)
        mem.update_context(execution_result="accuracy: 0.92")
        assert mem.get_context().execution_result == "accuracy: 0.92"

    def test_context_evaluation_feedback_field(self, memory):
        mem, _ = memory
        ctx = AgentContext(user_query="Run model")
        mem.set_context(ctx)
        mem.update_context(evaluation_feedback="PASS score=0.91")
        assert mem.get_context().evaluation_feedback == "PASS score=0.91"


# ──────────────────────────────────────────────────────────────────────────────
# Self-reflection memory
# ──────────────────────────────────────────────────────────────────────────────

class TestReflectionMemory:

    def test_store_reflection_calls_add_documents(self, memory):
        mem, mock_vs = memory
        mock_vs.add_documents.return_value = ["ref-1"]
        mem.store_reflection(
            task="train classifier",
            error="No train_test_split found",
            suggestion="Add train_test_split before model.fit()",
        )
        mock_vs.add_documents.assert_called_once()

    def test_store_reflection_content_includes_task(self, memory):
        mem, mock_vs = memory
        mock_vs.add_documents.return_value = ["ref-1"]
        mem.store_reflection(
            task="train classifier",
            error="NaN in output",
            suggestion="Clean the dataset",
        )
        stored_text = mock_vs.add_documents.call_args[0][0][0]
        assert "train classifier" in stored_text

    def test_store_reflection_metadata_type_is_reflection(self, memory):
        mem, mock_vs = memory
        mock_vs.add_documents.return_value = ["ref-1"]
        mem.store_reflection(task="t", error="e", suggestion="s")
        stored_meta = mock_vs.add_documents.call_args[0][1][0]
        assert stored_meta.get("type") == "reflection"

    def test_retrieve_reflections_returns_list(self, memory):
        mem, mock_vs = memory
        from langchain_core.documents import Document
        mock_vs.similarity_search.return_value = [
            Document(
                page_content="[REFLECTION]\nTask: train\nError: NaN\nSuggestion: clean data",
                metadata={"type": "reflection"},
            )
        ]
        reflections = mem.retrieve_reflections("train a model", k=1)
        assert isinstance(reflections, list)
        assert len(reflections) == 1

    def test_retrieve_reflections_on_empty_store(self, memory):
        mem, mock_vs = memory
        mock_vs.similarity_search.return_value = []
        reflections = mem.retrieve_reflections("some task", k=3)
        assert reflections == []


# ──────────────────────────────────────────────────────────────────────────────
# Session persistence
# ──────────────────────────────────────────────────────────────────────────────

class TestSessionPersistence:

    def test_save_session_calls_add_documents(self, memory):
        mem, mock_vs = memory
        mem.add_message("user", "What is ML?")
        mem.add_message("assistant", "Machine learning is...")
        mock_vs.add_documents.return_value = ["s-1", "s-2"]
        mem.save_session_to_long_term()
        mock_vs.add_documents.assert_called_once()

    def test_save_empty_session_skips_store(self, memory):
        mem, mock_vs = memory
        mem.save_session_to_long_term()
        mock_vs.add_documents.assert_not_called()

    def test_save_session_metadata_type_is_conversation(self, memory):
        mem, mock_vs = memory
        mem.add_message("user", "Hello")
        mem.add_message("assistant", "Hi")
        mock_vs.add_documents.return_value = ["s-1"]
        mem.save_session_to_long_term()
        stored_meta = mock_vs.add_documents.call_args[0][1]
        assert all(m.get("type") == "conversation" for m in stored_meta)

    def test_save_session_only_stores_user_and_assistant(self, memory):
        mem, mock_vs = memory
        mem.add_message("user", "Query")
        mem.add_message("system", "Internal system note")
        mem.add_message("assistant", "Answer")
        mock_vs.add_documents.return_value = ["s-1", "s-2"]
        mem.save_session_to_long_term()
        stored_texts = mock_vs.add_documents.call_args[0][0]
        assert all("system" not in t.lower() for t in stored_texts)
