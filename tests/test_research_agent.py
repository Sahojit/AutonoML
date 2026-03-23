import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import pytest
from langchain.docstore.document import Document

from agents.research_agent import ResearchAgent
from agents.base_agent import AgentMessage
from memory.memory_manager import AgentContext


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.get_formatted_history.return_value = ""
    mem.add_message = MagicMock()
    mem.update_context = MagicMock()
    mem.store_in_long_term = MagicMock(return_value=["id-1"])
    ctx = AgentContext(user_query="test")
    mem.get_context.return_value = ctx
    return mem


@pytest.fixture
def agent():
    with patch("agents.research_agent.get_llm"), \
         patch("agents.research_agent.get_vector_store") as mock_vs_factory, \
         patch("agents.research_agent.get_web_search"):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        mock_vs_factory.return_value = mock_vs
        return ResearchAgent(use_web_search=False)


@pytest.fixture
def agent_with_docs():
    with patch("agents.research_agent.get_llm"), \
         patch("agents.research_agent.get_vector_store") as mock_vs_factory, \
         patch("agents.research_agent.get_web_search"):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            Document(page_content="Random Forest uses ensemble of decision trees.", metadata={"source": "ml_docs"}),
            Document(page_content="Hyperparameters: n_estimators, max_depth.", metadata={"source": "ml_docs"}),
        ]
        mock_vs_factory.return_value = mock_vs
        return ResearchAgent(use_web_search=False), mock_vs


class TestResearchAgentRun:

    def test_run_returns_agent_message(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Research summary here.")
        agent._chain = chain
        msg = agent.run("What is Random Forest?", mock_memory)
        assert isinstance(msg, AgentMessage)

    def test_run_status_is_success(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Summary.")
        agent._chain = chain
        msg = agent.run("What is Random Forest?", mock_memory)
        assert msg.status == "success"

    def test_run_output_is_string(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Summary text.")
        agent._chain = chain
        msg = agent.run("Describe XGBoost", mock_memory)
        assert isinstance(msg.output, str)

    def test_run_stores_summary_in_memory(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Stored summary.")
        agent._chain = chain
        agent.run("Research query", mock_memory)
        mock_memory.store_in_long_term.assert_called_once()

    def test_run_updates_context_with_retrieved_docs(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Summary.")
        agent._chain = chain
        agent.run("Research query", mock_memory)
        mock_memory.update_context.assert_called()

    def test_run_metadata_contains_kb_docs_count(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Summary.")
        agent._chain = chain
        msg = agent.run("Research query", mock_memory)
        assert "kb_docs_retrieved" in msg.metadata

    def test_run_metadata_contains_duration(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Summary.")
        agent._chain = chain
        msg = agent.run("Research query", mock_memory)
        assert "duration_s" in msg.metadata
        assert msg.metadata["duration_s"] >= 0

    def test_run_with_kb_docs_reflects_count(self, agent_with_docs, mock_memory):
        agent, mock_vs = agent_with_docs
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Summary with docs.")
        agent._chain = chain
        msg = agent.run("Random Forest hyperparameters", mock_memory)
        assert msg.metadata["kb_docs_retrieved"] == 2

    def test_run_web_search_not_used_when_disabled(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Summary.")
        agent._chain = chain
        msg = agent.run("query", mock_memory, use_web=False)
        assert msg.metadata["web_search_used"] is False

    def test_run_llm_failure_returns_fallback_summary(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.side_effect = RuntimeError("LLM unavailable")
        agent._chain = chain
        msg = agent.run("Some query", mock_memory)
        assert msg.status == "success"
        assert isinstance(msg.output, str)
        assert len(msg.output) > 0


class TestResearchAgentFormatDocs:

    def test_format_empty_docs_returns_not_found(self, agent, mock_memory):
        result = agent._format_docs([])
        assert "No relevant" in result

    def test_format_single_doc_includes_content(self, agent, mock_memory):
        docs = [Document(page_content="Python is interpreted.", metadata={"source": "wiki"})]
        result = agent._format_docs(docs)
        assert "Python is interpreted." in result

    def test_format_doc_includes_source(self, agent, mock_memory):
        docs = [Document(page_content="Content.", metadata={"source": "my_source"})]
        result = agent._format_docs(docs)
        assert "my_source" in result

    def test_format_multiple_docs_numbered(self, agent, mock_memory):
        docs = [
            Document(page_content="Doc A", metadata={"source": "src"}),
            Document(page_content="Doc B", metadata={"source": "src"}),
        ]
        result = agent._format_docs(docs)
        assert "Doc 1" in result
        assert "Doc 2" in result
