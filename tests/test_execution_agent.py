import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import pytest

from agents.execution_agent import ExecutionAgent
from agents.base_agent import AgentMessage
from memory.memory_manager import AgentContext
from tools.tool_registry import ToolResult


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.get_formatted_history.return_value = ""
    mem.retrieve_reflections.return_value = []
    mem.add_message = MagicMock()
    mem.update_context = MagicMock()
    ctx = AgentContext(user_query="test")
    mem.get_context.return_value = ctx
    return mem


@pytest.fixture
def agent():
    with patch("agents.execution_agent.get_llm"):
        return ExecutionAgent()


def _make_tool_result(name="python_exec", success=True, output="42"):
    return ToolResult(tool_name=name, success=success, output=output)


class TestExecutionAgentDispatch:

    def test_python_code_block_dispatches_python_exec(self, agent):
        text = "```python\nprint(42)\n```"
        with patch.object(agent, "_dispatch", wraps=agent._dispatch) as spy, \
             patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.execute.return_value = _make_tool_result()
            mock_reg.__contains__ = MagicMock(return_value=True)
            result, called = agent._dispatch(text)
            mock_reg.execute.assert_called_once_with("python_exec", code="print(42)")
            assert called is True

    def test_web_search_pattern_dispatches_web_search(self, agent):
        text = "WEB_SEARCH: latest ML papers 2024"
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.execute.return_value = _make_tool_result("web_search", output="results")
            result, called = agent._dispatch(text)
            mock_reg.execute.assert_called_once_with("web_search", query="latest ML papers 2024")
            assert called is True

    def test_sql_query_pattern_dispatches_sql_query(self, agent):
        text = "SQL_QUERY: SELECT * FROM employees LIMIT 5"
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.execute.return_value = _make_tool_result("sql_query", output="rows")
            result, called = agent._dispatch(text)
            mock_reg.execute.assert_called_once_with(
                "sql_query", query="SELECT * FROM employees LIMIT 5"
            )
            assert called is True

    def test_dataset_analyze_pattern_dispatches_dataset_analyze(self, agent):
        text = "DATASET_ANALYZE: sklearn:iris"
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.execute.return_value = _make_tool_result("dataset_analyze", output="stats")
            result, called = agent._dispatch(text)
            mock_reg.execute.assert_called_once_with("dataset_analyze", source="sklearn:iris")
            assert called is True

    def test_auto_experiment_pattern_dispatches_auto_experiment(self, agent):
        text = "AUTO_EXPERIMENT: sklearn:wine"
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.execute.return_value = _make_tool_result("auto_experiment", output="leaderboard")
            result, called = agent._dispatch(text)
            mock_reg.execute.assert_called_once_with("auto_experiment", source="sklearn:wine")
            assert called is True

    def test_no_tool_pattern_returns_not_called(self, agent):
        text = "This is just a plain text response with no tool call."
        result, called = agent._dispatch(text)
        assert called is False
        assert result.tool_name == "none"

    def test_python_block_takes_priority_over_other_patterns(self, agent):
        text = "```python\nprint('hi')\n```\nWEB_SEARCH: something"
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.execute.return_value = _make_tool_result()
            _, called = agent._dispatch(text)
            call_args = mock_reg.execute.call_args
            assert call_args[0][0] == "python_exec"

    def test_tool_call_generic_pattern_dispatches(self, agent):
        text = 'TOOL_CALL: python_exec {"code": "print(1)"}'
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.execute.return_value = _make_tool_result()
            mock_reg.__contains__ = MagicMock(return_value=True)
            result, called = agent._dispatch(text)
            assert called is True


class TestExecutionAgentRun:

    def test_run_returns_agent_message(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="The answer is 42.")
        agent._chain = chain
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.tools_summary.return_value = "tool list"
            msg = agent.run("What is 6 * 7?", mock_memory)
        assert isinstance(msg, AgentMessage)

    def test_run_output_is_string(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Result: 42")
        agent._chain = chain
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.tools_summary.return_value = ""
            msg = agent.run("Compute 6*7", mock_memory)
        assert isinstance(msg.output, str)

    def test_run_includes_tool_trace_in_metadata(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Done.")
        agent._chain = chain
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.tools_summary.return_value = ""
            msg = agent.run("Simple task", mock_memory)
        assert "tool_trace" in msg.metadata

    def test_run_with_research_context(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Answer using context.")
        agent._chain = chain
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.tools_summary.return_value = ""
            msg = agent.run("Summarise", mock_memory, research_context="Some background docs.")
        assert msg.status == "success"

    def test_run_with_feedback_injected(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="Improved answer.")
        agent._chain = chain
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.tools_summary.return_value = ""
            msg = agent.run(
                "Fix the model", mock_memory,
                feedback="Add train_test_split to avoid leakage."
            )
        assert msg.status == "success"

    def test_run_status_success_on_normal_output(self, agent, mock_memory):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content="All good.")
        agent._chain = chain
        with patch("agents.execution_agent.tool_registry") as mock_reg:
            mock_reg.tools_summary.return_value = ""
            msg = agent.run("Normal task", mock_memory)
        assert msg.status == "success"
