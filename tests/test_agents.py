"""
Agent unit tests — use mocked LLMs and vector stores so they run without Ollama.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.planner_agent import PlannerAgent, Plan
from agents.evaluation_agent import EvaluationAgent, EvaluationResult
from memory.memory_manager import MemoryManager, AgentContext


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_memory(tmp_path):
    with patch("memory.vector_store.get_embedding_model"), \
         patch("chromadb.PersistentClient"):
        mem = MagicMock(spec=MemoryManager)
        mem.get_formatted_history.return_value = ""
        mem.add_message = MagicMock()
        mem.update_context = MagicMock()
        mem.store_in_long_term = MagicMock(return_value=["id-1"])
        ctx = AgentContext(user_query="test query")
        mem.get_context.return_value = ctx
        return mem


# ──────────────────────────────────────────────────────────────────────────────
# PlannerAgent tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPlannerAgent:

    VALID_JSON_PLAN = json.dumps([
        {"step_number": 1, "agent": "research", "instruction": "Find docs", "depends_on": []},
        {"step_number": 2, "agent": "execution", "instruction": "Run analysis", "depends_on": [1]},
        {"step_number": 3, "agent": "evaluation", "instruction": "Validate", "depends_on": [2]},
    ])

    @patch("agents.planner_agent.get_llm")
    def test_plan_parsed_correctly(self, mock_get_llm, mock_memory):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self.VALID_JSON_PLAN
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self.VALID_JSON_PLAN
        mock_get_llm.return_value = mock_llm

        with patch("agents.planner_agent.PLANNER_PROMPT.__or__", return_value=mock_chain):
            agent = PlannerAgent()
            agent._chain = mock_chain
            plan = agent.run("Analyse the iris dataset", mock_memory)

        assert isinstance(plan, Plan)
        assert len(plan.steps) == 3
        assert plan.steps[0].agent == "research"
        assert plan.steps[1].agent == "execution"
        assert plan.steps[2].agent == "evaluation"

    @patch("agents.planner_agent.get_llm")
    def test_fallback_plan_on_invalid_json(self, mock_get_llm, mock_memory):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "This is not JSON at all."
        mock_get_llm.return_value = MagicMock()

        agent = PlannerAgent()
        agent._chain = mock_chain
        plan = agent.run("Do something", mock_memory)

        assert isinstance(plan, Plan)
        assert len(plan.steps) >= 2  # fallback has at least 3 steps


# ──────────────────────────────────────────────────────────────────────────────
# EvaluationAgent tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluationAgent:

    PASS_VERDICT = json.dumps({
        "verdict": "PASS",
        "score": 0.95,
        "issues": [],
        "suggestions": [],
        "corrected_output": "",
    })

    RETRY_VERDICT = json.dumps({
        "verdict": "RETRY",
        "score": 0.60,
        "issues": ["Missing error handling"],
        "suggestions": ["Add try/except block"],
        "corrected_output": "try:\n    pass\nexcept Exception:\n    pass",
    })

    @patch("agents.evaluation_agent.get_llm")
    def test_pass_verdict(self, mock_get_llm, mock_memory):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self.PASS_VERDICT
        mock_get_llm.return_value = MagicMock()

        agent = EvaluationAgent()
        agent._chain = mock_chain
        result = agent.run("Run analysis", "Output here", mock_memory)

        assert isinstance(result, EvaluationResult)
        assert result.verdict == "PASS"
        assert result.score == 0.95
        assert result.passed() is True

    @patch("agents.evaluation_agent.get_llm")
    def test_retry_verdict(self, mock_get_llm, mock_memory):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self.RETRY_VERDICT
        mock_get_llm.return_value = MagicMock()

        agent = EvaluationAgent()
        agent._chain = mock_chain
        result = agent.run("Run analysis", "Incomplete output", mock_memory)

        assert result.verdict == "RETRY"
        assert result.passed() is False
        assert len(result.issues) == 1
        assert result.corrected_output != ""

    @patch("agents.evaluation_agent.get_llm")
    def test_default_pass_on_empty_response(self, mock_get_llm, mock_memory):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = ""
        mock_get_llm.return_value = MagicMock()

        agent = EvaluationAgent()
        agent._chain = mock_chain
        result = agent.run("instruction", "output", mock_memory)

        # Should fall back to PASS rather than crashing
        assert result.verdict == "PASS"
