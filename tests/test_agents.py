import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.planner_agent import PlannerAgent, PlannerOutput
from agents.evaluation_agent import EvaluationAgent
from memory.memory_manager import AgentContext


class TestPlannerAgent:

    VALID_PLAN = json.dumps({
        "goal": "Analyse the iris dataset",
        "tasks": [
            {"id": 1, "type": "research",   "task": "Find docs",      "depends_on": []},
            {"id": 2, "type": "execution",  "task": "Run analysis",   "depends_on": [1]},
            {"id": 3, "type": "evaluation", "task": "Validate result", "depends_on": [2]},
        ],
    })

    @pytest.fixture
    def planner(self):
        with patch("agents.planner_agent.get_llm"):
            return PlannerAgent()

    def test_parse_returns_planner_output(self, planner):
        plan = planner._parse_output(self.VALID_PLAN, "analyse iris")
        assert isinstance(plan, PlannerOutput)

    def test_parse_goal(self, planner):
        plan = planner._parse_output(self.VALID_PLAN, "analyse iris")
        assert plan.goal == "Analyse the iris dataset"

    def test_parse_task_count(self, planner):
        plan = planner._parse_output(self.VALID_PLAN, "analyse iris")
        assert len(plan.tasks) == 3

    def test_parse_task_types(self, planner):
        plan = planner._parse_output(self.VALID_PLAN, "analyse iris")
        types = [t.type for t in plan.tasks]
        assert types == ["research", "execution", "evaluation"]

    def test_parse_task_ids(self, planner):
        plan = planner._parse_output(self.VALID_PLAN, "analyse iris")
        assert [t.id for t in plan.tasks] == [1, 2, 3]

    def test_parse_dependencies(self, planner):
        plan = planner._parse_output(self.VALID_PLAN, "analyse iris")
        assert plan.tasks[0].depends_on == []
        assert plan.tasks[1].depends_on == [1]
        assert plan.tasks[2].depends_on == [2]

    def test_fallback_on_empty_string(self, planner):
        plan = planner._parse_output("", "do something")
        assert isinstance(plan, PlannerOutput)
        assert len(plan.tasks) == 3

    def test_fallback_on_invalid_json(self, planner):
        plan = planner._parse_output("{not json}", "do something")
        assert len(plan.tasks) == 3

    def test_fallback_task_types(self, planner):
        plan = planner._fallback("run the script")
        types = [t.type for t in plan.tasks]
        assert "research" in types
        assert "execution" in types
        assert "evaluation" in types

    def test_independent_tasks(self, planner):
        raw = json.dumps({
            "goal": "Parallel work",
            "tasks": [
                {"id": 1, "type": "research", "task": "A", "depends_on": []},
                {"id": 2, "type": "research", "task": "B", "depends_on": []},
                {"id": 3, "type": "execution", "task": "C", "depends_on": [1, 2]},
            ],
        })
        plan = planner._parse_output(raw, "parallel")
        assert len(plan.independent_tasks()) == 2

    def test_to_dict_has_tasks_key(self, planner):
        plan = planner._parse_output(self.VALID_PLAN, "analyse iris")
        d = plan.to_dict()
        assert "tasks" in d
        assert "goal" in d


class TestEvaluationAgent:

    PASS_JSON = json.dumps({
        "verdict": "PASS",
        "score": 0.92,
        "issues": [],
        "suggestions": [],
    })

    FAIL_JSON = json.dumps({
        "verdict": "FAIL",
        "score": 0.55,
        "issues": ["Missing metric"],
        "suggestions": ["Add accuracy score"],
    })

    @pytest.fixture
    def evaluator(self):
        with patch("agents.evaluation_agent.get_llm"):
            return EvaluationAgent()

    @pytest.fixture
    def mock_memory(self):
        mem = MagicMock()
        mem.add_message = MagicMock()
        mem.update_context = MagicMock()
        ctx = AgentContext(user_query="test")
        mem.get_context.return_value = ctx
        return mem

    def _run(self, evaluator, mock_memory, llm_json, instruction, output):
        chain = MagicMock()
        chain.invoke.return_value = MagicMock(content=llm_json)
        evaluator._chain = chain
        return evaluator.run(instruction, output, mock_memory)

    def test_pass_verdict_returned(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.PASS_JSON,
                        "explain recursion", "Recursion calls itself.")
        assert msg.metadata["verdict"] == "PASS"

    def test_fail_verdict_returned(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.FAIL_JSON,
                        "explain recursion", "Recursion calls itself.")
        assert msg.metadata["verdict"] == "FAIL"

    def test_score_preserved(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.PASS_JSON,
                        "explain recursion", "Recursion calls itself.")
        assert msg.metadata["score"] == 0.92

    def test_issues_list_preserved(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.FAIL_JSON,
                        "explain recursion", "Recursion calls itself.")
        assert "Missing metric" in msg.metadata["issues"]

    def test_empty_llm_response_defaults_pass(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, "",
                        "explain recursion", "Recursion calls itself.")
        assert msg.metadata["verdict"] == "PASS"

    def test_malformed_json_defaults_pass(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, "not json",
                        "explain recursion", "Recursion calls itself.")
        assert msg.metadata["verdict"] == "PASS"

    def test_programmatic_fail_overrides_llm_pass(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.PASS_JSON,
                        "train a classifier", "accuracy: 1.0",
                        )
        assert msg.metadata["verdict"] == "FAIL"

    def test_duration_in_metadata(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.PASS_JSON,
                        "explain recursion", "Recursion calls itself.")
        assert "duration_s" in msg.metadata
        assert msg.metadata["duration_s"] >= 0
