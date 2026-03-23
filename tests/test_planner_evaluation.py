import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import pytest

from agents.planner_agent import PlannerAgent, PlannerOutput
from agents.evaluation_agent import (
    EvaluationAgent,
    run_programmatic_checks,
    ml_sanity_checks,
)


@pytest.fixture
def planner() -> PlannerAgent:
    with patch("agents.planner_agent.get_llm"):
        return PlannerAgent()


@pytest.fixture
def evaluator() -> EvaluationAgent:
    with patch("agents.evaluation_agent.get_llm"):
        return EvaluationAgent()


VALID_PLAN_TASKS = json.dumps({
    "goal": "Analyse the iris dataset",
    "tasks": [
        {"id": 1, "type": "research",   "task": "Load iris data",      "depends_on": []},
        {"id": 2, "type": "execution",  "task": "Train random forest", "depends_on": [1]},
        {"id": 3, "type": "evaluation", "task": "Validate accuracy",   "depends_on": [2]},
    ],
})

VALID_PLAN_STEPS = json.dumps({
    "goal": "Classify flowers",
    "steps": [
        {"id": 1, "type": "research",  "task": "Research dataset", "depends_on": []},
        {"id": 2, "type": "execution", "task": "Run classifier",   "depends_on": [1]},
    ],
})


class TestPlannerAgentParser:

    def test_valid_json_tasks_key(self, planner):
        plan = planner._parse_output(VALID_PLAN_TASKS, "analyse iris")
        assert isinstance(plan, PlannerOutput)
        assert plan.goal == "Analyse the iris dataset"
        assert len(plan.tasks) == 3

    def test_valid_json_steps_key_backward_compat(self, planner):
        plan = planner._parse_output(VALID_PLAN_STEPS, "classify flowers")
        assert len(plan.tasks) == 2
        assert plan.tasks[0].type == "research"
        assert plan.tasks[1].type == "execution"

    def test_task_types_parsed_correctly(self, planner):
        plan = planner._parse_output(VALID_PLAN_TASKS, "analyse iris")
        assert [s.type for s in plan.tasks] == ["research", "execution", "evaluation"]

    def test_dependencies_preserved(self, planner):
        plan = planner._parse_output(VALID_PLAN_TASKS, "analyse iris")
        assert plan.tasks[0].depends_on == []
        assert plan.tasks[1].depends_on == [1]
        assert plan.tasks[2].depends_on == [2]

    def test_ids_preserved(self, planner):
        plan = planner._parse_output(VALID_PLAN_TASKS, "analyse iris")
        assert [s.id for s in plan.tasks] == [1, 2, 3]

    def test_json_wrapped_in_markdown_code_block(self, planner):
        wrapped = f"```json\n{VALID_PLAN_TASKS}\n```"
        plan = planner._parse_output(wrapped, "analyse iris")
        assert len(plan.tasks) == 3

    def test_json_with_surrounding_prose(self, planner):
        with_prose = f"Here is the plan:\n{VALID_PLAN_TASKS}\nHope this helps!"
        plan = planner._parse_output(with_prose, "analyse iris")
        assert len(plan.tasks) == 3

    def test_empty_response_returns_fallback(self, planner):
        plan = planner._parse_output("", "do something")
        assert isinstance(plan, PlannerOutput)
        assert len(plan.tasks) == 3
        assert plan.tasks[0].type == "research"
        assert plan.tasks[1].type == "execution"
        assert plan.tasks[2].type == "evaluation"

    def test_plain_text_response_returns_fallback(self, planner):
        plan = planner._parse_output("I cannot produce a plan.", "do something")
        assert len(plan.tasks) == 3

    def test_malformed_json_returns_fallback(self, planner):
        plan = planner._parse_output('{"goal": "x", "tasks": [}', "do something")
        assert len(plan.tasks) == 3

    def test_missing_tasks_field_returns_fallback(self, planner):
        raw = json.dumps({"goal": "Analyse", "steps": []})
        plan = planner._parse_output(raw, "analyse")
        assert len(plan.tasks) == 3

    def test_empty_tasks_list_returns_fallback(self, planner):
        raw = json.dumps({"goal": "Analyse", "tasks": []})
        plan = planner._parse_output(raw, "analyse")
        assert len(plan.tasks) == 3

    def test_invalid_task_type_normalised_to_execution(self, planner):
        raw = json.dumps({
            "goal": "Do something",
            "tasks": [{"id": 1, "type": "unknown_type", "task": "Do it", "depends_on": []}],
        })
        plan = planner._parse_output(raw, "do something")
        assert plan.tasks[0].type == "execution"

    def test_missing_goal_uses_query_fallback(self, planner):
        raw = json.dumps({
            "tasks": [{"id": 1, "type": "execution", "task": "Run it", "depends_on": []}],
        })
        plan = planner._parse_output(raw, "run the script")
        assert "run the script" in plan.goal.lower() or plan.goal.startswith("Solve:")

    def test_tasks_of_type_research(self, planner):
        plan = planner._parse_output(VALID_PLAN_TASKS, "analyse iris")
        assert len(plan.tasks_of_type("research")) == 1
        assert plan.tasks_of_type("research")[0].id == 1

    def test_tasks_of_type_evaluation(self, planner):
        plan = planner._parse_output(VALID_PLAN_TASKS, "analyse iris")
        assert len(plan.tasks_of_type("evaluation")) == 1

    def test_tasks_of_type_nonexistent_returns_empty(self, planner):
        plan = planner._parse_output(VALID_PLAN_TASKS, "analyse iris")
        assert plan.tasks_of_type("nonexistent") == []

    def test_independent_tasks_no_deps(self, planner):
        raw = json.dumps({
            "goal": "Parallel research",
            "tasks": [
                {"id": 1, "type": "research",  "task": "Task A", "depends_on": []},
                {"id": 2, "type": "research",  "task": "Task B", "depends_on": []},
                {"id": 3, "type": "execution", "task": "Task C", "depends_on": [1, 2]},
            ],
        })
        plan = planner._parse_output(raw, "parallel research")
        independent = plan.independent_tasks()
        assert len(independent) == 2
        assert all(s.depends_on == [] for s in independent)

    def test_to_dict_structure(self, planner):
        plan = planner._parse_output(VALID_PLAN_TASKS, "analyse iris")
        d = plan.to_dict()
        assert "goal" in d
        assert "tasks" in d
        assert isinstance(d["tasks"], list)
        assert d["tasks"][0]["type"] == "research"

    def test_fallback_plan_structure(self, planner):
        plan = planner._fallback("some query")
        assert plan.tasks[0].depends_on == []
        assert plan.tasks[1].depends_on == [1]
        assert plan.tasks[2].depends_on == [2]


class TestRunProgrammaticChecks:

    def test_clean_non_ml_output_passes(self):
        passed, issues = run_programmatic_checks(
            "Explain recursion", "Recursion is a technique where a function calls itself."
        )
        assert passed is True
        assert issues == []

    def test_nan_in_output_fails(self):
        passed, issues = run_programmatic_checks("Compute stats", "Result: nan")
        assert passed is False
        assert any("NaN" in i for i in issues)

    def test_traceback_in_output_fails(self):
        passed, issues = run_programmatic_checks(
            "Run code", "Traceback (most recent call last):\n  File error"
        )
        assert passed is False
        assert any("traceback" in i.lower() or "error" in i.lower() for i in issues)

    def test_ml_task_no_split_fails(self):
        code = "model.fit(X, y)\nprint(accuracy_score(y, model.predict(X)))"
        passed, issues = run_programmatic_checks("train a classifier", "accuracy: 0.92", code=code)
        assert passed is False
        assert any("split" in i.lower() or "leakage" in i.lower() for i in issues)

    def test_ml_task_with_split_passes_split_check(self):
        code = "X_train, X_test, y_train, y_test = train_test_split(X, y)\nmodel.fit(X_train, y_train)"
        _, issues = run_programmatic_checks("train a classifier", "accuracy: 0.88", code=code)
        assert not any("split" in i.lower() for i in issues)

    def test_ml_task_accuracy_one_fails(self):
        code = "X_train, X_test, y_train, y_test = train_test_split(X, y)"
        passed, issues = run_programmatic_checks("train a classifier", "accuracy: 1.0", code=code)
        assert passed is False
        assert any("1.0" in i or "leakage" in i.lower() for i in issues)

    def test_ml_task_accuracy_below_threshold_fails(self):
        code = "X_train, X_test, y_train, y_test = train_test_split(X, y)"
        passed, issues = run_programmatic_checks("train a classifier", "accuracy: 0.45", code=code)
        assert passed is False
        assert any("0.45" in i or "low" in i.lower() for i in issues)

    def test_ml_task_no_metric_reported_fails(self):
        code = "X_train, X_test, y_train, y_test = train_test_split(X, y)\nmodel.fit(X_train, y_train)"
        passed, issues = run_programmatic_checks(
            "train a classifier", "Model trained successfully.", code=code
        )
        assert passed is False
        assert any("metric" in i.lower() for i in issues)

    def test_non_ml_task_no_metric_check(self):
        passed, _ = run_programmatic_checks("Print hello world", "Hello, World!")
        assert passed is True

    def test_multiple_issues_accumulated(self):
        passed, issues = run_programmatic_checks(
            "train a classifier",
            "nan detected\nTraceback (most recent call last):\n  Error",
            code=None,
        )
        assert passed is False
        assert len(issues) >= 2


class TestMlSanityChecks:

    def test_no_split_fails(self):
        code = "model.fit(X, y)\nprint(model.score(X, y))"
        passed, reason = ml_sanity_checks(code, "accuracy: 0.95")
        assert passed is False
        assert reason is not None
        assert "split" in reason.lower() or "leakage" in reason.lower()

    def test_accuracy_one_fails(self):
        code = "X_train, X_test, y_train, y_test = train_test_split(X, y)"
        passed, reason = ml_sanity_checks(code, "accuracy: 1.0")
        assert passed is False
        assert reason is not None

    def test_traceback_fails(self):
        code = "X_train, X_test, y_train, y_test = train_test_split(X, y)"
        passed, reason = ml_sanity_checks(
            code, "Traceback (most recent call last):\nValueError: bad input"
        )
        assert passed is False
        assert reason is not None

    def test_clean_ml_output_passes(self):
        code = "X_train, X_test, y_train, y_test = train_test_split(X, y)\nmodel.fit(X_train, y_train)"
        passed, reason = ml_sanity_checks(code, "accuracy: 0.91")
        assert passed is True
        assert reason is None

    def test_none_code_no_split_check(self):
        passed, _ = ml_sanity_checks(None, "accuracy: 0.88")
        assert passed is True

    def test_accuracy_below_one_passes_sanity(self):
        code = "X_train, X_test, y_train, y_test = train_test_split(X, y)"
        passed, _ = ml_sanity_checks(code, "accuracy: 0.99")
        assert passed is True


class TestEvaluationAgentHybridVerdict:

    PASS_JSON = json.dumps({
        "verdict": "PASS", "score": 0.90,
        "issues": [], "suggestions": [],
    })

    FAIL_JSON = json.dumps({
        "verdict": "FAIL", "score": 0.50,
        "issues": ["Missing error handling"],
        "suggestions": ["Add try/except"],
    })

    def _run(self, evaluator, mock_memory, llm_response, instruction, output, code_in_output=""):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = MagicMock(content=llm_response)
        evaluator._chain = mock_chain
        full_output = f"```python\n{code_in_output}\n```\n{output}" if code_in_output else output
        return evaluator.run(instruction, full_output, mock_memory)

    def test_llm_pass_prog_pass_gives_pass(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.PASS_JSON,
                        "explain recursion", "Recursion is when a function calls itself.")
        assert msg.metadata["verdict"] == "PASS"

    def test_llm_pass_prog_fail_gives_fail(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.PASS_JSON,
                        "train a classifier", "accuracy: 1.0",
                        code_in_output="model.fit(X, y)")
        assert msg.metadata["verdict"] == "FAIL"

    def test_llm_fail_prog_pass_gives_fail(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.FAIL_JSON,
                        "explain recursion", "Recursion is when a function calls itself.")
        assert msg.metadata["verdict"] == "FAIL"

    def test_both_fail_gives_fail(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.FAIL_JSON,
                        "train a classifier", "nan\nTraceback (most recent call last):",
                        code_in_output="model.fit(X, y)")
        assert msg.metadata["verdict"] == "FAIL"

    def test_score_comes_from_llm(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.PASS_JSON,
                        "explain recursion", "Recursion is when a function calls itself.")
        assert msg.metadata["score"] == 0.90

    def test_programmatic_issues_exposed_in_metadata(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.PASS_JSON,
                        "train a classifier", "nan in result",
                        code_in_output="model.fit(X, y)")
        assert len(msg.metadata["programmatic_issues"]) > 0

    def test_empty_llm_response_defaults_to_pass_if_prog_passes(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, "",
                        "explain recursion", "Recursion is when a function calls itself.")
        assert msg.metadata["verdict"] == "PASS"

    def test_malformed_llm_json_defaults_to_pass_if_prog_passes(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, "not json at all",
                        "explain recursion", "Recursion is when a function calls itself.")
        assert msg.metadata["verdict"] == "PASS"

    def test_issues_merged_from_both_layers(self, evaluator, mock_memory):
        msg = self._run(evaluator, mock_memory, self.FAIL_JSON,
                        "train a classifier", "nan")
        assert len(msg.metadata["issues"]) >= 1
