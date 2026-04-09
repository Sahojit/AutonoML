"""
Unit tests for AgentController — all LLM/agent calls mocked.
"""
from unittest.mock import MagicMock, patch

import pytest

from agents.base_agent import AgentMessage
from agents.planner_agent import PlanStep, PlannerOutput
from orchestrator.agent_controller import AgentController
from orchestrator.task_manager import StepStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(id, type_, task="do task", depends_on=None):
    return PlanStep(id=id, type=type_, task=task, depends_on=depends_on or [])


def _plan(*steps, goal="test goal"):
    return PlannerOutput(goal=goal, tasks=list(steps))


def _agent_msg(output="result", agent="execution"):
    return AgentMessage(
        agent=agent, task_id="t1", input="q",
        output=output, status="success", metadata={},
    )


def _eval_msg(verdict="PASS", score=0.9, issues=None, suggestions=None):
    er = MagicMock()
    er.verdict = verdict
    er.score = score
    er.issues = issues or []
    er.suggestions = suggestions or []
    er.passed.return_value = verdict == "PASS"
    er.feedback_text.return_value = "needs improvement"
    msg = AgentMessage(
        agent="evaluation", task_id="e1", input="q",
        output="eval output", status="success",
        metadata={"evaluation": er},
    )
    return msg


def _planner_msg(plan):
    return AgentMessage(
        agent="planner", task_id="p1", input="q",
        output="plan output", status="success",
        metadata={"plan": plan},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_agents():
    with (
        patch("orchestrator.agent_controller.PlannerAgent") as MockPlanner,
        patch("orchestrator.agent_controller.ResearchAgent") as MockResearcher,
        patch("orchestrator.agent_controller.ExecutionAgent") as MockExecutor,
        patch("orchestrator.agent_controller.EvaluationAgent") as MockEvaluator,
        patch("orchestrator.agent_controller.MemoryManager") as MockMemory,
    ):
        planner  = MockPlanner.return_value
        researcher = MockResearcher.return_value
        executor = MockExecutor.return_value
        evaluator = MockEvaluator.return_value
        memory   = MockMemory.return_value
        memory.get_conversation_history.return_value = []
        memory.store_in_long_term.return_value = ["id-1"]
        memory.store_reflection = MagicMock()
        memory.save_session_to_long_term = MagicMock()
        yield {
            "planner": planner,
            "researcher": researcher,
            "executor": executor,
            "evaluator": evaluator,
            "memory": memory,
        }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestAgentControllerInit:
    def test_generates_session_id(self, mock_agents):
        ctrl = AgentController()
        assert ctrl.session_id is not None

    def test_accepts_explicit_session_id(self, mock_agents):
        ctrl = AgentController(session_id="my-session")
        assert ctrl.session_id == "my-session"


# ---------------------------------------------------------------------------
# run() — return structure
# ---------------------------------------------------------------------------

class TestAgentControllerRun:
    def _setup(self, mock_agents, plan):
        mock_agents["planner"].run.return_value = _planner_msg(plan)
        exec_msg = _agent_msg("execution output")
        exec_msg.metadata = {"tool_trace": []}
        mock_agents["executor"].run.return_value = exec_msg
        mock_agents["evaluator"].run.return_value = _eval_msg("PASS")
        mock_agents["researcher"].run.return_value = _agent_msg("research output", "research")

    def test_run_returns_final_answer(self, mock_agents):
        plan = _plan(_step(1, "execution"))
        self._setup(mock_agents, plan)
        ctrl = AgentController()
        result = ctrl.run("test query")
        assert "final_answer" in result

    def test_run_returns_session_id(self, mock_agents):
        plan = _plan(_step(1, "execution"))
        self._setup(mock_agents, plan)
        ctrl = AgentController(session_id="s1")
        result = ctrl.run("test query")
        assert result["session_id"] == "s1"

    def test_run_returns_goal(self, mock_agents):
        plan = _plan(_step(1, "execution"), goal="my goal")
        self._setup(mock_agents, plan)
        ctrl = AgentController()
        result = ctrl.run("test query")
        assert result["goal"] == "my goal"

    def test_run_returns_step_records(self, mock_agents):
        plan = _plan(_step(1, "execution"))
        self._setup(mock_agents, plan)
        ctrl = AgentController()
        result = ctrl.run("test query")
        assert isinstance(result["step_records"], list)

    def test_run_returns_duration(self, mock_agents):
        plan = _plan(_step(1, "execution"))
        self._setup(mock_agents, plan)
        ctrl = AgentController()
        result = ctrl.run("test query")
        assert result["duration_s"] >= 0

    def test_research_step_calls_researcher(self, mock_agents):
        plan = _plan(_step(1, "research"))
        self._setup(mock_agents, plan)
        ctrl = AgentController()
        ctrl.run("test query")
        mock_agents["researcher"].run.assert_called_once()

    def test_execution_step_calls_executor(self, mock_agents):
        plan = _plan(_step(1, "execution"))
        self._setup(mock_agents, plan)
        ctrl = AgentController()
        ctrl.run("test query")
        mock_agents["executor"].run.assert_called_once()

    def test_execution_step_calls_evaluator(self, mock_agents):
        plan = _plan(_step(1, "execution"))
        self._setup(mock_agents, plan)
        ctrl = AgentController()
        ctrl.run("test query")
        mock_agents["evaluator"].run.assert_called_once()

    def test_evaluation_step_uses_last_exec_output(self, mock_agents):
        plan = _plan(_step(1, "execution"), _step(2, "evaluation", depends_on=[1]))
        self._setup(mock_agents, plan)
        ctrl = AgentController()
        result = ctrl.run("test query")
        assert result["final_answer"] != ""

    def test_step_record_marked_done_on_pass(self, mock_agents):
        plan = _plan(_step(1, "execution"))
        self._setup(mock_agents, plan)
        ctrl = AgentController()
        result = ctrl.run("test query")
        assert result["step_records"][0]["status"] == "done"


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------

class TestRetryLoop:
    def test_retries_on_fail_then_passes(self, mock_agents):
        plan = _plan(_step(1, "execution"))
        mock_agents["planner"].run.return_value = _planner_msg(plan)

        fail_msg = _eval_msg("FAIL")
        pass_msg = _eval_msg("PASS")
        mock_agents["evaluator"].run.side_effect = [fail_msg, pass_msg]

        exec_msg = _agent_msg("output")
        exec_msg.metadata = {"tool_trace": []}
        mock_agents["executor"].run.return_value = exec_msg

        ctrl = AgentController()
        result = ctrl.run("test")
        assert mock_agents["executor"].run.call_count == 2

    def test_accepts_last_output_after_max_iterations(self, mock_agents):
        plan = _plan(_step(1, "execution"))
        mock_agents["planner"].run.return_value = _planner_msg(plan)

        exec_msg = _agent_msg("output")
        exec_msg.metadata = {"tool_trace": []}
        mock_agents["executor"].run.return_value = exec_msg
        mock_agents["evaluator"].run.return_value = _eval_msg("FAIL")

        ctrl = AgentController()
        result = ctrl.run("test")
        assert result["step_records"][0]["status"] == "done"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_get_history(self, mock_agents):
        ctrl = AgentController()
        history = ctrl.get_history()
        assert isinstance(history, list)

    def test_add_document(self, mock_agents):
        ctrl = AgentController()
        ids = ctrl.add_document("some text")
        assert ids == ["id-1"]
