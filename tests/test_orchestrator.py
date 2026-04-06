"""
Unit tests for AgentOrchestrator and TaskManager.
All LLM/agent calls are mocked — no real inference runs.
"""
import time
from unittest.mock import MagicMock, patch

import pytest

from agents.base_agent import AgentMessage
from agents.planner_agent import PlanStep, PlannerOutput
from orchestrator.agent_orchestrator import AgentOrchestrator, AgentController
from orchestrator.task_manager import StepRecord, StepStatus, TaskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(id, type_, task="do something", depends_on=None):
    return PlanStep(id=id, type=type_, task=task, depends_on=depends_on or [])


def _plan(*steps):
    return PlannerOutput(goal="test goal", tasks=list(steps))


def _agent_msg(output="ok"):
    return AgentMessage(
        agent="execution", task_id="t1", input="q",
        output=output, status="success",
    )


def _eval_result(verdict="PASS", score=0.9, issues=None, suggestions=None):
    r = MagicMock()
    r.verdict = verdict
    r.score = score
    r.issues = issues or []
    r.suggestions = suggestions or []
    r.passed.return_value = verdict == "PASS"
    r.feedback_text.return_value = "fix this"
    return r


# ---------------------------------------------------------------------------
# TaskManager — registration
# ---------------------------------------------------------------------------

class TestTaskManagerSetPlan:
    def test_registers_all_steps(self):
        tm = TaskManager(query="q", session_id="s1")
        tm.set_plan(_plan(_step(1, "research"), _step(2, "execution")))
        assert len(tm._records) == 2

    def test_step_initial_status_pending(self):
        tm = TaskManager(query="q", session_id="s1")
        tm.set_plan(_plan(_step(1, "research")))
        assert tm._records[1].status == StepStatus.PENDING

    def test_step_type_stored(self):
        tm = TaskManager(query="q", session_id="s1")
        tm.set_plan(_plan(_step(1, "execution")))
        assert tm._records[1].step_type == "execution"

    def test_step_task_stored(self):
        tm = TaskManager(query="q", session_id="s1")
        tm.set_plan(_plan(_step(1, "research", task="find data")))
        assert tm._records[1].task == "find data"


# ---------------------------------------------------------------------------
# TaskManager — lifecycle transitions
# ---------------------------------------------------------------------------

class TestTaskManagerLifecycle:
    def setup_method(self):
        self.tm = TaskManager(query="q", session_id="s1")
        self.tm.set_plan(_plan(_step(1, "execution"), _step(2, "research")))

    def test_start_step_sets_running(self):
        self.tm.start_step(1)
        assert self.tm._records[1].status == StepStatus.RUNNING

    def test_start_step_records_time(self):
        self.tm.start_step(1)
        assert self.tm._records[1].started_at is not None

    def test_complete_step_sets_done(self):
        self.tm.start_step(1)
        self.tm.complete_step(1, output="result")
        assert self.tm._records[1].status == StepStatus.DONE

    def test_complete_step_stores_output(self):
        self.tm.complete_step(1, output="my output")
        assert self.tm._records[1].output == "my output"

    def test_complete_step_stores_iterations(self):
        self.tm.complete_step(1, output="x", iterations=3)
        assert self.tm._records[1].iterations == 3

    def test_complete_step_stores_eval(self):
        er = _eval_result("PASS", 0.85, ["i1"])
        self.tm.complete_step(1, output="x", eval_result=er)
        assert self.tm._records[1].verdict == "PASS"
        assert self.tm._records[1].eval_score == 0.85

    def test_fail_step_sets_failed(self):
        self.tm.fail_step(1, error="boom")
        assert self.tm._records[1].status == StepStatus.FAILED

    def test_fail_step_stores_error(self):
        self.tm.fail_step(1, error="boom")
        assert self.tm._records[1].error == "boom"

    def test_skip_step_sets_skipped(self):
        self.tm.skip_step(1)
        assert self.tm._records[1].status == StepStatus.SKIPPED

    def test_record_iteration_updates_count(self):
        self.tm.start_step(1)
        self.tm.record_iteration(1, 2)
        assert self.tm._records[1].iterations == 2

    def test_unknown_step_id_is_safe(self):
        self.tm.start_step(999)  # should not raise


# ---------------------------------------------------------------------------
# TaskManager — queries
# ---------------------------------------------------------------------------

class TestTaskManagerQueries:
    def setup_method(self):
        self.tm = TaskManager(query="q", session_id="s1")
        self.tm.set_plan(_plan(
            _step(1, "research"),
            _step(2, "execution"),
            _step(3, "execution"),
        ))

    def test_get_step_output_returns_output(self):
        self.tm.complete_step(1, output="research out")
        assert self.tm.get_step_output(1) == "research out"

    def test_get_step_output_none_for_unknown(self):
        assert self.tm.get_step_output(999) is None

    def test_last_execution_output_picks_latest(self):
        self.tm.complete_step(2, output="exec1")
        self.tm.complete_step(3, output="exec2")
        assert self.tm.last_execution_output() == "exec2"

    def test_last_execution_output_none_if_no_exec(self):
        self.tm.complete_step(1, output="research out")
        assert self.tm.last_execution_output() is None

    def test_completed_records_filters_done(self):
        self.tm.complete_step(1, output="ok")
        self.tm.fail_step(2, error="err")
        done = self.tm.completed_records()
        assert len(done) == 1
        assert done[0].step_id == 1

    def test_all_records_as_dicts_count(self):
        assert len(self.tm.all_records_as_dicts()) == 3

    def test_all_records_as_dicts_has_step_id(self):
        dicts = self.tm.all_records_as_dicts()
        ids = {d["step_id"] for d in dicts}
        assert ids == {1, 2, 3}

    def test_get_progress_total(self):
        p = self.tm.get_progress()
        assert p["total_steps"] == 3

    def test_get_progress_completed(self):
        self.tm.complete_step(1, output="ok")
        assert self.tm.get_progress()["completed_steps"] == 1

    def test_get_progress_pct(self):
        self.tm.complete_step(1, output="ok")
        self.tm.complete_step(2, output="ok")
        assert self.tm.get_progress()["progress_pct"] == pytest.approx(66.7, abs=0.1)


# ---------------------------------------------------------------------------
# StepRecord
# ---------------------------------------------------------------------------

class TestStepRecord:
    def test_duration_none_if_not_started(self):
        rec = StepRecord(step_id=1, step_type="execution", task="t")
        assert rec.duration_s is None

    def test_duration_computed(self):
        rec = StepRecord(step_id=1, step_type="execution", task="t")
        rec.started_at = time.time()
        time.sleep(0.05)
        rec.finished_at = time.time()
        assert rec.duration_s > 0

    def test_to_dict_has_required_keys(self):
        rec = StepRecord(step_id=1, step_type="research", task="find it")
        d = rec.to_dict()
        for key in ("step_id", "step_type", "task", "status", "output_preview", "error"):
            assert key in d


# ---------------------------------------------------------------------------
# AgentOrchestrator._group_by_level
# ---------------------------------------------------------------------------

class TestGroupByLevel:
    def test_independent_steps_same_group(self):
        steps = [_step(1, "research"), _step(2, "research")]
        groups = AgentOrchestrator._group_by_level(steps)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_chain_produces_multiple_groups(self):
        steps = [_step(1, "research"), _step(2, "execution", depends_on=[1])]
        groups = AgentOrchestrator._group_by_level(steps)
        assert len(groups) == 2
        assert groups[0][0].id == 1
        assert groups[1][0].id == 2

    def test_dag_with_two_parallel_then_one(self):
        steps = [
            _step(1, "research"),
            _step(2, "research"),
            _step(3, "execution", depends_on=[1, 2]),
        ]
        groups = AgentOrchestrator._group_by_level(steps)
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert groups[1][0].id == 3

    def test_empty_steps(self):
        assert AgentOrchestrator._group_by_level([]) == []


# ---------------------------------------------------------------------------
# AgentOrchestrator._collect_context
# ---------------------------------------------------------------------------

class TestCollectContext:
    def setup_method(self):
        self.orch = AgentOrchestrator.__new__(AgentOrchestrator)
        self.tm = TaskManager(query="q", session_id="s")
        self.tm.set_plan(_plan(_step(1, "research"), _step(2, "execution", depends_on=[1])))
        self.tm.complete_step(1, output="research summary")

    def test_no_deps_returns_empty(self):
        step = _step(3, "execution", depends_on=[])
        ctx = self.orch._collect_context(step, {}, self.tm)
        assert ctx == ""

    def test_uses_research_context_dict(self):
        step = _step(2, "execution", depends_on=[1])
        ctx = self.orch._collect_context(step, {1: "from research"}, self.tm)
        assert "from research" in ctx

    def test_falls_back_to_task_mgr_output(self):
        step = _step(2, "execution", depends_on=[1])
        ctx = self.orch._collect_context(step, {}, self.tm)
        assert "research summary" in ctx


# ---------------------------------------------------------------------------
# AgentOrchestrator._synthesise
# ---------------------------------------------------------------------------

class TestSynthesise:
    def test_no_records_returns_fallback(self):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        tm = TaskManager(query="q", session_id="s")
        tm.set_plan(_plan(_step(1, "execution")))
        result = orch._synthesise(_plan(_step(1, "execution")), tm)
        assert "did not produce" in result.lower() or result != ""

    def test_includes_goal(self):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        plan = _plan(_step(1, "execution"))
        tm = TaskManager(query="q", session_id="s")
        tm.set_plan(plan)
        tm.complete_step(1, output="final result")
        result = orch._synthesise(plan, tm)
        assert "test goal" in result

    def test_includes_step_output(self):
        orch = AgentOrchestrator.__new__(AgentOrchestrator)
        plan = _plan(_step(1, "execution"))
        tm = TaskManager(query="q", session_id="s")
        tm.set_plan(plan)
        tm.complete_step(1, output="the answer is 42")
        result = orch._synthesise(plan, tm)
        assert "the answer is 42" in result


# ---------------------------------------------------------------------------
# AgentController alias
# ---------------------------------------------------------------------------

def test_agent_controller_is_subclass_of_orchestrator():
    assert issubclass(AgentController, AgentOrchestrator)
