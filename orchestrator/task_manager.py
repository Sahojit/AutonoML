"""
TaskManager
───────────
Tracks the full lifecycle of every step in a single query run.

Stores per-step:
  - status (pending → running → done / failed)
  - output text
  - agent message envelope
  - evaluation result
  - iteration count (for self-correction loop)
  - timing

The orchestrator uses this to build the rich response sent to the API/UI.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("multiagent.orchestrator.task_manager")


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Step record
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    step_id:   int
    step_type: str                            # "research" | "execution" | "evaluation"
    task:      str

    status:      StepStatus = StepStatus.PENDING
    output:      Optional[str] = None
    error:       Optional[str] = None
    iterations:  int = 0                      # self-correction rounds used
    verdict:     Optional[str] = None         # "PASS" | "FAIL" | None
    eval_score:  Optional[float] = None
    eval_issues: List[str] = field(default_factory=list)

    started_at:  Optional[float] = None
    finished_at: Optional[float] = None

    # Raw AgentMessage for detailed introspection
    agent_msg: Optional[Any] = None

    @property
    def duration_s(self) -> Optional[float]:
        if self.started_at and self.finished_at:
            return round(self.finished_at - self.started_at, 3)
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id":    self.step_id,
            "step_type":  self.step_type,
            "task":       self.task,
            "status":     self.status.value,
            "output_preview": (self.output or "")[:500],
            "output_full":    self.output or "",
            "error":      self.error,
            "iterations": self.iterations,
            "verdict":    self.verdict,
            "eval_score": self.eval_score,
            "eval_issues": self.eval_issues,
            "duration_s": self.duration_s,
        }


# ---------------------------------------------------------------------------
# TaskManager
# ---------------------------------------------------------------------------

class TaskManager:
    """
    Manages the execution state of a multi-step plan for one query.
    """

    def __init__(self, query: str, session_id: str) -> None:
        self.query      = query
        self.session_id = session_id
        self._records:  Dict[int, StepRecord] = {}
        self._started:  float = time.time()
        logger.info("TaskManager created for session=%s", session_id)

    # ------------------------------------------------------------------
    # Plan registration
    # ------------------------------------------------------------------

    def set_plan(self, plan: Any) -> None:
        """Register all tasks from a ``PlannerOutput``."""
        for step in plan.tasks:
            self._records[step.id] = StepRecord(
                step_id=step.id,
                step_type=step.type,
                task=step.task,
            )
        logger.info("TaskManager loaded %d tasks.", len(self._records))

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def start_step(self, step_id: int) -> None:
        rec = self._get(step_id)
        if rec:
            rec.status     = StepStatus.RUNNING
            rec.started_at = time.time()

    def complete_step(
        self,
        step_id: int,
        output: str,
        agent_msg: Any = None,
        eval_result: Any = None,
        iterations: int = 1,
    ) -> None:
        rec = self._get(step_id)
        if not rec:
            return
        rec.status      = StepStatus.DONE
        rec.output      = output
        rec.agent_msg   = agent_msg
        rec.iterations  = iterations
        rec.finished_at = time.time()
        if eval_result is not None:
            rec.verdict     = eval_result.verdict
            rec.eval_score  = eval_result.score
            rec.eval_issues = list(eval_result.issues)

    def fail_step(self, step_id: int, error: str) -> None:
        rec = self._get(step_id)
        if rec:
            rec.status      = StepStatus.FAILED
            rec.error       = error
            rec.finished_at = time.time()

    def skip_step(self, step_id: int) -> None:
        rec = self._get(step_id)
        if rec:
            rec.status      = StepStatus.SKIPPED
            rec.finished_at = time.time()

    def record_iteration(self, step_id: int, iteration: int) -> None:
        rec = self._get(step_id)
        if rec:
            rec.iterations = iteration

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_step_output(self, step_id: int) -> Optional[str]:
        rec = self._get(step_id)
        return rec.output if rec else None

    def last_execution_output(self) -> Optional[str]:
        """Return the output of the most recently completed execution step."""
        for rec in reversed(list(self._records.values())):
            if rec.step_type == "execution" and rec.output:
                return rec.output
        return None

    def completed_records(self) -> List[StepRecord]:
        return [r for r in self._records.values() if r.status == StepStatus.DONE]

    def all_records_as_dicts(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self._records.values()]

    def get_progress(self) -> Dict[str, Any]:
        total = len(self._records)
        done  = sum(1 for r in self._records.values() if r.status == StepStatus.DONE)
        return {
            "total_steps":     total,
            "completed_steps": done,
            "progress_pct":    round(done / total * 100 if total else 0, 1),
            "current_step":    next(
                (r.step_id for r in self._records.values() if r.status == StepStatus.RUNNING),
                None,
            ),
        }

    def summary(self) -> str:
        _icons = {
            "done": "✓", "failed": "✗", "running": "→",
            "pending": "○", "skipped": "–",
        }
        lines = [f"Task: {self.query[:80]}"]
        for rec in self._records.values():
            icon = _icons.get(rec.status.value, "?")
            dur  = f" ({rec.duration_s}s)" if rec.duration_s is not None else ""
            lines.append(
                f"  {icon} Step {rec.step_id} [{rec.step_type}]: "
                f"{rec.status.value}{dur}"
                + (f" | {rec.verdict}" if rec.verdict else "")
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, step_id: int) -> Optional[StepRecord]:
        rec = self._records.get(step_id)
        if rec is None:
            logger.warning("TaskManager: unknown step_id=%d", step_id)
        return rec
