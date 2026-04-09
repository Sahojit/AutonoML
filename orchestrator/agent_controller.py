"""
AgentController
───────────────
Standalone per-session orchestrator. Wires together all four agents and
drives the full execution pipeline:

  User Query
    → PlannerAgent       (decompose into typed steps)
    → For each step:
        → ResearchAgent  (type == "research")
        → ExecutionAgent (type == "execution", with eval retry loop)
        → EvaluationAgent (type == "evaluation", standalone)
    → Synthesise final answer
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Dict, List, Optional

from agents.evaluation_agent import EvaluationAgent, EvaluationResult
from agents.execution_agent import ExecutionAgent
from agents.planner_agent import PlanStep, PlannerAgent, PlannerOutput
from agents.research_agent import ResearchAgent
from config.settings import settings
from memory.memory_manager import MemoryManager
from orchestrator.task_manager import TaskManager

logger = logging.getLogger("multiagent.controller")


class AgentController:
    """
    Per-session controller. Each instance owns its own MemoryManager,
    so multiple requests can run concurrently without shared state.

    Usage::

        ctrl = AgentController(session_id="abc")
        result = ctrl.run("Analyse the Iris dataset")
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self._memory   = MemoryManager(self.session_id)
        self._planner  = PlannerAgent()
        self._researcher = ResearchAgent(use_web_search=False)
        self._executor = ExecutionAgent()
        self._evaluator = EvaluationAgent()
        logger.info("AgentController ready — session=%s", self.session_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str) -> Dict:
        """
        Synchronous end-to-end pipeline execution.

        Returns
        -------
        {
            "final_answer": str,
            "session_id":   str,
            "goal":         str,
            "step_records": list[dict],
            "duration_s":   float,
        }
        """
        t0 = time.perf_counter()
        logger.info("AgentController.run START session=%s", self.session_id)

        self._memory.add_message("user", query)

        # 1 ── Plan ────────────────────────────────────────────────────
        plan_msg = self._planner.run(query, self._memory)
        plan: PlannerOutput = plan_msg.metadata["plan"]
        task_mgr = TaskManager(query=query, session_id=self.session_id)
        task_mgr.set_plan(plan)

        # 2 ── Execute steps ───────────────────────────────────────────
        research_contexts: Dict[int, str] = {}

        for step in plan.tasks:
            ctx = self._collect_context(step, research_contexts, task_mgr)

            if step.type == "research":
                output = self._run_research(step, ctx, task_mgr)
                research_contexts[step.id] = output

            elif step.type == "execution":
                self._run_execution(step, ctx, task_mgr)

            elif step.type == "evaluation":
                self._run_evaluation(step, ctx, task_mgr)

            else:
                logger.warning("Unknown step type '%s' — treating as execution.", step.type)
                self._run_execution(step, ctx, task_mgr)

        # 3 ── Synthesise ──────────────────────────────────────────────
        final_answer = self._synthesise(plan, task_mgr)

        elapsed = round(time.perf_counter() - t0, 3)
        self._memory.add_message("assistant", final_answer)
        self._memory.save_session_to_long_term()

        logger.info("AgentController.run END session=%s  duration=%.3fs", self.session_id, elapsed)

        return {
            "final_answer":  final_answer,
            "session_id":    self.session_id,
            "goal":          plan.goal,
            "step_records":  task_mgr.all_records_as_dicts(),
            "duration_s":    elapsed,
        }

    def get_history(self) -> List[Dict]:
        return self._memory.get_conversation_history()

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> List[str]:
        return self._memory.store_in_long_term([text], [metadata or {}])

    # ------------------------------------------------------------------
    # Step runners
    # ------------------------------------------------------------------

    def _run_research(self, step: PlanStep, ctx: str, task_mgr: TaskManager) -> str:
        task_mgr.start_step(step.id)
        try:
            msg = self._researcher.run(
                instruction=step.task,
                memory=self._memory,
                task_id=f"step_{step.id}",
            )
            task_mgr.complete_step(step.id, output=msg.output, agent_msg=msg)
            return msg.output
        except Exception as exc:            # noqa: BLE001
            logger.error("Research step %d failed: %s", step.id, exc)
            task_mgr.fail_step(step.id, error=str(exc))
            return f"Research failed: {exc}"

    def _run_execution(self, step: PlanStep, ctx: str, task_mgr: TaskManager) -> None:
        task_mgr.start_step(step.id)
        max_iter = settings.MAX_AGENT_ITERATIONS
        feedback = ""

        for iteration in range(1, max_iter + 1):
            task_mgr.record_iteration(step.id, iteration)

            exec_msg = self._executor.run(
                instruction=step.task,
                memory=self._memory,
                research_context=ctx,
                feedback=feedback,
                task_id=f"step_{step.id}_iter{iteration}",
            )

            eval_msg = self._evaluator.run(
                instruction=step.task,
                execution_output=exec_msg.output,
                memory=self._memory,
                research_context=ctx,
                tool_trace=exec_msg.metadata.get("tool_trace", []),
                task_id=f"step_{step.id}_eval{iteration}",
            )
            eval_result: EvaluationResult = eval_msg.metadata["evaluation"]

            if eval_result.passed():
                task_mgr.complete_step(
                    step.id, output=exec_msg.output,
                    agent_msg=exec_msg, eval_result=eval_result, iterations=iteration,
                )
                return

            feedback = eval_result.feedback_text()
            self._memory.store_reflection(
                task=step.task,
                error="\n".join(eval_result.issues) or "Unknown issue",
                suggestion="\n".join(eval_result.suggestions) or "Review output carefully",
            )

        task_mgr.complete_step(
            step.id, output=exec_msg.output,         # type: ignore[possibly-undefined]
            agent_msg=exec_msg, eval_result=eval_result,  # type: ignore[possibly-undefined]
            iterations=max_iter,
        )

    def _run_evaluation(self, step: PlanStep, ctx: str, task_mgr: TaskManager) -> None:
        task_mgr.start_step(step.id)
        last_output = task_mgr.last_execution_output() or "No execution output."
        try:
            eval_msg = self._evaluator.run(
                instruction=step.task,
                execution_output=last_output,
                memory=self._memory,
                research_context=ctx,
                task_id=f"step_{step.id}",
            )
            eval_result: EvaluationResult = eval_msg.metadata["evaluation"]
            task_mgr.complete_step(
                step.id, output=eval_msg.output,
                agent_msg=eval_msg, eval_result=eval_result,
            )
        except Exception as exc:            # noqa: BLE001
            logger.error("Evaluation step %d failed: %s", step.id, exc)
            task_mgr.fail_step(step.id, error=str(exc))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collect_context(
        self,
        step: PlanStep,
        research_contexts: Dict[int, str],
        task_mgr: TaskManager,
    ) -> str:
        parts = []
        for dep_id in step.depends_on:
            if dep_id in research_contexts:
                parts.append(f"[Research from step {dep_id}]\n{research_contexts[dep_id]}")
            else:
                out = task_mgr.get_step_output(dep_id)
                if out:
                    parts.append(f"[Output from step {dep_id}]\n{out}")
        return "\n\n".join(parts)

    def _synthesise(self, plan: PlannerOutput, task_mgr: TaskManager) -> str:
        records = task_mgr.completed_records()
        if not records:
            return "The pipeline did not produce a result. Please try again."
        parts = [f"## Goal\n{plan.goal}\n"]
        for rec in records:
            if rec.output:
                parts.append(f"### Step {rec.step_id} [{rec.step_type}]\n{rec.output}")
        if len(parts) == 1:
            return task_mgr.last_execution_output() or "No result produced."
        return "\n\n".join(parts)
