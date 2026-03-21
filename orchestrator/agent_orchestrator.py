"""
AgentOrchestrator — Central Pipeline Controller
─────────────────────────────────────────────────
Responsibilities
────────────────
1. Accept a user query and create a per-session MemoryManager.
2. Invoke the PlannerAgent to produce a hierarchical plan.
3. Route each plan step to the correct agent (research / execution / evaluation).
4. Run independent research steps concurrently via asyncio.gather.
5. Wrap every execution step in an iterative self-correction loop
   (Execution → Evaluation → retry up to MAX_ITERATIONS).
6. Track per-step state via TaskManager.
7. Synthesise a final answer from all step outputs.
8. Return a rich result dict consumed by the FastAPI layer.

Pipeline
────────
User Query
  → PlannerAgent          → PlannerOutput (goal + typed steps)
  → Parallel research     → research summaries merged
  → ExecutionAgent        → execution output
  → EvaluationAgent       → verdict
  → (if FAIL) retry loop  → up to MAX_ITERATIONS
  → final answer synthesis
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List

from agents.evaluation_agent import EvaluationAgent, EvaluationResult
from agents.execution_agent import ExecutionAgent
from agents.planner_agent import PlanStep, PlannerAgent, PlannerOutput
from agents.research_agent import ResearchAgent
from config.settings import settings
from memory.memory_manager import MemoryManager
from orchestrator.task_manager import TaskManager

logger = logging.getLogger("multiagent.orchestrator")

# Alias for legacy import compatibility
AgentController = None   # set at bottom of file


class AgentOrchestrator:
    """
    Central controller that executes a full multi-agent pipeline for a
    single user query.

    Thread-safety
    ─────────────
    Each ``run()`` call creates its own ``MemoryManager`` and ``TaskManager``
    instance, so multiple requests can run concurrently in a thread pool.
    """

    def __init__(self) -> None:
        self._planner   = PlannerAgent()
        self._researcher = ResearchAgent(use_web_search=True)
        self._executor  = ExecutionAgent()
        self._evaluator = EvaluationAgent()
        logger.info("AgentOrchestrator initialised.")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, query: str, session_id: str | None = None) -> Dict[str, Any]:
        """
        Execute the full pipeline for *query* and return a result dict.

        Parameters
        ----------
        query      : str   User query.
        session_id : str   Reuse an existing session (for conversation continuity).

        Returns
        -------
        {
            "final_answer": str,
            "session_id":   str,
            "goal":         str,
            "plan":         dict,
            "step_records": list[dict],
            "duration_s":   float,
        }
        """
        sid = session_id or str(uuid.uuid4())
        memory = MemoryManager(session_id=sid)
        task_mgr = TaskManager(query=query, session_id=sid)

        t0 = time.perf_counter()
        logger.info("[Orchestrator] session=%s | query=%s", sid, query[:120])

        memory.add_message("user", query)

        # ── 1. Planning ────────────────────────────────────────────────
        plan_msg = self._planner.run(query, memory)
        plan: PlannerOutput = plan_msg.metadata["plan"]
        task_mgr.set_plan(plan)

        logger.info(
            "[Orchestrator] session=%s | goal=%s | tasks=%d",
            sid, plan.goal, len(plan.tasks),
        )

        # ── 2. Execute tasks ───────────────────────────────────────────
        # Collect research context keyed by task id
        research_contexts: Dict[int, str] = {}

        # Group tasks by level for parallel execution
        step_groups = self._group_by_level(plan.tasks)

        for group in step_groups:
            if len(group) > 1 and settings.PARALLEL_RESEARCH:
                # Run parallel research steps
                research_steps = [s for s in group if s.type == "research"]
                other_steps    = [s for s in group if s.type != "research"]

                if research_steps:
                    parallel_results = asyncio.run(
                        self._run_research_parallel(
                            research_steps, memory, task_mgr
                        )
                    )
                    research_contexts.update(parallel_results)

                # Run non-research steps sequentially (they depend on context)
                for step in other_steps:
                    ctx = self._collect_context(step, research_contexts, task_mgr)
                    self._execute_step(step, memory, task_mgr, ctx)
            else:
                for step in group:
                    ctx = self._collect_context(step, research_contexts, task_mgr)
                    if step.type == "research":
                        result = self._run_research_step(step, memory, task_mgr, use_web=False)
                        research_contexts[step.id] = result
                    else:
                        self._execute_step(step, memory, task_mgr, ctx)

        # ── 3. Synthesise final answer ─────────────────────────────────
        final_answer = self._synthesise(plan, task_mgr)

        elapsed = round(time.perf_counter() - t0, 3)
        logger.info(
            "[Orchestrator] session=%s | done in %.3fs", sid, elapsed
        )

        # Persist session
        memory.add_message("assistant", final_answer)
        memory.save_session_to_long_term()

        return {
            "final_answer": final_answer,
            "session_id": sid,
            "goal": plan.goal,
            "plan": plan.to_dict(),
            "step_records": task_mgr.all_records_as_dicts(),
            "duration_s": elapsed,
        }

    # ------------------------------------------------------------------
    # Step routing
    # ------------------------------------------------------------------

    def _execute_step(
        self,
        step: PlanStep,
        memory: MemoryManager,
        task_mgr: TaskManager,
        research_context: str,
    ) -> None:
        """Route a single non-research step to the correct agent."""
        if step.type == "execution":
            self._run_execution_with_loop(step, memory, task_mgr, research_context)
        elif step.type == "evaluation":
            self._run_evaluation_step(step, memory, task_mgr, research_context)
        else:
            # Unknown type — fall back to execution
            logger.warning("Unknown step type '%s'; treating as execution.", step.type)
            self._run_execution_with_loop(step, memory, task_mgr, research_context)

    # ------------------------------------------------------------------
    # Research
    # ------------------------------------------------------------------

    def _run_research_step(
        self,
        step: PlanStep,
        memory: MemoryManager,
        task_mgr: TaskManager,
        use_web: bool = False,
    ) -> str:
        task_id = f"step_{step.id}"
        task_mgr.start_step(step.id)
        try:
            msg = self._researcher.run(
                instruction=step.task,
                memory=memory,
                use_web=use_web,
                task_id=task_id,
            )
            task_mgr.complete_step(step.id, output=msg.output, agent_msg=msg)
            return msg.output
        except Exception as exc:           # noqa: BLE001
            logger.error("[Orchestrator] Research step %d failed: %s", step.id, exc)
            task_mgr.fail_step(step.id, error=str(exc))
            return f"Research failed: {exc}"

    async def _run_research_parallel(
        self,
        steps: List[PlanStep],
        memory: MemoryManager,
        task_mgr: TaskManager,
    ) -> Dict[int, str]:
        """Run multiple research steps concurrently using asyncio."""
        loop = asyncio.get_event_loop()

        async def _one(step: PlanStep) -> tuple[int, str]:
            result = await loop.run_in_executor(
                None,
                lambda s=step: self._run_research_step(s, memory, task_mgr),
            )
            return step.id, result

        pairs = await asyncio.gather(*[_one(s) for s in steps], return_exceptions=False)
        return dict(pairs)

    # ------------------------------------------------------------------
    # Execution + self-correction loop
    # ------------------------------------------------------------------

    def _run_execution_with_loop(
        self,
        step: PlanStep,
        memory: MemoryManager,
        task_mgr: TaskManager,
        research_context: str,
    ) -> None:
        """
        Execute a step with iterative self-correction.

        Loop
        ────
        Execution → Evaluation
          if FAIL and iterations < MAX_ITERATIONS:
              feedback → Execution (retry)
          else:
              break
        """
        task_id = f"step_{step.id}"
        task_mgr.start_step(step.id)
        max_iter = settings.MAX_AGENT_ITERATIONS
        feedback = ""

        for iteration in range(1, max_iter + 1):
            logger.info(
                "[Orchestrator] step=%d | execution iteration %d/%d",
                step.id, iteration, max_iter,
            )
            task_mgr.record_iteration(step.id, iteration)

            # ── Execute ────────────────────────────────────────────────
            exec_msg = self._executor.run(
                instruction=step.task,
                memory=memory,
                research_context=research_context,
                feedback=feedback,
                task_id=f"{task_id}_iter{iteration}",
            )

            # ── Evaluate ───────────────────────────────────────────────
            tool_trace = exec_msg.metadata.get("tool_trace", [])
            eval_msg = self._evaluator.run(
                instruction=step.task,
                execution_output=exec_msg.output,
                memory=memory,
                research_context=research_context,
                tool_trace=tool_trace,
                task_id=f"{task_id}_eval{iteration}",
            )
            eval_result: EvaluationResult = eval_msg.metadata["evaluation"]

            logger.info(
                "[Orchestrator] step=%d | iter=%d | verdict=%s | score=%.2f",
                step.id, iteration, eval_result.verdict, eval_result.score,
            )

            if eval_result.passed():
                task_mgr.complete_step(
                    step.id,
                    output=exec_msg.output,
                    agent_msg=exec_msg,
                    eval_result=eval_result,
                    iterations=iteration,
                )
                return

            # FAIL path — store reflection + prepare feedback for next iteration
            feedback = eval_result.feedback_text()
            logger.info(
                "[Orchestrator] step=%d | FAIL → feedback: %s",
                step.id, feedback[:200],
            )
            # Persist failure to self-reflection memory so future runs avoid this mistake
            memory.store_reflection(
                task=step.task,
                error="\n".join(eval_result.issues) or "Unknown issue",
                suggestion="\n".join(eval_result.suggestions) or "Review output carefully",
            )

        # Exhausted iterations — accept last output regardless
        logger.warning(
            "[Orchestrator] step=%d | max iterations reached; accepting last output.", step.id
        )
        task_mgr.complete_step(
            step.id,
            output=exec_msg.output,   # type: ignore[possibly-undefined]
            agent_msg=exec_msg,
            eval_result=eval_result,  # type: ignore[possibly-undefined]
            iterations=max_iter,
        )

    # ------------------------------------------------------------------
    # Standalone evaluation step
    # ------------------------------------------------------------------

    def _run_evaluation_step(
        self,
        step: PlanStep,
        memory: MemoryManager,
        task_mgr: TaskManager,
        research_context: str,
    ) -> None:
        """Handle a plan step explicitly typed as 'evaluation'."""
        task_id = f"step_{step.id}"
        task_mgr.start_step(step.id)

        # Find the most recent execution output to evaluate
        last_exec_output = task_mgr.last_execution_output() or "No execution output available."

        try:
            eval_msg = self._evaluator.run(
                instruction=step.task,
                execution_output=last_exec_output,
                memory=memory,
                research_context=research_context,
                task_id=task_id,
            )
            eval_result: EvaluationResult = eval_msg.metadata["evaluation"]
            task_mgr.complete_step(
                step.id,
                output=eval_msg.output,
                agent_msg=eval_msg,
                eval_result=eval_result,
            )
        except Exception as exc:           # noqa: BLE001
            logger.error("[Orchestrator] Evaluation step %d failed: %s", step.id, exc)
            task_mgr.fail_step(step.id, error=str(exc))

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _collect_context(
        self,
        step: PlanStep,
        research_contexts: Dict[int, str],
        task_mgr: TaskManager,
    ) -> str:
        """
        Build the research context string for a step by combining outputs
        from all its dependencies.
        """
        parts: List[str] = []
        for dep_id in step.depends_on:
            if dep_id in research_contexts:
                parts.append(f"[Research from step {dep_id}]\n{research_contexts[dep_id]}")
            else:
                dep_output = task_mgr.get_step_output(dep_id)
                if dep_output:
                    parts.append(f"[Output from step {dep_id}]\n{dep_output}")
        return "\n\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Dependency grouping (for parallel execution)
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_level(steps: List[PlanStep]) -> List[List[PlanStep]]:
        """
        Topologically sort steps into levels.  Steps within the same level
        have no inter-dependencies and can run concurrently.
        """
        remaining = {s.id: s for s in steps}
        completed: set = set()
        groups: List[List[PlanStep]] = []

        while remaining:
            ready = [
                s for s in remaining.values()
                if all(dep in completed for dep in s.depends_on)
            ]
            if not ready:
                # Cycle or unresolvable deps — flush remaining as one group
                ready = list(remaining.values())

            groups.append(ready)
            for s in ready:
                completed.add(s.id)
                del remaining[s.id]

        return groups

    # ------------------------------------------------------------------
    # Final answer synthesis
    # ------------------------------------------------------------------

    def _synthesise(
        self, plan: PlannerOutput, task_mgr: TaskManager
    ) -> str:
        """
        Build the final answer by combining successful step outputs.
        Falls back to the last non-empty step output.
        """
        records = task_mgr.completed_records()
        if not records:
            return "The pipeline did not produce a result. Please try again."

        parts = [f"## Goal\n{plan.goal}\n"]
        for rec in records:
            if rec.output:
                parts.append(f"### Step {rec.step_id} [{rec.step_type}]\n{rec.output}")

        if len(parts) == 1:
            # No outputs collected
            return task_mgr.last_execution_output() or "No result produced."

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Backward-compatibility alias (used by old api.py imports)
# ---------------------------------------------------------------------------

class AgentController(AgentOrchestrator):
    """Alias kept for backward compatibility with existing API code."""
    pass
