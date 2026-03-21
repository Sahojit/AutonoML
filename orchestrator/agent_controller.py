"""
Agent Controller
────────────────
Central orchestrator. Wires together all four agents and drives the
full execution pipeline:

  User Query
    → PlannerAgent  (decompose)
    → For each step:
        → ResearchAgent  (if needed)
        → ExecutionAgent (run)
        → EvaluationAgent (validate; retry up to EVALUATION_RETRY_LIMIT)
    → Synthesise final answer
"""

import logging
import uuid
from typing import AsyncGenerator, Dict, Generator, Optional

from agents.planner_agent import PlannerAgent
from agents.research_agent import ResearchAgent
from agents.execution_agent import ExecutionAgent
from agents.evaluation_agent import EvaluationAgent
from memory.memory_manager import AgentContext, MemoryManager
from orchestrator.task_manager import TaskManager
from config.settings import settings

logger = logging.getLogger(__name__)


class AgentController:
    """
    Singleton-per-request orchestrator.

    Usage:
        controller = AgentController(session_id="abc")
        result = controller.run(user_query)
    """

    def __init__(self, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self._memory = MemoryManager(self.session_id)
        self._planner = PlannerAgent()
        self._researcher = ResearchAgent()
        self._executor = ExecutionAgent()
        self._evaluator = EvaluationAgent()
        logger.info("AgentController ready — session=%s", self.session_id)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _needs_research(self, instruction: str) -> bool:
        """Heuristic: research before execution if query mentions retrieval keywords."""
        keywords = {
            "find", "search", "lookup", "retrieve", "document", "paper", "article",
            "what is", "explain", "describe", "context", "background", "information",
        }
        low = instruction.lower()
        return any(k in low for k in keywords)

    def _synthesise_final_answer(self, task_manager: TaskManager, query: str) -> str:
        """Merge results from all completed steps into a coherent final answer."""
        results = []
        for rec in task_manager.run.records:
            if rec.result:
                results.append(f"[{rec.step.agent.upper()} — Step {rec.step.step_number}]\n{rec.result}")

        if not results:
            return "I was unable to produce a result for your query. Please try again."

        # If only one result, return it directly
        if len(results) == 1:
            return results[0].split("\n", 1)[-1].strip()

        header = f"## Answer for: {query}\n\n"
        body = "\n\n---\n\n".join(results)
        return header + body

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, query: str) -> Dict:
        """
        Synchronous end-to-end execution.

        Returns dict with:
          final_answer, session_id, plan_summary, step_records, duration_s
        """
        logger.info("=== AgentController.run START session=%s ===", self.session_id)

        # 1. Initialise memory context
        self._memory.add_message("user", query)
        ctx = AgentContext(user_query=query)
        self._memory.set_context(ctx)

        task_manager = TaskManager(query=query, session_id=self.session_id)

        # 2. Plan
        logger.info("Step 0: Planning…")
        plan = self._planner.run(query, self._memory)
        task_manager.set_plan(plan)

        # 3. Execute plan step by step
        last_research = ""
        last_execution = ""

        for rec in task_manager.run.records:
            step = rec.step
            logger.info("Step %d [%s]: %s", step.step_number, step.agent, step.instruction[:80])
            task_manager.start_step(step.step_number)

            # ── Research ─────────────────────────────────────────────────────
            if step.agent == "research":
                research_output = self._researcher.run(step.instruction, self._memory)
                last_research = research_output
                task_manager.complete_step(step.step_number, research_output)
                continue

            # ── Execution ─────────────────────────────────────────────────────
            if step.agent == "execution":
                # Optionally pre-fetch research context if not already done
                context = last_research
                if not context and self._needs_research(step.instruction):
                    context = self._researcher.run(step.instruction, self._memory)
                    last_research = context

                retry_count = 0
                exec_output = ""

                while retry_count <= settings.EVALUATION_RETRY_LIMIT:
                    exec_output = self._executor.run(
                        instruction=step.instruction,
                        memory=self._memory,
                        research_context=context,
                    )
                    last_execution = exec_output

                    # Auto-evaluate every execution output
                    eval_result = self._evaluator.run(
                        instruction=step.instruction,
                        execution_output=exec_output,
                        memory=self._memory,
                        research_context=context,
                    )

                    if eval_result.passed():
                        logger.info("Step %d passed evaluation.", step.step_number)
                        break

                    if eval_result.verdict == "FAIL":
                        logger.warning("Step %d FAILED evaluation — no retry.", step.step_number)
                        task_manager.fail_step(step.step_number, eval_result.summary())
                        break

                    # RETRY — feed corrected output as additional context
                    retry_count += 1
                    logger.info("Step %d RETRY %d/%d.", step.step_number, retry_count, settings.EVALUATION_RETRY_LIMIT)
                    task_manager.fail_step(step.step_number, eval_result.summary(), retry=True)
                    task_manager.start_step(step.step_number)

                    if eval_result.corrected_output:
                        context = (
                            f"{context}\n\nPrevious attempt feedback:\n{eval_result.summary()}"
                            f"\n\nPartially corrected output:\n{eval_result.corrected_output}"
                        )
                    else:
                        context = f"{context}\n\nPrevious attempt feedback:\n{eval_result.summary()}"

                task_manager.complete_step(step.step_number, exec_output)
                continue

            # ── Evaluation (standalone step) ──────────────────────────────────
            if step.agent == "evaluation":
                eval_result = self._evaluator.run(
                    instruction=step.instruction,
                    execution_output=last_execution,
                    memory=self._memory,
                    research_context=last_research,
                )
                task_manager.complete_step(step.step_number, eval_result.summary())
                continue

            # Unknown agent
            logger.warning("Unknown agent type '%s' — skipping step %d.", step.agent, step.step_number)
            task_manager.skip_step(step.step_number)

        # 4. Synthesise final answer
        final_answer = self._synthesise_final_answer(task_manager, query)
        task_manager.set_final_answer(final_answer)

        # 5. Persist session
        self._memory.add_message("assistant", final_answer)
        self._memory.save_session_to_long_term()

        logger.info(
            "=== AgentController.run END session=%s  duration=%.1fs ===",
            self.session_id,
            task_manager.run.duration or 0,
        )

        return {
            "final_answer": final_answer,
            "session_id": self.session_id,
            "plan_summary": plan.summary(),
            "step_records": [r.to_dict() for r in task_manager.run.records],
            "duration_s": task_manager.run.duration,
        }

    def get_history(self) -> list:
        return self._memory.get_conversation_history()

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> list:
        """Ingest a document into long-term memory."""
        return self._memory.store_in_long_term([text], [metadata or {}])
