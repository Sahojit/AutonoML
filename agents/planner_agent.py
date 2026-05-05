"""
Planner Agent — Hierarchical Task Decomposer
─────────────────────────────────────────────
Decomposes a user query into atomic subtasks, labels each with a type
(research / execution / evaluation), and tracks dependencies between tasks.

Output contract
───────────────
The LLM is instructed to emit ONLY the following JSON:

{
  "goal": "<concise description of the overall objective>",
  "tasks": [
    {"id": 1, "type": "research",   "task": "...", "depends_on": []},
    {"id": 2, "type": "execution",  "task": "...", "depends_on": [1]},
    {"id": 3, "type": "evaluation", "task": "...", "depends_on": [2]}
  ]
}
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate

from agents.base_agent import AgentMessage, BaseAgent
from config.settings import settings
from memory.memory_manager import MemoryManager
from models.llm_loader import get_llm

logger = logging.getLogger("multiagent.agent.planner")

# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------

VALID_TYPES = {"research", "execution", "evaluation"}


@dataclass
class PlanStep:
    id: int
    type: str           # "research" | "execution" | "evaluation"
    task: str
    depends_on: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "task": self.task,
            "depends_on": self.depends_on,
        }


@dataclass
class PlannerOutput:
    goal: str
    tasks: List[PlanStep]
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "tasks": [s.to_dict() for s in self.tasks],
        }

    def summary(self) -> str:
        lines = [f"Goal: {self.goal}", "─" * 60]
        for s in self.tasks:
            dep = f" (after {s.depends_on})" if s.depends_on else ""
            lines.append(f"  [{s.id}] {s.type.upper()}{dep}: {s.task}")
        return "\n".join(lines)

    def tasks_of_type(self, task_type: str) -> List[PlanStep]:
        return [s for s in self.tasks if s.type == task_type]

    def independent_tasks(self) -> List[PlanStep]:
        """Return tasks with no dependencies (can run in parallel)."""
        return [s for s in self.tasks if not s.depends_on]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PLANNER_PROMPT = PromptTemplate(
    input_variables=["query", "conversation_history", "max_steps"],
    template="""You are a Hierarchical AI Task Planner. Your job is to decompose the user's request into a minimal, ordered set of atomic subtasks.

RULES:
1. Produce at most {max_steps} tasks. Prefer exactly 3: research → execution → evaluation.
2. Identify the correct type for each task:
   - "research"   : gather facts, retrieve context, look up information
   - "execution"  : write code, run computations, solve the problem, call tools
   - "evaluation" : validate correctness, check logic, verify output quality
3. Each task must have a single, unambiguous task description — minimize vagueness.
4. List explicit dependencies in "depends_on" (task IDs that must finish first).
5. Multiple independent research tasks (same level) will run in PARALLEL — split research when beneficial.
6. Skip "research" if the task is purely computational or the user provides all needed facts.
7. Return ONLY valid JSON — no explanation, no markdown, no extra text.

Conversation history (for context):
{conversation_history}

User request:
{query}

Required JSON format:
{{
  "goal": "<one-sentence description of the overall objective>",
  "tasks": [
    {{"id": 1, "type": "research",   "task": "<specific research task>",   "depends_on": []}},
    {{"id": 2, "type": "research",   "task": "<specific research task>",   "depends_on": []}},
    {{"id": 3, "type": "execution",  "task": "<specific execution task>",  "depends_on": [1, 2]}},
    {{"id": 4, "type": "evaluation", "task": "<specific evaluation task>", "depends_on": [3]}}
  ]
}}

JSON output:""",
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PlannerAgent(BaseAgent):
    """
    Hierarchical task planner.

    Produces a ``PlannerOutput`` (goal + typed tasks with dependencies)
    AND an ``AgentMessage`` for the orchestrator.
    """

    NAME = "planner"

    def __init__(self) -> None:
        super().__init__(self.NAME)
        self._llm = get_llm("planner")   # temp=0.2 — structured, deterministic
        self._chain = _PLANNER_PROMPT | self._llm
        self.logger.info("PlannerAgent initialised.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(self, query: str, memory: MemoryManager) -> AgentMessage:
        """
        Generate a structured hierarchical plan.

        Returns an ``AgentMessage`` whose ``output`` is the JSON-encoded plan
        and whose ``metadata["plan"]`` contains the ``PlannerOutput`` object.
        """
        task_id = self._make_task_id("plan")
        t0 = time.perf_counter()
        self.logger.info("[PlannerAgent] task=%s | query=%s", task_id, query[:120])

        memory.add_message("system", "PlannerAgent is decomposing the task…", agent_name=self.NAME)
        history = memory.get_formatted_history(last_n=6)

        raw = ""
        try:
            response = self._chain.invoke({
                "query": query,
                "conversation_history": history or "None",
                "max_steps": settings.PLAN_MAX_STEPS,
            })
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:           # noqa: BLE001
            self.logger.exception("PlannerAgent LLM call failed: %s", exc)

        plan = self._parse_output(raw, query)
        elapsed = round(time.perf_counter() - t0, 3)

        summary = plan.summary()
        memory.add_message("assistant", summary, agent_name=self.NAME)
        memory.update_context(plan=summary)

        self.logger.info(
            "[PlannerAgent] task=%s | %d tasks | %.3fs",
            task_id, len(plan.tasks), elapsed,
        )

        return self._success(
            task_id=task_id,
            input_text=query,
            output_text=json.dumps(plan.to_dict(), indent=2),
            metadata={
                "plan": plan,
                "goal": plan.goal,
                "task_count": len(plan.tasks),
                "duration_s": elapsed,
            },
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_output(self, raw: str, query: str) -> PlannerOutput:
        """Extract and validate the JSON plan from the LLM response."""
        clean = re.sub(r"```(?:json)?", "", raw).strip()

        # Locate outermost { … }
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end <= start:
            self.logger.warning("No JSON object found in planner output; using fallback.")
            return self._fallback(query, raw)

        try:
            data = json.loads(clean[start:end])
        except json.JSONDecodeError as exc:
            self.logger.warning("JSON parse error: %s — using fallback.", exc)
            return self._fallback(query, raw)

        goal = data.get("goal") or f"Solve: {query[:80]}"
        raw_steps = data.get("tasks") or data.get("steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            return self._fallback(query, raw)

        steps: List[PlanStep] = []
        for item in raw_steps:
            step_type = str(item.get("type", "execution")).lower()
            if step_type not in VALID_TYPES:
                step_type = "execution"
            steps.append(PlanStep(
                id=int(item.get("id", len(steps) + 1)),
                type=step_type,
                task=str(item.get("task", "")),
                depends_on=[int(x) for x in item.get("depends_on", [])],
            ))

        return PlannerOutput(goal=goal, tasks=steps, raw_response=raw)

    def _fallback(self, query: str, raw: str = "") -> PlannerOutput:
        """Safe three-step default plan when the LLM output cannot be parsed."""
        self.logger.warning("PlannerAgent: using fallback plan for query: %s", query[:80])
        return PlannerOutput(
            goal=f"Solve: {query[:100]}",
            tasks=[
                PlanStep(1, "research",   f"Research background for: {query}", depends_on=[]),
                PlanStep(2, "execution",  f"Execute and solve: {query}",        depends_on=[1]),
                PlanStep(3, "evaluation", "Validate the results from step 2.",  depends_on=[2]),
            ],
            raw_response=raw,
        )
