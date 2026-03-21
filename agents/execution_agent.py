"""
Execution Agent — Tool-Calling ReAct Loop
──────────────────────────────────────────
The "hands" of the system.  Resolves tool calls dynamically via the
ToolRegistry and iterates up to MAX_TOOL_ROUNDS times before returning a
final ``AgentMessage``.

Pipeline
────────
task
  → LLM decides which tool to use (or answers directly)
  → ToolRegistry.execute(tool_name, **args)
  → tool result fed back into next LLM round
  → repeat up to MAX_TOOL_ROUNDS
  → return AgentMessage with structured result + tool trace
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List

from langchain.prompts import PromptTemplate

from agents.base_agent import AgentMessage, BaseAgent
from config.settings import settings
from memory.memory_manager import MemoryManager
from models.llm_loader import get_llm
from tools.tool_registry import ToolResult, tool_registry

logger = logging.getLogger("multiagent.agent.execution")

# ---------------------------------------------------------------------------
# Regex patterns for tool-call detection
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE    = re.compile(r"```python\n(.*?)```", re.DOTALL | re.IGNORECASE)
_WEB_SEARCH_RE    = re.compile(r"WEB_SEARCH:\s*(.+)", re.IGNORECASE)
_SQL_QUERY_RE     = re.compile(r"SQL_QUERY:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)
_DATASET_RE       = re.compile(r"DATASET_ANALYZE:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_AUTO_EXPERIMENT_RE = re.compile(r"AUTO_EXPERIMENT:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_TOOL_CALL_RE     = re.compile(
    r"TOOL_CALL:\s*(\w+)\s*\{([^}]*)\}", re.DOTALL | re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_EXECUTION_PROMPT = PromptTemplate(
    input_variables=[
        "instruction", "research_context",
        "conversation_history", "tool_results",
        "tools_summary", "feedback", "past_reflections",
    ],
    template="""You are an expert AI Execution Agent. You solve tasks by reasoning step-by-step and calling the available tools.

{tools_summary}

Tool call syntax:
  Python code   → wrap in ```python\\n...``` fenced block
  Web search    → WEB_SEARCH: <query>
  SQL query     → SQL_QUERY: <SELECT ...>
  Dataset       → DATASET_ANALYZE: sklearn:<name>  OR  csv:<filepath>
  Auto-experiment → DATASET_ANALYZE: sklearn:<name>  (will auto-compare multiple ML models)

Conversation history:
{conversation_history}

Research context:
{research_context}

Previous tool outputs (this round):
{tool_results}

Feedback from Evaluation Agent (if retrying):
{feedback}

⚠ Previous mistakes to avoid (learned from past failures):
{past_reflections}

Current task:
{instruction}

Think carefully. Decide whether you need a tool or can answer directly.
If using a tool, emit the call and NOTHING else on that response.
If you have all the information needed, produce the final complete answer.

FORMATTING RULES (always follow):
- When providing code, ALWAYS wrap it in a fenced block with the language tag:
  ```python
  # your code here
  ```
- Never mix code inline with prose — keep code in its own fenced block.
- Use ```sql for SQL, ```bash for shell commands, ```python for Python.
- Explanations go before or after the code block, never inside it.

Response:""",
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ExecutionAgent(BaseAgent):
    """
    ReAct-style execution agent with multi-round tool calling.

    Supports:
    - Python code execution (PythonExecutor)
    - Web search (WebSearchTool)
    - SQL queries (SQLQueryTool)
    - Dataset analysis (DatasetAnalyzer)
    - Generic TOOL_CALL: <name> {args} dispatch
    """

    NAME = "execution"

    def __init__(self) -> None:
        super().__init__(self.NAME)
        self._llm = get_llm("execution")  # temp=0.1 — precise code generation
        self._chain = _EXECUTION_PROMPT | self._llm
        self._max_rounds = settings.MAX_TOOL_ROUNDS
        self.logger.info(
            "ExecutionAgent initialised (max_tool_rounds=%d).", self._max_rounds
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        instruction: str,
        memory: MemoryManager,
        research_context: str = "",
        feedback: str = "",
        task_id: str | None = None,
    ) -> AgentMessage:
        """
        Execute *instruction* and return an ``AgentMessage``.

        Parameters
        ----------
        instruction      : str   Task from the Planner.
        memory           : MemoryManager
        research_context : str   Output from the Research Agent (may be empty).
        feedback         : str   Issues / suggestions from Evaluation Agent (retry loop).
        task_id          : str   Optional caller-supplied step ID.
        """
        tid = task_id or self._make_task_id("exec")
        t0 = time.perf_counter()
        self.logger.info("[ExecutionAgent] task=%s | instruction=%s", tid, instruction[:120])

        memory.add_message(
            "system", "Execution Agent is working on the task…", agent_name=self.NAME
        )

        history = memory.get_formatted_history(last_n=6)
        tools_summary = tool_registry.tools_summary()

        # Retrieve past reflections for this task (self-reflection memory)
        reflections = memory.retrieve_reflections(instruction, k=3)
        past_reflections = (
            "\n".join(f"  • {r}" for r in reflections)
            if reflections else "None — no past failures recorded for this task type."
        )

        accumulated_tool_results = ""
        tool_trace: List[Dict[str, Any]] = []
        final_response = ""
        current_instruction = instruction

        for round_num in range(1, self._max_rounds + 1):
            self.logger.debug(
                "[ExecutionAgent] task=%s | ReAct round %d/%d",
                tid, round_num, self._max_rounds,
            )

            try:
                raw = self._chain.invoke({
                    "instruction": current_instruction,
                    "research_context": research_context or "No research context provided.",
                    "conversation_history": history,
                    "tool_results": accumulated_tool_results or "None",
                    "tools_summary": tools_summary,
                    "feedback": feedback or "None",
                    "past_reflections": past_reflections,
                })
                raw_text: str = raw.content if hasattr(raw, "content") else str(raw)
            except Exception as exc:           # noqa: BLE001
                self.logger.exception("[ExecutionAgent] LLM call failed: %s", exc)
                raw_text = f"Execution failed: {exc}"
                break

            tool_result, tool_called = self._dispatch(raw_text)

            if tool_called:
                result_str = tool_result.format()
                accumulated_tool_results += (
                    f"\n[Round {round_num} — {tool_result.tool_name}]\n{result_str}\n"
                )
                tool_trace.append(tool_result.to_dict())
                self.logger.info(
                    "[ExecutionAgent] task=%s | round=%d | tool=%s | success=%s",
                    tid, round_num, tool_result.tool_name, tool_result.success,
                )
                # Continue to next round with tool output incorporated
                current_instruction = (
                    f"{instruction}\n\n"
                    f"Tool output (round {round_num}):\n{result_str}\n\n"
                    "Using this output, provide the complete final answer."
                )
            else:
                final_response = raw_text
                break
        else:
            # All rounds exhausted — use last LLM response
            final_response = raw_text   # type: ignore[possibly-undefined]

        elapsed = round(time.perf_counter() - t0, 3)

        memory.add_message("assistant", final_response, agent_name=self.NAME)
        memory.update_context(execution_result=final_response)

        self.logger.info(
            "[ExecutionAgent] task=%s | %d tool calls | %.3fs",
            tid, len(tool_trace), elapsed,
        )

        return self._success(
            task_id=tid,
            input_text=instruction,
            output_text=final_response,
            metadata={
                "tool_trace": tool_trace,
                "rounds_used": len(tool_trace),
                "feedback_applied": bool(feedback),
                "duration_s": elapsed,
            },
        )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, text: str) -> tuple[ToolResult, bool]:
        """
        Detect a tool call in *text* and execute it via the ToolRegistry.

        Returns (ToolResult, was_called).
        """
        # 1. Python code block
        py_match = _CODE_BLOCK_RE.search(text)
        if py_match:
            code = py_match.group(1).strip()
            self.logger.info("[ExecutionAgent] dispatch → python_exec (%d chars)", len(code))
            return tool_registry.execute("python_exec", code=code), True

        # 2. Web search
        web_match = _WEB_SEARCH_RE.search(text)
        if web_match:
            query = web_match.group(1).strip()
            self.logger.info("[ExecutionAgent] dispatch → web_search: %s", query)
            return tool_registry.execute("web_search", query=query), True

        # 3. SQL query
        sql_match = _SQL_QUERY_RE.search(text)
        if sql_match:
            query = sql_match.group(1).strip()
            self.logger.info("[ExecutionAgent] dispatch → sql_query: %s", query[:80])
            return tool_registry.execute("sql_query", query=query), True

        # 4. Dataset analysis
        ds_match = _DATASET_RE.search(text)
        if ds_match:
            source = ds_match.group(1).strip()
            self.logger.info("[ExecutionAgent] dispatch → dataset_analyze: %s", source)
            return tool_registry.execute("dataset_analyze", source=source), True

        # 5. Auto-experimentation (multi-model comparison)
        ae_match = _AUTO_EXPERIMENT_RE.search(text)
        if ae_match:
            source = ae_match.group(1).strip()
            self.logger.info("[ExecutionAgent] dispatch → auto_experiment: %s", source)
            return tool_registry.execute("auto_experiment", source=source), True

        # 6. Generic TOOL_CALL: <name> {json_args}
        tc_match = _TOOL_CALL_RE.search(text)
        if tc_match:
            name = tc_match.group(1).strip()
            raw_args = "{" + tc_match.group(2).strip() + "}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            if name in tool_registry:
                self.logger.info("[ExecutionAgent] dispatch → %s(%s)", name, args)
                return tool_registry.execute(name, **args), True

        # No tool detected
        dummy = ToolResult(tool_name="none", success=True, output=None)
        return dummy, False
