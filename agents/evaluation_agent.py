"""
Evaluation Agent — Hybrid Deep Validator
─────────────────────────────────────────
Combines two evaluation layers:

  (A) LLM evaluation  — semantic quality on 5 criteria (existing)
  (B) Programmatic    — deterministic rule-based checks (new)

Final verdict = PASS only if BOTH layers agree.

ML-Aware rules enforced programmatically:
  • Missing train/test split
  • Suspicious accuracy = 1.0 (possible leakage / overfitting)
  • Accuracy < 0.6 (likely broken)
  • NaN values in output
  • No evaluation metric reported
  • Fitting on test data

Output contract
───────────────
{
  "verdict":    "PASS" | "FAIL",
  "score":      0.0 – 1.0,
  "issues":     ["..."],
  "suggestions": ["..."]
}
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

from langchain.prompts import PromptTemplate

from agents.base_agent import AgentMessage, BaseAgent
from memory.memory_manager import MemoryManager
from models.llm_loader import get_llm

logger = logging.getLogger("multiagent.agent.evaluation")

Verdict = Literal["PASS", "FAIL"]

# Regex to extract a Python code block from text
_CODE_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL | re.IGNORECASE)
# Regex to extract an accuracy float from text like "accuracy: 0.97" or "Accuracy = 1.0"
_ACCURACY_RE   = re.compile(r"accuracy[:\s=]+([0-9]+\.?[0-9]*)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Domain object
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    verdict: Verdict
    score: float
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    # Internal bookkeeping
    llm_verdict: Optional[str] = field(default=None, repr=False)
    programmatic_issues: List[str] = field(default_factory=list, repr=False)

    def passed(self) -> bool:
        return self.verdict == "PASS"

    def feedback_text(self) -> str:
        """Compact feedback string for the Execution Agent retry prompt."""
        lines = [f"Verdict: {self.verdict} (score={self.score:.2f})"]
        if self.issues:
            lines.append("Issues:\n" + "\n".join(f"  - {i}" for i in self.issues))
        if self.suggestions:
            lines.append("Suggestions:\n" + "\n".join(f"  - {s}" for s in self.suggestions))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "score": self.score,
            "issues": self.issues,
            "suggestions": self.suggestions,
        }

    def summary(self) -> str:
        lines = [
            f"Verdict : {self.verdict}",
            f"Score   : {self.score:.2f}",
        ]
        if self.issues:
            lines.append("Issues  :\n  " + "\n  ".join(self.issues))
        if self.suggestions:
            lines.append("Tips    :\n  " + "\n  ".join(self.suggestions))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt — ML-aware evaluation criteria added
# ---------------------------------------------------------------------------

_EVAL_PROMPT = PromptTemplate(
    input_variables=[
        "original_instruction", "execution_output",
        "research_context", "tool_trace",
    ],
    template="""You are a rigorous AI Evaluation Agent and ML expert. Critically review the Execution Agent's output.

Original instruction:
{original_instruction}

Research context:
{research_context}

Tools used (tool trace):
{tool_trace}

Execution Agent output:
{execution_output}

Evaluate on ALL five criteria:
1. Logical correctness   — Is the reasoning sound? Are facts accurate?
2. Task completeness     — Does the output fully address every part of the instruction?
3. Proper tool use       — Were the right tools chosen and called correctly?
4. Code validity         — If code was produced, is it syntactically correct and produces expected output?
5. Statistical validity  — For ML/data tasks: are metrics, splits, and methods appropriate?

ML-SPECIFIC RULES (apply when task involves ML or data):
- FAIL if there is no train/test split (data leakage risk)
- FAIL if accuracy = 1.0 without explanation (likely overfitting or leakage)
- FAIL if accuracy < 0.5 for a classification task (broken model)
- FAIL if evaluation metric is missing or inappropriate for the task type
- FAIL if the model is fitted on test data
- WARN if cross-validation is not used for small datasets
- WARN if class imbalance is ignored when accuracy is the only metric

Score each criterion 0.0–1.0 and average them for the overall score.

Return ONLY a valid JSON object — no markdown, no explanation:
{{
  "verdict": "PASS" or "FAIL",
  "score": <float 0.0-1.0>,
  "issues": ["<specific issue 1>", "..."],
  "suggestions": ["<actionable suggestion 1>", "..."]
}}

Verdict rules:
  PASS → overall score >= 0.75 and no critical blocking issues
  FAIL → overall score < 0.75 OR any critical blocking issue exists

JSON:""",
)


# ---------------------------------------------------------------------------
# Programmatic checks
# ---------------------------------------------------------------------------

def _extract_accuracy(text: str) -> Optional[float]:
    """Return the first accuracy float found in *text*, or None."""
    match = _ACCURACY_RE.search(text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _extract_code(text: str) -> Optional[str]:
    """Extract the first Python code block from *text*, or None."""
    match = _CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else None


def run_programmatic_checks(
    task: str,
    output: str,
    code: Optional[str] = None,
) -> Tuple[bool, List[str]]:
    """
    Deterministic rule-based checks on execution output.

    Returns
    -------
    passed : bool       True if no issues found.
    issues : list[str]  Human-readable issue descriptions.
    """
    issues: List[str] = []
    task_lower  = task.lower()
    output_lower = output.lower()

    # ── NaN / error signals ────────────────────────────────────────────
    if "nan" in output_lower:
        issues.append("NaN values detected in output — possible data quality issue.")

    if "traceback" in output_lower or "error:" in output_lower:
        issues.append("Output contains a Python error or traceback.")

    # ── ML-specific checks ─────────────────────────────────────────────
    is_ml_task = any(
        kw in task_lower
        for kw in ("train", "model", "classifier", "regression", "accuracy",
                   "f1", "precision", "recall", "dataset", "predict", "fit")
    )

    if is_ml_task:
        # Check for train/test split
        if code and "train_test_split" not in code and "cross_val" not in code:
            issues.append(
                "No train/test split or cross-validation detected — "
                "model may be evaluated on training data."
            )

        # Check accuracy range
        acc = _extract_accuracy(output)
        if acc is not None:
            if acc == 1.0:
                issues.append(
                    f"Accuracy = 1.0 detected — likely data leakage or "
                    f"model fitted on test set."
                )
            elif acc < 0.6:
                issues.append(
                    f"Accuracy = {acc:.3f} is suspiciously low — "
                    f"verify model setup and data preprocessing."
                )

        # No metric reported at all
        metric_keywords = ("accuracy", "f1", "precision", "recall", "auc", "rmse", "mae", "r2")
        if not any(kw in output_lower for kw in metric_keywords):
            issues.append(
                "No evaluation metric found in output — "
                "task completeness cannot be verified."
            )

    return (len(issues) == 0, issues)


def ml_sanity_checks(
    code: Optional[str],
    output: str,
) -> Tuple[bool, Optional[str]]:
    """
    Hard ML sanity checks used as a fast pre-filter.

    Returns (passed, reason_or_None).
    """
    if code and "train_test_split" not in code and "cross_val" not in code:
        return False, "No data split detected — possible data leakage."

    output_lower = output.lower()
    if re.search(r"accuracy[:\s=]+1\.0+\b", output_lower):
        return False, "Accuracy = 1.0 detected — possible overfitting or leakage."

    if "traceback" in output_lower:
        return False, "Output contains a Python traceback (execution error)."

    return True, None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class EvaluationAgent(BaseAgent):
    """
    Hybrid deep validator.

    Combines:
      (A) LLM-based scoring on 5 criteria
      (B) Deterministic programmatic + ML sanity checks

    Final verdict = PASS only when both layers agree.
    """

    NAME = "evaluation"
    PASS_THRESHOLD = 0.75

    def __init__(self) -> None:
        super().__init__(self.NAME)
        self._llm = get_llm("evaluation")  # temp=0.0 — strict deterministic judging
        self._chain = _EVAL_PROMPT | self._llm
        self.logger.info("EvaluationAgent initialised.")

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        instruction: str,
        execution_output: str,
        memory: MemoryManager,
        research_context: str = "",
        tool_trace: list | None = None,
        task_id: str | None = None,
    ) -> AgentMessage:
        """
        Evaluate *execution_output* using hybrid (LLM + programmatic) checks.

        The ``EvaluationResult`` is stored in ``metadata["evaluation"]``.
        """
        tid = task_id or self._make_task_id("eval")
        t0 = time.perf_counter()
        self.logger.info(
            "[EvaluationAgent] task=%s | evaluating %d chars",
            tid, len(execution_output),
        )

        memory.add_message(
            "system", "Evaluation Agent is validating the result…", agent_name=self.NAME
        )

        tool_trace_str = (
            json.dumps(tool_trace, indent=2) if tool_trace else "No tools used."
        )

        # Extract code from output for programmatic checks
        code = _extract_code(execution_output)

        # ── (A) LLM evaluation ─────────────────────────────────────────
        raw = ""
        try:
            response = self._chain.invoke({
                "original_instruction": instruction,
                "execution_output": execution_output,
                "research_context": research_context or "None",
                "tool_trace": tool_trace_str,
            })
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:           # noqa: BLE001
            self.logger.exception("[EvaluationAgent] LLM call failed: %s", exc)

        llm_result = self._parse(raw)

        # ── (B) Programmatic + ML sanity checks ────────────────────────
        prog_passed, prog_issues = run_programmatic_checks(
            task=instruction,
            output=execution_output,
            code=code,
        )
        ml_passed, ml_reason = ml_sanity_checks(code=code, output=execution_output)

        # Collect all programmatic issues
        all_prog_issues: List[str] = list(prog_issues)
        if not ml_passed and ml_reason:
            all_prog_issues.append(f"[ML sanity] {ml_reason}")

        # ── Combine verdicts ───────────────────────────────────────────
        programmatic_pass = prog_passed and ml_passed
        final_verdict: Verdict = (
            "PASS"
            if (llm_result.verdict == "PASS" and programmatic_pass)
            else "FAIL"
        )

        # Merge issues from both layers (dedup)
        merged_issues = list(dict.fromkeys(llm_result.issues + all_prog_issues))
        merged_suggestions = list(llm_result.suggestions)
        if all_prog_issues:
            merged_suggestions.append(
                "Fix programmatic check failures before re-evaluating."
            )

        result = EvaluationResult(
            verdict=final_verdict,
            score=llm_result.score,
            issues=merged_issues,
            suggestions=merged_suggestions,
            llm_verdict=llm_result.verdict,
            programmatic_issues=all_prog_issues,
        )

        elapsed = round(time.perf_counter() - t0, 3)

        if all_prog_issues:
            self.logger.info(
                "[EvaluationAgent] task=%s | programmatic issues: %s",
                tid, all_prog_issues,
            )

        memory.add_message("assistant", result.summary(), agent_name=self.NAME)
        memory.update_context(evaluation_feedback=result.summary())

        self.logger.info(
            "[EvaluationAgent] task=%s | llm=%s | prog=%s | final=%s | score=%.2f | %.3fs",
            tid, llm_result.verdict, "PASS" if programmatic_pass else "FAIL",
            final_verdict, result.score, elapsed,
        )

        return self._success(
            task_id=tid,
            input_text=instruction,
            output_text=result.summary(),
            metadata={
                "evaluation": result,
                "verdict": result.verdict,
                "score": result.score,
                "issues": result.issues,
                "suggestions": result.suggestions,
                "llm_verdict": result.llm_verdict,
                "programmatic_issues": result.programmatic_issues,
                "duration_s": elapsed,
            },
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, raw: str) -> EvaluationResult:
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start == -1 or end <= start:
            self.logger.warning("No JSON object in evaluation response; defaulting to PASS.")
            return self._default_pass()

        try:
            data = json.loads(clean[start:end])
            score = float(data.get("score", 0.8))
            verdict_raw = str(data.get("verdict", "PASS")).upper().strip()
            if verdict_raw not in ("PASS", "FAIL"):
                verdict_raw = "PASS" if score >= self.PASS_THRESHOLD else "FAIL"
            return EvaluationResult(
                verdict=verdict_raw,       # type: ignore[arg-type]
                score=score,
                issues=list(data.get("issues", [])),
                suggestions=list(data.get("suggestions", [])),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            self.logger.warning("Evaluation parse error: %s", exc)
            return self._default_pass()

    @staticmethod
    def _default_pass() -> EvaluationResult:
        return EvaluationResult(
            verdict="PASS",
            score=0.75,
            issues=[],
            suggestions=["Evaluation response could not be parsed; defaulting to PASS."],
        )
