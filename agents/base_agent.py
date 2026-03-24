"""
Base agent class and structured AgentMessage envelope.

All agents inherit BaseAgent and communicate exclusively via AgentMessage,
ensuring deterministic orchestration with a consistent JSON-serialisable
format throughout the pipeline.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    """Call once at application startup to configure structured logging."""
    import sys
    from config.settings import settings

    fmt = settings.LOG_FORMAT
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("./logs/agent_system.log", encoding="utf-8"),
        ],
    )


# ---------------------------------------------------------------------------
# Structured message envelope
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AgentMessage:
    """
    Canonical communication envelope shared by every agent.

    Fields
    ------
    agent      : name of the agent that produced this message
    task_id    : unique identifier for the step / sub-task
    input      : raw input the agent received
    output     : agent's response / result
    status     : "success" | "fail"
    metadata   : arbitrary extra data (tool calls, scores, verdicts, …)
    timestamp  : UTC ISO-8601 creation time
    """

    agent: str
    task_id: str
    input: str
    output: str
    status: str                                   # "success" | "fail"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "task_id": self.task_id,
            "input": self.input,
            "output": self.output,
            "status": self.status,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent:
    """
    Abstract base for all pipeline agents.

    Sub-classes must implement ``run()``.
    They inherit helper factories ``_success()`` and ``_fail()`` for
    constructing correctly-typed AgentMessage objects.
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"multiagent.agent.{agent_name}")

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    def _make_task_id(self, prefix: str = "") -> str:
        tag = prefix or self.agent_name
        return f"{tag}_{uuid.uuid4().hex[:8]}"

    def _success(
        self,
        task_id: str,
        input_text: str,
        output_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        self.logger.debug(
            "[%s] task=%s → success (%d chars)",
            self.agent_name,
            task_id,
            len(output_text),
        )
        return AgentMessage(
            agent=self.agent_name,
            task_id=task_id,
            input=input_text,
            output=output_text,
            status="success",
            metadata=metadata or {},
        )

    def _fail(
        self,
        task_id: str,
        input_text: str,
        error: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentMessage:
        self.logger.error(
            "[%s] task=%s → FAIL: %s",
            self.agent_name,
            task_id,
            error,
        )
        return AgentMessage(
            agent=self.agent_name,
            task_id=task_id,
            input=input_text,
            output=f"ERROR: {error}",
            status="fail",
            metadata=metadata or {"error": error},
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def run(self, *args: Any, **kwargs: Any) -> AgentMessage:
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement run()"
        )
