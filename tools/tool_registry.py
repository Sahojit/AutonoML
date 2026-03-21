"""
Central tool registry for the Execution Agent.

Tools register themselves once at import time via ``tool_registry.register_tool()``.
The Execution Agent resolves tool names dynamically at runtime, calls them, and
receives a ``ToolResult`` with uniform structure.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("multiagent.tools.registry")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToolDefinition:
    name: str
    description: str
    func: Callable[..., Any]
    parameters: Dict[str, str] = field(default_factory=dict)   # param_name → type hint


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
        }

    def format(self) -> str:
        if self.success:
            out = str(self.output) if not isinstance(self.output, str) else self.output
            return f"[{self.tool_name}] ✓ ({self.duration_s:.2f}s)\n{out}"
        return f"[{self.tool_name}] ✗ ERROR: {self.error}"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Singleton registry that maps tool names to callables.

    Usage
    -----
    # register
    tool_registry.register_tool("python_exec", "Run Python code", executor.execute,
                                 {"code": "str"})

    # execute
    result: ToolResult = tool_registry.execute("python_exec", code="print('hi')")
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Registration API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[Dict[str, str]] = None,
    ) -> Callable:
        """Decorator factory. Registers the decorated function as a tool."""

        def decorator(func: Callable) -> Callable:
            self.register_tool(name, description, func, parameters)
            return func

        return decorator

    def register_tool(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: Optional[Dict[str, str]] = None,
    ) -> None:
        if name in self._tools:
            logger.warning("Tool '%s' already registered — overwriting.", name)
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            func=func,
            parameters=parameters or {},
        )
        logger.info("Tool registered: %s", name)

    # ------------------------------------------------------------------
    # Execution API
    # ------------------------------------------------------------------

    def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name. Always returns a ToolResult — never raises."""
        if name not in self._tools:
            available = ", ".join(self._tools) or "none"
            return ToolResult(
                tool_name=name,
                success=False,
                output=None,
                error=f"Tool '{name}' not found. Available: {available}",
            )

        tool = self._tools[name]
        t0 = time.perf_counter()
        try:
            result = tool.func(**kwargs)
            elapsed = time.perf_counter() - t0
            logger.info("Tool '%s' finished in %.3fs", name, elapsed)
            return ToolResult(
                tool_name=name,
                success=True,
                output=result,
                duration_s=elapsed,
            )
        except Exception as exc:          # noqa: BLE001
            elapsed = time.perf_counter() - t0
            logger.error("Tool '%s' raised %s: %s", name, type(exc).__name__, exc)
            return ToolResult(
                tool_name=name,
                success=False,
                output=None,
                error=str(exc),
                duration_s=elapsed,
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": td.name,
                "description": td.description,
                "parameters": td.parameters,
            }
            for td in self._tools.values()
        ]

    def tools_summary(self) -> str:
        """One-line description of every registered tool, for prompt injection."""
        if not self._tools:
            return "No tools available."
        lines = ["Available tools:"]
        for td in self._tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in td.parameters.items())
            sig = f"{td.name}({params})" if params else td.name
            lines.append(f"  • {sig} — {td.description}")
        return "\n".join(lines)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# Module-level singleton shared across the whole application
tool_registry = ToolRegistry()
