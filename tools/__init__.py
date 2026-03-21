"""
Tools package — registers all tools with the central ToolRegistry at import time.

Import this package early in the application lifecycle so that the Execution Agent
can resolve tool names dynamically.
"""
from __future__ import annotations

import logging

from .dataset_analyzer import DatasetAnalyzer, get_dataset_analyzer
from .python_executor import PythonExecutor, get_python_executor
from .sql_tool import SQLQueryTool, get_sql_tool
from .tool_registry import ToolRegistry, ToolResult, tool_registry
from .web_search import WebSearchTool, get_web_search

logger = logging.getLogger("multiagent.tools")


def _register_all() -> None:
    """Wire up every tool singleton into the global ToolRegistry."""

    # ── PythonExecutor ──────────────────────────────────────────────────
    executor = get_python_executor()
    tool_registry.register_tool(
        name="python_exec",
        description="Execute Python code and return stdout/stderr/exit_code",
        func=executor.execute,
        parameters={"code": "str"},
    )

    # ── WebSearchTool ────────────────────────────────────────────────────
    searcher = get_web_search()
    tool_registry.register_tool(
        name="web_search",
        description="Search the web and return a list of results with titles, URLs, snippets",
        func=searcher.search,
        parameters={"query": "str", "num_results": "int (optional, default 5)"},
    )

    # ── SQLQueryTool ─────────────────────────────────────────────────────
    sql = get_sql_tool()
    tool_registry.register_tool(
        name="sql_query",
        description="Execute a read-only SQL SELECT query and return rows",
        func=sql.execute,
        parameters={"query": "str", "params": "dict (optional)"},
    )

    # ── DatasetAnalyzer ──────────────────────────────────────────────────
    analyzer = get_dataset_analyzer()
    tool_registry.register_tool(
        name="dataset_analyze",
        description=(
            "Load and analyze a dataset. "
            "source can be 'sklearn:<name>' (iris, wine, diabetes, …) "
            "or 'csv:<filepath>'. Returns shape, dtypes, stats, sample rows."
        ),
        func=analyzer.analyze,
        parameters={"source": "str", "dataset_name": "str (optional)", "max_rows": "int (optional)"},
    )

    # ── Auto-Experimentation ─────────────────────────────────────────────
    tool_registry.register_tool(
        name="auto_experiment",
        description=(
            "Automatically train and compare multiple ML classifiers "
            "(RandomForest, LogisticRegression, GradientBoosting, XGBoost) "
            "on a dataset. Returns a ranked leaderboard and selects the best model. "
            "source: 'sklearn:<name>' or 'csv:<filepath>'. "
            "Use this whenever the task asks to train, compare, or select a model."
        ),
        func=analyzer.auto_experiment,
        parameters={
            "source": "str",
            "target_column": "str (optional, default 'target')",
            "test_size": "float (optional, default 0.2)",
        },
    )

    logger.info("Tool registry ready: %s", tool_registry.list_tool_names())


# Register on package import
_register_all()

__all__ = [
    # Singletons
    "tool_registry",
    "get_python_executor",
    "get_web_search",
    "get_sql_tool",
    "get_dataset_analyzer",
    # Classes
    "ToolRegistry",
    "ToolResult",
    "PythonExecutor",
    "WebSearchTool",
    "SQLQueryTool",
    "DatasetAnalyzer",
]
