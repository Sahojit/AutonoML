"""
SQLQueryTool backed by SQLAlchemy.

Supports read-only SELECT queries by default; write access is opt-in.
Returns rows, columns, counts and timing in a JSON-serialisable dict.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import settings

logger = logging.getLogger("multiagent.tools.sql_tool")


class SQLQueryTool:
    """
    Execute SQL queries against any SQLAlchemy-compatible database.

    Default mode is read-only (SELECT only).  Set ``read_only=False``
    when write access is explicitly required by the task.
    """

    _WRITE_KEYWORDS = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE")

    def __init__(
        self,
        database_url: str | None = None,
        read_only: bool = True,
    ) -> None:
        self.database_url = database_url or settings.SQL_DATABASE_URL
        self.read_only = read_only
        self._engine = create_engine(self.database_url, future=True)
        logger.info(
            "SQLQueryTool: connected to %s (read_only=%s)",
            self.database_url, read_only,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a SQL query and return::

            {
                "success":    bool,
                "rows":       list[dict],
                "columns":    list[str],
                "row_count":  int,
                "duration_s": float,
                "error":      str | None,
            }
        """
        guard_error = self._guard(query)
        if guard_error:
            return {
                "success": False,
                "rows": [],
                "columns": [],
                "row_count": 0,
                "duration_s": 0.0,
                "error": guard_error,
            }

        logger.info("SQLQueryTool: executing: %s", query[:200])
        t0 = time.perf_counter()
        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(query), params or {})
                columns = list(result.keys())
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                elapsed = round(time.perf_counter() - t0, 3)
                logger.info(
                    "SQLQueryTool: %d rows in %.3fs", len(rows), elapsed
                )
                return {
                    "success": True,
                    "rows": rows,
                    "columns": columns,
                    "row_count": len(rows),
                    "duration_s": elapsed,
                    "error": None,
                }
        except SQLAlchemyError as exc:
            elapsed = round(time.perf_counter() - t0, 3)
            logger.warning("SQLQueryTool error: %s", exc)
            return {
                "success": False,
                "rows": [],
                "columns": [],
                "row_count": 0,
                "duration_s": elapsed,
                "error": str(exc),
            }

    def list_tables(self) -> List[str]:
        insp = inspect(self._engine)
        return insp.get_table_names()

    def describe_table(self, table_name: str) -> List[Dict[str, str]]:
        insp = inspect(self._engine)
        cols = insp.get_columns(table_name)
        return [{"name": c["name"], "type": str(c["type"])} for c in cols]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_result(self, result: Dict[str, Any], max_rows: int = 20) -> str:
        if not result["success"]:
            return f"SQL Error: {result['error']}"
        if not result["rows"]:
            return "Query returned 0 rows."
        header = " | ".join(result["columns"])
        lines = [header, "-" * len(header)]
        for row in result["rows"][:max_rows]:
            lines.append(" | ".join(str(v) for v in row.values()))
        extra = result["row_count"] - max_rows
        if extra > 0:
            lines.append(f"... ({extra} more rows)")
        lines.append(f"\n[{result['row_count']} rows total | {result.get('duration_s', 0):.3f}s]")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _guard(self, query: str) -> Optional[str]:
        if not self.read_only:
            return None
        normalized = query.strip().upper()
        for kw in self._WRITE_KEYWORDS:
            if normalized.startswith(kw):
                return (
                    f"Read-only mode: '{kw}' statements are not permitted. "
                    "Use a SELECT query."
                )
        return None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[SQLQueryTool] = None


def get_sql_tool() -> SQLQueryTool:
    global _instance
    if _instance is None:
        _instance = SQLQueryTool()
    return _instance
