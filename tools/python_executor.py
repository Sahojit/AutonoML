"""
PythonExecutor tool.

Runs arbitrary Python code in an isolated subprocess with a configurable
timeout.  Returns stdout, stderr, exit code, duration, and warnings — all
JSON-serialisable so the Execution Agent can reason about the result.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import settings

logger = logging.getLogger("multiagent.tools.python_executor")

# ---------------------------------------------------------------------------
# Safe prelude injected before every user snippet
# ---------------------------------------------------------------------------

_SAFE_PRELUDE = textwrap.dedent("""\
    import math, json, re, csv, io, datetime, itertools, functools
    from pathlib import Path
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        pass
    try:
        import sklearn
    except ImportError:
        pass
""")

# Patterns we flag in warnings (execution still proceeds — sandbox is subprocess)
_SENSITIVE_PATTERNS: List[str] = [
    "os.system",
    "subprocess.call",
    "subprocess.Popen",
    "shutil.rmtree",
    "__import__",
    "socket.connect",
]


class PythonExecutor:
    """
    Execute Python code snippets in a sandboxed subprocess.

    The code is written to a temp file (avoids shell-escape issues with -c),
    then run with the host Python interpreter so all installed packages are
    available.  A hard ``timeout`` is enforced.
    """

    def __init__(self, timeout: int | None = None) -> None:
        self.timeout = timeout or settings.PYTHON_EXEC_TIMEOUT

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute(self, code: str, inject_prelude: bool = True) -> Dict[str, Any]:
        """
        Run *code* and return::

            {
                "success":    bool,
                "stdout":     str,
                "stderr":     str,
                "exit_code":  int,
                "duration_s": float,
                "warnings":   list[str],
            }
        """
        warnings = self._check_sensitive(code)
        full_code = (_SAFE_PRELUDE + "\n" + code) if inject_prelude else code

        logger.info("PythonExecutor: executing snippet (%d chars)", len(full_code))

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(full_code)
            tmp_path = Path(fh.name)

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            elapsed = round(time.perf_counter() - t0, 3)
            success = proc.returncode == 0
            logger.info(
                "PythonExecutor: exit=%d, %.3fs, stdout=%d chars",
                proc.returncode, elapsed, len(proc.stdout),
            )
            return {
                "success": success,
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
                "exit_code": proc.returncode,
                "duration_s": elapsed,
                "warnings": warnings,
            }
        except subprocess.TimeoutExpired:
            elapsed = round(time.perf_counter() - t0, 3)
            logger.warning("PythonExecutor: timed out after %ds", self.timeout)
            return {
                "success": False,
                "stdout": "",
                "stderr": f"TimeoutError: execution exceeded {self.timeout}s limit.",
                "exit_code": -1,
                "duration_s": elapsed,
                "warnings": warnings,
            }
        except Exception as exc:           # noqa: BLE001
            elapsed = round(time.perf_counter() - t0, 3)
            logger.error("PythonExecutor unexpected error: %s", exc)
            return {
                "success": False,
                "stdout": "",
                "stderr": str(exc),
                "exit_code": -2,
                "duration_s": elapsed,
                "warnings": warnings,
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_result(self, result: Dict[str, Any]) -> str:
        status = "SUCCESS" if result["success"] else f"FAILED (exit {result['exit_code']})"
        parts = [
            f"=== PythonExecutor: {status} | {result.get('duration_s', 0):.3f}s ==="
        ]
        if result.get("warnings"):
            parts.append("⚠ Warnings: " + "; ".join(result["warnings"]))
        if result.get("stdout"):
            parts.append(f"STDOUT:\n{result['stdout']}")
        if result.get("stderr"):
            parts.append(f"STDERR:\n{result['stderr']}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_sensitive(self, code: str) -> List[str]:
        found = [p for p in _SENSITIVE_PATTERNS if p in code]
        if found:
            logger.warning("PythonExecutor: sensitive patterns detected: %s", found)
        return found


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_executor: Optional[PythonExecutor] = None


def get_python_executor() -> PythonExecutor:
    global _executor
    if _executor is None:
        _executor = PythonExecutor()
    return _executor
