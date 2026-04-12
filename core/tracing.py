"""
LangSmith Tracing Hook
──────────────────────
Enables LangSmith observability when LANGCHAIN_TRACING_V2=true is set in .env.

All LangChain calls (agent chains, LLM invocations, tool calls) are
automatically traced — no code changes needed in the agents.

Usage:
    Call setup_tracing() once at application startup (done in api.py lifespan).

Environment variables:
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls__...
    LANGCHAIN_PROJECT=AutonoML         (optional, default: AutonoML)
    LANGCHAIN_ENDPOINT=https://...     (optional, default: smith.langchain.com)
"""
from __future__ import annotations

import logging
import os

from config.settings import settings

logger = logging.getLogger("multiagent.tracing")


def setup_tracing() -> bool:
    """
    Configure LangSmith tracing via environment variables.

    Returns True if tracing was enabled, False if skipped.
    Fails silently so a missing API key never crashes the server.
    """
    if not settings.LANGCHAIN_TRACING_V2:
        logger.debug("LangSmith tracing disabled (LANGCHAIN_TRACING_V2=false).")
        return False

    if not settings.LANGCHAIN_API_KEY:
        logger.warning(
            "LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is not set — tracing skipped."
        )
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"]     = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"]     = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"]    = settings.LANGCHAIN_ENDPOINT

    logger.info(
        "LangSmith tracing enabled — project=%s  endpoint=%s",
        settings.LANGCHAIN_PROJECT,
        settings.LANGCHAIN_ENDPOINT,
    )
    return True


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing is currently active."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"


def get_tracing_status() -> dict:
    """Return a dict suitable for inclusion in the /status endpoint."""
    return {
        "tracing_enabled": is_tracing_enabled(),
        "project":         os.environ.get("LANGCHAIN_PROJECT", settings.LANGCHAIN_PROJECT),
        "endpoint":        os.environ.get("LANGCHAIN_ENDPOINT", settings.LANGCHAIN_ENDPOINT),
    }
