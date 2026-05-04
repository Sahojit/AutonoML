"""
Unified LLM loader — supports Ollama, vLLM, and OpenAI-compatible endpoints.

Multi-Model Strategy
────────────────────
Each agent type receives a dedicated LLM configuration via get_llm(agent_type).
Temperature profiles are tuned per agent role:
  planner    → 0.2  (structured, deterministic JSON planning)
  research   → 0.4  (creative synthesis of retrieved context)
  execution  → 0.1  (precise, literal code generation)
  evaluation → 0.0  (strict, reproducible judging)

Model overrides per agent (env vars, optional):
  PLANNER_MODEL, RESEARCH_MODEL, EXECUTION_MODEL, EVALUATION_MODEL
  If unset, all agents share LLM_MODEL.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-agent profiles
# ---------------------------------------------------------------------------

_AGENT_PROFILES: dict[str, dict] = {
    "planner":    {"temperature": 0.2, "model_env": "PLANNER_MODEL"},
    "research":   {"temperature": 0.4, "model_env": "RESEARCH_MODEL"},
    "execution":  {"temperature": 0.1, "model_env": "EXECUTION_MODEL"},
    "evaluation": {"temperature": 0.0, "model_env": "EVALUATION_MODEL"},
    "default":    {"temperature": settings.LLM_TEMPERATURE, "model_env": None},
}


def _resolve_model(agent_type: str) -> str:
    """Return model name for *agent_type*, honouring env-var overrides."""
    profile = _AGENT_PROFILES.get(agent_type, _AGENT_PROFILES["default"])
    env_key = profile.get("model_env")
    if env_key:
        override = os.environ.get(env_key)
        if override:
            return override
    return settings.LLM_MODEL


def _resolve_temp(agent_type: str) -> float:
    profile = _AGENT_PROFILES.get(agent_type, _AGENT_PROFILES["default"])
    return float(profile.get("temperature", settings.LLM_TEMPERATURE))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _build_ollama(agent_type: str) -> Ollama:
    model = _resolve_model(agent_type)
    temp  = _resolve_temp(agent_type)
    logger.info(
        "Loading Ollama LLM: agent=%-10s  model=%s  temp=%.1f",
        agent_type, model, temp,
    )
    return Ollama(
        model=model,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temp,
        num_predict=settings.LLM_MAX_TOKENS,
        timeout=settings.LLM_TIMEOUT,
    )


def _build_vllm(agent_type: str) -> ChatOpenAI:
    model = _resolve_model(agent_type)
    temp  = _resolve_temp(agent_type)
    logger.info(
        "Loading vLLM LLM:   agent=%-10s  model=%s  temp=%.1f",
        agent_type, model, temp,
    )
    return ChatOpenAI(
        model=model,
        openai_api_base=f"{settings.VLLM_BASE_URL}/v1",
        openai_api_key="vllm-no-key",
        temperature=temp,
        max_tokens=settings.LLM_MAX_TOKENS,
        timeout=settings.LLM_TIMEOUT,
    )


def _build_groq(agent_type: str) -> ChatOpenAI:
    if not settings.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY must be set when LLM_PROVIDER=groq")
    model = _resolve_model(agent_type)
    temp  = _resolve_temp(agent_type)
    logger.info(
        "Loading Groq LLM:   agent=%-10s  model=%s  temp=%.1f",
        agent_type, model, temp,
    )
    return ChatOpenAI(
        model=model,
        base_url=settings.GROQ_BASE_URL,
        api_key=settings.GROQ_API_KEY,
        temperature=temp,
        max_tokens=settings.LLM_MAX_TOKENS,
        timeout=settings.LLM_TIMEOUT,
    )


def _build_openai(agent_type: str) -> ChatOpenAI:
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")
    model = _resolve_model(agent_type)
    temp  = _resolve_temp(agent_type)
    logger.info(
        "Loading OpenAI LLM: agent=%-10s  model=%s  temp=%.1f",
        agent_type, model, temp,
    )
    return ChatOpenAI(
        model=model,
        api_key=settings.OPENAI_API_KEY,
        temperature=temp,
        max_tokens=settings.LLM_MAX_TOKENS,
        timeout=settings.LLM_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def get_llm(agent_type: str = "default") -> Any:
    """
    Return a cached LLM instance for *agent_type*.

    Each agent type is cached independently (maxsize=8 covers all agents
    plus future extensions).

    Usage in agents::

        self._llm = get_llm("execution")   # temp=0.1, code-precise
        self._llm = get_llm("evaluation")  # temp=0.0, strict judging
    """
    provider = settings.LLM_PROVIDER.lower()
    builders = {
        "ollama":  _build_ollama,
        "vllm":    _build_vllm,
        "openai":  _build_openai,
        "groq":    _build_groq,
    }
    if provider not in builders:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider!r}. Choose from {list(builders)}"
        )
    return builders[provider](agent_type)
