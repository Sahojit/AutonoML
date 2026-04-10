"""
Central configuration for the Multi-Agent AI System.
All settings are loaded from environment variables with sensible defaults.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ─── Project ──────────────────────────────────────────────────────────────
    PROJECT_NAME: str = "Multi-Agent AI System"
    VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ─── LLM ──────────────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "ollama"          # "ollama" | "vllm" | "openai"
    LLM_MODEL: str = "llama3.2:3b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    VLLM_BASE_URL: str = "http://localhost:8000"
    OPENAI_API_KEY: Optional[str] = None
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_TIMEOUT: int = 180

    # ─── Embeddings ───────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DEVICE: str = "cpu"         # "cpu" | "cuda" | "mps"

    # ─── Vector Store (Chroma) ────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "knowledge_base"
    CHROMA_HOST: Optional[str] = None
    CHROMA_PORT: int = 8001

    # ─── Memory ───────────────────────────────────────────────────────────────
    SHORT_TERM_MEMORY_LIMIT: int = 20
    LONG_TERM_MEMORY_TOP_K: int = 5

    # ─── Tools ────────────────────────────────────────────────────────────────
    SERPAPI_KEY: Optional[str] = None
    PYTHON_EXEC_TIMEOUT: int = 30
    SQL_DATABASE_URL: str = "sqlite:///./data/agent_data.db"

    # ─── API ──────────────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    CORS_ORIGINS: List[str] = ["*"]

    # ─── Session Store ────────────────────────────────────────────────────────
    REDIS_URL: Optional[str] = None          # e.g. redis://localhost:6379/0
    SESSION_TTL_SECONDS: int = 3600          # 1 hour; used by Redis backend

    # ─── Frontend ─────────────────────────────────────────────────────────────
    FRONTEND_API_URL: str = "http://localhost:8000"

    # ─── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    # ─── Agent Behaviour ──────────────────────────────────────────────────────
    MAX_AGENT_ITERATIONS: int = 3          # max self-correction retries
    EVALUATION_RETRY_LIMIT: int = 3        # alias kept for compatibility
    PLAN_MAX_STEPS: int = 6                # max steps per plan
    MAX_TOOL_ROUNDS: int = 3               # max tool-call rounds per execution
    PARALLEL_RESEARCH: bool = True         # run independent research tasks concurrently

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()

# Ensure required directories exist
Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path("./data").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
