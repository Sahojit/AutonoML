"""
Unit tests for llm_loader — provider routing, per-agent profiles, error paths.
All real LLM constructors are mocked; no Ollama or OpenAI calls are made.
"""
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_cache():
    from models.llm_loader import get_llm
    get_llm.cache_clear()


# ---------------------------------------------------------------------------
# _resolve_model
# ---------------------------------------------------------------------------

class TestResolveModel:
    def test_default_returns_settings_model(self):
        from models.llm_loader import _resolve_model
        from config.settings import settings
        assert _resolve_model("default") == settings.LLM_MODEL

    def test_known_agent_returns_settings_model_when_no_env_override(self):
        from models.llm_loader import _resolve_model
        from config.settings import settings
        assert _resolve_model("planner") == settings.LLM_MODEL

    def test_env_override_is_respected(self, monkeypatch):
        from models.llm_loader import _resolve_model
        monkeypatch.setenv("PLANNER_MODEL", "gpt-4o")
        assert _resolve_model("planner") == "gpt-4o"

    def test_unknown_agent_falls_back_to_default(self):
        from models.llm_loader import _resolve_model
        from config.settings import settings
        assert _resolve_model("unknown_agent") == settings.LLM_MODEL


# ---------------------------------------------------------------------------
# _resolve_temp
# ---------------------------------------------------------------------------

class TestResolveTemp:
    def test_planner_temp_is_low(self):
        from models.llm_loader import _resolve_temp
        assert _resolve_temp("planner") == pytest.approx(0.2)

    def test_evaluation_temp_is_zero(self):
        from models.llm_loader import _resolve_temp
        assert _resolve_temp("evaluation") == pytest.approx(0.0)

    def test_execution_temp_is_precise(self):
        from models.llm_loader import _resolve_temp
        assert _resolve_temp("execution") == pytest.approx(0.1)

    def test_research_temp_higher_than_execution(self):
        from models.llm_loader import _resolve_temp
        assert _resolve_temp("research") > _resolve_temp("execution")

    def test_unknown_agent_uses_settings_default(self):
        from models.llm_loader import _resolve_temp
        from config.settings import settings
        assert _resolve_temp("unknown") == pytest.approx(settings.LLM_TEMPERATURE)


# ---------------------------------------------------------------------------
# get_llm — Ollama provider
# ---------------------------------------------------------------------------

class TestGetLlmOllama:
    def setup_method(self):
        _reset_cache()

    def test_ollama_provider_returns_instance(self):
        mock_ollama = MagicMock()
        with patch("models.llm_loader.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.LLM_MODEL = "llama3.2:3b"
            mock_settings.LLM_TEMPERATURE = 0.1
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.LLM_MAX_TOKENS = 2048
            mock_settings.LLM_TIMEOUT = 180
            with patch("models.llm_loader.Ollama", return_value=mock_ollama):
                from models.llm_loader import get_llm
                get_llm.cache_clear()
                result = get_llm("execution")
                assert result is mock_ollama

    def test_ollama_called_with_correct_model(self):
        with patch("models.llm_loader.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "ollama"
            mock_settings.LLM_MODEL = "llama3.2:3b"
            mock_settings.LLM_TEMPERATURE = 0.1
            mock_settings.OLLAMA_BASE_URL = "http://localhost:11434"
            mock_settings.LLM_MAX_TOKENS = 2048
            mock_settings.LLM_TIMEOUT = 180
            with patch("models.llm_loader.Ollama") as MockOllama:
                MockOllama.return_value = MagicMock()
                from models.llm_loader import get_llm
                get_llm.cache_clear()
                get_llm("planner")
                _, kwargs = MockOllama.call_args
                assert kwargs.get("model") == "llama3.2:3b"


# ---------------------------------------------------------------------------
# get_llm — OpenAI provider (smoke test, fully mocked)
# ---------------------------------------------------------------------------

class TestGetLlmOpenAI:
    def setup_method(self):
        _reset_cache()

    def test_openai_provider_returns_instance(self):
        mock_chat = MagicMock()
        with patch("models.llm_loader.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.LLM_MODEL = "gpt-4o"
            mock_settings.LLM_TEMPERATURE = 0.1
            mock_settings.OPENAI_API_KEY = "sk-test-key"
            mock_settings.LLM_MAX_TOKENS = 2048
            mock_settings.LLM_TIMEOUT = 180
            with patch("models.llm_loader.ChatOpenAI", return_value=mock_chat):
                from models.llm_loader import get_llm
                get_llm.cache_clear()
                result = get_llm("execution")
                assert result is mock_chat

    def test_openai_raises_without_api_key(self):
        with patch("models.llm_loader.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.LLM_MODEL = "gpt-4o"
            mock_settings.LLM_TEMPERATURE = 0.1
            mock_settings.OPENAI_API_KEY = None
            mock_settings.LLM_MAX_TOKENS = 2048
            mock_settings.LLM_TIMEOUT = 180
            from models.llm_loader import get_llm
            get_llm.cache_clear()
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_llm("execution")

    def test_openai_called_with_api_key(self):
        with patch("models.llm_loader.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.LLM_MODEL = "gpt-4o"
            mock_settings.LLM_TEMPERATURE = 0.1
            mock_settings.OPENAI_API_KEY = "sk-test-key"
            mock_settings.LLM_MAX_TOKENS = 2048
            mock_settings.LLM_TIMEOUT = 180
            with patch("models.llm_loader.ChatOpenAI") as MockChat:
                MockChat.return_value = MagicMock()
                from models.llm_loader import get_llm
                get_llm.cache_clear()
                get_llm("evaluation")
                _, kwargs = MockChat.call_args
                assert kwargs.get("api_key") == "sk-test-key"

    def test_openai_evaluation_uses_zero_temp(self):
        with patch("models.llm_loader.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "openai"
            mock_settings.LLM_MODEL = "gpt-4o"
            mock_settings.LLM_TEMPERATURE = 0.1
            mock_settings.OPENAI_API_KEY = "sk-test-key"
            mock_settings.LLM_MAX_TOKENS = 2048
            mock_settings.LLM_TIMEOUT = 180
            with patch("models.llm_loader.ChatOpenAI") as MockChat:
                MockChat.return_value = MagicMock()
                from models.llm_loader import get_llm
                get_llm.cache_clear()
                get_llm("evaluation")
                _, kwargs = MockChat.call_args
                assert kwargs.get("temperature") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_llm — vLLM provider
# ---------------------------------------------------------------------------

class TestGetLlmVllm:
    def setup_method(self):
        _reset_cache()

    def test_vllm_provider_returns_instance(self):
        mock_chat = MagicMock()
        with patch("models.llm_loader.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "vllm"
            mock_settings.LLM_MODEL = "mistral-7b"
            mock_settings.LLM_TEMPERATURE = 0.1
            mock_settings.VLLM_BASE_URL = "http://localhost:8000"
            mock_settings.LLM_MAX_TOKENS = 2048
            mock_settings.LLM_TIMEOUT = 180
            with patch("models.llm_loader.ChatOpenAI", return_value=mock_chat):
                from models.llm_loader import get_llm
                get_llm.cache_clear()
                result = get_llm("research")
                assert result is mock_chat


# ---------------------------------------------------------------------------
# get_llm — unknown provider
# ---------------------------------------------------------------------------

class TestGetLlmUnknownProvider:
    def setup_method(self):
        _reset_cache()

    def test_unknown_provider_raises_value_error(self):
        with patch("models.llm_loader.settings") as mock_settings:
            mock_settings.LLM_PROVIDER = "anthropic"
            mock_settings.LLM_MODEL = "claude"
            mock_settings.LLM_TEMPERATURE = 0.1
            from models.llm_loader import get_llm
            get_llm.cache_clear()
            with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
                get_llm("default")
