"""
Unit tests for LangSmith tracing hook.
No real LangSmith calls — env vars and settings are mocked.
"""
import os
from unittest.mock import patch

import pytest

from core.tracing import get_tracing_status, is_tracing_enabled, setup_tracing


def _clean_env():
    for key in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT", "LANGCHAIN_ENDPOINT"):
        os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# setup_tracing
# ---------------------------------------------------------------------------

class TestSetupTracing:
    def setup_method(self):
        _clean_env()

    def teardown_method(self):
        _clean_env()

    def test_returns_false_when_disabled(self):
        with patch("core.tracing.settings") as s:
            s.LANGCHAIN_TRACING_V2 = False
            s.LANGCHAIN_API_KEY = "ls-key"
            s.LANGCHAIN_PROJECT = "proj"
            s.LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
            assert setup_tracing() is False

    def test_returns_false_when_no_api_key(self):
        with patch("core.tracing.settings") as s:
            s.LANGCHAIN_TRACING_V2 = True
            s.LANGCHAIN_API_KEY = None
            assert setup_tracing() is False

    def test_returns_true_when_enabled_with_key(self):
        with patch("core.tracing.settings") as s:
            s.LANGCHAIN_TRACING_V2 = True
            s.LANGCHAIN_API_KEY = "ls-test-key"
            s.LANGCHAIN_PROJECT = "AutonoML"
            s.LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
            result = setup_tracing()
        assert result is True

    def test_sets_env_vars_when_enabled(self):
        with patch("core.tracing.settings") as s:
            s.LANGCHAIN_TRACING_V2 = True
            s.LANGCHAIN_API_KEY = "ls-test-key"
            s.LANGCHAIN_PROJECT = "MyProject"
            s.LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
            setup_tracing()
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "ls-test-key"
        assert os.environ.get("LANGCHAIN_PROJECT") == "MyProject"

    def test_does_not_set_env_vars_when_disabled(self):
        with patch("core.tracing.settings") as s:
            s.LANGCHAIN_TRACING_V2 = False
            setup_tracing()
        assert os.environ.get("LANGCHAIN_TRACING_V2") is None

    def test_does_not_set_env_vars_when_no_key(self):
        with patch("core.tracing.settings") as s:
            s.LANGCHAIN_TRACING_V2 = True
            s.LANGCHAIN_API_KEY = None
            setup_tracing()
        assert os.environ.get("LANGCHAIN_TRACING_V2") is None


# ---------------------------------------------------------------------------
# is_tracing_enabled
# ---------------------------------------------------------------------------

class TestIsTracingEnabled:
    def setup_method(self):
        _clean_env()

    def teardown_method(self):
        _clean_env()

    def test_false_when_env_not_set(self):
        assert is_tracing_enabled() is False

    def test_true_when_env_set_to_true(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        assert is_tracing_enabled() is True

    def test_false_when_env_set_to_false(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        assert is_tracing_enabled() is False

    def test_case_insensitive(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "TRUE"
        assert is_tracing_enabled() is True


# ---------------------------------------------------------------------------
# get_tracing_status
# ---------------------------------------------------------------------------

class TestGetTracingStatus:
    def setup_method(self):
        _clean_env()

    def teardown_method(self):
        _clean_env()

    def test_returns_dict(self):
        assert isinstance(get_tracing_status(), dict)

    def test_has_tracing_enabled_key(self):
        assert "tracing_enabled" in get_tracing_status()

    def test_has_project_key(self):
        assert "project" in get_tracing_status()

    def test_has_endpoint_key(self):
        assert "endpoint" in get_tracing_status()

    def test_tracing_enabled_false_by_default(self):
        assert get_tracing_status()["tracing_enabled"] is False

    def test_tracing_enabled_true_after_env_set(self):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        assert get_tracing_status()["tracing_enabled"] is True
