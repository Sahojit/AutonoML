import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from tools.tool_registry import ToolRegistry, ToolResult


@pytest.fixture
def registry():
    return ToolRegistry()


def _echo(text: str = "") -> str:
    return f"echo: {text}"


def _add(a: int = 0, b: int = 0) -> int:
    return a + b


def _fail(**kwargs):
    raise ValueError("tool intentionally failed")


class TestToolRegistryRegistration:

    def test_register_tool_adds_to_registry(self, registry):
        registry.register_tool("echo", "Echoes input", _echo, {"text": "str"})
        assert "echo" in registry

    def test_register_two_tools(self, registry):
        registry.register_tool("echo", "Echoes", _echo)
        registry.register_tool("add", "Adds two numbers", _add)
        assert "echo" in registry
        assert "add" in registry

    def test_list_tool_names_returns_registered(self, registry):
        registry.register_tool("echo", "Echoes", _echo)
        assert "echo" in registry.list_tool_names()

    def test_list_tools_returns_dicts(self, registry):
        registry.register_tool("echo", "Echoes", _echo, {"text": "str"})
        tools = registry.list_tools()
        assert isinstance(tools, list)
        assert tools[0]["name"] == "echo"

    def test_tools_summary_contains_name(self, registry):
        registry.register_tool("echo", "Echoes text back", _echo)
        summary = registry.tools_summary()
        assert "echo" in summary

    def test_tools_summary_contains_description(self, registry):
        registry.register_tool("echo", "Echoes text back", _echo)
        summary = registry.tools_summary()
        assert "Echoes text back" in summary

    def test_contains_returns_false_for_unknown(self, registry):
        assert "nonexistent_tool" not in registry

    def test_overwrite_existing_tool(self, registry):
        registry.register_tool("echo", "v1", _echo)
        registry.register_tool("echo", "v2", _add)
        tools = registry.list_tools()
        descriptions = [t["description"] for t in tools]
        assert "v2" in descriptions


class TestToolRegistryExecute:

    def test_execute_returns_tool_result(self, registry):
        registry.register_tool("echo", "Echoes", _echo)
        result = registry.execute("echo", text="hello")
        assert isinstance(result, ToolResult)

    def test_execute_success_is_true(self, registry):
        registry.register_tool("echo", "Echoes", _echo)
        result = registry.execute("echo", text="hi")
        assert result.success is True

    def test_execute_output_is_correct(self, registry):
        registry.register_tool("echo", "Echoes", _echo)
        result = registry.execute("echo", text="world")
        assert result.output == "echo: world"

    def test_execute_tool_name_in_result(self, registry):
        registry.register_tool("echo", "Echoes", _echo)
        result = registry.execute("echo", text="x")
        assert result.tool_name == "echo"

    def test_execute_duration_is_non_negative(self, registry):
        registry.register_tool("echo", "Echoes", _echo)
        result = registry.execute("echo", text="y")
        assert result.duration_s >= 0

    def test_execute_unknown_tool_returns_failure(self, registry):
        result = registry.execute("unknown_tool")
        assert result.success is False
        assert result.error is not None

    def test_execute_failing_tool_returns_failure(self, registry):
        registry.register_tool("fail", "Always fails", _fail)
        result = registry.execute("fail")
        assert result.success is False
        assert "tool intentionally failed" in (result.error or "")

    def test_execute_error_field_is_none_on_success(self, registry):
        registry.register_tool("add", "Adds", _add)
        result = registry.execute("add", a=3, b=4)
        assert result.error is None

    def test_execute_numeric_output(self, registry):
        registry.register_tool("add", "Adds", _add)
        result = registry.execute("add", a=10, b=5)
        assert result.output == 15


class TestToolResult:

    def test_to_dict_keys(self):
        r = ToolResult(tool_name="t", success=True, output="ok")
        d = r.to_dict()
        assert set(d.keys()) == {"tool_name", "success", "output", "error", "duration_s"}

    def test_format_success_contains_tool_name(self):
        r = ToolResult(tool_name="echo", success=True, output="hello")
        assert "echo" in r.format()

    def test_format_failure_contains_error(self):
        r = ToolResult(tool_name="echo", success=False, output=None, error="crashed")
        assert "crashed" in r.format()

    def test_format_failure_contains_error_marker(self):
        r = ToolResult(tool_name="echo", success=False, output=None, error="oops")
        assert "ERROR" in r.format() or "✗" in r.format()
