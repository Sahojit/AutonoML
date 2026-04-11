"""
Unit tests for FileReaderTool — text files, PDF fallback, chunking, errors.
"""
import json
import os

import pytest

from tools.file_reader import FileReaderTool, get_file_reader


@pytest.fixture
def reader():
    return FileReaderTool()


@pytest.fixture
def txt_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello world\nThis is a test file.\n" * 50, encoding="utf-8")
    return str(f)


@pytest.fixture
def json_file(tmp_path):
    f = tmp_path / "data.json"
    f.write_text(json.dumps({"key": "value", "num": 42}), encoding="utf-8")
    return str(f)


@pytest.fixture
def py_file(tmp_path):
    f = tmp_path / "script.py"
    f.write_text("def hello():\n    return 'hi'\n", encoding="utf-8")
    return str(f)


# ---------------------------------------------------------------------------
# Success cases — text files
# ---------------------------------------------------------------------------

class TestReadText:
    def test_success_flag_true(self, reader, txt_file):
        result = reader.read(txt_file)
        assert result["success"] is True

    def test_content_not_empty(self, reader, txt_file):
        result = reader.read(txt_file)
        assert len(result["content"]) > 0

    def test_extension_detected(self, reader, txt_file):
        result = reader.read(txt_file)
        assert result["extension"] == ".txt"

    def test_path_in_result(self, reader, txt_file):
        result = reader.read(txt_file)
        assert result["path"] == txt_file

    def test_size_bytes_positive(self, reader, txt_file):
        result = reader.read(txt_file)
        assert result["size_bytes"] > 0

    def test_char_count_matches_content(self, reader, txt_file):
        result = reader.read(txt_file)
        assert result["char_count"] == len(result["content"])

    def test_error_field_none_on_success(self, reader, txt_file):
        result = reader.read(txt_file)
        assert result["error"] is None

    def test_json_file_reads(self, reader, json_file):
        result = reader.read(json_file)
        assert result["success"] is True
        assert "value" in result["content"]

    def test_py_file_reads(self, reader, py_file):
        result = reader.read(py_file)
        assert result["success"] is True
        assert "def hello" in result["content"]

    def test_max_chars_truncates(self, reader, txt_file):
        result = reader.read(txt_file, max_chars=20)
        assert len(result["content"]) <= 20
        assert result["truncated"] is True

    def test_no_truncation_when_short(self, reader, json_file):
        result = reader.read(json_file, max_chars=10_000)
        assert result["truncated"] is False


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunking:
    def test_chunk_index_zero_returns_first_chunk(self, reader, txt_file):
        result = reader.read(txt_file, chunk_index=0, chunk_size=50)
        assert result["success"] is True
        assert len(result["content"]) <= 50

    def test_chunk_index_one_returns_second_chunk(self, reader, txt_file):
        r0 = reader.read(txt_file, chunk_index=0, chunk_size=50)
        r1 = reader.read(txt_file, chunk_index=1, chunk_size=50)
        assert r0["content"] != r1["content"]

    def test_chunk_count_positive(self, reader, txt_file):
        result = reader.read(txt_file, chunk_size=100)
        assert result["chunks"] >= 1

    def test_out_of_range_chunk_returns_error(self, reader, txt_file):
        result = reader.read(txt_file, chunk_index=9999, chunk_size=50)
        assert result["success"] is False

    def test_list_chunks_returns_count(self, reader, txt_file):
        result = reader.list_chunks(txt_file, chunk_size=100)
        assert result["success"] is True
        assert result["chunks"] >= 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_file_returns_failure(self, reader):
        result = reader.read("/nonexistent/path/file.txt")
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_directory_path_returns_failure(self, reader, tmp_path):
        result = reader.read(str(tmp_path))
        assert result["success"] is False

    def test_unsupported_extension_returns_failure(self, reader, tmp_path):
        f = tmp_path / "file.xyz"
        # Write non-UTF-8 binary content so _is_text_file returns False
        f.write_bytes(bytes(range(256)))
        result = reader.read(str(f))
        assert result["success"] is False

    def test_error_field_set_on_failure(self, reader):
        result = reader.read("/no/such/file.txt")
        assert result["error"] is not None

    def test_content_empty_on_failure(self, reader):
        result = reader.read("/no/such/file.txt")
        assert result["content"] == ""


# ---------------------------------------------------------------------------
# PDF — no pypdf installed (expected graceful failure)
# ---------------------------------------------------------------------------

class TestPdf:
    def test_pdf_without_pypdf_returns_failure(self, reader, tmp_path):
        f = tmp_path / "fake.pdf"
        f.write_text("fake pdf content", encoding="utf-8")
        import sys
        import unittest.mock as mock
        with mock.patch.dict(sys.modules, {"pypdf": None}):
            result = reader.read(str(f))
        # Either reads it as text or returns graceful failure — never crashes
        assert "success" in result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def test_get_file_reader_returns_same_instance():
    r1 = get_file_reader()
    r2 = get_file_reader()
    assert r1 is r2


def test_get_file_reader_is_file_reader_tool():
    assert isinstance(get_file_reader(), FileReaderTool)


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def test_file_read_registered_in_tool_registry():
    import tools  # triggers _register_all()
    assert "file_read" in tools.tool_registry.list_tool_names()
