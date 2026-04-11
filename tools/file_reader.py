"""
FileReaderTool

Reads plain-text (.txt, .md, .csv, .json, .py, .log) and PDF files,
returning the content as a string for ingestion into agent memory or
for direct use as research context.

Supports:
  - Plain text files  — any UTF-8 encoded file
  - PDF files         — requires pypdf (optional); falls back gracefully
  - Chunking          — large files can be split into fixed-size chunks

Usage::

    from tools.file_reader import get_file_reader
    reader = get_file_reader()
    result = reader.read("path/to/file.txt")
    result = reader.read("path/to/report.pdf", max_chars=5000)
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("multiagent.tools.file_reader")

# Extensions treated as plain text
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".py",
    ".log", ".yaml", ".yml", ".toml", ".rst",
}

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB hard limit


class FileReaderTool:
    """Read text and PDF files and return their content as a string."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(
        self,
        path: str,
        max_chars: int = 20_000,
        chunk_index: Optional[int] = None,
        chunk_size: int = 4_000,
    ) -> Dict[str, Any]:
        """
        Read a file and return its content.

        Parameters
        ----------
        path        : Absolute or relative path to the file.
        max_chars   : Maximum characters to return (default 20 000).
                      Truncation is noted in the result.
        chunk_index : If set, return only the nth chunk (0-based) of
                      ``chunk_size`` characters instead of the full file.
        chunk_size  : Characters per chunk when ``chunk_index`` is used.

        Returns
        -------
        {
            "success":    bool,
            "path":       str,
            "extension":  str,
            "size_bytes": int,
            "char_count": int,
            "content":    str,
            "truncated":  bool,
            "chunks":     int,   # total available chunks
            "error":      str | None,
        }
        """
        path = os.path.expanduser(path.strip())

        if not os.path.exists(path):
            return self._error(path, f"File not found: {path}")

        if not os.path.isfile(path):
            return self._error(path, f"Path is not a file: {path}")

        size = os.path.getsize(path)
        if size > MAX_FILE_SIZE_BYTES:
            return self._error(
                path,
                f"File too large ({size // (1024*1024)} MB). "
                f"Limit is {MAX_FILE_SIZE_BYTES // (1024*1024)} MB.",
            )

        ext = os.path.splitext(path)[1].lower()

        if ext == ".pdf":
            content = self._read_pdf(path)
        elif ext in _TEXT_EXTENSIONS or self._is_text_file(path):
            content = self._read_text(path)
        else:
            return self._error(path, f"Unsupported file type: '{ext}'")

        if isinstance(content, dict):  # error dict from sub-reader
            return content

        total_chunks = max(1, -(-len(content) // chunk_size))  # ceiling div

        if chunk_index is not None:
            start = chunk_index * chunk_size
            end = start + chunk_size
            content = content[start:end]
            if not content:
                return self._error(path, f"Chunk index {chunk_index} out of range (total={total_chunks})")
            truncated = False
        else:
            truncated = len(content) > max_chars
            content = content[:max_chars]

        logger.info(
            "FileReaderTool: read '%s' | ext=%s | size=%d | chars=%d | chunks=%d",
            path, ext, size, len(content), total_chunks,
        )

        return {
            "success":   True,
            "path":      path,
            "extension": ext,
            "size_bytes": size,
            "char_count": len(content),
            "content":   content,
            "truncated": truncated,
            "chunks":    total_chunks,
            "error":     None,
        }

    def list_chunks(self, path: str, chunk_size: int = 4_000) -> Dict[str, Any]:
        """Return chunk count and metadata without reading full content."""
        result = self.read(path, max_chars=0, chunk_size=chunk_size)
        if not result["success"]:
            return result
        return {
            "success":    True,
            "path":       path,
            "size_bytes": result["size_bytes"],
            "chunks":     result["chunks"],
            "chunk_size": chunk_size,
        }

    # ------------------------------------------------------------------
    # Plain-text reader
    # ------------------------------------------------------------------

    def _read_text(self, path: str) -> str | Dict:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception as exc:            # noqa: BLE001
            return self._error(path, f"Failed to read text file: {exc}")

    # ------------------------------------------------------------------
    # PDF reader
    # ------------------------------------------------------------------

    def _read_pdf(self, path: str) -> str | Dict:
        try:
            from pypdf import PdfReader
        except ImportError:
            return self._error(
                path,
                "pypdf is not installed. Install it with: pip install pypdf",
            )
        try:
            reader = PdfReader(path)
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(pages)
        except Exception as exc:            # noqa: BLE001
            return self._error(path, f"Failed to read PDF: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_text_file(path: str, sample_bytes: int = 1024) -> bool:
        """Heuristic: try reading a small sample as UTF-8."""
        try:
            with open(path, "rb") as fh:
                sample = fh.read(sample_bytes)
            sample.decode("utf-8")
            return True
        except (UnicodeDecodeError, OSError):
            return False

    @staticmethod
    def _error(path: str, msg: str) -> Dict[str, Any]:
        logger.warning("FileReaderTool error: %s", msg)
        return {
            "success":    False,
            "path":       path,
            "extension":  os.path.splitext(path)[1].lower(),
            "size_bytes": 0,
            "char_count": 0,
            "content":    "",
            "truncated":  False,
            "chunks":     0,
            "error":      msg,
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[FileReaderTool] = None


def get_file_reader() -> FileReaderTool:
    global _instance
    if _instance is None:
        _instance = FileReaderTool()
    return _instance
