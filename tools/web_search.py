"""
WebSearchTool.

Uses SerpAPI when a key is configured; falls back to DuckDuckGo Instant
Answer API (no key required).  Returns a list of result dicts and a
structured summary suitable for LLM consumption.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import requests

from config.settings import settings

logger = logging.getLogger("multiagent.tools.web_search")


class WebSearchTool:
    """
    Retrieve web search results.

    Return value of ``search()``::

        [
            {"title": str, "url": str, "snippet": str},
            ...
        ]
    """

    MAX_RESULTS = 5

    def search(self, query: str, num_results: int | None = None) -> Dict[str, Any]:
        """
        Search the web and return a result dict::

            {
                "success": bool,
                "query": str,
                "results": list[dict],
                "provider": str,
                "duration_s": float,
                "error": str | None,
            }
        """
        n = num_results or self.MAX_RESULTS
        t0 = time.perf_counter()
        provider = "serpapi" if settings.SERPAPI_KEY else "duckduckgo"
        try:
            if settings.SERPAPI_KEY:
                results = self._serpapi_search(query, n)
            else:
                results = self._ddg_search(query, n)
            elapsed = round(time.perf_counter() - t0, 3)
            logger.info(
                "WebSearch [%s]: '%s' → %d results in %.3fs",
                provider, query, len(results), elapsed,
            )
            return {
                "success": True,
                "query": query,
                "results": results,
                "provider": provider,
                "duration_s": elapsed,
                "error": None,
            }
        except Exception as exc:           # noqa: BLE001
            elapsed = round(time.perf_counter() - t0, 3)
            logger.error("WebSearch [%s] failed: %s", provider, exc)
            return {
                "success": False,
                "query": query,
                "results": [],
                "provider": provider,
                "duration_s": elapsed,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Providers
    # ------------------------------------------------------------------

    def _serpapi_search(self, query: str, n: int) -> List[Dict[str, str]]:
        logger.debug("SerpAPI search: '%s'", query)
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "num": n,
            "api_key": settings.SERPAPI_KEY,
            "engine": "google",
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("organic_results", [])[:n]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", ""),
            })
        return results

    def _ddg_search(self, query: str, n: int) -> List[Dict[str, str]]:
        """DuckDuckGo Instant Answer API — free, no key needed."""
        logger.debug("DuckDuckGo search: '%s'", query)
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results: List[Dict[str, str]] = []

        if data.get("AbstractText"):
            results.append({
                "title": data.get("Heading", "DDG Abstract"),
                "url": data.get("AbstractURL", ""),
                "snippet": data["AbstractText"],
            })

        for topic in data.get("RelatedTopics", []):
            if len(results) >= n:
                break
            if isinstance(topic, dict) and "Text" in topic:
                results.append({
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                })

        return results[:n]

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_results(self, result: Dict[str, Any]) -> str:
        """Convert search result dict to a numbered text block."""
        if not result.get("success"):
            return f"Web search failed: {result.get('error', 'unknown')}"
        items = result.get("results", [])
        if not items:
            return "No web results found."
        lines = [f"Web search results for: '{result['query']}'", ""]
        for i, r in enumerate(items, 1):
            lines.append(f"{i}. {r['title']}")
            if r.get("url"):
                lines.append(f"   URL: {r['url']}")
            lines.append(f"   {r['snippet']}")
            lines.append("")
        return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[WebSearchTool] = None


def get_web_search() -> WebSearchTool:
    global _instance
    if _instance is None:
        _instance = WebSearchTool()
    return _instance
