"""
Research Agent
──────────────
Queries vector memory FIRST, then optionally supplements with a live web
search.  Returns an ``AgentMessage`` with a structured research summary
that downstream agents can consume directly.

Pipeline
────────
query → embedding → Chroma vector search → retrieved context
      → (optional) web search
      → LLM synthesis → AgentMessage(output=summary)
"""
from __future__ import annotations

import logging
import time
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from agents.base_agent import AgentMessage, BaseAgent
from config.settings import settings
from memory.memory_manager import MemoryManager
from memory.vector_store import get_vector_store
from models.llm_loader import get_llm
from tools.web_search import get_web_search

logger = logging.getLogger("multiagent.agent.research")

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_RESEARCH_PROMPT = PromptTemplate(
    input_variables=["query", "retrieved_docs", "web_results"],
    template="""You are an expert Research Agent. Using the provided knowledge-base documents and web search results, produce a concise, factual research summary.

Query:
{query}

Knowledge Base Documents:
{retrieved_docs}

Web Search Results:
{web_results}

Instructions:
- Summarise only the most relevant facts for the query.
- Do NOT hallucinate — only reference information from the sources above.
- Keep the summary under 600 words.
- Explicitly state what information is missing or uncertain if the sources are insufficient.
- Structure the output clearly with short paragraphs.

Research Summary:""",
)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ResearchAgent(BaseAgent):
    """
    RAG-powered research agent.

    Workflow
    --------
    1. Retrieve top-k documents from the Chroma vector store.
    2. Optionally run a web search for up-to-date information.
    3. LLM synthesises a concise research summary.
    4. Summary stored back into long-term memory for future sessions.
    """

    NAME = "research"

    def __init__(self, use_web_search: bool = True) -> None:
        super().__init__(self.NAME)
        self._llm = get_llm("research")  # temp=0.4 — creative synthesis
        self._chain = _RESEARCH_PROMPT | self._llm
        self._vector_store = get_vector_store()
        self._web_search = get_web_search() if use_web_search else None
        self.logger.info("ResearchAgent initialised (web_search=%s).", use_web_search)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        instruction: str,
        memory: MemoryManager,
        use_web: bool = False,
        task_id: str | None = None,
    ) -> AgentMessage:
        """
        Execute a research step and return an ``AgentMessage``.

        Parameters
        ----------
        instruction : str       Research query from the Planner.
        memory      : MemoryManager
        use_web     : bool      Supplement KB results with a web search.
        task_id     : str       Optional caller-supplied ID (e.g. "step_1").
        """
        tid = task_id or self._make_task_id("research")
        t0 = time.perf_counter()
        self.logger.info("[ResearchAgent] task=%s | query=%s", tid, instruction[:120])

        memory.add_message(
            "system", "Research Agent is retrieving knowledge…", agent_name=self.NAME
        )

        # 1. Vector DB retrieval
        kb_docs = self._retrieve_from_kb(instruction)
        formatted_docs = self._format_docs(kb_docs)

        # 2. Optional web search
        web_text = "Web search not requested."
        web_results_raw = None
        if use_web and self._web_search:
            web_result = self._web_search.search(instruction, num_results=5)
            web_text = self._web_search.format_results(web_result)
            web_results_raw = web_result

        # 3. LLM synthesis
        try:
            response = self._chain.invoke({
                "query": instruction,
                "retrieved_docs": formatted_docs,
                "web_results": web_text,
            })
            summary: str = (
                response.content if hasattr(response, "content") else str(response)
            )
        except Exception as exc:           # noqa: BLE001
            self.logger.exception("[ResearchAgent] LLM synthesis failed: %s", exc)
            summary = f"Research retrieval completed (LLM synthesis failed).\n\n{formatted_docs}"

        elapsed = round(time.perf_counter() - t0, 3)

        # 4. Persist in memory
        memory.add_message("assistant", summary, agent_name=self.NAME)
        memory.update_context(retrieved_docs=kb_docs)
        memory.store_in_long_term(
            [summary],
            [{"type": "research_summary", "query": instruction[:200]}],
        )

        self.logger.info(
            "[ResearchAgent] task=%s | summary=%d chars | kb_docs=%d | %.3fs",
            tid, len(summary), len(kb_docs), elapsed,
        )

        return self._success(
            task_id=tid,
            input_text=instruction,
            output_text=summary,
            metadata={
                "kb_docs_retrieved": len(kb_docs),
                "web_search_used": use_web and self._web_search is not None,
                "web_results": web_results_raw,
                "duration_s": elapsed,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _retrieve_from_kb(self, query: str, k: int | None = None) -> List[Document]:
        k = k or settings.LONG_TERM_MEMORY_TOP_K
        try:
            docs = self._vector_store.similarity_search(query, k=k)
            self.logger.debug("[ResearchAgent] KB returned %d docs.", len(docs))
            return docs
        except Exception as exc:           # noqa: BLE001
            self.logger.warning("[ResearchAgent] KB retrieval failed: %s", exc)
            return []

    def _format_docs(self, docs: List[Document]) -> str:
        if not docs:
            return "No relevant documents found in knowledge base."
        parts = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            parts.append(f"[Doc {i} | source: {src}]\n{doc.page_content}")
        return "\n\n".join(parts)
