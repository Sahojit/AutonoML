"""
Chroma-backed vector store.
Supports adding documents, similarity search, and metadata filtering.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma

from config.settings import settings
from models.embedding_model import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Thin wrapper around LangChain's Chroma integration.

    Provides:
    - add_documents  : embed & persist text chunks
    - similarity_search : retrieve top-k relevant docs
    - delete_by_ids  : remove documents
    - get_all        : list all stored documents
    """

    def __init__(self) -> None:
        self._embeddings = get_embedding_model()
        self._client = self._build_client()
        self._store = self._init_store()
        logger.info(
            "VectorStore ready — collection=%s  persist_dir=%s",
            settings.CHROMA_COLLECTION_NAME,
            settings.CHROMA_PERSIST_DIR,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _build_client(self) -> chromadb.Client:
        if settings.CHROMA_HOST:
            logger.info("Connecting to remote Chroma at %s:%s", settings.CHROMA_HOST, settings.CHROMA_PORT)
            return chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
            )
        return chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _init_store(self) -> Chroma:
        return Chroma(
            client=self._client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self._embeddings,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        """Embed texts and persist them to Chroma. Returns list of doc IDs."""
        if not texts:
            return []
        ids = [str(uuid.uuid4()) for _ in texts]
        metadatas = metadatas or [{} for _ in texts]
        docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(texts, metadatas)
        ]
        self._store.add_documents(docs, ids=ids)
        logger.info("Added %d document(s) to vector store.", len(docs))
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Return top-k documents most similar to *query*."""
        k = k or settings.LONG_TERM_MEMORY_TOP_K
        results = self._store.similarity_search(query, k=k, filter=filter)
        logger.debug("similarity_search returned %d docs for query='%s'", len(results), query[:80])
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = None,
    ) -> List[tuple[Document, float]]:
        """Return (doc, score) pairs — lower score = more similar for L2."""
        k = k or settings.LONG_TERM_MEMORY_TOP_K
        return self._store.similarity_search_with_score(query, k=k)

    def delete_by_ids(self, ids: List[str]) -> None:
        """Remove documents by their Chroma IDs."""
        self._store.delete(ids=ids)
        logger.info("Deleted %d document(s) from vector store.", len(ids))

    def get_collection_count(self) -> int:
        """Return total number of embeddings stored."""
        collection = self._client.get_collection(settings.CHROMA_COLLECTION_NAME)
        return collection.count()

    def reset_collection(self) -> None:
        """Delete and recreate the collection — useful for testing."""
        logger.warning("Resetting vector store collection: %s", settings.CHROMA_COLLECTION_NAME)
        self._client.delete_collection(settings.CHROMA_COLLECTION_NAME)
        self._store = self._init_store()


# Module-level singleton
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
