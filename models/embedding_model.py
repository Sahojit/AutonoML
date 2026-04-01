"""
Embedding model loader using HuggingFace BGE-large or Instructor XL.
Returns a LangChain Embeddings object for use with Chroma.
"""

import logging
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a cached embedding model instance."""
    logger.info(
        "Loading embedding model: %s  device=%s",
        settings.EMBEDDING_MODEL,
        settings.EMBEDDING_DEVICE,
    )
    encode_kwargs = {"normalize_embeddings": True}  # cosine similarity
    model_kwargs = {"device": settings.EMBEDDING_DEVICE}

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    logger.info("Embedding model loaded successfully.")
    return embeddings
