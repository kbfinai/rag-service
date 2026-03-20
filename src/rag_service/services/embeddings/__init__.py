"""Embedding providers for generating vector embeddings."""

from rag_service.services.embeddings.base import EmbeddingProvider, EmbeddingResult
from rag_service.services.embeddings.factory import EmbeddingProviderFactory
from rag_service.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from rag_service.services.embeddings.huggingface_provider import HuggingFaceEmbeddingProvider
from rag_service.services.embeddings.ollama_provider import OllamaEmbeddingProvider
from rag_service.services.embeddings.legacy import EmbeddingService

__all__ = [
    "EmbeddingProvider",
    "EmbeddingResult",
    "EmbeddingProviderFactory",
    "OpenAIEmbeddingProvider",
    "HuggingFaceEmbeddingProvider",
    "OllamaEmbeddingProvider",
    "EmbeddingService",
]
