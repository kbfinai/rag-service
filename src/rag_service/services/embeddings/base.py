"""Base protocol for embedding providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embedding: list[float]
    model: str
    dimension: int
    token_count: int | None = None


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    @property
    def name(self) -> str:
        """Provider name (e.g., 'openai', 'huggingface')."""
        ...

    @property
    def model_name(self) -> str:
        """Model name (e.g., 'text-embedding-3-small')."""
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        ...

    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        ...


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, model_name: str, config: dict | None = None):
        self._model_name = model_name
        self._config = config or {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @property
    def model_name(self) -> str:
        """Model name."""
        return self._model_name

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension for the current model."""
        pass

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text."""
        pass

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts.

        Default implementation calls embed() for each text.
        Override for batch optimization.
        """
        results = []
        for text in texts:
            result = await self.embed(text)
            results.append(result)
        return results

    def _normalize_text(self, text: str) -> str:
        """Normalize text before embedding."""
        # Replace newlines with spaces
        text = text.replace("\n", " ")
        # Remove excessive whitespace
        text = " ".join(text.split())
        return text.strip()
