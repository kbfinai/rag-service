"""Base protocol for chunking strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ChunkResult:
    """Result of chunking a document."""

    content: str
    metadata: dict = field(default_factory=dict)
    # Optional fields for specific chunkers
    chunk_index: int = 0
    start_char: int | None = None
    end_char: int | None = None


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Protocol for chunking strategies."""

    @property
    def name(self) -> str:
        """Strategy name."""
        ...

    def chunk(self, text: str, config: dict | None = None) -> list[ChunkResult]:
        """Split text into chunks."""
        ...


class BaseChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @abstractmethod
    def chunk(self, text: str, config: dict | None = None) -> list[ChunkResult]:
        """Split text into chunks."""
        pass

    def _clean_chunk(self, text: str) -> str:
        """Clean a chunk by stripping whitespace."""
        return text.strip()

    def _is_valid_chunk(self, text: str, min_length: int = 10) -> bool:
        """Check if chunk has meaningful content."""
        cleaned = self._clean_chunk(text)
        return len(cleaned) >= min_length
