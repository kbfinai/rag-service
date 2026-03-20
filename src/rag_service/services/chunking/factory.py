"""Factory for creating chunking strategies."""

from rag_service.services.chunking.base import ChunkingStrategy
from rag_service.services.chunking.fixed import FixedSizeChunker
from rag_service.services.chunking.recursive import RecursiveChunker
from rag_service.services.chunking.semantic import SemanticChunker
from rag_service.services.chunking.token_based import TokenBasedChunker
from rag_service.services.chunking.markdown import MarkdownChunker
from rag_service.services.chunking.invoice import InvoiceChunker


# Default configurations for each strategy
DEFAULT_CONFIGS = {
    "fixed": {
        "chunk_size": 512,
        "overlap": 50,
        "respect_sentences": True,
    },
    "recursive": {
        "chunk_size": 1000,
        "overlap": 100,
        "separators": ["\n\n", "\n", ". ", " "],
    },
    "semantic": {
        "min_chunk_size": 100,
        "max_chunk_size": 1500,
        "combine_short_paragraphs": True,
    },
    "token": {
        "max_tokens": 512,
        "overlap_tokens": 50,
        "tokenizer": "simple",
    },
    "markdown": {
        "max_chunk_size": 1500,
        "min_chunk_size": 100,
        "respect_headers": True,
        "include_header_in_chunks": True,
    },
    "invoice": {
        "extract_line_items": True,
        "extract_totals": True,
        "include_raw_text": False,
    },
}


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies."""

    _strategies = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "token": TokenBasedChunker,
        "markdown": MarkdownChunker,
        "invoice": InvoiceChunker,
    }

    @classmethod
    def create(cls, strategy_name: str) -> ChunkingStrategy:
        """
        Create a chunking strategy instance.

        Args:
            strategy_name: Strategy name (fixed, recursive, semantic, token, markdown, invoice)

        Returns:
            ChunkingStrategy instance

        Raises:
            ValueError: If strategy name is not supported
        """
        if strategy_name not in cls._strategies:
            raise ValueError(
                f"Unknown chunking strategy: {strategy_name}. "
                f"Supported strategies: {list(cls._strategies.keys())}"
            )

        strategy_class = cls._strategies[strategy_name]
        return strategy_class()

    @classmethod
    def get_default_config(cls, strategy_name: str) -> dict:
        """
        Get the default configuration for a strategy.

        Args:
            strategy_name: Strategy name

        Returns:
            Default configuration dict
        """
        return DEFAULT_CONFIGS.get(strategy_name, {}).copy()

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List available strategy names."""
        return list(cls._strategies.keys())

    @classmethod
    def get_strategy_info(cls) -> list[dict]:
        """Get information about all available strategies."""
        return [
            {
                "name": "fixed",
                "description": "Fixed-size chunks with overlap",
                "best_for": "General text, documents without clear structure",
                "default_config": DEFAULT_CONFIGS["fixed"],
            },
            {
                "name": "recursive",
                "description": "Recursive splitting respecting document structure",
                "best_for": "Documents with hierarchical structure, long-form content",
                "default_config": DEFAULT_CONFIGS["recursive"],
            },
            {
                "name": "semantic",
                "description": "Paragraph and section-aware chunking",
                "best_for": "Articles, reports, documents with clear paragraphs",
                "default_config": DEFAULT_CONFIGS["semantic"],
            },
            {
                "name": "token",
                "description": "Token-based chunking for LLM context limits",
                "best_for": "LLM applications with strict token limits",
                "default_config": DEFAULT_CONFIGS["token"],
            },
            {
                "name": "markdown",
                "description": "Markdown structure-aware chunking",
                "best_for": "Technical documentation, README files, markdown content",
                "default_config": DEFAULT_CONFIGS["markdown"],
            },
            {
                "name": "invoice",
                "description": "Invoice and billing document chunking",
                "best_for": "Invoices, bills, financial documents with line items",
                "default_config": DEFAULT_CONFIGS["invoice"],
            },
        ]

    @classmethod
    def register_strategy(cls, name: str, strategy_class: type, default_config: dict | None = None) -> None:
        """
        Register a custom chunking strategy.

        Args:
            name: Strategy name
            strategy_class: Strategy class implementing ChunkingStrategy protocol
            default_config: Optional default configuration
        """
        cls._strategies[name] = strategy_class
        if default_config:
            DEFAULT_CONFIGS[name] = default_config
