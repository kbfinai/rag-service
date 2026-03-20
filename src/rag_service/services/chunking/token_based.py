"""Token-based chunking strategy for LLM context limits."""

import re

from rag_service.services.chunking.base import BaseChunkingStrategy, ChunkResult


class TokenBasedChunker(BaseChunkingStrategy):
    """
    Token-based chunking for LLM context limits.

    Uses token counting to ensure chunks fit within LLM token limits.
    Approximates token count using word-based heuristics or actual tokenizers.
    """

    DEFAULT_MAX_TOKENS = 512
    DEFAULT_OVERLAP_TOKENS = 50
    # Average characters per token (for OpenAI tokenizers)
    CHARS_PER_TOKEN = 4

    @property
    def name(self) -> str:
        return "token"

    def chunk(self, text: str, config: dict | None = None) -> list[ChunkResult]:
        """
        Split text based on token count.

        Config options:
            max_tokens: Maximum tokens per chunk (default: 512)
            overlap_tokens: Overlap in tokens (default: 50)
            tokenizer: Tokenizer to use ('simple', 'tiktoken') (default: 'simple')
        """
        config = config or {}
        max_tokens = config.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        overlap_tokens = config.get("overlap_tokens", self.DEFAULT_OVERLAP_TOKENS)
        tokenizer_type = config.get("tokenizer", "simple")

        if not text or not text.strip():
            return []

        # Get tokenizer
        tokenizer = self._get_tokenizer(tokenizer_type)

        # Tokenize the text
        tokens = tokenizer(text)

        # If text is small enough, return as single chunk
        if len(tokens) <= max_tokens:
            return [
                ChunkResult(
                    content=text.strip(),
                    chunk_index=0,
                    metadata={
                        "total_chunks": 1,
                        "token_count": len(tokens),
                    },
                )
            ]

        # Split into chunks
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))

            # Try to find a good break point (sentence or paragraph boundary)
            if end < len(tokens):
                end = self._find_token_break(tokens, start, end)

            chunk_tokens = tokens[start:end]
            chunk_text = self._tokens_to_text(chunk_tokens)

            if chunk_text.strip():
                chunks.append(
                    ChunkResult(
                        content=chunk_text.strip(),
                        chunk_index=chunk_index,
                        metadata={
                            "token_count": len(chunk_tokens),
                        },
                    )
                )
                chunk_index += 1

            # Move with overlap
            start = end - overlap_tokens
            if start < 0:
                start = 0
            if start >= end:
                start = end

        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_chunks"] = total

        return chunks

    def _get_tokenizer(self, tokenizer_type: str):
        """Get the appropriate tokenizer function."""
        if tokenizer_type == "tiktoken":
            return self._tiktoken_tokenize
        else:
            return self._simple_tokenize

    def _simple_tokenize(self, text: str) -> list[str]:
        """Simple word-based tokenization."""
        # Split on whitespace and punctuation while keeping punctuation
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return tokens

    def _tiktoken_tokenize(self, text: str) -> list[str]:
        """Tokenize using tiktoken (OpenAI's tokenizer)."""
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            token_ids = enc.encode(text)
            # Return token strings for easier handling
            return [enc.decode([t]) for t in token_ids]
        except ImportError:
            # Fallback to simple tokenization
            return self._simple_tokenize(text)

    def _tokens_to_text(self, tokens: list[str]) -> str:
        """Convert tokens back to text."""
        # Simple reconstruction - join with spaces, then clean up
        text = " ".join(tokens)
        # Remove spaces before punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        return text

    def _find_token_break(
        self,
        tokens: list[str],
        start: int,
        end: int,
    ) -> int:
        """Find a good break point in the token sequence."""
        # Look for sentence endings
        sentence_endings = {".", "!", "?"}

        # Search backwards from end
        for i in range(end - 1, max(start, end - 50), -1):
            if tokens[i] in sentence_endings:
                return i + 1

        # No good break point, return original end
        return end

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count for a text."""
        # Simple estimation: characters / 4
        return len(text) // self.CHARS_PER_TOKEN
