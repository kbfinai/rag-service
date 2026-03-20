"""Fixed-size chunking strategy with overlap."""

from rag_service.services.chunking.base import BaseChunkingStrategy, ChunkResult


class FixedSizeChunker(BaseChunkingStrategy):
    """
    Fixed-size chunking with overlap.

    Splits text into chunks of a fixed size with optional overlap.
    Tries to break at sentence or word boundaries when possible.
    """

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_OVERLAP = 50

    @property
    def name(self) -> str:
        return "fixed"

    def chunk(self, text: str, config: dict | None = None) -> list[ChunkResult]:
        """
        Split text into fixed-size chunks.

        Config options:
            chunk_size: Maximum chunk size in characters (default: 512)
            overlap: Number of overlapping characters (default: 50)
            respect_sentences: Try to break at sentence boundaries (default: True)
        """
        config = config or {}
        chunk_size = config.get("chunk_size", self.DEFAULT_CHUNK_SIZE)
        overlap = config.get("overlap", self.DEFAULT_OVERLAP)
        respect_sentences = config.get("respect_sentences", True)

        if not text or not text.strip():
            return []

        text = text.strip()

        # If text is smaller than chunk size, return as single chunk
        if len(text) <= chunk_size:
            return [
                ChunkResult(
                    content=text,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(text),
                    metadata={"total_chunks": 1},
                )
            ]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + chunk_size, len(text))

            # If not at the end, try to find a good break point
            if end < len(text) and respect_sentences:
                end = self._find_break_point(text, start, end)

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    ChunkResult(
                        content=chunk_text,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                        metadata={},
                    )
                )
                chunk_index += 1

            # Move start position with overlap
            if end >= len(text):
                break

            start = end - overlap
            if start < 0:
                start = 0
            # Ensure we make progress
            if start >= end:
                start = end

        # Add total_chunks to metadata
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_chunks"] = total

        return chunks

    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find a good break point near the end position."""
        # Sentence boundaries to look for
        sentence_endings = [". ", ".\n", "! ", "!\n", "? ", "?\n"]

        # Look backwards from end for sentence boundary
        search_start = max(start, end - 100)  # Search in last 100 chars
        best_break = end

        for ending in sentence_endings:
            pos = text.rfind(ending, search_start, end)
            if pos != -1:
                # Include the period but not the space
                candidate = pos + 1
                if candidate > start and candidate < best_break:
                    best_break = candidate
                    break

        # If no sentence boundary found, try word boundary
        if best_break == end:
            space_pos = text.rfind(" ", search_start, end)
            if space_pos != -1 and space_pos > start:
                best_break = space_pos

        return best_break
