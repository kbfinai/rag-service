"""Recursive chunking strategy that respects document structure."""

from rag_service.services.chunking.base import BaseChunkingStrategy, ChunkResult


class RecursiveChunker(BaseChunkingStrategy):
    """
    Recursive chunking that respects document structure.

    Splits text hierarchically using different separators,
    starting with larger structures (paragraphs) and falling back
    to smaller ones (sentences, words) as needed.
    """

    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_OVERLAP = 100
    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " "]

    @property
    def name(self) -> str:
        return "recursive"

    def chunk(self, text: str, config: dict | None = None) -> list[ChunkResult]:
        """
        Split text recursively respecting structure.

        Config options:
            chunk_size: Target chunk size (default: 1000)
            overlap: Overlap between chunks (default: 100)
            separators: List of separators in priority order
        """
        config = config or {}
        chunk_size = config.get("chunk_size", self.DEFAULT_CHUNK_SIZE)
        overlap = config.get("overlap", self.DEFAULT_OVERLAP)
        separators = config.get("separators", self.DEFAULT_SEPARATORS)

        if not text or not text.strip():
            return []

        text = text.strip()

        # Recursively split
        raw_chunks = self._split_recursive(text, separators, chunk_size)

        # Add overlap between chunks
        chunks_with_overlap = self._add_overlap(raw_chunks, overlap)

        # Convert to ChunkResult objects
        results = []
        for i, chunk_text in enumerate(chunks_with_overlap):
            if self._is_valid_chunk(chunk_text):
                results.append(
                    ChunkResult(
                        content=chunk_text.strip(),
                        chunk_index=i,
                        metadata={"total_chunks": len(chunks_with_overlap)},
                    )
                )

        # Update total_chunks in metadata
        total = len(results)
        for result in results:
            result.metadata["total_chunks"] = total

        return results

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
        chunk_size: int,
    ) -> list[str]:
        """Recursively split text using separators."""
        if not separators:
            # No more separators, split by character count
            return self._split_by_size(text, chunk_size)

        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by current separator
        splits = text.split(separator)

        chunks = []
        current_chunk = ""

        for split in splits:
            # Add separator back (except for space)
            piece = split + (separator if separator != " " else " ")

            # If adding this piece exceeds chunk size
            if len(current_chunk) + len(piece) > chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # If single piece is too large, recursively split it
                if len(piece) > chunk_size:
                    sub_chunks = self._split_recursive(
                        piece.strip(), remaining_separators, chunk_size
                    )
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = piece
            else:
                current_chunk += piece

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_size(self, text: str, chunk_size: int) -> list[str]:
        """Split text by size when no separators work."""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i : i + chunk_size])
        return chunks

    def _add_overlap(self, chunks: list[str], overlap: int) -> list[str]:
        """Add overlap between consecutive chunks."""
        if not chunks or overlap <= 0:
            return chunks

        result = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                result.append(chunk)
            else:
                # Add end of previous chunk as prefix
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-overlap:] if len(prev_chunk) >= overlap else prev_chunk
                result.append(overlap_text + " " + chunk)

        return result
