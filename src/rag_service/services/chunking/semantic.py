"""Semantic chunking strategy based on paragraph and section structure."""

import re

from rag_service.services.chunking.base import BaseChunkingStrategy, ChunkResult


class SemanticChunker(BaseChunkingStrategy):
    """
    Semantic chunking that respects paragraph and section boundaries.

    Identifies natural semantic boundaries in text such as paragraphs,
    sections, and topic changes to create more meaningful chunks.
    """

    DEFAULT_MIN_CHUNK_SIZE = 100
    DEFAULT_MAX_CHUNK_SIZE = 1500

    @property
    def name(self) -> str:
        return "semantic"

    def chunk(self, text: str, config: dict | None = None) -> list[ChunkResult]:
        """
        Split text based on semantic boundaries.

        Config options:
            min_chunk_size: Minimum chunk size (default: 100)
            max_chunk_size: Maximum chunk size (default: 1500)
            combine_short_paragraphs: Combine short paragraphs (default: True)
        """
        config = config or {}
        min_size = config.get("min_chunk_size", self.DEFAULT_MIN_CHUNK_SIZE)
        max_size = config.get("max_chunk_size", self.DEFAULT_MAX_CHUNK_SIZE)
        combine_short = config.get("combine_short_paragraphs", True)

        if not text or not text.strip():
            return []

        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)

        # Combine or split paragraphs based on size constraints
        chunks = self._process_paragraphs(paragraphs, min_size, max_size, combine_short)

        # Convert to ChunkResult objects
        results = []
        for i, chunk_text in enumerate(chunks):
            if self._is_valid_chunk(chunk_text, min_length=min_size // 2):
                results.append(
                    ChunkResult(
                        content=chunk_text.strip(),
                        chunk_index=i,
                        metadata={
                            "semantic_type": self._detect_semantic_type(chunk_text),
                        },
                    )
                )

        # Update total_chunks
        total = len(results)
        for result in results:
            result.metadata["total_chunks"] = total

        return results

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split on multiple newlines (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _process_paragraphs(
        self,
        paragraphs: list[str],
        min_size: int,
        max_size: int,
        combine_short: bool,
    ) -> list[str]:
        """Process paragraphs to create appropriately sized chunks."""
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para_size = len(para)

            # If paragraph itself is too large, split it
            if para_size > max_size:
                # First, save any accumulated content
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # Split large paragraph by sentences
                sentences = self._split_into_sentences(para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) <= max_size:
                        current_chunk += (" " if current_chunk else "") + sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sent

            # If paragraph can be added to current chunk
            elif combine_short and len(current_chunk) + para_size + 2 <= max_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

            # If current chunk is big enough, save it and start new
            else:
                if current_chunk and len(current_chunk) >= min_size:
                    chunks.append(current_chunk)
                    current_chunk = para
                elif current_chunk:
                    # Chunk too small, combine with next
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with nltk or spacy)
        sentence_endings = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _detect_semantic_type(self, text: str) -> str:
        """Detect the semantic type of a chunk."""
        text_lower = text.lower()

        # Check for common patterns
        if re.match(r"^(chapter|section|\d+\.)\s", text, re.IGNORECASE):
            return "heading"
        elif text.startswith("•") or text.startswith("-") or text.startswith("*"):
            return "list"
        elif re.search(r"\d+%|\$[\d,]+|[\d,]+\s*(dollars|usd)", text_lower):
            return "financial"
        elif re.search(r"(table|figure)\s*\d+", text_lower):
            return "reference"
        elif len(text) < 100:
            return "short"
        else:
            return "paragraph"
