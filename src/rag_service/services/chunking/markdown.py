"""Markdown-aware chunking strategy."""

import re

from rag_service.services.chunking.base import BaseChunkingStrategy, ChunkResult


class MarkdownChunker(BaseChunkingStrategy):
    """
    Markdown-aware chunking that respects document structure.

    Preserves markdown structure including headers, code blocks,
    lists, and other markdown elements.
    """

    DEFAULT_MAX_CHUNK_SIZE = 1500
    DEFAULT_MIN_CHUNK_SIZE = 100

    @property
    def name(self) -> str:
        return "markdown"

    def chunk(self, text: str, config: dict | None = None) -> list[ChunkResult]:
        """
        Split markdown text respecting structure.

        Config options:
            max_chunk_size: Maximum chunk size (default: 1500)
            min_chunk_size: Minimum chunk size (default: 100)
            respect_headers: Keep sections under headers together (default: True)
            include_header_in_chunks: Include section header in each chunk (default: True)
        """
        config = config or {}
        max_size = config.get("max_chunk_size", self.DEFAULT_MAX_CHUNK_SIZE)
        min_size = config.get("min_chunk_size", self.DEFAULT_MIN_CHUNK_SIZE)
        respect_headers = config.get("respect_headers", True)
        include_header = config.get("include_header_in_chunks", True)

        if not text or not text.strip():
            return []

        if respect_headers:
            # Split by headers
            sections = self._split_by_headers(text)
            chunks = self._process_sections(sections, max_size, min_size, include_header)
        else:
            # Simple paragraph-based splitting
            chunks = self._split_simple(text, max_size, min_size)

        # Convert to ChunkResult objects
        results = []
        for i, (chunk_text, metadata) in enumerate(chunks):
            if self._is_valid_chunk(chunk_text, min_length=min_size // 2):
                results.append(
                    ChunkResult(
                        content=chunk_text.strip(),
                        chunk_index=i,
                        metadata=metadata,
                    )
                )

        # Update total_chunks
        total = len(results)
        for result in results:
            result.metadata["total_chunks"] = total

        return results

    def _split_by_headers(self, text: str) -> list[dict]:
        """Split text into sections by markdown headers."""
        # Match markdown headers (# to ######)
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        sections = []
        last_end = 0
        current_headers = {}  # Track header hierarchy

        for match in header_pattern.finditer(text):
            # Save content before this header
            if last_end < match.start():
                content = text[last_end : match.start()].strip()
                if content:
                    sections.append(
                        {
                            "content": content,
                            "headers": dict(current_headers),
                            "level": 0,
                        }
                    )

            # Parse header
            level = len(match.group(1))
            header_text = match.group(2).strip()

            # Update header hierarchy
            current_headers[level] = header_text
            # Clear lower-level headers
            for l in list(current_headers.keys()):
                if l > level:
                    del current_headers[l]

            last_end = match.end()

        # Don't forget content after last header
        if last_end < len(text):
            content = text[last_end:].strip()
            if content:
                sections.append(
                    {
                        "content": content,
                        "headers": dict(current_headers),
                        "level": 0,
                    }
                )

        return sections

    def _process_sections(
        self,
        sections: list[dict],
        max_size: int,
        min_size: int,
        include_header: bool,
    ) -> list[tuple[str, dict]]:
        """Process sections into appropriately sized chunks."""
        chunks = []

        for section in sections:
            content = section["content"]
            headers = section["headers"]

            # Build header prefix if needed
            header_prefix = ""
            if include_header and headers:
                # Use the deepest (most specific) header
                max_level = max(headers.keys()) if headers else 0
                if max_level > 0:
                    header_prefix = f"{'#' * max_level} {headers[max_level]}\n\n"

            # If section fits, add as single chunk
            if len(header_prefix) + len(content) <= max_size:
                chunks.append(
                    (
                        header_prefix + content,
                        {
                            "section_headers": headers,
                            "md_type": self._detect_md_type(content),
                        },
                    )
                )
            else:
                # Split large section
                sub_chunks = self._split_large_section(content, max_size - len(header_prefix))
                for sub in sub_chunks:
                    chunks.append(
                        (
                            header_prefix + sub,
                            {
                                "section_headers": headers,
                                "md_type": self._detect_md_type(sub),
                            },
                        )
                    )

        return chunks

    def _split_large_section(self, content: str, max_size: int) -> list[str]:
        """Split a large section into smaller chunks."""
        # Try to split by code blocks first
        code_block_pattern = re.compile(r"```[\s\S]*?```", re.MULTILINE)

        chunks = []
        last_end = 0

        for match in code_block_pattern.finditer(content):
            # Content before code block
            if last_end < match.start():
                pre_content = content[last_end : match.start()].strip()
                if pre_content:
                    chunks.extend(self._split_by_paragraphs(pre_content, max_size))

            # Code block itself (keep intact if possible)
            code_block = match.group()
            if len(code_block) <= max_size:
                chunks.append(code_block)
            else:
                # Code block too large, split by lines
                chunks.extend(self._split_code_block(code_block, max_size))

            last_end = match.end()

        # Remaining content
        if last_end < len(content):
            remaining = content[last_end:].strip()
            if remaining:
                chunks.extend(self._split_by_paragraphs(remaining, max_size))

        return chunks

    def _split_by_paragraphs(self, text: str, max_size: int) -> list[str]:
        """Split text by paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 2 <= max_size:
                if current:
                    current += "\n\n" + para
                else:
                    current = para
            else:
                if current:
                    chunks.append(current)
                current = para

        if current:
            chunks.append(current)

        return chunks

    def _split_code_block(self, code_block: str, max_size: int) -> list[str]:
        """Split a large code block by lines."""
        lines = code_block.split("\n")
        chunks = []
        current = ""

        # Preserve opening fence
        fence_match = re.match(r"```\w*", lines[0])
        fence = fence_match.group() if fence_match else "```"

        for i, line in enumerate(lines):
            if i == 0:
                current = line
            elif len(current) + len(line) + 1 <= max_size:
                current += "\n" + line
            else:
                # Close current chunk and start new
                if not current.endswith("```"):
                    current += "\n```"
                chunks.append(current)
                current = fence + "\n" + line

        # Add closing fence if needed
        if current and not current.endswith("```"):
            current += "\n```"
        if current:
            chunks.append(current)

        return chunks

    def _split_simple(self, text: str, max_size: int, min_size: int) -> list[tuple[str, dict]]:
        """Simple splitting without header awareness."""
        chunks = self._split_by_paragraphs(text, max_size)
        return [(c, {"md_type": self._detect_md_type(c)}) for c in chunks]

    def _detect_md_type(self, text: str) -> str:
        """Detect the type of markdown content."""
        if text.startswith("```"):
            return "code_block"
        elif re.match(r"^[-*+]\s", text) or re.match(r"^\d+\.\s", text):
            return "list"
        elif text.startswith("|") and "|" in text:
            return "table"
        elif text.startswith(">"):
            return "blockquote"
        else:
            return "text"
