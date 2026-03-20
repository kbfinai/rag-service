"""Markdown file parser."""

import re

from fastapi import UploadFile

from rag_service.services.parsers.base import BaseFileParser, ParseResult


class MarkdownParser(BaseFileParser):
    """Parser for Markdown files."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".MD", ".markdown", ".MARKDOWN"]

    @property
    def supported_mime_types(self) -> list[str]:
        return [
            "text/markdown",
            "text/x-markdown",
        ]

    async def parse(self, file: UploadFile) -> ParseResult:
        """Parse Markdown file."""
        content = await self._read_file_content(file)

        # Markdown is typically UTF-8
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("utf-8", errors="replace")

        # Extract structure information
        structure = self._analyze_structure(text)

        # Keep markdown as-is for chunking (markdown chunker handles it)
        clean_text = text.strip()

        return ParseResult(
            content=clean_text,
            file_type="markdown",
            word_count=self._count_words(self._strip_markdown(clean_text)),
            char_count=len(clean_text),
            metadata={
                "filename": file.filename,
                "file_size": len(content),
                "structure": structure,
            },
        )

    def _analyze_structure(self, text: str) -> dict:
        """Analyze markdown structure."""
        structure = {
            "headers": [],
            "code_blocks": 0,
            "links": 0,
            "images": 0,
            "lists": 0,
            "tables": 0,
        }

        # Count headers
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        for match in header_pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()
            structure["headers"].append({"level": level, "title": title})

        # Count code blocks
        structure["code_blocks"] = len(re.findall(r"```[\s\S]*?```", text))

        # Count links
        structure["links"] = len(re.findall(r"\[([^\]]+)\]\([^)]+\)", text))

        # Count images
        structure["images"] = len(re.findall(r"!\[([^\]]*)\]\([^)]+\)", text))

        # Count list items
        structure["lists"] = len(re.findall(r"^[\s]*[-*+]\s", text, re.MULTILINE))
        structure["lists"] += len(re.findall(r"^[\s]*\d+\.\s", text, re.MULTILINE))

        # Check for tables
        structure["tables"] = len(re.findall(r"^\|.+\|$", text, re.MULTILINE))

        return structure

    def _strip_markdown(self, text: str) -> str:
        """Strip markdown formatting for word counting."""
        # Remove code blocks
        text = re.sub(r"```[\s\S]*?```", "", text)
        # Remove inline code
        text = re.sub(r"`[^`]+`", "", text)
        # Remove links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        # Remove images
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", "", text)
        # Remove headers markers
        text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
        # Remove emphasis markers
        text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
        return text
