"""Base protocol for file parsers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import BinaryIO, Protocol, runtime_checkable

from fastapi import UploadFile


@dataclass
class ParseResult:
    """Result of parsing a file."""

    content: str
    file_type: str
    metadata: dict = field(default_factory=dict)
    # Optional fields
    page_count: int | None = None
    word_count: int | None = None
    char_count: int | None = None


@runtime_checkable
class FileParser(Protocol):
    """Protocol for file parsers."""

    @property
    def supported_extensions(self) -> list[str]:
        """List of supported file extensions (e.g., ['.pdf', '.PDF'])."""
        ...

    @property
    def supported_mime_types(self) -> list[str]:
        """List of supported MIME types."""
        ...

    async def parse(self, file: UploadFile) -> ParseResult:
        """Parse a file and extract text content."""
        ...

    async def validate(self, file: UploadFile) -> bool:
        """Validate if file can be parsed."""
        ...


class BaseFileParser(ABC):
    """Abstract base class for file parsers."""

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """List of supported file extensions."""
        pass

    @property
    @abstractmethod
    def supported_mime_types(self) -> list[str]:
        """List of supported MIME types."""
        pass

    @abstractmethod
    async def parse(self, file: UploadFile) -> ParseResult:
        """Parse a file and extract text content."""
        pass

    async def validate(self, file: UploadFile) -> bool:
        """Validate if file can be parsed."""
        if not file.filename:
            return False

        # Check extension
        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in [e.lower() for e in self.supported_extensions]:
            return False

        # Check content type if available
        if file.content_type:
            # Some content types include charset
            content_type = file.content_type.split(";")[0].strip()
            if content_type not in self.supported_mime_types:
                # Allow generic binary type
                if content_type != "application/octet-stream":
                    return False

        return True

    async def _read_file_content(self, file: UploadFile) -> bytes:
        """Read file content as bytes."""
        content = await file.read()
        # Reset file position for potential re-reads
        await file.seek(0)
        return content

    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        import re

        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        return "\n".join(line for line in lines if line)
