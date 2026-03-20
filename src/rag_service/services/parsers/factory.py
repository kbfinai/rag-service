"""Factory for creating file parsers."""

from pathlib import Path

from rag_service.services.parsers.base import FileParser
from rag_service.services.parsers.pdf import PDFParser
from rag_service.services.parsers.docx import DocxParser
from rag_service.services.parsers.txt import TextParser
from rag_service.services.parsers.markdown import MarkdownParser
from rag_service.services.parsers.csv import CSVParser
from rag_service.services.parsers.excel import ExcelParser
from rag_service.services.parsers.html import HTMLParser
from rag_service.services.parsers.rtf import RTFParser


# Extension to parser mapping
EXTENSION_PARSERS = {
    ".pdf": PDFParser,
    ".docx": DocxParser,
    ".txt": TextParser,
    ".text": TextParser,
    ".log": TextParser,
    ".md": MarkdownParser,
    ".markdown": MarkdownParser,
    ".csv": CSVParser,
    ".tsv": CSVParser,
    ".xlsx": ExcelParser,
    ".xls": ExcelParser,
    ".html": HTMLParser,
    ".htm": HTMLParser,
    ".rtf": RTFParser,
}

# MIME type to parser mapping
MIME_PARSERS = {
    "application/pdf": PDFParser,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxParser,
    "text/plain": TextParser,
    "text/markdown": MarkdownParser,
    "text/x-markdown": MarkdownParser,
    "text/csv": CSVParser,
    "text/tab-separated-values": CSVParser,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ExcelParser,
    "application/vnd.ms-excel": ExcelParser,
    "text/html": HTMLParser,
    "application/xhtml+xml": HTMLParser,
    "application/rtf": RTFParser,
    "text/rtf": RTFParser,
}


class FileParserFactory:
    """Factory for creating file parsers based on file type."""

    @classmethod
    def get_parser(cls, filename: str) -> FileParser:
        """
        Get appropriate parser for a file based on its extension.

        Args:
            filename: Name of the file

        Returns:
            FileParser instance

        Raises:
            ValueError: If file type is not supported
        """
        ext = Path(filename).suffix.lower()

        if ext not in EXTENSION_PARSERS:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported types: {cls.list_supported_extensions()}"
            )

        parser_class = EXTENSION_PARSERS[ext]
        return parser_class()

    @classmethod
    def get_parser_by_mime(cls, mime_type: str) -> FileParser | None:
        """
        Get appropriate parser for a file based on its MIME type.

        Args:
            mime_type: MIME type of the file

        Returns:
            FileParser instance or None if not found
        """
        # Strip charset info
        mime_type = mime_type.split(";")[0].strip()

        parser_class = MIME_PARSERS.get(mime_type)
        if parser_class:
            return parser_class()
        return None

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        """Check if a file type is supported."""
        ext = Path(filename).suffix.lower()
        return ext in EXTENSION_PARSERS

    @classmethod
    def list_supported_extensions(cls) -> list[str]:
        """List all supported file extensions."""
        return sorted(set(EXTENSION_PARSERS.keys()))

    @classmethod
    def list_supported_mime_types(cls) -> list[str]:
        """List all supported MIME types."""
        return sorted(set(MIME_PARSERS.keys()))

    @classmethod
    def get_parser_info(cls) -> list[dict]:
        """Get information about all supported parsers."""
        parsers_info = {}

        for ext, parser_class in EXTENSION_PARSERS.items():
            name = parser_class.__name__
            if name not in parsers_info:
                instance = parser_class()
                parsers_info[name] = {
                    "name": name,
                    "extensions": instance.supported_extensions,
                    "mime_types": instance.supported_mime_types,
                    "file_type": instance.supported_extensions[0].lstrip("."),
                }

        return list(parsers_info.values())

    @classmethod
    def register_parser(
        cls,
        extensions: list[str],
        mime_types: list[str],
        parser_class: type,
    ) -> None:
        """
        Register a custom file parser.

        Args:
            extensions: List of file extensions (e.g., ['.xyz'])
            mime_types: List of MIME types
            parser_class: Parser class implementing FileParser protocol
        """
        for ext in extensions:
            EXTENSION_PARSERS[ext.lower()] = parser_class
        for mime in mime_types:
            MIME_PARSERS[mime] = parser_class
