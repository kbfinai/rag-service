"""Plain text file parser with encoding detection."""

from fastapi import UploadFile

from rag_service.services.parsers.base import BaseFileParser, ParseResult


class TextParser(BaseFileParser):
    """Parser for plain text files with automatic encoding detection."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt", ".TXT", ".text", ".log", ".LOG"]

    @property
    def supported_mime_types(self) -> list[str]:
        return [
            "text/plain",
            "text/x-log",
        ]

    async def parse(self, file: UploadFile) -> ParseResult:
        """Parse text file with encoding detection."""
        content = await self._read_file_content(file)

        # Detect encoding
        encoding = self._detect_encoding(content)

        try:
            text = content.decode(encoding)
        except UnicodeDecodeError:
            # Fallback encodings
            for enc in ["utf-8", "latin-1", "cp1252", "ascii"]:
                try:
                    text = content.decode(enc)
                    encoding = enc
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # Last resort: decode with errors replaced
                text = content.decode("utf-8", errors="replace")
                encoding = "utf-8 (with replacements)"

        clean_text = self._clean_text(text)

        return ParseResult(
            content=clean_text,
            file_type="txt",
            word_count=self._count_words(clean_text),
            char_count=len(clean_text),
            metadata={
                "filename": file.filename,
                "file_size": len(content),
                "encoding": encoding,
                "line_count": clean_text.count("\n") + 1,
            },
        )

    def _detect_encoding(self, content: bytes) -> str:
        """Detect text encoding using chardet."""
        try:
            import chardet

            result = chardet.detect(content)
            if result and result.get("encoding"):
                return result["encoding"]
        except ImportError:
            pass

        # Check for BOM markers
        if content.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        elif content.startswith(b"\xff\xfe"):
            return "utf-16-le"
        elif content.startswith(b"\xfe\xff"):
            return "utf-16-be"

        # Default to UTF-8
        return "utf-8"
