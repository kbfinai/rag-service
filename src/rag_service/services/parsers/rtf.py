"""RTF (Rich Text Format) file parser."""

from fastapi import UploadFile

from rag_service.services.parsers.base import BaseFileParser, ParseResult


class RTFParser(BaseFileParser):
    """Parser for RTF files."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".rtf", ".RTF"]

    @property
    def supported_mime_types(self) -> list[str]:
        return [
            "application/rtf",
            "text/rtf",
        ]

    async def parse(self, file: UploadFile) -> ParseResult:
        """Parse RTF file and extract text content."""
        from striprtf.striprtf import rtf_to_text

        content = await self._read_file_content(file)

        # RTF is typically ASCII-based with escapes for other characters
        try:
            rtf_text = content.decode("utf-8")
        except UnicodeDecodeError:
            rtf_text = content.decode("latin-1")

        try:
            # Extract text using striprtf
            plain_text = rtf_to_text(rtf_text)
        except Exception as e:
            raise ValueError(f"Failed to parse RTF: {e}") from e

        clean_text = self._clean_text(plain_text)

        return ParseResult(
            content=clean_text,
            file_type="rtf",
            word_count=self._count_words(clean_text),
            char_count=len(clean_text),
            metadata={
                "filename": file.filename,
                "file_size": len(content),
            },
        )
