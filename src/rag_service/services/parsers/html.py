"""HTML file parser."""

import re

from fastapi import UploadFile

from rag_service.services.parsers.base import BaseFileParser, ParseResult


class HTMLParser(BaseFileParser):
    """Parser for HTML files."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".HTML", ".htm", ".HTM"]

    @property
    def supported_mime_types(self) -> list[str]:
        return [
            "text/html",
            "application/xhtml+xml",
        ]

    async def parse(self, file: UploadFile) -> ParseResult:
        """Parse HTML file and extract text content."""
        from bs4 import BeautifulSoup

        content = await self._read_file_content(file)

        # Detect encoding from content or default to UTF-8
        encoding = self._detect_encoding(content)
        html_text = content.decode(encoding, errors="replace")

        try:
            soup = BeautifulSoup(html_text, "lxml")
        except Exception:
            # Fallback to html.parser if lxml fails
            soup = BeautifulSoup(html_text, "html.parser")

        # Extract metadata
        title = soup.title.string if soup.title else None
        meta_description = None
        meta_keywords = None

        for meta in soup.find_all("meta"):
            name = meta.get("name", "").lower()
            if name == "description":
                meta_description = meta.get("content")
            elif name == "keywords":
                meta_keywords = meta.get("content")

        # Remove script and style elements
        for element in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            element.decompose()

        # Get text content
        text = soup.get_text(separator="\n")

        # Clean up the text
        clean_text = self._clean_html_text(text)

        return ParseResult(
            content=clean_text,
            file_type="html",
            word_count=self._count_words(clean_text),
            char_count=len(clean_text),
            metadata={
                "filename": file.filename,
                "file_size": len(content),
                "title": title,
                "meta_description": meta_description,
                "meta_keywords": meta_keywords,
                "encoding": encoding,
            },
        )

    def _detect_encoding(self, content: bytes) -> str:
        """Detect HTML encoding from meta tags or use chardet."""
        # Check for meta charset
        content_start = content[:1024].decode("ascii", errors="ignore")

        # HTML5 charset
        match = re.search(r'charset=["\']?([^"\'\s>]+)', content_start, re.IGNORECASE)
        if match:
            return match.group(1)

        # Try chardet
        try:
            import chardet

            result = chardet.detect(content)
            if result and result.get("encoding"):
                return result["encoding"]
        except ImportError:
            pass

        return "utf-8"

    def _clean_html_text(self, text: str) -> str:
        """Clean extracted HTML text."""
        # Remove excessive whitespace
        lines = []
        for line in text.split("\n"):
            line = line.strip()
            if line:
                # Collapse multiple spaces
                line = re.sub(r"\s+", " ", line)
                lines.append(line)

        # Remove duplicate empty lines
        result = []
        prev_empty = False
        for line in lines:
            if not line:
                if not prev_empty:
                    result.append(line)
                    prev_empty = True
            else:
                result.append(line)
                prev_empty = False

        return "\n".join(result)
