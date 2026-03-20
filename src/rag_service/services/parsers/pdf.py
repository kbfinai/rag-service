"""PDF file parser."""

import io

from fastapi import UploadFile

from rag_service.services.parsers.base import BaseFileParser, ParseResult


class PDFParser(BaseFileParser):
    """Parser for PDF files using pypdf."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf", ".PDF"]

    @property
    def supported_mime_types(self) -> list[str]:
        return ["application/pdf"]

    async def parse(self, file: UploadFile) -> ParseResult:
        """Parse PDF and extract text."""
        from pypdf import PdfReader

        content = await self._read_file_content(file)
        pdf_file = io.BytesIO(content)

        try:
            reader = PdfReader(pdf_file)
            pages_text = []
            page_count = len(reader.pages)

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)

            full_text = "\n\n".join(pages_text)

            # If no text extracted, might be scanned PDF
            if not full_text.strip():
                full_text = await self._try_ocr(content)

            clean_text = self._clean_text(full_text) if full_text else ""

            return ParseResult(
                content=clean_text,
                file_type="pdf",
                page_count=page_count,
                word_count=self._count_words(clean_text),
                char_count=len(clean_text),
                metadata={
                    "filename": file.filename,
                    "file_size": len(content),
                    "pdf_info": self._extract_pdf_info(reader),
                },
            )
        except Exception as e:
            raise ValueError(f"Failed to parse PDF: {e}") from e

    async def _try_ocr(self, pdf_content: bytes) -> str:
        """Try OCR for scanned PDFs."""
        try:
            # Check if OCR dependencies are available
            import pytesseract
            from pdf2image import convert_from_bytes
            from PIL import Image

            images = convert_from_bytes(pdf_content)
            texts = []

            for image in images:
                text = pytesseract.image_to_string(image)
                if text:
                    texts.append(text)

            return "\n\n".join(texts)
        except ImportError:
            # OCR dependencies not installed
            return ""
        except Exception:
            # OCR failed
            return ""

    def _extract_pdf_info(self, reader) -> dict:
        """Extract PDF metadata."""
        info = {}
        if reader.metadata:
            for key in ["/Title", "/Author", "/Subject", "/Creator", "/Producer"]:
                if key in reader.metadata:
                    value = reader.metadata[key]
                    if value:
                        info[key.lstrip("/")] = str(value)
        return info
