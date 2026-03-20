"""CSV file parser."""

import csv
import io

from fastapi import UploadFile

from rag_service.services.parsers.base import BaseFileParser, ParseResult


class CSVParser(BaseFileParser):
    """Parser for CSV files."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".csv", ".CSV", ".tsv", ".TSV"]

    @property
    def supported_mime_types(self) -> list[str]:
        return [
            "text/csv",
            "text/tab-separated-values",
            "application/csv",
        ]

    async def parse(self, file: UploadFile) -> ParseResult:
        """Parse CSV file and convert to text representation."""
        content = await self._read_file_content(file)

        # Detect encoding
        encoding = self._detect_encoding(content)
        text = content.decode(encoding, errors="replace")

        # Detect delimiter
        delimiter = self._detect_delimiter(text, file.filename)

        try:
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows = list(reader)
        except csv.Error as e:
            raise ValueError(f"Failed to parse CSV: {e}") from e

        if not rows:
            return ParseResult(
                content="",
                file_type="csv",
                metadata={"filename": file.filename, "file_size": len(content)},
            )

        # Get headers (first row)
        headers = rows[0]
        data_rows = rows[1:]

        # Convert to readable text format
        text_parts = []

        # Add header info
        text_parts.append(f"Columns: {', '.join(headers)}")
        text_parts.append(f"Total rows: {len(data_rows)}")
        text_parts.append("")

        # Add data as readable rows
        for i, row in enumerate(data_rows):
            row_text = []
            for j, value in enumerate(row):
                if j < len(headers):
                    row_text.append(f"{headers[j]}: {value}")
                else:
                    row_text.append(f"Column {j + 1}: {value}")
            text_parts.append(f"Row {i + 1}: {'; '.join(row_text)}")

        full_text = "\n".join(text_parts)

        return ParseResult(
            content=full_text,
            file_type="csv",
            word_count=self._count_words(full_text),
            char_count=len(full_text),
            metadata={
                "filename": file.filename,
                "file_size": len(content),
                "encoding": encoding,
                "delimiter": delimiter,
                "column_count": len(headers),
                "row_count": len(data_rows),
                "headers": headers,
            },
        )

    def _detect_encoding(self, content: bytes) -> str:
        """Detect file encoding."""
        try:
            import chardet

            result = chardet.detect(content)
            if result and result.get("encoding"):
                return result["encoding"]
        except ImportError:
            pass
        return "utf-8"

    def _detect_delimiter(self, text: str, filename: str | None) -> str:
        """Detect CSV delimiter."""
        # Check file extension
        if filename and filename.lower().endswith(".tsv"):
            return "\t"

        # Use csv.Sniffer to detect delimiter
        try:
            sample = text[:4096]  # Use first 4KB
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            # Default to comma
            return ","
