"""Microsoft Excel file parser."""

import io

from fastapi import UploadFile

from rag_service.services.parsers.base import BaseFileParser, ParseResult


class ExcelParser(BaseFileParser):
    """Parser for Microsoft Excel files (xlsx, xls)."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".xlsx", ".XLSX", ".xls", ".XLS"]

    @property
    def supported_mime_types(self) -> list[str]:
        return [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel",
        ]

    async def parse(self, file: UploadFile) -> ParseResult:
        """Parse Excel file and convert to text representation."""
        from openpyxl import load_workbook

        content = await self._read_file_content(file)
        excel_file = io.BytesIO(content)

        try:
            wb = load_workbook(excel_file, data_only=True, read_only=True)
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file: {e}") from e

        text_parts = []
        sheet_info = []

        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]

            # Get sheet dimensions
            try:
                rows = list(sheet.rows)
            except Exception:
                continue

            if not rows:
                continue

            sheet_info.append({
                "name": sheet_name,
                "row_count": len(rows),
            })

            text_parts.append(f"=== Sheet: {sheet_name} ===")
            text_parts.append("")

            # Get headers (first row)
            headers = []
            first_row = rows[0] if rows else []
            for cell in first_row:
                value = cell.value
                headers.append(str(value) if value is not None else "")

            if headers:
                text_parts.append(f"Columns: {', '.join(h for h in headers if h)}")

            # Process data rows
            for i, row in enumerate(rows[1:], start=1):
                row_values = []
                for j, cell in enumerate(row):
                    value = cell.value
                    if value is not None:
                        header = headers[j] if j < len(headers) and headers[j] else f"Column {j + 1}"
                        row_values.append(f"{header}: {value}")

                if row_values:
                    text_parts.append(f"Row {i}: {'; '.join(row_values)}")

            text_parts.append("")

        wb.close()

        full_text = "\n".join(text_parts)

        return ParseResult(
            content=full_text,
            file_type="excel",
            word_count=self._count_words(full_text),
            char_count=len(full_text),
            metadata={
                "filename": file.filename,
                "file_size": len(content),
                "sheet_count": len(wb.sheetnames),
                "sheets": sheet_info,
            },
        )
