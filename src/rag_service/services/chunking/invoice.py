"""Invoice and billing document chunking strategy."""

import re
from dataclasses import dataclass
from typing import Literal

from rag_service.services.chunking.base import BaseChunkingStrategy, ChunkResult


@dataclass
class InvoiceField:
    """Extracted invoice field."""

    name: str
    value: str
    field_type: Literal["header", "line_item", "summary", "other"]


class InvoiceChunker(BaseChunkingStrategy):
    """
    Invoice and billing document chunking.

    Extracts structured data from invoices including:
    - Header information (vendor, invoice number, dates)
    - Line items (products/services, quantities, prices)
    - Summary (subtotal, tax, total, payment terms)

    Creates separate chunks for each logical section with rich metadata.
    """

    @property
    def name(self) -> str:
        return "invoice"

    def chunk(self, text: str, config: dict | None = None) -> list[ChunkResult]:
        """
        Split invoice/billing document into structured chunks.

        Config options:
            extract_line_items: Extract individual line items (default: True)
            extract_totals: Extract totals as separate chunk (default: True)
            include_raw_text: Include raw text in metadata (default: False)
        """
        config = config or {}
        extract_line_items = config.get("extract_line_items", True)
        extract_totals = config.get("extract_totals", True)
        include_raw = config.get("include_raw_text", False)

        if not text or not text.strip():
            return []

        chunks = []
        chunk_index = 0

        # Extract header information
        header_chunk = self._extract_header(text)
        if header_chunk:
            chunks.append(
                ChunkResult(
                    content=header_chunk["content"],
                    chunk_index=chunk_index,
                    metadata={
                        "invoice_section": "header",
                        "extracted_fields": header_chunk["fields"],
                    },
                )
            )
            chunk_index += 1

        # Extract line items
        if extract_line_items:
            line_items = self._extract_line_items(text)
            for item in line_items:
                chunks.append(
                    ChunkResult(
                        content=item["content"],
                        chunk_index=chunk_index,
                        metadata={
                            "invoice_section": "line_item",
                            "extracted_fields": item["fields"],
                        },
                    )
                )
                chunk_index += 1

        # Extract summary/totals
        if extract_totals:
            summary_chunk = self._extract_summary(text)
            if summary_chunk:
                chunks.append(
                    ChunkResult(
                        content=summary_chunk["content"],
                        chunk_index=chunk_index,
                        metadata={
                            "invoice_section": "summary",
                            "extracted_fields": summary_chunk["fields"],
                        },
                    )
                )
                chunk_index += 1

        # If no structured content found, fall back to simple chunking
        if not chunks:
            chunks = self._fallback_chunk(text)

        # Update total_chunks and add raw text if requested
        total = len(chunks)
        for chunk in chunks:
            chunk.metadata["total_chunks"] = total
            if include_raw:
                chunk.metadata["raw_text"] = text

        return chunks

    def _extract_header(self, text: str) -> dict | None:
        """Extract invoice header information."""
        fields = {}

        # Invoice number patterns
        invoice_patterns = [
            r"invoice\s*(?:#|no\.?|number)?:?\s*([A-Z0-9-]+)",
            r"inv\s*(?:#|no\.?)?:?\s*([A-Z0-9-]+)",
            r"bill\s*(?:#|no\.?)?:?\s*([A-Z0-9-]+)",
        ]
        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["invoice_number"] = match.group(1)
                break

        # Date patterns
        date_patterns = [
            r"(?:invoice\s+)?date:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"(?:invoice\s+)?date:?\s*(\w+\s+\d{1,2},?\s+\d{4})",
            r"dated?:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["invoice_date"] = match.group(1)
                break

        # Due date
        due_patterns = [
            r"due\s*(?:date)?:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
            r"payment\s+due:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})",
        ]
        for pattern in due_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["due_date"] = match.group(1)
                break

        # Vendor/company name (often at the top)
        lines = text.strip().split("\n")
        if lines:
            # First non-empty line is often company name
            for line in lines[:5]:
                line = line.strip()
                if line and not re.match(r"^(invoice|bill|statement)", line, re.IGNORECASE):
                    if len(line) > 3 and len(line) < 100:
                        fields["vendor_name"] = line
                        break

        # Customer/Bill To
        bill_to_match = re.search(
            r"(?:bill\s+to|customer|client):?\s*\n?([\w\s,.]+?)(?:\n|$)",
            text,
            re.IGNORECASE,
        )
        if bill_to_match:
            fields["customer"] = bill_to_match.group(1).strip()

        if not fields:
            return None

        # Build content string
        content_parts = []
        if "vendor_name" in fields:
            content_parts.append(f"Vendor: {fields['vendor_name']}")
        if "invoice_number" in fields:
            content_parts.append(f"Invoice Number: {fields['invoice_number']}")
        if "invoice_date" in fields:
            content_parts.append(f"Invoice Date: {fields['invoice_date']}")
        if "due_date" in fields:
            content_parts.append(f"Due Date: {fields['due_date']}")
        if "customer" in fields:
            content_parts.append(f"Customer: {fields['customer']}")

        return {
            "content": "\n".join(content_parts),
            "fields": fields,
        }

    def _extract_line_items(self, text: str) -> list[dict]:
        """Extract line items from invoice."""
        items = []

        # Common line item patterns
        # Pattern: Description, Quantity, Unit Price, Total
        line_patterns = [
            # Tabular format with quantity and amounts
            r"(.+?)\s+(\d+(?:\.\d+)?)\s+\$?([\d,]+\.?\d*)\s+\$?([\d,]+\.?\d*)",
            # Simple item with price
            r"(.+?)\s+\$?([\d,]+\.?\d*)\s*$",
        ]

        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip header/footer lines
            if re.match(
                r"^(description|item|qty|quantity|price|amount|total|subtotal)",
                line,
                re.IGNORECASE,
            ):
                continue

            # Try to match line item patterns
            for pattern in line_patterns:
                match = re.match(pattern, line)
                if match:
                    groups = match.groups()
                    fields = {"description": groups[0].strip()}

                    if len(groups) == 4:
                        # Full line item
                        fields["quantity"] = groups[1]
                        fields["unit_price"] = groups[2]
                        fields["total"] = groups[3]
                        content = (
                            f"{fields['description']} - "
                            f"Qty: {fields['quantity']}, "
                            f"Price: ${fields['unit_price']}, "
                            f"Total: ${fields['total']}"
                        )
                    elif len(groups) == 2:
                        # Simple item
                        fields["amount"] = groups[1]
                        content = f"{fields['description']} - ${fields['amount']}"
                    else:
                        content = line

                    items.append({"content": content, "fields": fields})
                    break

        return items

    def _extract_summary(self, text: str) -> dict | None:
        """Extract invoice summary (totals)."""
        fields = {}

        # Subtotal
        subtotal_match = re.search(
            r"subtotal:?\s*\$?([\d,]+\.?\d*)",
            text,
            re.IGNORECASE,
        )
        if subtotal_match:
            fields["subtotal"] = subtotal_match.group(1)

        # Tax
        tax_match = re.search(
            r"(?:tax|vat|gst):?\s*\$?([\d,]+\.?\d*)",
            text,
            re.IGNORECASE,
        )
        if tax_match:
            fields["tax"] = tax_match.group(1)

        # Discount
        discount_match = re.search(
            r"discount:?\s*-?\$?([\d,]+\.?\d*)",
            text,
            re.IGNORECASE,
        )
        if discount_match:
            fields["discount"] = discount_match.group(1)

        # Total (look for "Total" that's not "Subtotal")
        total_match = re.search(
            r"(?:grand\s+)?total(?:\s+due)?:?\s*\$?([\d,]+\.?\d*)",
            text,
            re.IGNORECASE,
        )
        if total_match:
            fields["total"] = total_match.group(1)

        # Amount Due
        due_match = re.search(
            r"(?:amount|balance)\s+due:?\s*\$?([\d,]+\.?\d*)",
            text,
            re.IGNORECASE,
        )
        if due_match:
            fields["amount_due"] = due_match.group(1)

        # Payment terms
        terms_match = re.search(
            r"(?:payment\s+)?terms?:?\s*(.+?)(?:\n|$)",
            text,
            re.IGNORECASE,
        )
        if terms_match:
            fields["payment_terms"] = terms_match.group(1).strip()

        if not fields:
            return None

        # Build content string
        content_parts = []
        if "subtotal" in fields:
            content_parts.append(f"Subtotal: ${fields['subtotal']}")
        if "discount" in fields:
            content_parts.append(f"Discount: ${fields['discount']}")
        if "tax" in fields:
            content_parts.append(f"Tax: ${fields['tax']}")
        if "total" in fields:
            content_parts.append(f"Total: ${fields['total']}")
        if "amount_due" in fields:
            content_parts.append(f"Amount Due: ${fields['amount_due']}")
        if "payment_terms" in fields:
            content_parts.append(f"Payment Terms: {fields['payment_terms']}")

        return {
            "content": "\n".join(content_parts),
            "fields": fields,
        }

    def _fallback_chunk(self, text: str) -> list[ChunkResult]:
        """Fallback to simple chunking if no structure detected."""
        # Simple paragraph-based chunking
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if para:
                chunks.append(
                    ChunkResult(
                        content=para,
                        chunk_index=i,
                        metadata={
                            "invoice_section": "unstructured",
                        },
                    )
                )

        return chunks
