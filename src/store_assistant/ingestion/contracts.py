"""Supplier-contract PDF ingestion with normalized filter metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from store_assistant.ingestion.models import DocumentChunk, MetadataValue
from store_assistant.ingestion.normalization import (
    NormalizationError,
    normalize_date,
    normalize_product_category,
    normalize_sku,
    normalize_store_id,
    normalize_supplier,
)

CONTRACT_CHUNK_TOKENS: Final = 800
_FIELDS: Final = {
    "Contract ID": "contract_id",
    "Effective Date": "effective_date",
    "Store ID": "store_id",
    "Supplier": "supplier",
    "Product Category": "product_category",
    "SKU": "sku",
}


class ContractIngestionError(ValueError):
    """Raised when a supplier contract cannot be safely ingested."""


def ingest_supplier_contract(path: str | Path) -> list[DocumentChunk]:
    """Extract a PDF contract and split it into chunks of about 800 tokens."""
    source_path = Path(path)
    try:
        reader = PdfReader(source_path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except (OSError, PdfReadError, ValueError) as exc:
        raise ContractIngestionError(f"unable to read contract PDF {source_path}") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise ContractIngestionError("contract PDF contains no extractable text")

    fields, body_lines = _extract_fields(text)
    missing = [label for label, key in _FIELDS.items() if not fields.get(key)]
    if missing:
        raise ContractIngestionError(f"contract is missing metadata: {', '.join(missing)}")
    body = "\n".join(body_lines).strip()
    if not body:
        raise ContractIngestionError("contract contains no body text")

    try:
        metadata: dict[str, MetadataValue] = {
            "source_type": "supplier_contract",
            "contract_id": fields["contract_id"],
            "effective_date": normalize_date(fields["effective_date"]),
            "store_id": normalize_store_id(fields["store_id"]),
            "supplier": normalize_supplier(fields["supplier"]),
            "product_category": normalize_product_category(fields["product_category"]),
            "sku": normalize_sku(fields["sku"]),
        }
    except NormalizationError as exc:
        raise ContractIngestionError(f"contract has invalid metadata: {exc}") from exc

    chunks = _split_words(body, CONTRACT_CHUNK_TOKENS)
    return [
        DocumentChunk(
            document_id=f"contract:{fields['contract_id']}:{index}",
            text=chunk,
            metadata={**metadata, "chunk_index": index},
        )
        for index, chunk in enumerate(chunks)
    ]


def _extract_fields(text: str) -> tuple[dict[str, str], list[str]]:
    fields: dict[str, str] = {}
    body: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        matched = False
        for label, key in _FIELDS.items():
            prefix = f"{label}:"
            if stripped.startswith(prefix):
                fields[key] = stripped.removeprefix(prefix).strip()
                matched = True
                break
        if not matched and stripped:
            body.append(stripped)
    return fields, body


def _split_words(text: str, maximum: int) -> list[str]:
    words = text.split()
    return [" ".join(words[start : start + maximum]) for start in range(0, len(words), maximum)]
