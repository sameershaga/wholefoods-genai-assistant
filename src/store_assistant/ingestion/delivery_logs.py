"""Ingestion for synthetic delivery and on-hand inventory JSON logs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from store_assistant.ingestion.models import DocumentChunk, MetadataValue
from store_assistant.ingestion.normalization import (
    NormalizationError,
    normalize_date,
    normalize_product_category,
    normalize_sku,
    normalize_store_id,
    normalize_supplier,
)

DELIVERY_CHUNK_TOKENS: Final = 50
_REQUIRED_FIELDS: Final = {
    "delivery_id",
    "delivered_at",
    "store_id",
    "sku",
    "product_name",
    "product_category",
    "supplier",
    "quantity",
    "unit",
}


class DeliveryLogError(ValueError):
    """Raised when a delivery log cannot be ingested safely."""


def ingest_delivery_logs(path: str | Path) -> list[DocumentChunk]:
    """Load delivery records and emit chunks of at most 50 whitespace tokens."""
    source_path = Path(path)
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeliveryLogError(f"unable to read delivery log {source_path}") from exc

    if not isinstance(payload, list):
        raise DeliveryLogError("delivery log root must be a JSON array")

    chunks: list[DocumentChunk] = []
    seen_ids: set[str] = set()
    for index, raw_record in enumerate(payload):
        record = _validate_record(raw_record, index)
        delivery_id = _required_text(record, "delivery_id", index)
        if delivery_id in seen_ids:
            raise DeliveryLogError(f"record {index} has duplicate delivery_id {delivery_id!r}")
        seen_ids.add(delivery_id)

        try:
            metadata = _normalize_metadata(record)
        except NormalizationError as exc:
            raise DeliveryLogError(f"record {index} has invalid metadata: {exc}") from exc
        text = _render_record(record, metadata)
        text_parts = _split_words(text, DELIVERY_CHUNK_TOKENS)
        for part_number, text_part in enumerate(text_parts):
            chunks.append(
                DocumentChunk(
                    document_id=f"delivery:{delivery_id}:{part_number}",
                    text=text_part,
                    metadata={**metadata, "chunk_index": part_number},
                )
            )
    return chunks


def _validate_record(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeliveryLogError(f"record {index} must be a JSON object")
    missing = sorted(_REQUIRED_FIELDS - value.keys())
    if missing:
        raise DeliveryLogError(f"record {index} is missing fields: {', '.join(missing)}")
    quantity = value["quantity"]
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 0:
        raise DeliveryLogError(f"record {index} quantity must be a non-negative integer")
    for field in _REQUIRED_FIELDS - {"quantity"}:
        _required_text(value, field, index)
    return value


def _required_text(record: dict[str, Any], field: str, index: int) -> str:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise DeliveryLogError(f"record {index} field {field!r} must be non-empty text")
    return value.strip()


def _normalize_metadata(record: dict[str, Any]) -> dict[str, MetadataValue]:
    return {
        "source_type": "delivery_log",
        "delivery_id": str(record["delivery_id"]).strip(),
        "delivered_at": normalize_date(str(record["delivered_at"])),
        "store_id": normalize_store_id(str(record["store_id"])),
        "sku": normalize_sku(str(record["sku"])),
        "product_category": normalize_product_category(str(record["product_category"])),
        "supplier": normalize_supplier(str(record["supplier"])),
        "quantity": int(record["quantity"]),
    }


def _render_record(record: dict[str, Any], metadata: dict[str, MetadataValue]) -> str:
    return (
        f"Delivery {metadata['delivery_id']} at store {metadata['store_id']} on "
        f"{metadata['delivered_at']}: {record['product_name']} (SKU {metadata['sku']}), "
        f"{metadata['quantity']} {record['unit']}, supplied by {record['supplier']}."
    )


def _split_words(text: str, maximum: int) -> list[str]:
    words = text.split()
    return [" ".join(words[start : start + maximum]) for start in range(0, len(words), maximum)]
