"""Source-neutral records emitted by ingestion pipelines."""

from __future__ import annotations

from dataclasses import dataclass

type MetadataValue = str | int | float | bool


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A retrievable text unit with normalized, filterable metadata."""

    document_id: str
    text: str
    metadata: dict[str, MetadataValue]
