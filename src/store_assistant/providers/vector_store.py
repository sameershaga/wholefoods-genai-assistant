"""Vector-store contract and an in-memory implementation for local operation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence, runtime_checkable

from store_assistant.ingestion.models import MetadataValue
from store_assistant.providers.embeddings import Embedding


class VectorStoreError(ValueError):
    """Raised when vector records or search parameters are invalid."""


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """A chunk and its embedding, ready for storage."""

    document_id: str
    text: str
    embedding: Embedding
    metadata: Mapping[str, MetadataValue]


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """A matching stored record with its vector similarity score."""

    document_id: str
    text: str
    score: float
    metadata: Mapping[str, MetadataValue]


@runtime_checkable
class VectorStore(Protocol):
    """Boundary implemented by local and Pinecone-backed vector indexes."""

    @property
    def dimension(self) -> int:
        """Return the vector dimension accepted by this index."""

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        """Insert records, replacing existing records with the same document ID."""

    def query(
        self,
        embedding: Embedding,
        *,
        top_k: int,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[VectorSearchResult]:
        """Return the best matches satisfying all exact-match metadata filters."""


class InMemoryVectorStore:
    """Deterministic cosine-similarity index for offline use and tests."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise VectorStoreError("vector-store dimension must be positive")
        self._dimension = dimension
        self._records: dict[str, VectorRecord] = {}

    @property
    def dimension(self) -> int:
        return self._dimension

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        validated: list[VectorRecord] = []
        for position, record in enumerate(records):
            if not record.document_id.strip():
                raise VectorStoreError(f"record at position {position} has an empty document ID")
            if not record.text.strip():
                raise VectorStoreError(f"record {record.document_id!r} has empty text")
            self._validate_embedding(record.embedding, label=f"record {record.document_id!r}")
            validated.append(
                VectorRecord(
                    document_id=record.document_id,
                    text=record.text,
                    embedding=tuple(record.embedding),
                    metadata=dict(record.metadata),
                )
            )

        # Validate the entire batch before mutating the index.
        for record in validated:
            self._records[record.document_id] = record

    def query(
        self,
        embedding: Embedding,
        *,
        top_k: int,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[VectorSearchResult]:
        if top_k < 1:
            raise VectorStoreError("top_k must be positive")
        query_magnitude = self._validate_embedding(embedding, label="query")
        required = filters or {}

        matches: list[VectorSearchResult] = []
        for record in self._records.values():
            if not all(record.metadata.get(key) == value for key, value in required.items()):
                continue
            record_magnitude = math.sqrt(sum(value * value for value in record.embedding))
            score = sum(
                left * right for left, right in zip(embedding, record.embedding, strict=True)
            ) / (query_magnitude * record_magnitude)
            matches.append(
                VectorSearchResult(
                    document_id=record.document_id,
                    text=record.text,
                    score=score,
                    metadata=dict(record.metadata),
                )
            )

        matches.sort(key=lambda result: (-result.score, result.document_id))
        return matches[:top_k]

    def _validate_embedding(self, embedding: Embedding, *, label: str) -> float:
        if len(embedding) != self._dimension:
            raise VectorStoreError(
                f"{label} vector has dimension {len(embedding)}; expected {self._dimension}"
            )
        if any(not math.isfinite(value) for value in embedding):
            raise VectorStoreError(f"{label} vector contains a non-finite value")
        magnitude = math.sqrt(sum(value * value for value in embedding))
        if magnitude == 0:
            raise VectorStoreError(f"{label} vector must not be all zeros")
        return magnitude
