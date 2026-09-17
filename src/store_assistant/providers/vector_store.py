"""Vector-store contract and an in-memory implementation for local operation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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


class PineconeIndex(Protocol):
    """Narrow subset of a Pinecone index used by the hosted adapter."""

    def upsert(self, *, vectors: Sequence[Mapping[str, object]], **kwargs: object) -> object:
        """Write vectors to the index."""

    def query(self, **kwargs: object) -> object:
        """Query vectors from the index."""


class PineconeVectorStore:
    """Pinecone-compatible vector store using an injected SDK index client."""

    _TEXT_KEY = "_store_assistant_text"

    def __init__(
        self,
        index: PineconeIndex,
        dimension: int,
        *,
        namespace: str | None = None,
    ) -> None:
        if dimension < 1:
            raise VectorStoreError("vector-store dimension must be positive")
        if namespace is not None and not namespace.strip():
            raise VectorStoreError("Pinecone namespace must be non-empty when provided")
        self._index = index
        self._dimension = dimension
        self._namespace = namespace

    @property
    def dimension(self) -> int:
        return self._dimension

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        vectors: list[Mapping[str, object]] = []
        for position, record in enumerate(records):
            if not record.document_id.strip():
                raise VectorStoreError(f"record at position {position} has an empty document ID")
            if not record.text.strip():
                raise VectorStoreError(f"record {record.document_id!r} has empty text")
            self._validate_embedding(record.embedding, label=f"record {record.document_id!r}")
            if self._TEXT_KEY in record.metadata:
                raise VectorStoreError(f"record {record.document_id!r} uses reserved metadata key")
            metadata = dict(record.metadata)
            metadata[self._TEXT_KEY] = record.text
            vectors.append(
                {"id": record.document_id, "values": list(record.embedding), "metadata": metadata}
            )

        if not vectors:
            return
        try:
            if self._namespace is None:
                self._index.upsert(vectors=vectors)
            else:
                self._index.upsert(vectors=vectors, namespace=self._namespace)
        except Exception as exc:
            raise VectorStoreError("Pinecone upsert failed") from exc

    def query(
        self,
        embedding: Embedding,
        *,
        top_k: int,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[VectorSearchResult]:
        if top_k < 1:
            raise VectorStoreError("top_k must be positive")
        self._validate_embedding(embedding, label="query")
        kwargs: dict[str, object] = {
            "vector": list(embedding),
            "top_k": top_k,
            "include_metadata": True,
            "include_values": False,
        }
        if filters:
            kwargs["filter"] = dict(filters)
        if self._namespace is not None:
            kwargs["namespace"] = self._namespace
        try:
            response = self._index.query(**kwargs)
            matches = self._response_matches(response)
            return [self._parse_match(match) for match in matches]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Pinecone query failed") from exc

    def _validate_embedding(self, embedding: Embedding, *, label: str) -> None:
        if len(embedding) != self._dimension:
            raise VectorStoreError(
                f"{label} vector has dimension {len(embedding)}; expected {self._dimension}"
            )
        if any(not math.isfinite(value) for value in embedding):
            raise VectorStoreError(f"{label} vector contains a non-finite value")
        if not any(value != 0 for value in embedding):
            raise VectorStoreError(f"{label} vector must not be all zeros")

    @staticmethod
    def _response_matches(response: object) -> Sequence[object]:
        data: Any = response
        matches = data.get("matches") if isinstance(data, Mapping) else data.matches
        if not isinstance(matches, Sequence) or isinstance(matches, (str, bytes)):
            raise VectorStoreError("Pinecone returned malformed matches")
        return matches

    def _parse_match(self, match: object) -> VectorSearchResult:
        data: Any = match
        document_id = data.get("id") if isinstance(data, Mapping) else data.id
        score = data.get("score") if isinstance(data, Mapping) else data.score
        metadata = data.get("metadata") if isinstance(data, Mapping) else data.metadata
        if not isinstance(document_id, str) or not document_id.strip():
            raise VectorStoreError("Pinecone returned a match without a document ID")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise VectorStoreError("Pinecone returned a match with an invalid score")
        if not isinstance(metadata, Mapping):
            raise VectorStoreError("Pinecone returned a match without metadata")
        copied = dict(metadata)
        text = copied.pop(self._TEXT_KEY, None)
        if not isinstance(text, str) or not text.strip():
            raise VectorStoreError("Pinecone returned a match without stored text")
        return VectorSearchResult(document_id, text, float(score), copied)


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
