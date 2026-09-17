"""Retrieval orchestration from query embedding through filtered reranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from store_assistant.ingestion.models import MetadataValue
from store_assistant.providers.embeddings import EmbeddingProvider
from store_assistant.providers.reranking import RerankCandidate, Reranker, RerankResult
from store_assistant.providers.vector_store import VectorStore


class RetrievalError(ValueError):
    """Raised when retrieval configuration or a request is invalid."""


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Controls candidate breadth and the amount of context sent downstream."""

    candidate_count: int = 25
    context_count: int = 5

    def __post_init__(self) -> None:
        if self.candidate_count < 1:
            raise RetrievalError("candidate_count must be positive")
        if self.context_count < 1:
            raise RetrievalError("context_count must be positive")
        if self.context_count > self.candidate_count:
            raise RetrievalError("context_count cannot exceed candidate_count")


class RetrievalService:
    """Coordinates provider boundaries while applying filters before reranking."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        reranker: Reranker,
        *,
        config: RetrievalConfig | None = None,
    ) -> None:
        if embedding_provider.dimension != vector_store.dimension:
            raise RetrievalError(
                "embedding provider and vector store dimensions do not match "
                f"({embedding_provider.dimension} != {vector_store.dimension})"
            )
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._reranker = reranker
        self._config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        *,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> list[RerankResult]:
        """Return strongest filtered context with retrieval and reranking scores."""
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("query must be a non-empty string")

        query_embedding = self._embedding_provider.embed([query])[0]
        matches = self._vector_store.query(
            query_embedding,
            top_k=self._config.candidate_count,
            filters=dict(filters) if filters is not None else None,
        )
        candidates = [
            RerankCandidate(
                document_id=match.document_id,
                text=match.text,
                retrieval_score=match.score,
                metadata=match.metadata,
            )
            for match in matches
        ]
        if not candidates:
            return []
        return self._reranker.rerank(query, candidates, top_k=self._config.context_count)
