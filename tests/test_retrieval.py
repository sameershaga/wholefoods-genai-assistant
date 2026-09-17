from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from store_assistant.ingestion.models import MetadataValue
from store_assistant.providers.embeddings import Embedding, LocalHashEmbeddingProvider
from store_assistant.providers.reranking import LocalLexicalReranker, RerankCandidate, RerankResult
from store_assistant.providers.vector_store import InMemoryVectorStore, VectorRecord
from store_assistant.retrieval import RetrievalConfig, RetrievalError, RetrievalService


def _service_with_inventory() -> RetrievalService:
    embeddings = LocalHashEmbeddingProvider(dimensions=32)
    store = InMemoryVectorStore(dimension=32)
    texts = [
        "Brooklyn inventory: oat milk SKU OAT-001 has 12 cartons available.",
        "Manhattan inventory: oat milk SKU OAT-001 has 3 cartons available.",
        "Brooklyn inventory: almond milk SKU ALM-002 has 8 cartons available.",
    ]
    metadata = [
        {"store_id": "BROOKLYN", "sku": "OAT-001", "source_type": "delivery_log"},
        {"store_id": "MANHATTAN", "sku": "OAT-001", "source_type": "delivery_log"},
        {"store_id": "BROOKLYN", "sku": "ALM-002", "source_type": "delivery_log"},
    ]
    store.upsert(
        [
            VectorRecord(f"doc-{index}", text, vector, item_metadata)
            for index, (text, vector, item_metadata) in enumerate(
                zip(texts, embeddings.embed(texts), metadata, strict=True)
            )
        ]
    )
    return RetrievalService(embeddings, store, LocalLexicalReranker())


def test_retrieval_preserves_store_isolation_and_scores() -> None:
    results = _service_with_inventory().retrieve(
        "Do we have oat milk?", filters={"store_id": "BROOKLYN", "sku": "OAT-001"}
    )

    assert [result.document_id for result in results] == ["doc-0"]
    assert results[0].metadata["store_id"] == "BROOKLYN"
    assert "12 cartons" in results[0].text
    assert isinstance(results[0].retrieval_score, float)
    assert isinstance(results[0].reranking_score, float)


def test_retrieval_returns_empty_without_calling_reranker() -> None:
    @dataclass
    class FailingReranker:
        def rerank(
            self, query: str, candidates: Sequence[RerankCandidate], *, top_k: int
        ) -> list[RerankResult]:
            raise AssertionError("reranker should not be called")

    embeddings = LocalHashEmbeddingProvider(dimensions=8)
    service = RetrievalService(embeddings, InMemoryVectorStore(8), FailingReranker())
    assert service.retrieve("oat milk", filters={"store_id": "BROOKLYN"}) == []


def test_retrieval_uses_configured_candidate_and_context_counts() -> None:
    @dataclass
    class RecordingStore:
        dimension: int = 8
        top_k: int | None = None
        filters: Mapping[str, MetadataValue] | None = None

        def upsert(self, records: Sequence[VectorRecord]) -> None:
            pass

        def query(
            self,
            embedding: Embedding,
            *,
            top_k: int,
            filters: Mapping[str, MetadataValue] | None = None,
        ) -> list[object]:
            self.top_k = top_k
            self.filters = filters
            return []

    store = RecordingStore()
    service = RetrievalService(
        LocalHashEmbeddingProvider(8),
        store,  # type: ignore[arg-type]
        LocalLexicalReranker(),
        config=RetrievalConfig(candidate_count=25, context_count=3),
    )
    service.retrieve("oat milk", filters={"store_id": "BROOKLYN"})
    assert store.top_k == 25
    assert store.filters == {"store_id": "BROOKLYN"}


@pytest.mark.parametrize("query", ["", "   "])
def test_retrieval_rejects_empty_query(query: str) -> None:
    with pytest.raises(RetrievalError, match="non-empty"):
        _service_with_inventory().retrieve(query)


def test_retrieval_rejects_dimension_mismatch() -> None:
    with pytest.raises(RetrievalError, match="dimensions do not match"):
        RetrievalService(
            LocalHashEmbeddingProvider(8), InMemoryVectorStore(9), LocalLexicalReranker()
        )


@pytest.mark.parametrize(("candidate_count", "context_count"), [(0, 1), (5, 0), (2, 3)])
def test_retrieval_rejects_invalid_configuration(candidate_count: int, context_count: int) -> None:
    with pytest.raises(RetrievalError):
        RetrievalConfig(candidate_count=candidate_count, context_count=context_count)
