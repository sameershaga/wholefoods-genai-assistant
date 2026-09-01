from __future__ import annotations

import pytest

from store_assistant.providers.embeddings import LocalHashEmbeddingProvider
from store_assistant.providers.vector_store import (
    InMemoryVectorStore,
    VectorRecord,
    VectorStore,
    VectorStoreError,
)


def _record(
    provider: LocalHashEmbeddingProvider,
    document_id: str,
    text: str,
    **metadata: str | int | float | bool,
) -> VectorRecord:
    return VectorRecord(document_id, text, provider.embed([text])[0], metadata)


def test_local_store_ranks_by_similarity_and_respects_top_k() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=256)
    store = InMemoryVectorStore(provider.dimension)
    store.upsert(
        [
            _record(provider, "oat", "Oat milk inventory has 12 cartons", store_id="BROOKLYN"),
            _record(provider, "salmon", "Salmon delivery arrives Friday", store_id="BROOKLYN"),
        ]
    )

    results = store.query(provider.embed(["Do we have oat milk?"])[0], top_k=1)

    assert [result.document_id for result in results] == ["oat"]
    assert results[0].score > 0


def test_exact_metadata_filters_enforce_cross_store_and_sku_isolation() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=256)
    store = InMemoryVectorStore(provider.dimension)
    store.upsert(
        [
            _record(
                provider,
                "brooklyn-oat",
                "Oat milk inventory has 12 cartons",
                store_id="BROOKLYN",
                sku="OATMILK-001",
                source_type="delivery_log",
            ),
            _record(
                provider,
                "manhattan-oat",
                "Oat milk inventory has 3 cartons",
                store_id="MANHATTAN",
                sku="OATMILK-001",
                source_type="delivery_log",
            ),
            _record(
                provider,
                "brooklyn-almond",
                "Almond milk inventory has 8 cartons",
                store_id="BROOKLYN",
                sku="ALMOND-001",
                source_type="delivery_log",
            ),
        ]
    )

    results = store.query(
        provider.embed(["Do we have oat milk?"])[0],
        top_k=25,
        filters={"store_id": "BROOKLYN", "sku": "OATMILK-001"},
    )

    assert [result.document_id for result in results] == ["brooklyn-oat"]
    assert all(result.metadata["store_id"] == "BROOKLYN" for result in results)


def test_upsert_replaces_existing_record_and_copies_metadata() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=64)
    store = InMemoryVectorStore(provider.dimension)
    metadata = {"store_id": "BROOKLYN"}
    store.upsert([_record(provider, "inventory", "Old oat milk count", **metadata)])
    metadata["store_id"] = "MANHATTAN"
    store.upsert([_record(provider, "inventory", "Updated oat milk count", store_id="BROOKLYN")])

    results = store.query(provider.embed(["updated oat milk count"])[0], top_k=10)

    assert len(results) == 1
    assert results[0].text == "Updated oat milk count"
    assert results[0].metadata["store_id"] == "BROOKLYN"


def test_local_store_satisfies_contract_and_empty_index_returns_no_results() -> None:
    store = InMemoryVectorStore(8)

    assert isinstance(store, VectorStore)
    assert store.query((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), top_k=1) == []


def test_invalid_batch_is_atomic_and_query_parameters_are_validated() -> None:
    store = InMemoryVectorStore(2)
    valid = VectorRecord("valid", "valid text", (1.0, 0.0), {})
    invalid = VectorRecord("invalid", "invalid text", (1.0,), {})

    with pytest.raises(VectorStoreError, match="expected 2"):
        store.upsert([valid, invalid])
    assert store.query((1.0, 0.0), top_k=1) == []

    with pytest.raises(VectorStoreError, match="top_k"):
        store.query((1.0, 0.0), top_k=0)
    with pytest.raises(VectorStoreError, match="all zeros"):
        store.query((0.0, 0.0), top_k=1)
