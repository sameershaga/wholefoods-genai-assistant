from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import pytest

from store_assistant.providers.vector_store import (
    PineconeVectorStore,
    VectorRecord,
    VectorStore,
    VectorStoreError,
)


class FakeIndex:
    def __init__(self, query_response: object | None = None) -> None:
        self.query_response = query_response or {"matches": []}
        self.upsert_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []

    def upsert(self, *, vectors: Sequence[Mapping[str, object]], **kwargs: object) -> object:
        self.upsert_calls.append({"vectors": vectors, **kwargs})
        return {}

    def query(self, **kwargs: object) -> object:
        self.query_calls.append(kwargs)
        return self.query_response


def test_pinecone_upsert_and_filtered_query_preserve_text_and_metadata() -> None:
    index = FakeIndex(
        {
            "matches": [
                {
                    "id": "brooklyn-oat",
                    "score": 0.91,
                    "metadata": {
                        "store_id": "BROOKLYN",
                        "sku": "OAT001",
                        "_store_assistant_text": "12 cartons of oat milk",
                    },
                }
            ]
        }
    )
    store = PineconeVectorStore(index, 2, namespace="inventory")
    store.upsert(
        [
            VectorRecord(
                "brooklyn-oat",
                "12 cartons of oat milk",
                (1.0, 0.0),
                {"store_id": "BROOKLYN"},
            )
        ]
    )

    results = store.query((1.0, 0.0), top_k=25, filters={"store_id": "BROOKLYN"})

    vectors = cast(Sequence[Mapping[str, object]], index.upsert_calls[0]["vectors"])
    vector = vectors[0]
    metadata = cast(Mapping[str, object], vector["metadata"])
    assert metadata["_store_assistant_text"] == "12 cartons of oat milk"
    assert index.query_calls == [
        {
            "vector": [1.0, 0.0],
            "top_k": 25,
            "include_metadata": True,
            "include_values": False,
            "filter": {"store_id": "BROOKLYN"},
            "namespace": "inventory",
        }
    ]
    assert results[0].text == "12 cartons of oat milk"
    assert results[0].metadata == {"store_id": "BROOKLYN", "sku": "OAT001"}
    assert isinstance(store, VectorStore)


def test_pinecone_validates_entire_batch_before_upsert() -> None:
    index = FakeIndex()
    store = PineconeVectorStore(index, 2)
    records = [
        VectorRecord("valid", "valid", (1.0, 0.0), {}),
        VectorRecord("invalid", "invalid", (1.0,), {}),
    ]

    with pytest.raises(VectorStoreError, match="expected 2"):
        store.upsert(records)
    assert index.upsert_calls == []


@pytest.mark.parametrize(
    "response, message",
    [
        ({"matches": "bad"}, "malformed matches"),
        ({"matches": [{"id": "doc", "score": 0.5, "metadata": {}}]}, "stored text"),
        (
            {
                "matches": [
                    {
                        "id": "doc",
                        "score": float("nan"),
                        "metadata": {"_store_assistant_text": "x"},
                    }
                ]
            },
            "invalid score",
        ),
    ],
)
def test_pinecone_rejects_malformed_query_responses(response: object, message: str) -> None:
    store = PineconeVectorStore(FakeIndex(response), 2)
    with pytest.raises(VectorStoreError, match=message):
        store.query((1.0, 0.0), top_k=1)


class FailingIndex(FakeIndex):
    def upsert(self, *, vectors: Sequence[Mapping[str, object]], **kwargs: object) -> object:
        raise RuntimeError("secret provider detail")


def test_pinecone_sanitizes_provider_errors() -> None:
    store = PineconeVectorStore(FailingIndex(), 2)
    with pytest.raises(VectorStoreError, match="Pinecone upsert failed") as error:
        store.upsert([VectorRecord("doc", "text", (1.0, 0.0), {})])
    assert "secret provider detail" not in str(error.value)
