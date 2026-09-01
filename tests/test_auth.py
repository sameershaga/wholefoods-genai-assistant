from __future__ import annotations

from typing import cast

import pytest

from store_assistant.auth import (
    AuthenticationError,
    AuthProvider,
    MockAuthProvider,
    UserContext,
)
from store_assistant.providers.embeddings import LocalHashEmbeddingProvider
from store_assistant.providers.reranking import LocalLexicalReranker
from store_assistant.providers.vector_store import InMemoryVectorStore, VectorRecord
from store_assistant.retrieval import RetrievalError, RetrievalService
from store_assistant.services import AuthenticatedRetrievalService


def _service() -> AuthenticatedRetrievalService:
    embeddings = LocalHashEmbeddingProvider(dimensions=64)
    store = InMemoryVectorStore(dimension=64)
    texts = ["Brooklyn has 12 cartons of oat milk", "Manhattan has 3 cartons of oat milk"]
    store.upsert(
        [
            VectorRecord(
                document_id=f"delivery-{index}",
                text=text,
                embedding=embedding,
                metadata={"store_id": store_id, "sku": "OAT001"},
            )
            for index, (text, embedding, store_id) in enumerate(
                zip(texts, embeddings.embed(texts), ("BROOKLYN", "MANHATTAN"), strict=True)
            )
        ]
    )
    retrieval = RetrievalService(embeddings, store, LocalLexicalReranker())
    auth = MockAuthProvider(
        {"brooklyn-token": UserContext("manager-1", "brooklyn", "Brooklyn Manager")}
    )
    return AuthenticatedRetrievalService(auth, retrieval)


def test_mock_provider_conforms_and_normalizes_claims() -> None:
    provider = MockAuthProvider({"token": UserContext(" user-1 ", "ny-brooklyn")})

    assert isinstance(provider, AuthProvider)
    assert provider.authenticate(" token ") == UserContext("user-1", "NY-BROOKLYN")


@pytest.mark.parametrize("token", ["", "unknown"])
def test_mock_provider_rejects_missing_or_unknown_tokens(token: str) -> None:
    provider = MockAuthProvider({"valid": UserContext("user", "brooklyn")})

    with pytest.raises(AuthenticationError):
        provider.authenticate(token)


def test_authenticated_retrieval_propagates_store_context() -> None:
    context, results = _service().retrieve(
        "brooklyn-token", "Do we have oat milk?", filters={"sku": "OAT001"}
    )

    assert context.store_id == "BROOKLYN"
    assert [result.metadata["store_id"] for result in results] == ["BROOKLYN"]
    assert "12 cartons" in results[0].text


def test_authenticated_retrieval_rejects_cross_store_override() -> None:
    with pytest.raises(RetrievalError, match="conflicts"):
        _service().retrieve(
            "brooklyn-token", "Do we have oat milk?", filters={"store_id": "Manhattan"}
        )


def test_user_context_rejects_invalid_claims() -> None:
    with pytest.raises(AuthenticationError):
        UserContext("", "brooklyn")
    with pytest.raises(AuthenticationError):
        UserContext("user", cast(str, None))
