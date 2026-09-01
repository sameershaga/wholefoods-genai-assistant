from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from store_assistant.auth import (
    AuthenticationError,
    AuthProvider,
    MockAuthProvider,
    OktaOIDCAuthProvider,
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


class StubOIDCVerifier:
    def __init__(self, claims: Mapping[str, Any] | None = None, *, fails: bool = False) -> None:
        self.claims = claims or {}
        self.fails = fails
        self.call: tuple[str, str, str] | None = None

    def verify(self, token: str, *, issuer: str, audience: str) -> Mapping[str, Any]:
        self.call = (token, issuer, audience)
        if self.fails:
            raise RuntimeError("sensitive SDK failure")
        return self.claims


def test_okta_provider_verifies_token_and_maps_normalized_context() -> None:
    verifier = StubOIDCVerifier({"sub": " manager-7 ", "store_id": "ny-brooklyn", "name": " Ada "})
    provider = OktaOIDCAuthProvider(
        verifier, issuer=" https://example.okta.com/oauth2/default/ ", audience="api://stores"
    )

    assert isinstance(provider, AuthProvider)
    assert provider.authenticate(" bearer-token ") == UserContext("manager-7", "NY-BROOKLYN", "Ada")
    assert verifier.call == (
        "bearer-token",
        "https://example.okta.com/oauth2/default",
        "api://stores",
    )


def test_okta_provider_supports_configurable_identity_claims() -> None:
    verifier = StubOIDCVerifier({"uid": "manager", "location": 42, "full_name": "Sam"})
    provider = OktaOIDCAuthProvider(
        verifier,
        issuer="https://example.okta.com",
        audience="stores",
        user_id_claim="uid",
        store_id_claim="location",
        display_name_claim="full_name",
    )

    assert provider.authenticate("token") == UserContext("manager", "42", "Sam")


@pytest.mark.parametrize(
    "claims",
    [
        {"store_id": "brooklyn"},
        {"sub": "manager"},
        {"sub": 7, "store_id": "brooklyn"},
        {"sub": "manager", "store_id": True},
        {"sub": "manager", "store_id": "brooklyn", "name": 7},
    ],
)
def test_okta_provider_rejects_missing_or_malformed_identity_claims(
    claims: Mapping[str, Any],
) -> None:
    provider = OktaOIDCAuthProvider(
        StubOIDCVerifier(claims), issuer="https://example.okta.com", audience="stores"
    )

    with pytest.raises(AuthenticationError, match="claim"):
        provider.authenticate("token")


def test_okta_provider_sanitizes_verifier_failures() -> None:
    provider = OktaOIDCAuthProvider(
        StubOIDCVerifier(fails=True), issuer="https://example.okta.com", audience="stores"
    )

    with pytest.raises(AuthenticationError, match="access token is invalid") as caught:
        provider.authenticate("token")

    assert "sensitive" not in str(caught.value)
