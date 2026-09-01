from __future__ import annotations

import math

import pytest

from store_assistant.providers.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    LocalHashEmbeddingProvider,
)


def _similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_local_embeddings_are_deterministic_normalized_and_ordered() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=64)
    texts = ["oat milk cartons", "Tuesday produce delivery"]

    first = provider.embed(texts)
    second = provider.embed(texts)

    assert first == second
    assert len(first) == len(texts)
    assert all(len(vector) == provider.dimension for vector in first)
    assert all(math.isclose(math.sqrt(_similarity(vector, vector)), 1.0) for vector in first)


def test_local_embeddings_preserve_useful_lexical_similarity() -> None:
    provider = LocalHashEmbeddingProvider(dimensions=256)
    query, relevant, unrelated = provider.embed(
        ["Do we have oat milk?", "Oat milk inventory has 12 cartons", "Salmon delivery Friday"]
    )

    assert _similarity(query, relevant) > _similarity(query, unrelated)


def test_local_provider_satisfies_embedding_contract() -> None:
    assert isinstance(LocalHashEmbeddingProvider(), EmbeddingProvider)


@pytest.mark.parametrize("text", ["", "   ", "---"])
def test_local_provider_rejects_unembeddable_text(text: str) -> None:
    with pytest.raises(EmbeddingError):
        LocalHashEmbeddingProvider().embed([text])


def test_local_provider_validates_dimension_and_supports_empty_batches() -> None:
    with pytest.raises(EmbeddingError, match="at least 8"):
        LocalHashEmbeddingProvider(dimensions=4)

    assert LocalHashEmbeddingProvider().embed([]) == []
