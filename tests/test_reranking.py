from __future__ import annotations

import pytest

from store_assistant.providers.reranking import (
    LocalLexicalReranker,
    RerankCandidate,
    Reranker,
    RerankingError,
)


def _candidate(document_id: str, text: str, score: float = 0.5) -> RerankCandidate:
    return RerankCandidate(document_id, text, score, {"store_id": "BROOKLYN"})


def test_local_reranker_promotes_query_relevance_over_vector_score() -> None:
    reranker = LocalLexicalReranker()
    candidates = [
        _candidate("salmon", "Salmon delivery is scheduled Friday", score=0.99),
        _candidate("oat", "Oat milk inventory has 12 cartons", score=0.40),
    ]

    results = reranker.rerank("Do we have oat milk inventory?", candidates, top_k=2)

    assert [result.document_id for result in results] == ["oat", "salmon"]
    assert results[0].reranking_score > results[1].reranking_score
    assert results[0].retrieval_score == 0.40


def test_reranker_respects_top_k_and_copies_metadata() -> None:
    reranker = LocalLexicalReranker()
    metadata = {"store_id": "BROOKLYN"}
    candidate = RerankCandidate("oat", "Oat milk is available", 0.8, metadata)

    results = reranker.rerank("oat milk", [candidate], top_k=1)
    metadata["store_id"] = "MANHATTAN"

    assert len(results) == 1
    assert results[0].metadata["store_id"] == "BROOKLYN"
    assert 0.0 <= results[0].reranking_score <= 1.0


def test_retrieval_score_and_document_id_make_ties_deterministic() -> None:
    reranker = LocalLexicalReranker()
    candidates = [
        _candidate("b", "oat milk", score=0.5),
        _candidate("c", "oat milk", score=0.8),
        _candidate("a", "oat milk", score=0.5),
    ]

    results = reranker.rerank("oat milk", candidates, top_k=3)

    assert [result.document_id for result in results] == ["c", "a", "b"]


def test_local_reranker_satisfies_provider_contract_and_handles_empty_candidates() -> None:
    reranker = LocalLexicalReranker()

    assert isinstance(reranker, Reranker)
    assert reranker.rerank("oat milk", [], top_k=5) == []


@pytest.mark.parametrize(
    ("query", "candidates", "top_k", "message"),
    [
        ("", [], 1, "query"),
        ("!!!", [], 1, "alphanumeric"),
        ("oat", [], 0, "top_k"),
        ("oat", [_candidate("", "text")], 1, "document ID"),
        ("oat", [_candidate("a", "")], 1, "empty text"),
        ("oat", [_candidate("a", "text", float("nan"))], 1, "non-finite"),
        ("oat", [_candidate("a", "one"), _candidate("a", "two")], 2, "duplicate"),
    ],
)
def test_invalid_reranking_requests_are_rejected(
    query: str, candidates: list[RerankCandidate], top_k: int, message: str
) -> None:
    with pytest.raises(RerankingError, match=message):
        LocalLexicalReranker().rerank(query, candidates, top_k=top_k)
