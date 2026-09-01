"""Reranking provider contract and a deterministic local implementation."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from store_assistant.ingestion.models import MetadataValue

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class RerankingError(ValueError):
    """Raised when a reranking request is invalid."""


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """A retrieved document offered to a reranker."""

    document_id: str
    text: str
    retrieval_score: float
    metadata: Mapping[str, MetadataValue]


@dataclass(frozen=True, slots=True)
class RerankResult:
    """A candidate with both retrieval and reranking scores."""

    document_id: str
    text: str
    retrieval_score: float
    reranking_score: float
    metadata: Mapping[str, MetadataValue]


@runtime_checkable
class Reranker(Protocol):
    """Boundary implemented by local and hosted cross-encoder rerankers."""

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_k: int
    ) -> list[RerankResult]:
        """Return the strongest candidates in descending relevance order."""


@dataclass(frozen=True, slots=True)
class LocalLexicalReranker:
    """Offline query-document scorer used in local development and tests.

    It approximates a reranking stage using query token coverage and ordered
    token-pair matches. Production deployments can replace it with a
    cross-encoder without changing retrieval orchestration.
    """

    def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], *, top_k: int
    ) -> list[RerankResult]:
        if not isinstance(query, str) or not query.strip():
            raise RerankingError("query must be a non-empty string")
        if top_k < 1:
            raise RerankingError("top_k must be positive")

        query_tokens = _TOKEN_PATTERN.findall(query.casefold())
        if not query_tokens:
            raise RerankingError("query must contain at least one alphanumeric token")
        query_terms = set(query_tokens)
        query_pairs = set(zip(query_tokens, query_tokens[1:], strict=False))

        results: list[RerankResult] = []
        seen_ids: set[str] = set()
        for position, candidate in enumerate(candidates):
            self._validate_candidate(candidate, position=position, seen_ids=seen_ids)
            document_tokens = _TOKEN_PATTERN.findall(candidate.text.casefold())
            document_terms = set(document_tokens)
            term_coverage = len(query_terms & document_terms) / len(query_terms)
            pair_coverage = (
                len(query_pairs & set(zip(document_tokens, document_tokens[1:], strict=False)))
                / len(query_pairs)
                if query_pairs
                else 0.0
            )
            score = (0.8 * term_coverage) + (0.2 * pair_coverage)
            results.append(
                RerankResult(
                    document_id=candidate.document_id,
                    text=candidate.text,
                    retrieval_score=candidate.retrieval_score,
                    reranking_score=score,
                    metadata=dict(candidate.metadata),
                )
            )

        results.sort(
            key=lambda result: (
                -result.reranking_score,
                -result.retrieval_score,
                result.document_id,
            )
        )
        return results[:top_k]

    @staticmethod
    def _validate_candidate(
        candidate: RerankCandidate, *, position: int, seen_ids: set[str]
    ) -> None:
        if not candidate.document_id.strip():
            raise RerankingError(f"candidate at position {position} has an empty document ID")
        if candidate.document_id in seen_ids:
            raise RerankingError(f"duplicate candidate document ID {candidate.document_id!r}")
        seen_ids.add(candidate.document_id)
        if not candidate.text.strip():
            raise RerankingError(f"candidate {candidate.document_id!r} has empty text")
        if not math.isfinite(candidate.retrieval_score):
            raise RerankingError(
                f"candidate {candidate.document_id!r} has a non-finite retrieval score"
            )
