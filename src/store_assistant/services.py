"""Application services that propagate trusted request context."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from store_assistant.answering import Answer, AnswerService
from store_assistant.auth import AuthProvider, UserContext
from store_assistant.ingestion.models import MetadataValue
from store_assistant.ingestion.normalization import normalize_store_id
from store_assistant.providers.reranking import RerankResult
from store_assistant.request_logging import (
    RequestLog,
    RequestLogRepository,
    RetrievedDocumentLog,
)
from store_assistant.retrieval import RetrievalError, RetrievalService


class AuthenticatedRetrievalService:
    """Authenticate requests and enforce identity-derived store isolation."""

    def __init__(
        self,
        auth_provider: AuthProvider,
        retrieval_service: RetrievalService,
        *,
        source_router: Callable[[str], str | None] | None = None,
    ) -> None:
        self._auth_provider = auth_provider
        self._retrieval_service = retrieval_service
        self._source_router = source_router

    def retrieve(
        self,
        access_token: str,
        query: str,
        *,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> tuple[UserContext, list[RerankResult]]:
        context = self._auth_provider.authenticate(access_token)
        scoped_filters = dict(filters or {})
        if "source_type" not in scoped_filters and self._source_router is not None:
            inferred_source = self._source_router(query)
            if inferred_source is not None:
                scoped_filters["source_type"] = inferred_source
        requested_store = scoped_filters.get("store_id")
        if requested_store is not None:
            if not isinstance(requested_store, (str, int)):
                raise RetrievalError("store_id filter must be a string or integer")
            if normalize_store_id(requested_store) != context.store_id:
                raise RetrievalError("store_id filter conflicts with authenticated store")
        scoped_filters["store_id"] = context.store_id
        return context, self._retrieval_service.retrieve(query, filters=scoped_filters)


@dataclass(frozen=True, slots=True)
class AssistantResponse:
    """Transport-neutral result returned to HTTP and Slack adapters."""

    request_id: str
    user: UserContext
    answer: Answer


class AssistantService:
    """Run one authenticated, grounded request and persist its telemetry."""

    def __init__(
        self,
        retrieval_service: AuthenticatedRetrievalService,
        answer_service: AnswerService,
        log_repository: RequestLogRepository,
        *,
        input_cost_per_million_tokens: float = 0.0,
        output_cost_per_million_tokens: float = 0.0,
        request_id_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if input_cost_per_million_tokens < 0 or output_cost_per_million_tokens < 0:
            raise ValueError("token prices must not be negative")
        self._retrieval_service = retrieval_service
        self._answer_service = answer_service
        self._log_repository = log_repository
        self._input_cost = input_cost_per_million_tokens
        self._output_cost = output_cost_per_million_tokens
        self._request_id_factory = request_id_factory or (lambda: str(uuid4()))
        self._clock = clock

    def ask(
        self,
        access_token: str,
        query: str,
        *,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> AssistantResponse:
        started_at = self._clock()
        user, context = self._retrieval_service.retrieve(access_token, query, filters=filters)
        answer = self._answer_service.answer(query, context)
        request_id = self._request_id_factory().strip()
        if not request_id:
            raise ValueError("request ID factory returned an empty value")
        elapsed_ms = max(0.0, (self._clock() - started_at) * 1000)
        cost = (
            answer.prompt_tokens * self._input_cost + answer.completion_tokens * self._output_cost
        ) / 1_000_000
        self._log_repository.append(
            RequestLog.create(
                request_id=request_id,
                query=query,
                user_id=user.user_id,
                store_id=user.store_id,
                retrieved_documents=[
                    RetrievedDocumentLog(
                        document_id=item.document_id,
                        retrieval_score=item.retrieval_score,
                        reranking_score=item.reranking_score,
                    )
                    for item in context
                ],
                model=answer.model,
                prompt_tokens=answer.prompt_tokens,
                completion_tokens=answer.completion_tokens,
                estimated_cost_usd=cost,
                latency_ms=elapsed_ms,
                final_answer=answer.text,
            )
        )
        return AssistantResponse(request_id=request_id, user=user, answer=answer)
