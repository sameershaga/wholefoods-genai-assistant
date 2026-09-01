"""Application services that propagate trusted request context."""

from __future__ import annotations

from typing import Mapping

from store_assistant.auth import AuthProvider, UserContext
from store_assistant.ingestion.models import MetadataValue
from store_assistant.ingestion.normalization import normalize_store_id
from store_assistant.providers.reranking import RerankResult
from store_assistant.retrieval import RetrievalError, RetrievalService


class AuthenticatedRetrievalService:
    """Authenticate requests and enforce identity-derived store isolation."""

    def __init__(self, auth_provider: AuthProvider, retrieval_service: RetrievalService) -> None:
        self._auth_provider = auth_provider
        self._retrieval_service = retrieval_service

    def retrieve(
        self,
        access_token: str,
        query: str,
        *,
        filters: Mapping[str, MetadataValue] | None = None,
    ) -> tuple[UserContext, list[RerankResult]]:
        context = self._auth_provider.authenticate(access_token)
        scoped_filters = dict(filters or {})
        requested_store = scoped_filters.get("store_id")
        if requested_store is not None:
            if not isinstance(requested_store, (str, int)):
                raise RetrievalError("store_id filter must be a string or integer")
            if normalize_store_id(requested_store) != context.store_id:
                raise RetrievalError("store_id filter conflicts with authenticated store")
        scoped_filters["store_id"] = context.store_id
        return context, self._retrieval_service.retrieve(query, filters=scoped_filters)

