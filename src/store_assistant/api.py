"""FastAPI transport adapter for the store assistant application services."""

from collections.abc import Mapping
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from store_assistant.auth import AuthenticationError, AuthProvider, UserContext
from store_assistant.feedback import (
    FeedbackError,
    FeedbackRating,
    FeedbackRepository,
)
from store_assistant.providers.llm import LLMError
from store_assistant.retrieval import RetrievalError
from store_assistant.services import AssistantService
from store_assistant.slack import SlackCommandHandler, create_slack_router


class QueryRequest(BaseModel):
    """Validated HTTP query payload."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2_000)
    filters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Grounded answer returned to an HTTP client."""

    request_id: str
    store_id: str
    answer: str
    citations: list[str]
    model: str


class DemoStoreResponse(BaseModel):
    """Synthetic store option exposed by the explicitly enabled demo adapter."""

    store_id: str
    label: str


class DemoQueryRequest(QueryRequest):
    """Public-demo query whose store must match a server-side allowlist."""

    store_id: str = Field(min_length=1, max_length=100)


class FeedbackRequest(BaseModel):
    """Validated feedback payload."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=2_000)


class DemoFeedbackRequest(FeedbackRequest):
    """Public-demo feedback scoped through a server-side synthetic identity."""

    store_id: str = Field(min_length=1, max_length=100)


class FeedbackResponse(BaseModel):
    """Acknowledgement of persisted feedback."""

    request_id: str
    rating: FeedbackRating


def create_app(
    assistant_service: AssistantService,
    auth_provider: AuthProvider,
    feedback_repository: FeedbackRepository,
    *,
    slack_handler: SlackCommandHandler | None = None,
    demo_store_tokens: Mapping[str, str] | None = None,
) -> FastAPI:
    """Create an HTTP adapter with dependencies supplied by the composition root."""

    app = FastAPI(title="Store Operations Assistant", version="0.1.0")
    if slack_handler is not None:
        app.include_router(create_slack_router(slack_handler))
    demo_tokens = dict(demo_store_tokens or {})

    def access_token(authorization: str | None = Header(default=None)) -> str:
        scheme, _, token = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer access token is required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token.strip()

    def current_user(token: Annotated[str, Depends(access_token)]) -> UserContext:
        try:
            return auth_provider.authenticate(token)
        except AuthenticationError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/query", response_model=QueryResponse)
    def query(payload: QueryRequest, token: Annotated[str, Depends(access_token)]) -> QueryResponse:
        return run_query(token, payload)

    def run_query(token: str, payload: QueryRequest) -> QueryResponse:
        try:
            result = assistant_service.ask(token, payload.query.strip(), filters=payload.filters)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        except RetrievalError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return QueryResponse(
            request_id=result.request_id,
            store_id=result.user.store_id,
            answer=result.answer.text,
            citations=list(result.answer.citations),
            model=result.answer.model,
        )

    if demo_tokens:

        @app.get("/v1/demo/stores", response_model=list[DemoStoreResponse])
        def demo_stores() -> list[DemoStoreResponse]:
            return [
                DemoStoreResponse(store_id=store_id, label=_demo_store_label(store_id))
                for store_id in demo_tokens
            ]

        @app.post("/v1/demo/query", response_model=QueryResponse)
        def demo_query(payload: DemoQueryRequest) -> QueryResponse:
            token = demo_tokens.get(payload.store_id)
            if token is None:
                raise HTTPException(status_code=404, detail="Synthetic demo store not found")
            return run_query(token, payload)

        @app.post("/v1/demo/feedback", response_model=FeedbackResponse)
        def demo_feedback(payload: DemoFeedbackRequest) -> FeedbackResponse:
            token = demo_tokens.get(payload.store_id)
            if token is None:
                raise HTTPException(status_code=404, detail="Synthetic demo store not found")
            try:
                user = auth_provider.authenticate(token)
            except AuthenticationError as exc:
                raise HTTPException(status_code=401, detail=str(exc)) from exc
            return save_feedback(payload, user)

    @app.post("/v1/feedback", response_model=FeedbackResponse)
    def feedback(
        payload: FeedbackRequest,
        user: Annotated[UserContext, Depends(current_user)],
    ) -> FeedbackResponse:
        return save_feedback(payload, user)

    def save_feedback(payload: FeedbackRequest, user: UserContext) -> FeedbackResponse:
        try:
            saved = feedback_repository.save(
                request_id=payload.request_id,
                user_id=user.user_id,
                rating=payload.rating,
                comment=payload.comment,
            )
        except FeedbackError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FeedbackResponse(request_id=saved.request_id, rating=saved.rating)

    return app


def _demo_store_label(store_id: str) -> str:
    """Create a human-readable label without adding store facts beyond its identifier."""
    return " ".join(part.capitalize() for part in store_id.replace("_", "-").split("-"))
