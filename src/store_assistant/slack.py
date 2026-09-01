"""Slack slash-command transport with request verification and identity mapping."""

from __future__ import annotations

import hmac
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from time import time
from types import MappingProxyType
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request

from store_assistant.auth import AuthenticationError
from store_assistant.retrieval import RetrievalError
from store_assistant.services import AssistantService


class SlackRequestError(ValueError):
    """Raised when a Slack request is invalid or cannot be trusted."""


class SlackSignatureVerifier:
    """Verify Slack's v0 HMAC signature and reject replayed requests."""

    def __init__(
        self,
        signing_secret: str,
        *,
        clock: Callable[[], float] = time,
        tolerance_seconds: int = 300,
    ) -> None:
        if not signing_secret.strip():
            raise ValueError("Slack signing secret must not be empty")
        if tolerance_seconds <= 0:
            raise ValueError("signature tolerance must be positive")
        self._secret = signing_secret.encode()
        self._clock = clock
        self._tolerance = tolerance_seconds

    def verify(self, body: bytes, timestamp: str | None, signature: str | None) -> None:
        try:
            request_time = int(timestamp or "")
        except ValueError as exc:
            raise SlackRequestError("invalid Slack request timestamp") from exc
        if abs(self._clock() - request_time) > self._tolerance:
            raise SlackRequestError("stale Slack request")
        if not signature:
            raise SlackRequestError("missing Slack request signature")
        base = b"v0:" + str(request_time).encode() + b":" + body
        expected = "v0=" + hmac.new(self._secret, base, sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise SlackRequestError("invalid Slack request signature")


@dataclass(frozen=True, slots=True)
class SlackCommandResponse:
    """Slack-compatible slash-command response payload."""

    text: str
    response_type: str = "ephemeral"

    def as_dict(self) -> dict[str, str]:
        return {"response_type": self.response_type, "text": self.text}


class SlackCommandHandler:
    """Translate trusted slash commands into assistant requests."""

    def __init__(
        self,
        assistant_service: AssistantService,
        verifier: SlackSignatureVerifier,
        access_tokens_by_slack_user: Mapping[str, str],
    ) -> None:
        tokens = {
            user_id.strip(): token.strip()
            for user_id, token in access_tokens_by_slack_user.items()
            if user_id.strip() and token.strip()
        }
        if not tokens:
            raise ValueError("at least one Slack user mapping is required")
        self._assistant = assistant_service
        self._verifier = verifier
        self._tokens = MappingProxyType(tokens)

    def handle(
        self,
        body: bytes,
        *,
        timestamp: str | None,
        signature: str | None,
    ) -> SlackCommandResponse:
        self._verifier.verify(body, timestamp, signature)
        try:
            fields = parse_qs(body.decode("utf-8"), strict_parsing=True)
        except (UnicodeDecodeError, ValueError) as exc:
            raise SlackRequestError("invalid Slack command payload") from exc
        user_id = _single_field(fields, "user_id")
        query = _single_field(fields, "text").strip()
        if not query:
            raise SlackRequestError("Slack command text must not be empty")
        try:
            access_token = self._tokens[user_id]
        except KeyError as exc:
            raise SlackRequestError("Slack user is not authorized") from exc
        try:
            result = self._assistant.ask(access_token, query)
        except (AuthenticationError, RetrievalError) as exc:
            raise SlackRequestError(str(exc)) from exc
        references = ", ".join(result.answer.citations) or "none"
        return SlackCommandResponse(
            text=(
                f"{result.answer.text}\n"
                f"Sources: {references}\n"
                f"Request ID: {result.request_id}"
            )
        )


def create_slack_router(handler: SlackCommandHandler) -> APIRouter:
    """Create the FastAPI endpoint configured as a Slack slash-command URL."""

    router = APIRouter()

    @router.post("/slack/commands")
    async def slack_command(request: Request) -> dict[str, str]:
        body = await request.body()
        try:
            response = handler.handle(
                body,
                timestamp=request.headers.get("X-Slack-Request-Timestamp"),
                signature=request.headers.get("X-Slack-Signature"),
            )
        except SlackRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return response.as_dict()

    return router


def _single_field(fields: Mapping[str, list[str]], name: str) -> str:
    values = fields.get(name)
    if values is None or len(values) != 1 or not values[0].strip():
        raise SlackRequestError(f"Slack command requires one {name}")
    return values[0].strip()
