"""Slack slash-command transport with request verification and identity mapping."""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from time import time
from types import MappingProxyType
from urllib.parse import parse_qs

from fastapi import APIRouter, HTTPException, Request

from store_assistant.auth import AuthenticationError
from store_assistant.feedback import FeedbackError, FeedbackRating, FeedbackRepository
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

    request_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"response_type": self.response_type, "text": self.text}
        if self.request_id is not None:
            payload["blocks"] = [
                {"type": "section", "text": {"type": "mrkdwn", "text": self.text}},
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "👍"},
                            "action_id": "feedback_up",
                            "value": self.request_id,
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "👎"},
                            "action_id": "feedback_down",
                            "value": self.request_id,
                        },
                    ],
                },
            ]
        return payload


class SlackCommandHandler:
    """Translate trusted slash commands into assistant requests."""

    MAX_QUERY_LENGTH = 2_000

    def __init__(
        self,
        assistant_service: AssistantService,
        verifier: SlackSignatureVerifier,
        access_tokens_by_slack_user: Mapping[str, str],
        feedback_repository: FeedbackRepository | None = None,
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
        self._feedback = feedback_repository

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
        if len(query) > self.MAX_QUERY_LENGTH:
            raise SlackRequestError(
                f"Slack command text must not exceed {self.MAX_QUERY_LENGTH} characters"
            )
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
            text=(f"{result.answer.text}\nSources: {references}\nRequest ID: {result.request_id}"),
            request_id=result.request_id,
        )

    def handle_interaction(
        self, body: bytes, *, timestamp: str | None, signature: str | None
    ) -> SlackCommandResponse:
        """Verify and persist a Slack Block Kit thumbs interaction."""
        self._verifier.verify(body, timestamp, signature)
        if self._feedback is None:
            raise SlackRequestError("Slack feedback is not configured")
        try:
            fields = parse_qs(body.decode("utf-8"), strict_parsing=True)
            payload = json.loads(_single_field(fields, "payload"))
            slack_user_id = payload["user"]["id"]
            actions = payload["actions"]
            if len(actions) != 1:
                raise ValueError("interaction must contain one action")
            action = actions[0]
            action_id = action["action_id"]
            request_id = action["value"]
            rating = {
                "feedback_up": FeedbackRating.UP,
                "feedback_down": FeedbackRating.DOWN,
            }[action_id]
        except (KeyError, TypeError, IndexError, UnicodeDecodeError, ValueError) as exc:
            raise SlackRequestError("invalid Slack interaction payload") from exc
        if slack_user_id not in self._tokens:
            raise SlackRequestError("Slack user is not authorized")
        try:
            self._feedback.save(request_id=request_id, user_id=slack_user_id, rating=rating)
        except FeedbackError as exc:
            raise SlackRequestError(str(exc)) from exc
        return SlackCommandResponse(text="Thanks for your feedback.")


def create_slack_router(handler: SlackCommandHandler) -> APIRouter:
    """Create the FastAPI endpoint configured as a Slack slash-command URL."""

    router = APIRouter()

    @router.post("/slack/commands")
    async def slack_command(request: Request) -> dict[str, object]:
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

    @router.post("/slack/interactions")
    async def slack_interaction(request: Request) -> dict[str, object]:
        body = await request.body()
        try:
            response = handler.handle_interaction(
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
