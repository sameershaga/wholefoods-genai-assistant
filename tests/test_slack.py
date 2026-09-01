from __future__ import annotations

import hmac
import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlencode

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from store_assistant.answering import AnswerService
from store_assistant.auth import MockAuthProvider, UserContext
from store_assistant.feedback import SQLiteFeedbackRepository
from store_assistant.providers.embeddings import LocalHashEmbeddingProvider
from store_assistant.providers.llm import (
    GeneratedAnswer,
    LLMError,
    LLMProvider,
    LocalExtractiveLLM,
)
from store_assistant.providers.reranking import LocalLexicalReranker, RerankResult
from store_assistant.providers.vector_store import InMemoryVectorStore, VectorRecord
from store_assistant.request_logging import JSONLRequestLogRepository
from store_assistant.retrieval import RetrievalService
from store_assistant.services import AssistantService, AuthenticatedRetrievalService
from store_assistant.slack import (
    SlackCommandHandler,
    SlackRequestError,
    SlackSignatureVerifier,
    create_slack_router,
)

NOW = 1_700_000_000
SECRET = "local-signing-secret"


def _handler(
    tmp_path: Path,
    feedback: SQLiteFeedbackRepository | None = None,
    llm: LLMProvider | None = None,
) -> SlackCommandHandler:
    embeddings = LocalHashEmbeddingProvider(dimensions=32)
    store = InMemoryVectorStore(dimension=32)
    texts = ("Brooklyn has 12 cartons of oat milk", "Manhattan has 3 cartons of oat milk")
    store.upsert(
        [
            VectorRecord(f"delivery-{store_id.lower()}", text, vector, {"store_id": store_id})
            for store_id, text, vector in zip(
                ("BROOKLYN", "MANHATTAN"), texts, embeddings.embed(texts), strict=True
            )
        ]
    )
    auth = MockAuthProvider({"brooklyn-token": UserContext("manager-1", "BROOKLYN")})
    assistant = AssistantService(
        AuthenticatedRetrievalService(
            auth, RetrievalService(embeddings, store, LocalLexicalReranker())
        ),
        AnswerService(llm or LocalExtractiveLLM()),
        JSONLRequestLogRepository(tmp_path / "requests.jsonl"),
        request_id_factory=lambda: "request-1",
    )
    return SlackCommandHandler(
        assistant,
        SlackSignatureVerifier(SECRET, clock=lambda: NOW),
        {"U123": "brooklyn-token"},
        feedback,
    )


def _signed(body: bytes, timestamp: int = NOW) -> tuple[str, str]:
    stamp = str(timestamp)
    digest = hmac.new(SECRET.encode(), b"v0:" + stamp.encode() + b":" + body, sha256)
    return stamp, "v0=" + digest.hexdigest()


def test_slack_command_returns_cited_store_isolated_answer(tmp_path: Path) -> None:
    body = urlencode({"user_id": "U123", "text": "Do we have oat milk?"}).encode()
    timestamp, signature = _signed(body)

    response = _handler(tmp_path).handle(body, timestamp=timestamp, signature=signature)

    assert response.response_type == "ephemeral"
    assert "12 cartons" in response.text
    assert "3 cartons" not in response.text
    assert "Sources: delivery-brooklyn" in response.text
    assert "Request ID: request-1" in response.text
    assert response.as_dict()["blocks"]


@pytest.mark.parametrize(
    ("timestamp", "signature", "match"),
    [(str(NOW), "v0=wrong", "invalid"), (str(NOW - 301), "v0=wrong", "stale")],
)
def test_slack_command_rejects_untrusted_requests(
    tmp_path: Path, timestamp: str, signature: str, match: str
) -> None:
    with pytest.raises(SlackRequestError, match=match):
        _handler(tmp_path).handle(
            b"user_id=U123&text=oat+milk",
            timestamp=timestamp,
            signature=signature,
        )


def test_slack_command_rejects_unknown_user_and_empty_query(tmp_path: Path) -> None:
    for fields, match in [
        ({"user_id": "OTHER", "text": "oats"}, "not authorized"),
        ({"user_id": "U123", "text": " "}, "text"),
    ]:
        body = urlencode(fields).encode()
        timestamp, signature = _signed(body)
        with pytest.raises(SlackRequestError, match=match):
            _handler(tmp_path).handle(body, timestamp=timestamp, signature=signature)


def test_slack_command_rejects_oversized_query(tmp_path: Path) -> None:
    body = urlencode({"user_id": "U123", "text": "x" * 2_001}).encode()
    timestamp, signature = _signed(body)

    with pytest.raises(SlackRequestError, match="must not exceed 2000 characters"):
        _handler(tmp_path).handle(body, timestamp=timestamp, signature=signature)


@pytest.mark.parametrize("interaction", [False, True])
def test_slack_handler_rejects_oversized_request_body(tmp_path: Path, interaction: bool) -> None:
    handler = _handler(tmp_path)
    body = b"x" * (handler.MAX_BODY_BYTES + 1)
    timestamp, signature = _signed(body)

    with pytest.raises(SlackRequestError, match="must not exceed 16384 bytes"):
        if interaction:
            handler.handle_interaction(body, timestamp=timestamp, signature=signature)
        else:
            handler.handle(body, timestamp=timestamp, signature=signature)


def test_slack_router_exposes_signed_command_endpoint(tmp_path: Path) -> None:
    app = FastAPI()
    app.include_router(create_slack_router(_handler(tmp_path)))
    body = urlencode({"user_id": "U123", "text": "oat milk"}).encode()
    timestamp, signature = _signed(body)

    response = TestClient(app).post(
        "/slack/commands",
        content=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        },
    )

    assert response.status_code == 200
    assert response.json()["response_type"] == "ephemeral"
    assert "12 cartons" in response.json()["text"]


def test_slack_router_reports_llm_provider_failure_as_bad_gateway(tmp_path: Path) -> None:
    class FailingLLM:
        def generate(self, query: str, context: Sequence[RerankResult]) -> GeneratedAnswer:
            raise LLMError("answer provider unavailable")

    app = FastAPI()
    app.include_router(create_slack_router(_handler(tmp_path, llm=FailingLLM())))
    body = urlencode({"user_id": "U123", "text": "oat milk"}).encode()
    timestamp, signature = _signed(body)

    response = TestClient(app).post(
        "/slack/commands",
        content=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        },
    )

    assert response.status_code == 502
    assert response.json() == {"detail": "answer provider unavailable"}


def test_slack_interaction_persists_signed_feedback(tmp_path: Path) -> None:
    feedback = SQLiteFeedbackRepository(tmp_path / "feedback.sqlite3")
    app = FastAPI()
    app.include_router(create_slack_router(_handler(tmp_path, feedback)))
    payload = {
        "user": {"id": "U123"},
        "actions": [{"action_id": "feedback_down", "value": "request-1"}],
    }
    body = urlencode({"payload": json.dumps(payload)}).encode()
    timestamp, signature = _signed(body)

    response = TestClient(app).post(
        "/slack/interactions",
        content=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        },
    )

    assert response.status_code == 200
    assert response.json()["text"] == "Thanks for your feedback."
    saved = feedback.get(request_id="request-1", user_id="U123")
    assert saved is not None and saved.rating.value == "down"
    feedback.close()


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"user": {"id": ["U123"]}, "actions": []},
        {"user": {"id": "U123"}, "actions": {"action_id": "feedback_up"}},
        {
            "user": {"id": "U123"},
            "actions": [{"action_id": "feedback_up", "value": 123}],
        },
    ],
)
def test_slack_interaction_rejects_malformed_field_types(tmp_path: Path, payload: object) -> None:
    body = urlencode({"payload": json.dumps(payload)}).encode()
    timestamp, signature = _signed(body)

    with pytest.raises(SlackRequestError, match="invalid Slack interaction payload"):
        _handler(
            tmp_path, SQLiteFeedbackRepository(tmp_path / "feedback.sqlite3")
        ).handle_interaction(body, timestamp=timestamp, signature=signature)
