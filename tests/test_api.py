from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from fastapi.testclient import TestClient

from store_assistant.answering import AnswerService
from store_assistant.api import create_app
from store_assistant.auth import MockAuthProvider, UserContext
from store_assistant.feedback import SQLiteFeedbackRepository
from store_assistant.providers.embeddings import LocalHashEmbeddingProvider
from store_assistant.providers.llm import GeneratedAnswer, LLMError, LLMProvider, LocalExtractiveLLM
from store_assistant.providers.reranking import LocalLexicalReranker, RerankResult
from store_assistant.providers.vector_store import InMemoryVectorStore, VectorRecord
from store_assistant.request_logging import JSONLRequestLogRepository
from store_assistant.retrieval import RetrievalService
from store_assistant.services import AssistantService, AuthenticatedRetrievalService


def _client(
    tmp_path: Path, llm: LLMProvider | None = None, *, enable_demo: bool = False
) -> tuple[TestClient, SQLiteFeedbackRepository]:
    embeddings = LocalHashEmbeddingProvider(dimensions=32)
    vector_store = InMemoryVectorStore(dimension=32)
    texts = (
        "Brooklyn has 12 cartons of oat milk",
        "Manhattan has 3 cartons of oat milk",
    )
    vector_store.upsert(
        [
            VectorRecord(
                document_id=f"delivery-{store.lower()}",
                text=text,
                embedding=embedding,
                metadata={"store_id": store, "sku": "OAT001"},
            )
            for store, text, embedding in zip(
                ("BROOKLYN", "MANHATTAN"), texts, embeddings.embed(texts), strict=True
            )
        ]
    )
    auth = MockAuthProvider({"brooklyn-token": UserContext("manager-1", "BROOKLYN")})
    retrieval = AuthenticatedRetrievalService(
        auth, RetrievalService(embeddings, vector_store, LocalLexicalReranker())
    )
    assistant = AssistantService(
        retrieval,
        AnswerService(llm or LocalExtractiveLLM()),
        JSONLRequestLogRepository(tmp_path / "requests.jsonl"),
        request_id_factory=lambda: "request-1",
    )
    feedback = SQLiteFeedbackRepository(tmp_path / "feedback.sqlite3")
    demo_tokens = {"BROOKLYN": "brooklyn-token"} if enable_demo else None
    app = create_app(assistant, auth, feedback, demo_store_tokens=demo_tokens)
    return TestClient(app), feedback


def test_health_is_public(tmp_path: Path) -> None:
    client, feedback = _client(tmp_path)
    try:
        assert client.get("/health").json() == {"status": "ok"}
    finally:
        feedback.close()


def test_query_propagates_auth_and_never_leaks_another_store(tmp_path: Path) -> None:
    client, feedback = _client(tmp_path)
    try:
        response = client.post(
            "/v1/query",
            headers={"Authorization": "Bearer brooklyn-token"},
            json={"query": "Do we have oat milk?", "filters": {"sku": "OAT001"}},
        )
        assert response.status_code == 200
        assert response.json()["request_id"] == "request-1"
        assert response.json()["store_id"] == "BROOKLYN"
        assert "12 cartons" in response.json()["answer"]
        assert "3 cartons" not in response.json()["answer"]
        assert response.json()["citations"] == ["delivery-brooklyn"]
    finally:
        feedback.close()


def test_query_reports_auth_validation_and_store_conflict_errors(tmp_path: Path) -> None:
    client, feedback = _client(tmp_path)
    try:
        assert client.post("/v1/query", json={"query": "oat milk"}).status_code == 401
        assert (
            client.post(
                "/v1/query",
                headers={"Authorization": "Bearer invalid"},
                json={"query": "oat milk"},
            ).status_code
            == 401
        )
        conflict = client.post(
            "/v1/query",
            headers={"Authorization": "Bearer brooklyn-token"},
            json={"query": "oat milk", "filters": {"store_id": "MANHATTAN"}},
        )
        assert conflict.status_code == 400
    finally:
        feedback.close()


def test_feedback_uses_authenticated_user_and_persists_vote(tmp_path: Path) -> None:
    client, feedback = _client(tmp_path)
    try:
        response = client.post(
            "/v1/feedback",
            headers={"Authorization": "Bearer brooklyn-token"},
            json={"request_id": "request-1", "rating": "up", "comment": "Useful"},
        )
        assert response.status_code == 200
        assert response.json() == {"request_id": "request-1", "rating": "up"}
        saved = feedback.get(request_id="request-1", user_id="manager-1")
        assert saved is not None and saved.comment == "Useful"
    finally:
        feedback.close()


def test_malformed_payload_returns_validation_error(tmp_path: Path) -> None:
    client, feedback = _client(tmp_path)
    try:
        response = client.post(
            "/v1/query",
            headers={"Authorization": "Bearer brooklyn-token"},
            json={"query": "", "unexpected": True},
        )
        assert response.status_code == 422
    finally:
        feedback.close()


def test_query_reports_llm_provider_failure_as_bad_gateway(tmp_path: Path) -> None:
    class FailingLLM:
        def generate(self, query: str, context: Sequence[RerankResult]) -> GeneratedAnswer:
            raise LLMError("answer provider unavailable")

    client, feedback = _client(tmp_path, FailingLLM())
    try:
        response = client.post(
            "/v1/query",
            headers={"Authorization": "Bearer brooklyn-token"},
            json={"query": "Do we have oat milk?"},
        )
        assert response.status_code == 502
        assert response.json() == {"detail": "answer provider unavailable"}
    finally:
        feedback.close()


def test_demo_adapter_lists_allowlisted_stores_and_queries_with_trusted_identity(
    tmp_path: Path,
) -> None:
    client, feedback = _client(tmp_path, enable_demo=True)
    try:
        stores = client.get("/v1/demo/stores")
        assert stores.status_code == 200
        assert stores.json() == [{"store_id": "BROOKLYN", "label": "Brooklyn"}]

        response = client.post(
            "/v1/demo/query",
            json={"store_id": "BROOKLYN", "query": "Do we have oat milk?"},
        )
        assert response.status_code == 200
        assert response.json()["store_id"] == "BROOKLYN"
        assert "12 cartons" in response.json()["answer"]

        vote = client.post(
            "/v1/demo/feedback",
            json={
                "store_id": "BROOKLYN",
                "request_id": response.json()["request_id"],
                "rating": "up",
                "comment": "Grounded and useful",
            },
        )
        assert vote.status_code == 200
        assert vote.json() == {"request_id": "request-1", "rating": "up"}
        saved = feedback.get(request_id="request-1", user_id="manager-1")
        assert saved is not None and saved.comment == "Grounded and useful"
    finally:
        feedback.close()


def test_demo_adapter_rejects_stores_outside_server_allowlist(tmp_path: Path) -> None:
    client, feedback = _client(tmp_path, enable_demo=True)
    try:
        response = client.post(
            "/v1/demo/query",
            json={"store_id": "MANHATTAN", "query": "Do we have oat milk?"},
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "Synthetic demo store not found"}
        feedback_response = client.post(
            "/v1/demo/feedback",
            json={"store_id": "MANHATTAN", "request_id": "request-1", "rating": "up"},
        )
        assert feedback_response.status_code == 404
        assert feedback_response.json() == {"detail": "Synthetic demo store not found"}
    finally:
        feedback.close()


def test_demo_adapter_rejects_malformed_query_inputs(tmp_path: Path) -> None:
    client, feedback = _client(tmp_path, enable_demo=True)
    try:
        payloads = [
            {"store_id": "BROOKLYN", "query": "   "},
            {"store_id": "BROOKLYN", "query": "x" * 2_001},
            {"store_id": "BROOKLYN"},
            {"store_id": "BROOKLYN", "query": "oat milk", "unexpected": True},
        ]
        for payload in payloads:
            assert client.post("/v1/demo/query", json=payload).status_code == 422

        invalid_json = client.post(
            "/v1/demo/query",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert invalid_json.status_code == 422
    finally:
        feedback.close()


def test_demo_adapter_rejects_malformed_feedback_inputs(tmp_path: Path) -> None:
    client, feedback = _client(tmp_path, enable_demo=True)
    try:
        payloads = [
            {"store_id": "BROOKLYN", "request_id": "   ", "rating": "up"},
            {"store_id": "BROOKLYN", "request_id": "x" * 201, "rating": "up"},
            {"store_id": "BROOKLYN", "request_id": "request-1", "rating": "sideways"},
            {
                "store_id": "BROOKLYN",
                "request_id": "request-1",
                "rating": "up",
                "comment": "x" * 2_001,
            },
            {
                "store_id": "BROOKLYN",
                "request_id": "request-1",
                "rating": "up",
                "unexpected": True,
            },
        ]
        for payload in payloads:
            assert client.post("/v1/demo/feedback", json=payload).status_code == 422
    finally:
        feedback.close()


def test_demo_adapter_is_absent_unless_explicitly_enabled(tmp_path: Path) -> None:
    client, feedback = _client(tmp_path)
    try:
        assert client.get("/v1/demo/stores").status_code == 404
        assert client.post("/v1/demo/query", json={}).status_code == 404
        assert client.post("/v1/demo/feedback", json={}).status_code == 404
    finally:
        feedback.close()
