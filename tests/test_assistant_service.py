from __future__ import annotations

from pathlib import Path

import pytest

from store_assistant.answering import AnswerService
from store_assistant.auth import MockAuthProvider, UserContext
from store_assistant.providers.embeddings import LocalHashEmbeddingProvider
from store_assistant.providers.llm import LocalExtractiveLLM
from store_assistant.providers.reranking import LocalLexicalReranker
from store_assistant.providers.vector_store import InMemoryVectorStore, VectorRecord
from store_assistant.query_routing import infer_source_type
from store_assistant.request_logging import JSONLRequestLogRepository
from store_assistant.retrieval import RetrievalService
from store_assistant.services import (
    AssistantService,
    AuthenticatedRetrievalService,
)


def _service(tmp_path: Path) -> tuple[AssistantService, JSONLRequestLogRepository]:
    embeddings = LocalHashEmbeddingProvider(dimensions=64)
    store = InMemoryVectorStore(dimension=64)
    texts = [
        "Brooklyn has 12 cartons of oat milk",
        "Manhattan has 3 cartons of oat milk",
    ]
    store.upsert(
        [
            VectorRecord(
                document_id=f"delivery-{store_id.lower()}",
                text=text,
                embedding=embedding,
                metadata={
                    "store_id": store_id,
                    "sku": "OAT001",
                    "source_type": "delivery_log",
                },
            )
            for text, embedding, store_id in zip(
                texts, embeddings.embed(texts), ("BROOKLYN", "MANHATTAN"), strict=True
            )
        ]
    )
    recipe_text = "Recipe: make overnight oats with oat milk"
    store.upsert(
        [
            VectorRecord(
                document_id="recipe-brooklyn",
                text=recipe_text,
                embedding=embeddings.embed([recipe_text])[0],
                metadata={"store_id": "BROOKLYN", "sku": "OAT001", "source_type": "recipe"},
            )
        ]
    )
    retrieval = AuthenticatedRetrievalService(
        MockAuthProvider({"token": UserContext("manager-1", "brooklyn")}),
        RetrievalService(embeddings, store, LocalLexicalReranker()),
        source_router=infer_source_type,
    )
    logs = JSONLRequestLogRepository(tmp_path / "requests.jsonl")
    ticks = iter((10.0, 10.125))
    return (
        AssistantService(
            retrieval,
            AnswerService(LocalExtractiveLLM()),
            logs,
            input_cost_per_million_tokens=2.0,
            output_cost_per_million_tokens=4.0,
            request_id_factory=lambda: "request-1",
            clock=lambda: next(ticks),
        ),
        logs,
    )


def test_ask_enforces_store_scope_answers_and_persists_complete_log(tmp_path: Path) -> None:
    service, logs = _service(tmp_path)

    response = service.ask("token", "Do we have oat milk?", filters={"sku": "OAT001"})

    assert response.request_id == "request-1"
    assert response.user.store_id == "BROOKLYN"
    assert "12 cartons" in response.answer.text
    assert "3 cartons" not in response.answer.text
    assert response.answer.citations == ("delivery-brooklyn",)
    [record] = logs.read_all()
    assert record.request_id == response.request_id
    assert record.user_id == "manager-1"
    assert record.store_id == "BROOKLYN"
    assert record.latency_ms == 125.0
    assert record.retrieved_documents[0].document_id == "delivery-brooklyn"
    assert record.retrieved_documents[0].retrieval_score > 0
    assert record.retrieved_documents[0].reranking_score > 0
    expected_cost = (
        response.answer.prompt_tokens * 2.0 + response.answer.completion_tokens * 4.0
    ) / 1_000_000
    assert record.estimated_cost_usd == expected_cost
    assert record.final_answer == response.answer.text


def test_ask_logs_safe_no_result_answer(tmp_path: Path) -> None:
    service, logs = _service(tmp_path)

    response = service.ask("token", "Do we have oat milk?", filters={"sku": "MISSING"})

    assert response.answer.citations == ()
    assert response.answer.model == "none"
    [record] = logs.read_all()
    assert record.retrieved_documents == ()
    assert record.estimated_cost_usd == 0


def test_ask_routes_inventory_away_from_recipe_content(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)

    response = service.ask("token", "Do we have oat milk?", filters={"sku": "OAT001"})

    assert response.answer.citations == ("delivery-brooklyn",)
    assert "12 cartons" in response.answer.text


def test_service_rejects_negative_token_prices(tmp_path: Path) -> None:
    service, logs = _service(tmp_path)

    with pytest.raises(ValueError, match="token prices"):
        AssistantService(
            service._retrieval_service,  # type: ignore[attr-defined]
            service._answer_service,  # type: ignore[attr-defined]
            logs,
            input_cost_per_million_tokens=-1,
        )
