"""Local composition root for a fully offline store assistant application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from store_assistant.answering import AnswerService
from store_assistant.api import create_app
from store_assistant.auth import MockAuthProvider, UserContext
from store_assistant.feedback import SQLiteFeedbackRepository
from store_assistant.ingestion.delivery_logs import ingest_delivery_logs
from store_assistant.providers.embeddings import LocalHashEmbeddingProvider
from store_assistant.providers.llm import LocalExtractiveLLM
from store_assistant.providers.reranking import LocalLexicalReranker
from store_assistant.providers.vector_store import InMemoryVectorStore, VectorRecord
from store_assistant.request_logging import JSONLRequestLogRepository
from store_assistant.retrieval import RetrievalService
from store_assistant.services import AssistantService, AuthenticatedRetrievalService


@dataclass(frozen=True, slots=True)
class LocalSettings:
    """Filesystem and identity settings used by the offline runtime."""

    delivery_logs_path: Path = Path("data/delivery_logs.json")
    state_directory: Path = Path(".local")
    embedding_dimensions: int = 384


def create_local_app(settings: LocalSettings | None = None) -> FastAPI:
    """Build an API using synthetic data and providers requiring no paid services."""
    config = settings or LocalSettings()
    chunks = ingest_delivery_logs(config.delivery_logs_path)
    embeddings = LocalHashEmbeddingProvider(config.embedding_dimensions)
    vector_store = InMemoryVectorStore(embeddings.dimension)
    vectors = embeddings.embed([chunk.text for chunk in chunks])
    vector_store.upsert(
        [
            VectorRecord(chunk.document_id, chunk.text, vector, chunk.metadata)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )

    auth = MockAuthProvider(
        {
            "local-brooklyn-token": UserContext("brooklyn-manager", "brooklyn-01"),
            "local-manhattan-token": UserContext("manhattan-manager", "manhattan-01"),
            "local-queens-token": UserContext("queens-manager", "queens-01"),
        }
    )
    retrieval = RetrievalService(embeddings, vector_store, LocalLexicalReranker())
    assistant = AssistantService(
        AuthenticatedRetrievalService(auth, retrieval),
        AnswerService(LocalExtractiveLLM()),
        JSONLRequestLogRepository(config.state_directory / "requests.jsonl"),
    )
    feedback = SQLiteFeedbackRepository(config.state_directory / "feedback.sqlite3")
    return create_app(assistant, auth, feedback)


app = create_local_app()
