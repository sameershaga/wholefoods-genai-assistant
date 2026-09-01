"""Local composition root for a fully offline store assistant application."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from store_assistant.answering import AnswerService
from store_assistant.api import create_app
from store_assistant.auth import MockAuthProvider, UserContext
from store_assistant.feedback import SQLiteFeedbackRepository
from store_assistant.ingestion.contracts import ingest_supplier_contracts
from store_assistant.ingestion.delivery_logs import ingest_delivery_logs
from store_assistant.ingestion.recipes import ingest_recipe_html
from store_assistant.providers.embeddings import (
    AmazonTitanEmbeddingProvider,
    EmbeddingProvider,
    LocalHashEmbeddingProvider,
)
from store_assistant.providers.llm import LocalExtractiveLLM
from store_assistant.providers.reranking import LocalLexicalReranker
from store_assistant.providers.vector_store import InMemoryVectorStore, VectorRecord
from store_assistant.query_routing import infer_source_type
from store_assistant.request_logging import JSONLRequestLogRepository
from store_assistant.retrieval import RetrievalService
from store_assistant.services import AssistantService, AuthenticatedRetrievalService
from store_assistant.slack import SlackCommandHandler, SlackSignatureVerifier


@dataclass(frozen=True, slots=True)
class LocalSettings:
    """Filesystem and identity settings used by the offline runtime."""

    delivery_logs_path: Path = Path("data/delivery_logs.json")
    recipe_path: Path = Path("data/recipes/oat_milk_overnight_oats.html")
    supplier_contract_path: Path | None = None
    state_directory: Path = Path(".local")
    embedding_dimensions: int = 384
    embedding_provider: str = "local"
    aws_region: str = "us-east-1"
    bedrock_embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    slack_signing_secret: str | None = None
    slack_user_tokens: Mapping[str, str] | None = None

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> LocalSettings:
        """Load local runtime settings from environment variables."""
        values = os.environ if environ is None else environ
        raw_dimensions = values.get("STORE_ASSISTANT_EMBEDDING_DIMENSIONS", "384")
        try:
            dimensions = int(raw_dimensions)
        except ValueError as exc:
            raise ValueError("STORE_ASSISTANT_EMBEDDING_DIMENSIONS must be an integer") from exc
        if dimensions <= 0:
            raise ValueError("STORE_ASSISTANT_EMBEDDING_DIMENSIONS must be positive")
        embedding_provider = values.get("STORE_ASSISTANT_EMBEDDING_PROVIDER", "local").strip()
        if embedding_provider not in {"local", "bedrock"}:
            raise ValueError("STORE_ASSISTANT_EMBEDDING_PROVIDER must be local or bedrock")
        if embedding_provider == "bedrock" and dimensions not in {256, 512, 1024}:
            raise ValueError(
                "STORE_ASSISTANT_EMBEDDING_DIMENSIONS must be 256, 512, or 1024 for bedrock"
            )
        aws_region = values.get("AWS_REGION", "us-east-1").strip()
        model_id = values.get("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0").strip()
        if not aws_region or not model_id:
            raise ValueError("AWS_REGION and BEDROCK_EMBEDDING_MODEL_ID must be non-empty")
        signing_secret = values.get("SLACK_SIGNING_SECRET", "").strip() or None
        slack_user_tokens = _load_slack_user_tokens(
            values.get("STORE_ASSISTANT_SLACK_USER_TOKENS", "")
        )
        if (signing_secret is None) != (slack_user_tokens is None):
            raise ValueError(
                "SLACK_SIGNING_SECRET and STORE_ASSISTANT_SLACK_USER_TOKENS "
                "must be configured together"
            )
        return cls(
            delivery_logs_path=Path(
                values.get("STORE_ASSISTANT_DELIVERY_LOGS_PATH", "data/delivery_logs.json")
            ),
            recipe_path=Path(
                values.get(
                    "STORE_ASSISTANT_RECIPE_PATH",
                    "data/recipes/oat_milk_overnight_oats.html",
                )
            ),
            supplier_contract_path=_optional_path(
                values.get("STORE_ASSISTANT_SUPPLIER_CONTRACT_PATH", "")
            ),
            state_directory=Path(values.get("STORE_ASSISTANT_STATE_DIRECTORY", ".local")),
            embedding_dimensions=dimensions,
            embedding_provider=embedding_provider,
            aws_region=aws_region,
            bedrock_embedding_model_id=model_id,
            slack_signing_secret=signing_secret,
            slack_user_tokens=slack_user_tokens,
        )


def create_local_app(settings: LocalSettings | None = None) -> FastAPI:
    """Build an API using synthetic data and providers requiring no paid services."""
    config = settings or LocalSettings.from_environment()
    chunks = [
        *ingest_delivery_logs(config.delivery_logs_path),
        *ingest_recipe_html(config.recipe_path),
    ]
    if config.supplier_contract_path is not None:
        chunks.extend(ingest_supplier_contracts(config.supplier_contract_path))
    embeddings = _create_embedding_provider(config)
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
        AuthenticatedRetrievalService(auth, retrieval, source_router=infer_source_type),
        AnswerService(LocalExtractiveLLM()),
        JSONLRequestLogRepository(config.state_directory / "requests.jsonl"),
    )
    feedback = SQLiteFeedbackRepository(config.state_directory / "feedback.sqlite3")
    slack_handler = None
    if config.slack_signing_secret is not None and config.slack_user_tokens is not None:
        slack_handler = SlackCommandHandler(
            assistant,
            SlackSignatureVerifier(config.slack_signing_secret),
            config.slack_user_tokens,
        )
    return create_app(assistant, auth, feedback, slack_handler=slack_handler)


def _load_slack_user_tokens(raw_value: str) -> dict[str, str] | None:
    if not raw_value.strip():
        return None
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError("STORE_ASSISTANT_SLACK_USER_TOKENS must be a JSON object") from exc
    if (
        not isinstance(value, dict)
        or not value
        or any(
            not isinstance(user_id, str)
            or not user_id.strip()
            or not isinstance(token, str)
            or not token.strip()
            for user_id, token in value.items()
        )
    ):
        raise ValueError("STORE_ASSISTANT_SLACK_USER_TOKENS must map Slack user IDs to tokens")
    return {user_id.strip(): token.strip() for user_id, token in value.items()}


def _optional_path(raw_value: str) -> Path | None:
    value = raw_value.strip()
    return Path(value) if value else None


def _create_embedding_provider(config: LocalSettings) -> EmbeddingProvider:
    if config.embedding_provider == "local":
        return LocalHashEmbeddingProvider(config.embedding_dimensions)
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Bedrock embeddings require the optional AWS dependencies; "
            "install the project with [aws]"
        ) from exc
    client: Any = boto3.client("bedrock-runtime", region_name=config.aws_region)
    return AmazonTitanEmbeddingProvider(
        client=client,
        model_id=config.bedrock_embedding_model_id,
        dimensions=config.embedding_dimensions,
    )


app = create_local_app()
