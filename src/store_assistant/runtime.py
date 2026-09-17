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
from store_assistant.auth import (
    AuthProvider,
    MockAuthProvider,
    OIDCTokenVerifier,
    OktaOIDCAuthProvider,
    UserContext,
)
from store_assistant.feedback import SQLiteFeedbackRepository
from store_assistant.ingestion.contracts import ingest_supplier_contracts
from store_assistant.ingestion.delivery_logs import ingest_delivery_logs
from store_assistant.ingestion.recipes import ingest_recipe_html
from store_assistant.providers.embeddings import (
    AmazonTitanEmbeddingProvider,
    EmbeddingProvider,
    LocalHashEmbeddingProvider,
)
from store_assistant.providers.llm import AmazonBedrockLLM, LLMProvider, LocalExtractiveLLM
from store_assistant.providers.reranking import LocalLexicalReranker
from store_assistant.providers.vector_store import (
    InMemoryVectorStore,
    PineconeVectorStore,
    VectorRecord,
    VectorStore,
)
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
    llm_provider: str = "local"
    bedrock_llm_model_id: str = "amazon.nova-lite-v1:0"
    bedrock_llm_max_tokens: int = 300
    bedrock_llm_temperature: float = 0.0
    vector_store_provider: str = "local"
    pinecone_api_key: str | None = None
    pinecone_index_host: str | None = None
    pinecone_namespace: str | None = None
    auth_provider: str = "mock"
    okta_issuer: str | None = None
    okta_audience: str | None = None
    okta_store_id_claim: str = "store_id"
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
        llm_provider = values.get("STORE_ASSISTANT_LLM_PROVIDER", "local").strip()
        if llm_provider not in {"local", "bedrock"}:
            raise ValueError("STORE_ASSISTANT_LLM_PROVIDER must be local or bedrock")
        llm_model_id = values.get("BEDROCK_LLM_MODEL_ID", "amazon.nova-lite-v1:0").strip()
        try:
            llm_max_tokens = int(values.get("BEDROCK_LLM_MAX_TOKENS", "300"))
            llm_temperature = float(values.get("BEDROCK_LLM_TEMPERATURE", "0"))
        except ValueError as exc:
            raise ValueError("Bedrock LLM token and temperature settings must be numeric") from exc
        if not llm_model_id or llm_max_tokens < 1 or not 0 <= llm_temperature <= 1:
            raise ValueError("Bedrock LLM settings are invalid")
        vector_store_provider = values.get("STORE_ASSISTANT_VECTOR_STORE_PROVIDER", "local").strip()
        if vector_store_provider not in {"local", "pinecone"}:
            raise ValueError("STORE_ASSISTANT_VECTOR_STORE_PROVIDER must be local or pinecone")
        pinecone_api_key = values.get("PINECONE_API_KEY", "").strip() or None
        pinecone_index_host = values.get("PINECONE_INDEX_HOST", "").strip() or None
        pinecone_namespace = values.get("PINECONE_NAMESPACE", "").strip() or None
        if vector_store_provider == "pinecone" and (
            pinecone_api_key is None or pinecone_index_host is None
        ):
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_HOST are required for pinecone")
        auth_provider = values.get("STORE_ASSISTANT_AUTH_PROVIDER", "mock").strip()
        if auth_provider not in {"mock", "okta"}:
            raise ValueError("STORE_ASSISTANT_AUTH_PROVIDER must be mock or okta")
        okta_issuer = values.get("OKTA_ISSUER", "").strip() or None
        okta_audience = values.get("OKTA_AUDIENCE", "").strip() or None
        okta_store_id_claim = values.get("OKTA_STORE_ID_CLAIM", "store_id").strip()
        if auth_provider == "okta" and (okta_issuer is None or okta_audience is None):
            raise ValueError("OKTA_ISSUER and OKTA_AUDIENCE are required for okta")
        if not okta_store_id_claim:
            raise ValueError("OKTA_STORE_ID_CLAIM must be non-empty")
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
            llm_provider=llm_provider,
            bedrock_llm_model_id=llm_model_id,
            bedrock_llm_max_tokens=llm_max_tokens,
            bedrock_llm_temperature=llm_temperature,
            vector_store_provider=vector_store_provider,
            pinecone_api_key=pinecone_api_key,
            pinecone_index_host=pinecone_index_host,
            pinecone_namespace=pinecone_namespace,
            auth_provider=auth_provider,
            okta_issuer=okta_issuer,
            okta_audience=okta_audience,
            okta_store_id_claim=okta_store_id_claim,
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
    vector_store = _create_vector_store(config, embeddings.dimension)
    vectors = embeddings.embed([chunk.text for chunk in chunks])
    vector_store.upsert(
        [
            VectorRecord(chunk.document_id, chunk.text, vector, chunk.metadata)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )

    auth = _create_auth_provider(config)
    retrieval = RetrievalService(embeddings, vector_store, LocalLexicalReranker())
    assistant = AssistantService(
        AuthenticatedRetrievalService(auth, retrieval, source_router=infer_source_type),
        AnswerService(_create_llm_provider(config)),
        JSONLRequestLogRepository(config.state_directory / "requests.jsonl"),
    )
    feedback = SQLiteFeedbackRepository(config.state_directory / "feedback.sqlite3")
    slack_handler = None
    if config.slack_signing_secret is not None and config.slack_user_tokens is not None:
        slack_handler = SlackCommandHandler(
            assistant,
            SlackSignatureVerifier(config.slack_signing_secret),
            config.slack_user_tokens,
            feedback,
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


def _create_llm_provider(config: LocalSettings) -> LLMProvider:
    if config.llm_provider == "local":
        return LocalExtractiveLLM()
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "Bedrock answer generation requires the optional AWS dependencies; "
            "install the project with [aws]"
        ) from exc
    client: Any = boto3.client("bedrock-runtime", region_name=config.aws_region)
    return AmazonBedrockLLM(
        client=client,
        model=config.bedrock_llm_model_id,
        max_tokens=config.bedrock_llm_max_tokens,
        temperature=config.bedrock_llm_temperature,
    )


def _create_vector_store(config: LocalSettings, dimension: int) -> VectorStore:
    if config.vector_store_provider == "local":
        return InMemoryVectorStore(dimension)
    try:
        from pinecone import Pinecone  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Pinecone vector storage requires the optional Pinecone dependencies; "
            "install the project with [pinecone]"
        ) from exc
    if config.pinecone_api_key is None or config.pinecone_index_host is None:
        raise ValueError("Pinecone API key and index host must be configured")
    client = Pinecone(api_key=config.pinecone_api_key)
    index: Any = client.Index(host=config.pinecone_index_host)
    return PineconeVectorStore(index, dimension, namespace=config.pinecone_namespace)


class _PyJWTOIDCTokenVerifier:
    """Verify Okta access tokens using discovery-compatible JWKS keys."""

    def __init__(self, jwt_module: Any, issuer: str) -> None:
        self._jwt = jwt_module
        self._jwks = jwt_module.PyJWKClient(f"{issuer.rstrip('/')}/v1/keys")

    def verify(self, token: str, *, issuer: str, audience: str) -> Mapping[str, Any]:
        signing_key = self._jwks.get_signing_key_from_jwt(token)
        claims: Any = self._jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
        )
        if not isinstance(claims, Mapping):
            raise ValueError("verified token claims are invalid")
        return claims


def _create_auth_provider(config: LocalSettings) -> AuthProvider:
    if config.auth_provider == "mock":
        return MockAuthProvider(
            {
                "local-brooklyn-token": UserContext("brooklyn-manager", "brooklyn-01"),
                "local-manhattan-token": UserContext("manhattan-manager", "manhattan-01"),
                "local-queens-token": UserContext("queens-manager", "queens-01"),
            }
        )
    try:
        import jwt
    except ImportError as exc:
        raise RuntimeError(
            "Okta authentication requires the optional OIDC dependencies; "
            "install the project with [okta]"
        ) from exc
    if config.okta_issuer is None or config.okta_audience is None:
        raise ValueError("Okta issuer and audience must be configured")
    verifier: OIDCTokenVerifier = _PyJWTOIDCTokenVerifier(jwt, config.okta_issuer)
    return OktaOIDCAuthProvider(
        verifier,
        issuer=config.okta_issuer,
        audience=config.okta_audience,
        store_id_claim=config.okta_store_id_claim,
    )


app = create_local_app()
