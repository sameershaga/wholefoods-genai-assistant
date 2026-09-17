import hmac
import sys
from hashlib import sha256
from pathlib import Path
from time import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from urllib.parse import urlencode

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from store_assistant.auth import OktaOIDCAuthProvider
from store_assistant.providers.embeddings import AmazonTitanEmbeddingProvider
from store_assistant.providers.llm import AmazonBedrockLLM
from store_assistant.providers.vector_store import PineconeVectorStore
from store_assistant.runtime import (
    LocalSettings,
    _create_auth_provider,
    _create_embedding_provider,
    _create_llm_provider,
    _create_vector_store,
    create_local_app,
)


def test_local_settings_load_from_environment(tmp_path: Path) -> None:
    settings = LocalSettings.from_environment(
        {
            "STORE_ASSISTANT_DELIVERY_LOGS_PATH": "fixtures/deliveries.json",
            "STORE_ASSISTANT_RECIPE_PATH": "fixtures/recipe.html",
            "STORE_ASSISTANT_SUPPLIER_CONTRACT_PATH": "fixtures/contract.pdf",
            "STORE_ASSISTANT_STATE_DIRECTORY": str(tmp_path),
            "STORE_ASSISTANT_EMBEDDING_DIMENSIONS": "128",
        }
    )

    assert settings.delivery_logs_path == Path("fixtures/deliveries.json")
    assert settings.recipe_path == Path("fixtures/recipe.html")
    assert settings.supplier_contract_path == Path("fixtures/contract.pdf")
    assert settings.state_directory == tmp_path
    assert settings.embedding_dimensions == 128
    assert settings.embedding_provider == "local"
    assert settings.vector_store_provider == "local"
    assert settings.auth_provider == "mock"
    assert settings.slack_signing_secret is None
    assert settings.slack_user_tokens is None


def test_local_settings_reject_invalid_embedding_dimensions() -> None:
    for value in ("zero", "0", "-1"):
        try:
            LocalSettings.from_environment({"STORE_ASSISTANT_EMBEDDING_DIMENSIONS": value})
        except ValueError as exc:
            assert "STORE_ASSISTANT_EMBEDDING_DIMENSIONS" in str(exc)
        else:
            raise AssertionError(f"expected invalid dimensions {value!r} to fail")


def test_settings_and_factory_select_bedrock_from_environment(monkeypatch: MonkeyPatch) -> None:
    client = object()
    fake_boto3 = SimpleNamespace(client=lambda service, region_name: client)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    settings = LocalSettings.from_environment(
        {
            "STORE_ASSISTANT_EMBEDDING_PROVIDER": "bedrock",
            "STORE_ASSISTANT_EMBEDDING_DIMENSIONS": "512",
            "AWS_REGION": "us-west-2",
            "BEDROCK_EMBEDDING_MODEL_ID": "test-titan-model",
        }
    )

    provider = _create_embedding_provider(settings)

    assert isinstance(provider, AmazonTitanEmbeddingProvider)
    assert provider.client is client
    assert provider.dimension == 512
    assert provider.model_id == "test-titan-model"


def test_settings_reject_invalid_embedding_provider_configuration() -> None:
    invalid_environments = (
        {"STORE_ASSISTANT_EMBEDDING_PROVIDER": "unknown"},
        {
            "STORE_ASSISTANT_EMBEDDING_PROVIDER": "bedrock",
            "STORE_ASSISTANT_EMBEDDING_DIMENSIONS": "384",
        },
    )
    for environment in invalid_environments:
        try:
            LocalSettings.from_environment(environment)
        except ValueError as exc:
            assert "EMBEDDING" in str(exc)
        else:
            raise AssertionError("expected invalid embedding provider configuration to fail")


def test_settings_and_factory_select_bedrock_llm_from_environment(
    monkeypatch: MonkeyPatch,
) -> None:
    client = object()
    boto3_client = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=boto3_client))
    settings = LocalSettings.from_environment(
        {
            "STORE_ASSISTANT_LLM_PROVIDER": "bedrock",
            "AWS_REGION": "us-west-2",
            "BEDROCK_LLM_MODEL_ID": "test-answer-model",
            "BEDROCK_LLM_MAX_TOKENS": "180",
            "BEDROCK_LLM_TEMPERATURE": "0.25",
        }
    )

    provider = _create_llm_provider(settings)

    assert isinstance(provider, AmazonBedrockLLM)
    assert provider.client is client
    assert provider.model == "test-answer-model"
    assert provider.max_tokens == 180
    assert provider.temperature == 0.25
    boto3_client.assert_called_once_with("bedrock-runtime", region_name="us-west-2")


def test_settings_reject_invalid_llm_configuration() -> None:
    invalid_environments = (
        {"STORE_ASSISTANT_LLM_PROVIDER": "unknown"},
        {"BEDROCK_LLM_MODEL_ID": ""},
        {"BEDROCK_LLM_MAX_TOKENS": "0"},
        {"BEDROCK_LLM_MAX_TOKENS": "many"},
        {"BEDROCK_LLM_TEMPERATURE": "1.1"},
    )
    for environment in invalid_environments:
        try:
            LocalSettings.from_environment(environment)
        except ValueError as exc:
            assert "LLM" in str(exc) or "token" in str(exc)
        else:
            raise AssertionError("expected invalid LLM configuration to fail")


def test_settings_and_factory_select_pinecone_from_environment(monkeypatch: MonkeyPatch) -> None:
    index = object()
    client = MagicMock()
    client.Index.return_value = index
    pinecone_constructor = MagicMock(return_value=client)
    monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=pinecone_constructor))
    settings = LocalSettings.from_environment(
        {
            "STORE_ASSISTANT_VECTOR_STORE_PROVIDER": "pinecone",
            "PINECONE_API_KEY": "test-key",
            "PINECONE_INDEX_HOST": "test-index.example.pinecone.io",
            "PINECONE_NAMESPACE": "store-assistant",
        }
    )

    store = _create_vector_store(settings, 384)

    assert isinstance(store, PineconeVectorStore)
    assert store.dimension == 384
    pinecone_constructor.assert_called_once_with(api_key="test-key")
    client.Index.assert_called_once_with(host="test-index.example.pinecone.io")


def test_settings_reject_invalid_pinecone_configuration() -> None:
    invalid_environments = (
        {"STORE_ASSISTANT_VECTOR_STORE_PROVIDER": "unknown"},
        {"STORE_ASSISTANT_VECTOR_STORE_PROVIDER": "pinecone"},
        {
            "STORE_ASSISTANT_VECTOR_STORE_PROVIDER": "pinecone",
            "PINECONE_API_KEY": "key",
        },
    )
    for environment in invalid_environments:
        try:
            LocalSettings.from_environment(environment)
        except ValueError as exc:
            assert "PINECONE" in str(exc) or "VECTOR_STORE_PROVIDER" in str(exc)
        else:
            raise AssertionError("expected invalid Pinecone configuration to fail")


def test_settings_and_factory_select_okta_from_environment(monkeypatch: MonkeyPatch) -> None:
    signing_key = SimpleNamespace(key="public-key")
    jwks_client = MagicMock()
    jwks_client.get_signing_key_from_jwt.return_value = signing_key
    jwks_constructor = MagicMock(return_value=jwks_client)
    decode = MagicMock(
        return_value={"sub": "manager-9", "location": "brooklyn-01", "name": "Morgan"}
    )
    monkeypatch.setitem(
        sys.modules,
        "jwt",
        SimpleNamespace(PyJWKClient=jwks_constructor, decode=decode),
    )
    settings = LocalSettings.from_environment(
        {
            "STORE_ASSISTANT_AUTH_PROVIDER": "okta",
            "OKTA_ISSUER": "https://example.okta.com/oauth2/default/",
            "OKTA_AUDIENCE": "api://stores",
            "OKTA_STORE_ID_CLAIM": "location",
        }
    )

    provider = _create_auth_provider(settings)
    context = provider.authenticate("signed-access-token")

    assert isinstance(provider, OktaOIDCAuthProvider)
    assert context.store_id == "BROOKLYN-01"
    jwks_constructor.assert_called_once_with("https://example.okta.com/oauth2/default/v1/keys")
    decode.assert_called_once_with(
        "signed-access-token",
        "public-key",
        algorithms=["RS256"],
        audience="api://stores",
        issuer="https://example.okta.com/oauth2/default",
    )


def test_settings_reject_invalid_okta_configuration() -> None:
    invalid_environments = (
        {"STORE_ASSISTANT_AUTH_PROVIDER": "unknown"},
        {"STORE_ASSISTANT_AUTH_PROVIDER": "okta"},
        {
            "STORE_ASSISTANT_AUTH_PROVIDER": "okta",
            "OKTA_ISSUER": "https://example.okta.com",
        },
        {"OKTA_STORE_ID_CLAIM": " "},
    )
    for environment in invalid_environments:
        try:
            LocalSettings.from_environment(environment)
        except ValueError as exc:
            assert "AUTH_PROVIDER" in str(exc) or "OKTA" in str(exc)
        else:
            raise AssertionError("expected invalid Okta configuration to fail")


def test_local_runtime_loads_synthetic_data_and_enforces_store_isolation(
    tmp_path: Path,
) -> None:
    app = create_local_app(LocalSettings(state_directory=tmp_path))

    with TestClient(app) as client:
        brooklyn = client.post(
            "/v1/query",
            headers={"Authorization": "Bearer local-brooklyn-token"},
            json={"query": "Do we have oat milk?", "filters": {"sku": "OAT001"}},
        )
        manhattan = client.post(
            "/v1/query",
            headers={"Authorization": "Bearer local-manhattan-token"},
            json={"query": "Do we have oat milk?", "filters": {"sku": "OAT001"}},
        )

    assert brooklyn.status_code == 200
    assert "12 cartons" in brooklyn.json()["answer"]
    assert "3 cartons" not in brooklyn.json()["answer"]
    assert manhattan.status_code == 200
    assert "3 cartons" in manhattan.json()["answer"]
    assert (tmp_path / "requests.jsonl").exists()
    assert (tmp_path / "feedback.sqlite3").exists()


def test_local_runtime_rejects_unknown_token(tmp_path: Path) -> None:
    client = TestClient(create_local_app(LocalSettings(state_directory=tmp_path)))

    response = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer unknown"},
        json={"query": "Do we have oat milk?"},
    )

    assert response.status_code == 401


def test_local_runtime_indexes_recipe_sections(tmp_path: Path) -> None:
    client = TestClient(create_local_app(LocalSettings(state_directory=tmp_path)))

    response = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer local-brooklyn-token"},
        json={"query": "How do I prepare oat milk overnight oats?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "oat" in payload["answer"].lower()
    assert any(citation.startswith("recipe:") for citation in payload["citations"])


@patch("store_assistant.runtime.ingest_supplier_contracts")
def test_local_runtime_indexes_configured_supplier_contract(
    ingest_contract: MagicMock, tmp_path: Path
) -> None:
    from store_assistant.ingestion.models import DocumentChunk

    ingest_contract.return_value = [
        DocumentChunk(
            document_id="contract:CON-2026-001:0",
            text="Oat milk pricing is fixed at $24 per case through December.",
            metadata={
                "source_type": "supplier_contract",
                "store_id": "BROOKLYN-01",
                "sku": "OAT001",
                "supplier": "green valley foods",
                "product_category": "dairy_alternatives",
            },
        )
    ]
    contract_path = tmp_path / "contract.pdf"
    client = TestClient(
        create_local_app(
            LocalSettings(
                state_directory=tmp_path,
                supplier_contract_path=contract_path,
            )
        )
    )

    response = client.post(
        "/v1/query",
        headers={"Authorization": "Bearer local-brooklyn-token"},
        json={"query": "What are the oat milk supplier contract terms?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "fixed at $24 per case" in payload["answer"]
    assert payload["citations"] == ["contract:CON-2026-001:0"]
    ingest_contract.assert_called_once_with(contract_path)


def test_local_runtime_enables_store_isolated_slack_from_environment(
    tmp_path: Path,
) -> None:
    secret = "test-signing-secret"
    settings = LocalSettings.from_environment(
        {
            "STORE_ASSISTANT_STATE_DIRECTORY": str(tmp_path),
            "SLACK_SIGNING_SECRET": secret,
            "STORE_ASSISTANT_SLACK_USER_TOKENS": ('{"U-BROOKLYN":"local-brooklyn-token"}'),
        }
    )
    body = urlencode({"user_id": "U-BROOKLYN", "text": "Do we have oat milk?"}).encode()
    timestamp = str(int(time()))
    signature = (
        "v0="
        + hmac.new(secret.encode(), b"v0:" + timestamp.encode() + b":" + body, sha256).hexdigest()
    )

    response = TestClient(create_local_app(settings)).post(
        "/slack/commands",
        content=body,
        headers={
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": signature,
        },
    )

    assert response.status_code == 200
    assert "12 cartons" in response.json()["text"]
    assert "3 cartons" not in response.json()["text"]


def test_local_settings_require_complete_valid_slack_configuration() -> None:
    invalid_environments = (
        {"SLACK_SIGNING_SECRET": "secret"},
        {"STORE_ASSISTANT_SLACK_USER_TOKENS": '{"U1":"token"}'},
        {
            "SLACK_SIGNING_SECRET": "secret",
            "STORE_ASSISTANT_SLACK_USER_TOKENS": "not-json",
        },
    )

    for environment in invalid_environments:
        try:
            LocalSettings.from_environment(environment)
        except ValueError as exc:
            assert "SLACK" in str(exc)
        else:
            raise AssertionError("expected invalid Slack configuration to fail")
