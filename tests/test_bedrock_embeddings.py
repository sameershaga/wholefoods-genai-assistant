import io
import json
from typing import Any

import pytest

from store_assistant.providers.embeddings import (
    AmazonTitanEmbeddingProvider,
    EmbeddingError,
    EmbeddingProvider,
)


class FakeBedrockClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def invoke_model(self, **kwargs: object) -> dict[str, Any]:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return {"body": io.BytesIO(json.dumps({"embedding": response}).encode())}


def test_titan_embeds_in_order_with_expected_bedrock_request() -> None:
    client = FakeBedrockClient([[0.5] * 256, [0.25] * 256])
    provider = AmazonTitanEmbeddingProvider(client, dimensions=256)

    vectors = provider.embed(["oat milk", "delivery schedule"])

    assert vectors == [(0.5,) * 256, (0.25,) * 256]
    assert [json.loads(str(call["body"]))["inputText"] for call in client.calls] == [
        "oat milk",
        "delivery schedule",
    ]
    assert client.calls[0]["modelId"] == "amazon.titan-embed-text-v2:0"
    assert json.loads(str(client.calls[0]["body"]))["dimensions"] == 256
    assert isinstance(provider, EmbeddingProvider)


@pytest.mark.parametrize("dimensions", [0, 384, 2048])
def test_titan_rejects_unsupported_dimensions(dimensions: int) -> None:
    with pytest.raises(EmbeddingError, match="256, 512, or 1024"):
        AmazonTitanEmbeddingProvider(FakeBedrockClient([]), dimensions=dimensions)


def test_titan_wraps_provider_errors_without_exposing_payload() -> None:
    provider = AmazonTitanEmbeddingProvider(
        FakeBedrockClient([RuntimeError("credential secret")]), dimensions=256
    )
    with pytest.raises(EmbeddingError, match="text at position 0") as error:
        provider.embed(["sensitive query"])
    assert "credential secret" not in str(error.value)


@pytest.mark.parametrize("response", [[1.0] * 255, [float("nan")] * 256, {"bad": "shape"}])
def test_titan_rejects_malformed_embeddings(response: object) -> None:
    provider = AmazonTitanEmbeddingProvider(FakeBedrockClient([response]), dimensions=256)
    with pytest.raises(EmbeddingError, match="Bedrock"):
        provider.embed(["oat milk"])


def test_titan_rejects_empty_text_before_calling_bedrock() -> None:
    client = FakeBedrockClient([])
    provider = AmazonTitanEmbeddingProvider(client, dimensions=256)
    with pytest.raises(EmbeddingError, match="position 0"):
        provider.embed(["  "])
    assert client.calls == []
