from typing import Any

import pytest

from store_assistant.providers.llm import AmazonBedrockLLM, LLMError, LLMProvider
from store_assistant.providers.reranking import RerankResult


class FakeBedrockClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def converse(self, **kwargs: object) -> Any:
        self.calls.append(kwargs)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _context() -> list[RerankResult]:
    return [RerankResult("delivery:BK-1:0", "Oat milk: 12 cartons.", 0.8, 0.9, {})]


def test_bedrock_llm_generates_grounded_answer_and_reports_usage() -> None:
    client = FakeBedrockClient(
        {
            "output": {"message": {"content": [{"text": "There are 12 cartons."}]}},
            "usage": {"inputTokens": 42, "outputTokens": 6},
        }
    )
    provider = AmazonBedrockLLM(client, model="amazon.nova-lite-v1:0")

    answer = provider.generate("Do we have oat milk?", _context())

    assert answer.text == "There are 12 cartons."
    assert answer.prompt_tokens == 42
    assert answer.completion_tokens == 6
    assert isinstance(provider, LLMProvider)
    call = client.calls[0]
    assert call["modelId"] == "amazon.nova-lite-v1:0"
    prompt = call["messages"][0]["content"][0]["text"]  # type: ignore[index]
    assert "delivery:BK-1:0" in prompt
    assert "Oat milk: 12 cartons." in prompt


def test_bedrock_llm_wraps_client_errors_without_exposing_details() -> None:
    provider = AmazonBedrockLLM(FakeBedrockClient(RuntimeError("secret credential")))
    with pytest.raises(LLMError, match="generation failed") as error:
        provider.generate("Do we have oat milk?", _context())
    assert "secret credential" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"output": {"message": {"content": []}}, "usage": {"inputTokens": 1, "outputTokens": 0}},
        {
            "output": {"message": {"content": [{"text": "answer"}]}},
            "usage": {"inputTokens": -1, "outputTokens": 1},
        },
    ],
)
def test_bedrock_llm_rejects_malformed_responses(response: object) -> None:
    with pytest.raises(LLMError, match="Bedrock"):
        AmazonBedrockLLM(FakeBedrockClient(response)).generate("question", _context())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": " "}, "model"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"temperature": 1.1}, "temperature"),
    ],
)
def test_bedrock_llm_validates_configuration(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(LLMError, match=message):
        AmazonBedrockLLM(FakeBedrockClient({}), **kwargs)  # type: ignore[arg-type]
