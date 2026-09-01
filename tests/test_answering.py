from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from store_assistant.answering import AnswerService
from store_assistant.providers.llm import (
    GeneratedAnswer,
    LLMError,
    LLMProvider,
    LocalExtractiveLLM,
)
from store_assistant.providers.reranking import RerankResult


def _result(document_id: str, text: str) -> RerankResult:
    return RerankResult(document_id, text, 0.8, 0.9, {"store_id": "BROOKLYN"})


def test_local_answer_is_grounded_and_cited() -> None:
    service = AnswerService(LocalExtractiveLLM())
    answer = service.answer(
        "Do we have oat milk?",
        [_result("delivery:BRK-001:0", "Oat milk has 12 cartons available.")],
    )

    assert answer.text == ("Oat milk has 12 cartons available. [source: delivery:BRK-001:0]")
    assert answer.citations == ("delivery:BRK-001:0",)
    assert answer.model == "local-extractive-v1"
    assert answer.prompt_tokens > 0
    assert answer.completion_tokens == 6


def test_answer_service_enforces_all_context_citations() -> None:
    @dataclass
    class ProviderWithoutCitations:
        def generate(self, query: str, context: Sequence[RerankResult]) -> GeneratedAnswer:
            return GeneratedAnswer("Use the first delivery.", "test-model", 10, 4)

    answer = AnswerService(ProviderWithoutCitations()).answer(
        "What arrived?", [_result("doc-1", "one"), _result("doc-2", "two")]
    )
    assert answer.text.endswith("[source: doc-1] [source: doc-2]")
    assert answer.citations == ("doc-1", "doc-2")


def test_empty_context_returns_safe_answer_without_calling_provider() -> None:
    @dataclass
    class FailingProvider:
        def generate(self, query: str, context: Sequence[RerankResult]) -> GeneratedAnswer:
            raise AssertionError("provider should not be called")

    answer = AnswerService(FailingProvider()).answer("unknown item", [])
    assert "couldn't find" in answer.text
    assert answer.citations == ()
    assert answer.model == "none"


def test_local_provider_conforms_to_protocol() -> None:
    assert isinstance(LocalExtractiveLLM(), LLMProvider)


@pytest.mark.parametrize("query", ["", "   "])
def test_local_provider_rejects_empty_query(query: str) -> None:
    with pytest.raises(LLMError, match="non-empty"):
        LocalExtractiveLLM().generate(query, [_result("doc", "context")])


def test_local_provider_rejects_empty_context() -> None:
    with pytest.raises(LLMError, match="at least one"):
        LocalExtractiveLLM().generate("question", [])


@pytest.mark.parametrize(
    "response",
    [
        GeneratedAnswer("", "model", 1, 1),
        GeneratedAnswer("answer", "", 1, 1),
        GeneratedAnswer("answer", "model", -1, 1),
    ],
)
def test_answer_service_rejects_invalid_provider_response(response: GeneratedAnswer) -> None:
    @dataclass
    class InvalidProvider:
        def generate(self, query: str, context: Sequence[RerankResult]) -> GeneratedAnswer:
            return response

    with pytest.raises(ValueError):
        AnswerService(InvalidProvider()).answer("question", [_result("doc", "context")])
