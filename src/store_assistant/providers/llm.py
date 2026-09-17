"""Language-model boundary and a deterministic offline answer provider."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from store_assistant.providers.reranking import RerankResult


class LLMError(ValueError):
    """Raised when an answer-generation request is invalid."""


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Answer text and usage details returned by a language-model provider."""

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


@runtime_checkable
class LLMProvider(Protocol):
    """Replaceable boundary for local and hosted language models."""

    def generate(self, query: str, context: Sequence[RerankResult]) -> GeneratedAnswer:
        """Generate a concise answer grounded only in the supplied context."""


@dataclass(frozen=True, slots=True)
class AmazonBedrockLLM:
    """Hosted Bedrock Converse adapter with an injected runtime client."""

    client: Any
    model: str = "amazon.nova-lite-v1:0"
    max_tokens: int = 300
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise LLMError("model must be non-empty")
        if self.max_tokens < 1:
            raise LLMError("max_tokens must be positive")
        if not 0.0 <= self.temperature <= 1.0:
            raise LLMError("temperature must be between 0 and 1")

    def generate(self, query: str, context: Sequence[RerankResult]) -> GeneratedAnswer:
        if not isinstance(query, str) or not query.strip():
            raise LLMError("query must be a non-empty string")
        if not context:
            raise LLMError("context must contain at least one document")
        if any(not item.text.strip() for item in context):
            raise LLMError("context documents must contain text")

        passages = "\n\n".join(
            f"Source {item.document_id}:\n{item.text.strip()}" for item in context
        )
        prompt = (
            "Answer the store manager's question concisely using only the supplied "
            "sources. If the sources do not support an answer, say so.\n\n"
            f"Question: {query.strip()}\n\nSources:\n{passages}"
        )
        try:
            response = self.client.converse(
                modelId=self.model,
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            content = response["output"]["message"]["content"]
            usage = response["usage"]
            text = "".join(part.get("text", "") for part in content).strip()
            prompt_tokens = usage["inputTokens"]
            completion_tokens = usage["outputTokens"]
        except Exception as exc:
            raise LLMError("Bedrock answer generation failed") from exc
        if (
            not text
            or not isinstance(prompt_tokens, int)
            or isinstance(prompt_tokens, bool)
            or prompt_tokens < 0
            or not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens < 0
        ):
            raise LLMError("Bedrock returned an invalid answer response")
        return GeneratedAnswer(text, self.model, prompt_tokens, completion_tokens)


@dataclass(frozen=True, slots=True)
class LocalExtractiveLLM:
    """Deterministic, no-cost provider for local development and evaluation.

    The strongest reranked passage is returned verbatim. Citation formatting is
    deliberately owned by the answer service so hosted providers cannot omit it.
    """

    model: str = "local-extractive-v1"

    def generate(self, query: str, context: Sequence[RerankResult]) -> GeneratedAnswer:
        if not isinstance(query, str) or not query.strip():
            raise LLMError("query must be a non-empty string")
        if not context:
            raise LLMError("context must contain at least one document")
        if any(not item.text.strip() for item in context):
            raise LLMError("context documents must contain text")

        answer = context[0].text.strip()
        prompt_tokens = len(query.split()) + sum(len(item.text.split()) for item in context)
        return GeneratedAnswer(
            text=answer,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(answer.split()),
        )
