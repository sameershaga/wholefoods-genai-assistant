"""Language-model boundary and a deterministic offline answer provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

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
