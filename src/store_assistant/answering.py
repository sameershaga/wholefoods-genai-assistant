"""Grounded operational answer generation with enforced source citations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from store_assistant.providers.llm import GeneratedAnswer, LLMProvider
from store_assistant.providers.reranking import RerankResult


@dataclass(frozen=True, slots=True)
class Answer:
    """A user-facing answer with explicit source and model usage details."""

    text: str
    citations: tuple[str, ...]
    model: str
    prompt_tokens: int
    completion_tokens: int


class AnswerService:
    """Generates answers and appends citations independently of the LLM."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    def answer(self, query: str, context: Sequence[RerankResult]) -> Answer:
        if not context:
            return Answer(
                text="I couldn't find relevant information for this store.",
                citations=(),
                model="none",
                prompt_tokens=0,
                completion_tokens=0,
            )

        generated = self._provider.generate(query, context)
        self._validate_generation(generated)
        citations = tuple(item.document_id for item in context)
        references = " ".join(f"[source: {document_id}]" for document_id in citations)
        return Answer(
            text=f"{generated.text.strip()} {references}",
            citations=citations,
            model=generated.model,
            prompt_tokens=generated.prompt_tokens,
            completion_tokens=generated.completion_tokens,
        )

    @staticmethod
    def _validate_generation(generated: GeneratedAnswer) -> None:
        if not generated.text.strip():
            raise ValueError("LLM provider returned an empty answer")
        if not generated.model.strip():
            raise ValueError("LLM provider returned an empty model name")
        if generated.prompt_tokens < 0 or generated.completion_tokens < 0:
            raise ValueError("LLM provider returned negative token usage")
