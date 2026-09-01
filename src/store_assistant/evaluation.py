"""Golden-dataset evaluation for the fully local assistant pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from store_assistant.answering import AnswerService
from store_assistant.ingestion.delivery_logs import ingest_delivery_logs
from store_assistant.ingestion.normalization import normalize_store_id
from store_assistant.providers.embeddings import LocalHashEmbeddingProvider
from store_assistant.providers.llm import LocalExtractiveLLM
from store_assistant.providers.reranking import LocalLexicalReranker
from store_assistant.providers.vector_store import InMemoryVectorStore, VectorRecord
from store_assistant.retrieval import RetrievalService


class EvaluationError(ValueError):
    """Raised when a golden evaluation dataset is invalid."""


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """Expected retrieval and answer behavior for one synthetic query."""

    case_id: str
    query: str
    store_id: str
    filters: dict[str, str]
    expected_document_ids: tuple[str, ...]
    answer_must_contain: tuple[str, ...]
    answer_must_not_contain: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Aggregate quality, isolation, performance, and cost measurements."""

    case_count: int
    retrieval_hit_rate: float
    correct_store_retrieval_rate: float
    answer_correctness: float
    citation_correctness: float
    average_latency_ms: float
    estimated_cost_per_query_usd: float


def load_golden_cases(path: str | Path) -> list[GoldenCase]:
    """Load and validate golden cases from JSON."""
    try:
        payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot load evaluation dataset: {path}") from exc
    if not isinstance(payload, list) or not payload:
        raise EvaluationError("evaluation dataset must be a non-empty JSON array")

    cases: list[GoldenCase] = []
    try:
        for item in payload:
            if not isinstance(item, dict):
                raise TypeError
            case = GoldenCase(
                case_id=_required_string(item, "case_id"),
                query=_required_string(item, "query"),
                store_id=normalize_store_id(_required_string(item, "store_id")),
                filters=_string_mapping(item.get("filters", {})),
                expected_document_ids=_string_tuple(item, "expected_document_ids"),
                answer_must_contain=_string_tuple(item, "answer_must_contain"),
                answer_must_not_contain=_string_tuple(
                    item, "answer_must_not_contain", required=False
                ),
            )
            cases.append(case)
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError("evaluation dataset contains an invalid case") from exc
    if len({case.case_id for case in cases}) != len(cases):
        raise EvaluationError("evaluation case IDs must be unique")
    return cases


def evaluate(
    cases: list[GoldenCase],
    delivery_logs_path: str | Path,
    *,
    embedding_dimensions: int = 384,
    input_cost_per_million_tokens: float = 0.0,
    output_cost_per_million_tokens: float = 0.0,
) -> EvaluationMetrics:
    """Run golden cases through local retrieval and grounded answering."""
    if not cases:
        raise EvaluationError("at least one evaluation case is required")
    if input_cost_per_million_tokens < 0 or output_cost_per_million_tokens < 0:
        raise EvaluationError("token prices must not be negative")

    chunks = ingest_delivery_logs(delivery_logs_path)
    embeddings = LocalHashEmbeddingProvider(embedding_dimensions)
    vector_store = InMemoryVectorStore(embeddings.dimension)
    vectors = embeddings.embed([chunk.text for chunk in chunks])
    vector_store.upsert(
        [
            VectorRecord(chunk.document_id, chunk.text, vector, chunk.metadata)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
    )
    retrieval = RetrievalService(embeddings, vector_store, LocalLexicalReranker())
    answering = AnswerService(LocalExtractiveLLM())

    hits = correct_store = correct_answers = correct_citations = 0
    total_latency_ms = total_cost = 0.0
    for case in cases:
        filters = dict(case.filters)
        filters["store_id"] = case.store_id
        started_at = perf_counter()
        context = retrieval.retrieve(case.query, filters=filters)
        answer = answering.answer(case.query, context)
        total_latency_ms += (perf_counter() - started_at) * 1000

        document_ids = {item.document_id for item in context}
        expected_ids = set(case.expected_document_ids)
        hits += bool(document_ids & expected_ids)
        store_is_correct = bool(context) and all(
            item.metadata.get("store_id") == case.store_id for item in context
        )
        correct_store += store_is_correct
        normalized_answer = answer.text.casefold()
        correct_answers += all(
            phrase.casefold() in normalized_answer for phrase in case.answer_must_contain
        ) and all(
            phrase.casefold() not in normalized_answer
            for phrase in case.answer_must_not_contain
        )
        correct_citations += bool(set(answer.citations) & expected_ids) and store_is_correct
        total_cost += (
            answer.prompt_tokens * input_cost_per_million_tokens
            + answer.completion_tokens * output_cost_per_million_tokens
        ) / 1_000_000

    count = len(cases)
    return EvaluationMetrics(
        case_count=count,
        retrieval_hit_rate=hits / count,
        correct_store_retrieval_rate=correct_store / count,
        answer_correctness=correct_answers / count,
        citation_correctness=correct_citations / count,
        average_latency_ms=total_latency_ms / count,
        estimated_cost_per_query_usd=total_cost / count,
    )


def _required_string(item: dict[str, Any], key: str) -> str:
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError
    return value.strip()


def _string_tuple(
    item: dict[str, Any], key: str, *, required: bool = True
) -> tuple[str, ...]:
    value = item[key] if required else item.get(key, [])
    if not isinstance(value, list) or (required and not value):
        raise TypeError
    if any(not isinstance(entry, str) or not entry.strip() for entry in value):
        raise TypeError
    return tuple(entry.strip() for entry in value)


def _string_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise TypeError
    return dict(value)


def main() -> None:
    """Run the bundled golden evaluation and print machine-readable metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evaluation/golden.json")
    parser.add_argument("--delivery-logs", default="data/delivery_logs.json")
    args = parser.parse_args()
    metrics = evaluate(load_golden_cases(args.dataset), args.delivery_logs)
    print(json.dumps(asdict(metrics), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
