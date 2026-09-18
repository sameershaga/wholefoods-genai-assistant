import json
from dataclasses import replace
from pathlib import Path

import pytest

from store_assistant.evaluation import (
    EvaluationError,
    evaluate,
    load_golden_cases,
)

DETERMINISTIC_METRIC_NAMES = (
    "case_count",
    "retrieval_hit_rate",
    "correct_store_retrieval_rate",
    "abstention_success_rate",
    "answer_correctness",
    "citation_correctness",
)


def test_bundled_golden_evaluation_meets_local_quality_targets() -> None:
    cases = load_golden_cases("evaluation/golden.json")

    metrics = evaluate(cases, "data/delivery_logs.json")

    assert metrics.case_count == 6
    assert metrics.retrieval_hit_rate == 1.0
    assert metrics.abstention_success_rate == 1.0
    assert metrics.correct_store_retrieval_rate == 1.0
    assert metrics.answer_correctness == 1.0
    assert metrics.citation_correctness == 1.0
    assert metrics.average_latency_ms >= 0
    assert metrics.estimated_cost_per_query_usd == 0


def test_checked_in_bundled_results_match_evaluation() -> None:
    expected = json.loads(Path("evaluation/bundled-results.json").read_text(encoding="utf-8"))
    metrics = evaluate(load_golden_cases("evaluation/golden.json"), "data/delivery_logs.json")

    actual = {name: getattr(metrics, name) for name in DETERMINISTIC_METRIC_NAMES}

    assert expected == actual


def test_evaluation_indexes_recipe_source_for_recipe_case() -> None:
    cases = load_golden_cases("evaluation/golden.json")
    recipe_case = next(case for case in cases if case.case_id == "recipe-query-brooklyn")

    metrics = evaluate([recipe_case], "data/delivery_logs.json")

    assert metrics.retrieval_hit_rate == 1.0
    assert metrics.answer_correctness == 1.0
    assert metrics.citation_correctness == 1.0


def test_evaluation_uses_production_source_routing_for_mixed_corpus() -> None:
    cases = load_golden_cases("evaluation/golden.json")
    inventory_case = next(case for case in cases if case.case_id == "brooklyn-oat-milk-isolation")

    metrics = evaluate([inventory_case], "data/delivery_logs.json")

    assert metrics.retrieval_hit_rate == 1.0
    assert metrics.answer_correctness == 1.0
    assert metrics.citation_correctness == 1.0


def test_positive_retrieval_miss_reduces_only_positive_hit_rate() -> None:
    cases = load_golden_cases("evaluation/golden.json")
    positive_case = next(case for case in cases if case.expected_document_ids)
    missed_case = replace(positive_case, expected_document_ids=("delivery:missing:0",))

    metrics = evaluate([positive_case, missed_case], "data/delivery_logs.json")

    assert metrics.retrieval_hit_rate == 0.5
    assert metrics.abstention_success_rate == 1.0


def test_expected_abstention_has_its_own_metric_and_does_not_lower_hit_rate() -> None:
    cases = load_golden_cases("evaluation/golden.json")
    positive_case = next(case for case in cases if case.expected_document_ids)
    abstention_case = next(case for case in cases if not case.expected_document_ids)
    leaking_case = replace(
        positive_case,
        case_id="unexpected-retrieval",
        expected_document_ids=(),
    )

    metrics = evaluate([positive_case, abstention_case, leaking_case], "data/delivery_logs.json")

    assert metrics.retrieval_hit_rate == 1.0
    assert metrics.abstention_success_rate == 0.5


def test_evaluation_exercises_cross_store_isolation_and_empty_retrieval() -> None:
    cases = load_golden_cases("evaluation/golden.json")

    isolation_cases = [case for case in cases if not case.expected_document_ids]
    assert len(isolation_cases) >= 2
    isolation_queries = {case.case_id for case in isolation_cases}
    assert "cross-store-isolation-brooklyn-cannot-see-manhattan" in isolation_queries
    assert "unknown-sku-empty-retrieval" in isolation_queries


def test_cross_store_case_abstains_from_existing_other_store_document() -> None:
    cases = load_golden_cases("evaluation/golden.json")
    isolation_case = next(
        case
        for case in cases
        if case.case_id == "cross-store-isolation-brooklyn-cannot-see-manhattan"
    )
    manhattan_case = replace(
        isolation_case,
        case_id="manhattan-can-see-own-delivery",
        store_id="MANHATTAN-01",
        expected_document_ids=("delivery:DLV-MN-1001:0",),
        answer_must_contain=("3 cartons",),
        answer_must_not_contain=(),
    )

    metrics = evaluate([manhattan_case, isolation_case], "data/delivery_logs.json")

    assert metrics.retrieval_hit_rate == 1.0
    assert metrics.abstention_success_rate == 1.0
    assert metrics.correct_store_retrieval_rate == 1.0
    assert metrics.answer_correctness == 1.0
    assert metrics.citation_correctness == 1.0


def test_evaluation_calculates_configured_token_cost() -> None:
    cases = load_golden_cases("evaluation/golden.json")[:1]

    metrics = evaluate(
        cases,
        "data/delivery_logs.json",
        input_cost_per_million_tokens=1.0,
        output_cost_per_million_tokens=2.0,
    )

    assert metrics.estimated_cost_per_query_usd > 0


def test_golden_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    case = {
        "case_id": "duplicate",
        "query": "Do we have oat milk?",
        "store_id": "brooklyn-01",
        "expected_document_ids": ["delivery:DLV-BK-1001:0"],
        "answer_must_contain": ["12 cartons"],
    }
    path = tmp_path / "golden.json"
    path.write_text(json.dumps([case, case]), encoding="utf-8")

    with pytest.raises(EvaluationError, match="unique"):
        load_golden_cases(path)
