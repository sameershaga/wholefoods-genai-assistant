import json
from pathlib import Path

import pytest

from store_assistant.evaluation import EvaluationError, evaluate, load_golden_cases


def test_bundled_golden_evaluation_meets_local_quality_targets() -> None:
    cases = load_golden_cases("evaluation/golden.json")

    metrics = evaluate(cases, "data/delivery_logs.json")

    assert metrics.case_count == 3
    assert metrics.retrieval_hit_rate == 1.0
    assert metrics.correct_store_retrieval_rate == 1.0
    assert metrics.answer_correctness == 1.0
    assert metrics.citation_correctness == 1.0
    assert metrics.average_latency_ms >= 0
    assert metrics.estimated_cost_per_query_usd == 0


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
