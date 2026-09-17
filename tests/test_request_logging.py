from datetime import UTC, datetime

import pytest

from store_assistant.request_logging import (
    JSONLRequestLogRepository,
    RequestLog,
    RequestLogError,
    RequestLogRepository,
    RetrievedDocumentLog,
)


def make_record() -> RequestLog:
    return RequestLog(
        request_id="req-123",
        query="Do we have oat milk?",
        user_id="manager-brooklyn",
        store_id="BROOKLYN",
        retrieved_documents=(RetrievedDocumentLog("delivery:brooklyn:oat", 0.91, 0.98),),
        model="local-extractive-v1",
        prompt_tokens=42,
        completion_tokens=12,
        estimated_cost_usd=0.00003,
        latency_ms=18.5,
        final_answer="Brooklyn has 12 cartons. [source: delivery:brooklyn:oat]",
        timestamp=datetime(2026, 9, 1, 12, 0, tzinfo=UTC),
    )


def test_jsonl_repository_round_trips_complete_structured_record(tmp_path) -> None:
    repository = JSONLRequestLogRepository(tmp_path / "logs" / "requests.jsonl")

    repository.append(make_record())

    assert repository.read_all() == [make_record()]
    raw = (tmp_path / "logs" / "requests.jsonl").read_text()
    assert '"retrieval_score":0.91' in raw
    assert '"reranking_score":0.98' in raw
    assert raw.endswith("\n")


def test_repository_appends_without_replacing_prior_requests(tmp_path) -> None:
    repository = JSONLRequestLogRepository(tmp_path / "requests.jsonl")
    first = make_record()
    second = RequestLog.create(
        request_id="req-456",
        query="When is the delivery?",
        user_id="manager-brooklyn",
        store_id="BROOKLYN",
        retrieved_documents=[],
        model="none",
        prompt_tokens=0,
        completion_tokens=0,
        estimated_cost_usd=0,
        latency_ms=2,
        final_answer="No relevant information was found.",
    )

    repository.append(first)
    repository.append(second)

    assert [record.request_id for record in repository.read_all()] == ["req-123", "req-456"]
    assert second.timestamp.tzinfo is not None


def test_repository_conforms_to_protocol(tmp_path) -> None:
    assert isinstance(JSONLRequestLogRepository(tmp_path / "requests.jsonl"), RequestLogRepository)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"query": " "}, "query"),
        ({"prompt_tokens": -1}, "token counts"),
        ({"estimated_cost_usd": -0.01}, "estimated_cost_usd"),
        ({"estimated_cost_usd": float("nan")}, "estimated_cost_usd"),
        ({"latency_ms": -1}, "latency_ms"),
        ({"timestamp": datetime(2026, 1, 1)}, "timezone-aware"),
    ],
)
def test_request_log_rejects_invalid_telemetry(changes, message) -> None:
    values = {field: getattr(make_record(), field) for field in make_record().__dataclass_fields__}
    values.update(changes)
    with pytest.raises(RequestLogError, match=message):
        RequestLog(**values)


def test_repository_reports_corrupt_local_records(tmp_path) -> None:
    path = tmp_path / "requests.jsonl"
    path.write_text("not json\n")

    with pytest.raises(RequestLogError, match="invalid record"):
        JSONLRequestLogRepository(path).read_all()
