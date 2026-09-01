"""Structured request telemetry and a dependency-free local JSONL sink."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Protocol, Sequence, runtime_checkable


class RequestLogError(ValueError):
    """Raised when request telemetry is invalid or cannot be decoded."""


@dataclass(frozen=True, slots=True)
class RetrievedDocumentLog:
    """Scores retained for one document selected as answer context."""

    document_id: str
    retrieval_score: float
    reranking_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise RequestLogError("document_id must be a non-empty string")
        if not isinstance(self.retrieval_score, (int, float)) or not math.isfinite(
            self.retrieval_score
        ):
            raise RequestLogError("retrieval_score must be numeric")
        if not isinstance(self.reranking_score, (int, float)) or not math.isfinite(
            self.reranking_score
        ):
            raise RequestLogError("reranking_score must be numeric")
        object.__setattr__(self, "document_id", self.document_id.strip())
        object.__setattr__(self, "retrieval_score", float(self.retrieval_score))
        object.__setattr__(self, "reranking_score", float(self.reranking_score))


@dataclass(frozen=True, slots=True)
class RequestLog:
    """Complete operational telemetry for one assistant request."""

    request_id: str
    query: str
    user_id: str
    store_id: str
    retrieved_documents: tuple[RetrievedDocumentLog, ...]
    model: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    final_answer: str
    timestamp: datetime

    def __post_init__(self) -> None:
        for field in ("request_id", "query", "user_id", "store_id", "model", "final_answer"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise RequestLogError(f"{field} must be a non-empty string")
            object.__setattr__(self, field, value.strip())
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise RequestLogError("token counts must not be negative")
        if not math.isfinite(self.estimated_cost_usd) or self.estimated_cost_usd < 0:
            raise RequestLogError("estimated_cost_usd must not be negative")
        if not math.isfinite(self.latency_ms) or self.latency_ms < 0:
            raise RequestLogError("latency_ms must not be negative")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise RequestLogError("timestamp must be timezone-aware")

    @classmethod
    def create(
        cls,
        *,
        request_id: str,
        query: str,
        user_id: str,
        store_id: str,
        retrieved_documents: Sequence[RetrievedDocumentLog],
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        estimated_cost_usd: float,
        latency_ms: float,
        final_answer: str,
    ) -> RequestLog:
        """Build a timestamped immutable request record."""
        return cls(
            request_id=request_id,
            query=query,
            user_id=user_id,
            store_id=store_id,
            retrieved_documents=tuple(retrieved_documents),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost_usd,
            latency_ms=latency_ms,
            final_answer=final_answer,
            timestamp=datetime.now(UTC),
        )


@runtime_checkable
class RequestLogRepository(Protocol):
    """Append-only persistence boundary for request telemetry."""

    def append(self, record: RequestLog) -> None:
        """Persist one structured request record."""


class JSONLRequestLogRepository:
    """Thread-safe local JSON Lines implementation suitable for local mode."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, record: RequestLog) -> None:
        if not isinstance(record, RequestLog):
            raise RequestLogError("record must be a RequestLog")
        payload = asdict(record)
        payload["timestamp"] = record.timestamp.astimezone(UTC).isoformat()
        line = json.dumps(payload, separators=(",", ":"), sort_keys=True, allow_nan=False)
        with self._lock, self._path.open("a", encoding="utf-8") as output:
            output.write(f"{line}\n")

    def read_all(self) -> list[RequestLog]:
        """Read local records, primarily for diagnostics and tests."""
        if not self._path.exists():
            return []
        records: list[RequestLog] = []
        try:
            with self._lock, self._path.open(encoding="utf-8") as source:
                for line in source:
                    payload = json.loads(line)
                    payload["retrieved_documents"] = tuple(
                        RetrievedDocumentLog(**item)
                        for item in payload["retrieved_documents"]
                    )
                    payload["timestamp"] = datetime.fromisoformat(payload["timestamp"])
                    records.append(RequestLog(**payload))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RequestLogError("request log contains an invalid record") from exc
        return records
