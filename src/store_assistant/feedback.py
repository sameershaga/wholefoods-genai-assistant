"""Feedback persistence boundary and a local SQLite implementation."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Protocol, runtime_checkable


class FeedbackError(ValueError):
    """Raised when feedback cannot be validated or persisted."""


class FeedbackRating(StrEnum):
    """Ratings supported by the Slack thumbs controls."""

    UP = "up"
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class Feedback:
    """The latest rating a user submitted for one assistant response."""

    request_id: str
    user_id: str
    rating: FeedbackRating
    comment: str | None
    created_at: datetime
    updated_at: datetime


@runtime_checkable
class FeedbackRepository(Protocol):
    """Persistence boundary for response feedback."""

    def save(
        self,
        *,
        request_id: str,
        user_id: str,
        rating: FeedbackRating,
        comment: str | None = None,
    ) -> Feedback:
        """Create feedback or replace the same user's rating for a request."""

    def get(self, *, request_id: str, user_id: str) -> Feedback | None:
        """Return a user's current feedback for a request, if present."""


class SQLiteFeedbackRepository:
    """Small durable repository used by local mode and automated tests."""

    def __init__(self, database: str | Path) -> None:
        self._connection = sqlite3.connect(str(database), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = Lock()
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
                    comment TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (request_id, user_id)
                )
                """
            )

    def save(
        self,
        *,
        request_id: str,
        user_id: str,
        rating: FeedbackRating,
        comment: str | None = None,
    ) -> Feedback:
        request_id = self._required(request_id, "request_id")
        user_id = self._required(user_id, "user_id")
        if not isinstance(rating, FeedbackRating):
            raise FeedbackError("rating must be FeedbackRating.UP or FeedbackRating.DOWN")
        normalized_comment = comment.strip() if comment and comment.strip() else None
        now = datetime.now(UTC).isoformat()

        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO feedback (
                    request_id, user_id, rating, comment, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(request_id, user_id) DO UPDATE SET
                    rating = excluded.rating,
                    comment = excluded.comment,
                    updated_at = excluded.updated_at
                """,
                (request_id, user_id, rating.value, normalized_comment, now, now),
            )
            row = self._connection.execute(
                "SELECT * FROM feedback WHERE request_id = ? AND user_id = ?",
                (request_id, user_id),
            ).fetchone()
        if row is None:  # pragma: no cover - defensive check for SQLite failures
            raise FeedbackError("feedback was not persisted")
        return self._from_row(row)

    def get(self, *, request_id: str, user_id: str) -> Feedback | None:
        request_id = self._required(request_id, "request_id")
        user_id = self._required(user_id, "user_id")
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM feedback WHERE request_id = ? AND user_id = ?",
                (request_id, user_id),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def close(self) -> None:
        """Release the underlying database connection."""

        with self._lock:
            self._connection.close()

    @staticmethod
    def _required(value: str, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise FeedbackError(f"{field} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Feedback:
        return Feedback(
            request_id=str(row["request_id"]),
            user_id=str(row["user_id"]),
            rating=FeedbackRating(row["rating"]),
            comment=row["comment"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
