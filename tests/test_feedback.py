from pathlib import Path

import pytest

from store_assistant.feedback import (
    FeedbackError,
    FeedbackRating,
    FeedbackRepository,
    SQLiteFeedbackRepository,
)


def test_feedback_is_persisted_across_repository_instances(tmp_path: Path) -> None:
    database = tmp_path / "feedback.sqlite3"
    repository = SQLiteFeedbackRepository(database)
    saved = repository.save(
        request_id="request-123",
        user_id="U-BROOKLYN-1",
        rating=FeedbackRating.UP,
        comment="  Helpful answer  ",
    )
    repository.close()

    reopened = SQLiteFeedbackRepository(database)
    assert reopened.get(request_id="request-123", user_id="U-BROOKLYN-1") == saved
    assert saved.comment == "Helpful answer"
    assert saved.created_at.tzinfo is not None
    reopened.close()


def test_new_rating_replaces_prior_user_rating_without_losing_creation_time() -> None:
    repository = SQLiteFeedbackRepository(":memory:")
    original = repository.save(request_id="request-123", user_id="U1", rating=FeedbackRating.UP)
    replacement = repository.save(
        request_id="request-123",
        user_id="U1",
        rating=FeedbackRating.DOWN,
        comment="Inventory was stale",
    )

    assert replacement.rating is FeedbackRating.DOWN
    assert replacement.comment == "Inventory was stale"
    assert replacement.created_at == original.created_at
    assert replacement.updated_at >= original.updated_at
    repository.close()


def test_feedback_is_isolated_by_request_and_user() -> None:
    repository = SQLiteFeedbackRepository(":memory:")
    repository.save(request_id="request-1", user_id="U1", rating=FeedbackRating.UP)

    assert repository.get(request_id="request-1", user_id="U2") is None
    assert repository.get(request_id="request-2", user_id="U1") is None
    repository.close()


def test_sqlite_repository_conforms_to_boundary() -> None:
    repository = SQLiteFeedbackRepository(":memory:")
    assert isinstance(repository, FeedbackRepository)
    repository.close()


@pytest.mark.parametrize(
    ("request_id", "user_id"),
    [("", "U1"), ("request-1", " ")],
)
def test_feedback_rejects_missing_identifiers(request_id: str, user_id: str) -> None:
    repository = SQLiteFeedbackRepository(":memory:")
    with pytest.raises(FeedbackError, match="non-empty"):
        repository.save(
            request_id=request_id,
            user_id=user_id,
            rating=FeedbackRating.UP,
        )
    repository.close()


def test_feedback_rejects_untyped_rating() -> None:
    repository = SQLiteFeedbackRepository(":memory:")
    with pytest.raises(FeedbackError, match="FeedbackRating"):
        repository.save(request_id="request-1", user_id="U1", rating="up")  # type: ignore[arg-type]
    repository.close()
