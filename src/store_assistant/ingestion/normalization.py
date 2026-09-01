"""Canonical metadata normalization shared by all ingestion sources."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Final

_SEPARATOR_RE: Final = re.compile(r"[^A-Z0-9]+")
_WHITESPACE_RE: Final = re.compile(r"\s+")


class NormalizationError(ValueError):
    """Raised when required source metadata cannot be normalized safely."""


def normalize_date(value: str | date | datetime) -> str:
    """Return an ISO-8601 UTC timestamp for common source date representations."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    else:
        cleaned = value.strip()
        if not cleaned:
            raise NormalizationError("date must not be empty")
        try:
            parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        except ValueError:
            parsed = _parse_us_date(cleaned)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_us_date(value: str) -> datetime:
    for date_format in ("%m/%d/%Y", "%m-%d-%Y"):
        try:
            return datetime.strptime(value, date_format).replace(tzinfo=UTC)
        except ValueError:
            continue
    raise NormalizationError(f"unsupported date format: {value!r}")


def normalize_sku(value: str | int) -> str:
    """Return a comparison-safe SKU while retaining meaningful leading zeroes."""
    normalized = _SEPARATOR_RE.sub("", str(value).strip().upper())
    if not normalized:
        raise NormalizationError("SKU must contain letters or digits")
    return normalized


def normalize_store_id(value: str | int) -> str:
    """Return the canonical store identifier used by authorization and filtering."""
    normalized = _SEPARATOR_RE.sub("-", str(value).strip().upper()).strip("-")
    if not normalized:
        raise NormalizationError("store ID must contain letters or digits")
    return normalized


def normalize_supplier(value: str) -> str:
    """Normalize supplier display names without discarding punctuation."""
    normalized = _WHITESPACE_RE.sub(" ", value.strip())
    if not normalized:
        raise NormalizationError("supplier must not be empty")
    return normalized.casefold()


def normalize_product_category(value: str) -> str:
    """Return a stable snake-case product category suitable for metadata filters."""
    normalized = _SEPARATOR_RE.sub("_", value.strip().upper()).strip("_").lower()
    if not normalized:
        raise NormalizationError("product category must contain letters or digits")
    return normalized

