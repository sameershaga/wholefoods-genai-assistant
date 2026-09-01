from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from store_assistant.ingestion.normalization import (
    NormalizationError,
    normalize_date,
    normalize_product_category,
    normalize_sku,
    normalize_store_id,
    normalize_supplier,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2026-08-31T14:30:00Z", "2026-08-31T14:30:00Z"),
        ("08/31/2026", "2026-08-31T00:00:00Z"),
        (date(2026, 8, 31), "2026-08-31T00:00:00Z"),
        (
            datetime(2026, 8, 31, 10, 30, tzinfo=timezone(timedelta(hours=-4))),
            "2026-08-31T14:30:00Z",
        ),
        (datetime(2026, 8, 31, 14, 30), "2026-08-31T14:30:00Z"),
    ],
)
def test_normalize_date(source: str | date | datetime, expected: str) -> None:
    assert normalize_date(source) == expected


def test_normalize_identifiers_and_filter_values() -> None:
    assert normalize_sku(" oat-001 ") == "OAT001"
    assert normalize_sku("00123") == "00123"
    assert normalize_store_id(" brooklyn 01 ") == "BROOKLYN-01"
    assert normalize_store_id(123) == "123"
    assert normalize_supplier("  Hudson   Valley Co-op ") == "hudson valley co-op"
    assert normalize_product_category(" Dairy & Alternatives ") == "dairy_alternatives"


@pytest.mark.parametrize(
    "normalizer,value",
    [
        (normalize_date, "yesterday"),
        (normalize_sku, "---"),
        (normalize_store_id, "  "),
        (normalize_supplier, "  "),
        (normalize_product_category, "&"),
    ],
)
def test_invalid_values_raise_domain_error(
    normalizer: Callable[[str], object], value: str
) -> None:
    with pytest.raises(NormalizationError):
        normalizer(value)


def test_utc_constant_is_available_on_supported_python() -> None:
    assert UTC.utcoffset(None) == timedelta(0)
