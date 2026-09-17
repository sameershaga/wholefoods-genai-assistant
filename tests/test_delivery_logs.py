import json
from pathlib import Path

import pytest

from store_assistant.ingestion.delivery_logs import DeliveryLogError, ingest_delivery_logs


def test_ingests_synthetic_multi_store_inventory_with_normalized_metadata() -> None:
    chunks = ingest_delivery_logs(Path("data/delivery_logs.json"))

    assert len(chunks) == 3
    by_store = {chunk.metadata["store_id"]: chunk for chunk in chunks}
    assert set(by_store) == {"BROOKLYN-01", "MANHATTAN-01", "QUEENS-01"}
    assert by_store["BROOKLYN-01"].metadata["quantity"] == 12
    assert by_store["MANHATTAN-01"].metadata["quantity"] == 3
    assert all(chunk.metadata["sku"] == "OAT001" for chunk in chunks)
    assert all(chunk.metadata["source_type"] == "delivery_log" for chunk in chunks)
    assert all(len(chunk.text.split()) <= 50 for chunk in chunks)
    assert len({chunk.document_id for chunk in chunks}) == len(chunks)


def test_long_delivery_record_is_split_at_source_specific_limit(tmp_path: Path) -> None:
    record = {
        "delivery_id": "DLV-1",
        "delivered_at": "08/31/2026",
        "store_id": "brooklyn-01",
        "sku": "OAT-001",
        "product_name": "very " * 60 + "long product",
        "product_category": "Dairy Alternatives",
        "supplier": "Supplier One",
        "quantity": 2,
        "unit": "cases",
    }
    path = tmp_path / "deliveries.json"
    path.write_text(json.dumps([record]), encoding="utf-8")

    chunks = ingest_delivery_logs(path)

    assert len(chunks) == 2
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1]
    assert all(len(chunk.text.split()) <= 50 for chunk in chunks)


@pytest.mark.parametrize(
    "payload,error",
    [
        ({}, "root must be a JSON array"),
        ([{"delivery_id": "DLV-1"}], "missing fields"),
        ([{"delivery_id": "DLV-1", "delivered_at": "bad"}], "missing fields"),
    ],
)
def test_rejects_malformed_records(tmp_path: Path, payload: object, error: str) -> None:
    path = tmp_path / "deliveries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DeliveryLogError, match=error):
        ingest_delivery_logs(path)
