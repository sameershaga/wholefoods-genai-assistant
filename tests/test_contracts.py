from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from store_assistant.ingestion.contracts import (
    ContractIngestionError,
    ingest_supplier_contract,
    ingest_supplier_contracts,
)
from store_assistant.ingestion.models import DocumentChunk


def _reader_with_text(text: str) -> MagicMock:
    page = MagicMock()
    page.extract_text.return_value = text
    reader = MagicMock()
    reader.pages = [page]
    return reader


def _contract(body: str = "Pricing is fixed for the stated term.") -> str:
    return "\n".join(
        [
            "Contract ID: CON-2026-001",
            "Effective Date: 09/01/2026",
            "Store ID: brooklyn 01",
            "Supplier:  Green Valley   Foods ",
            "Product Category: Dairy Alternatives",
            "SKU: oat-001",
            body,
        ]
    )


@patch("store_assistant.ingestion.contracts.PdfReader")
def test_ingests_pdf_contract_with_normalized_filter_metadata(reader: MagicMock) -> None:
    reader.return_value = _reader_with_text(_contract())

    chunks = ingest_supplier_contract(Path("contract.pdf"))

    assert len(chunks) == 1
    assert chunks[0].document_id == "contract:CON-2026-001:0"
    assert chunks[0].metadata == {
        "source_type": "supplier_contract",
        "contract_id": "CON-2026-001",
        "effective_date": "2026-09-01T00:00:00Z",
        "store_id": "BROOKLYN-01",
        "supplier": "green valley foods",
        "product_category": "dairy_alternatives",
        "sku": "OAT001",
        "chunk_index": 0,
    }
    assert chunks[0].text == "Pricing is fixed for the stated term."


@patch("store_assistant.ingestion.contracts.PdfReader")
def test_contract_chunks_respect_source_specific_limit(reader: MagicMock) -> None:
    reader.return_value = _reader_with_text(_contract("term " * 1601))

    chunks = ingest_supplier_contract("contract.pdf")

    assert [len(chunk.text.split()) for chunk in chunks] == [800, 800, 1]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [0, 1, 2]


@pytest.mark.parametrize(
    "text,error",
    [
        ("", "no extractable text"),
        ("Contract ID: CON-1\nSome terms", "missing metadata"),
        (_contract(""), "no body text"),
    ],
)
@patch("store_assistant.ingestion.contracts.PdfReader")
def test_rejects_unusable_contracts(reader: MagicMock, text: str, error: str) -> None:
    reader.return_value = _reader_with_text(text)

    with pytest.raises(ContractIngestionError, match=error):
        ingest_supplier_contract("contract.pdf")


@patch("store_assistant.ingestion.contracts.ingest_supplier_contract")
def test_ingests_contract_directory_in_deterministic_order(
    ingest_one: MagicMock, tmp_path: Path
) -> None:
    (tmp_path / "zebra.PDF").touch()
    (tmp_path / "Alpha.pdf").touch()
    (tmp_path / "notes.txt").touch()
    ingest_one.side_effect = lambda path: [
        DocumentChunk(f"contract:{path.stem}:0", f"Terms for {path.stem}", {})
    ]

    chunks = ingest_supplier_contracts(tmp_path)

    assert [chunk.document_id for chunk in chunks] == [
        "contract:Alpha:0",
        "contract:zebra:0",
    ]
    assert [call.args[0].name for call in ingest_one.call_args_list] == ["Alpha.pdf", "zebra.PDF"]


def test_rejects_contract_directory_without_pdfs(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").touch()

    with pytest.raises(ContractIngestionError, match="contains no PDF files"):
        ingest_supplier_contracts(tmp_path)


@patch("store_assistant.ingestion.contracts.ingest_supplier_contract")
def test_rejects_duplicate_document_ids_across_contracts(
    ingest_one: MagicMock, tmp_path: Path
) -> None:
    (tmp_path / "one.pdf").touch()
    (tmp_path / "two.pdf").touch()
    ingest_one.return_value = [DocumentChunk("contract:duplicate:0", "Terms", {})]

    with pytest.raises(ContractIngestionError, match="duplicate contract document ID"):
        ingest_supplier_contracts(tmp_path)
