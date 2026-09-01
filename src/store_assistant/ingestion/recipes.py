"""Recipe HTML ingestion with section-aware chunking."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Final

from store_assistant.ingestion.models import DocumentChunk, MetadataValue
from store_assistant.ingestion.normalization import (
    NormalizationError,
    normalize_date,
    normalize_product_category,
    normalize_sku,
    normalize_store_id,
    normalize_supplier,
)

_REQUIRED_METADATA: Final = {
    "recipe-id",
    "updated-at",
    "store-id",
    "sku",
    "product-category",
    "supplier",
}


class RecipeIngestionError(ValueError):
    """Raised when a recipe HTML document cannot be safely ingested."""


class _RecipeParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.metadata: dict[str, str] = {}
        self.title_parts: list[str] = []
        self.sections: list[tuple[str, str]] = []
        self._in_title = False
        self._section_depth = 0
        self._section_heading: list[str] = []
        self._section_body: list[str] = []
        self._in_section_heading = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        raw_name = attributes.get("name")
        if tag == "meta" and raw_name is not None and raw_name.startswith("recipe:"):
            name = raw_name.removeprefix("recipe:")
            self.metadata[name] = attributes.get("content") or ""
        if tag == "h1":
            self._in_title = True
        if tag == "section":
            if self._section_depth == 0:
                self._section_heading = []
                self._section_body = []
            self._section_depth += 1
        elif self._section_depth and tag in {"h2", "h3"}:
            self._in_section_heading = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "h1":
            self._in_title = False
        if self._section_depth and tag in {"h2", "h3"}:
            self._in_section_heading = False
        if tag == "section" and self._section_depth:
            self._section_depth -= 1
            if self._section_depth == 0:
                heading = _clean_text(self._section_heading)
                body = _clean_text(self._section_body)
                if heading or body:
                    self.sections.append((heading, body))

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(data)
        if self._section_depth:
            target = self._section_heading if self._in_section_heading else self._section_body
            target.append(data)


def ingest_recipe_html(path: str | Path) -> list[DocumentChunk]:
    """Load one recipe and emit one retrievable chunk per logical HTML section."""
    source_path = Path(path)
    try:
        source = source_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RecipeIngestionError(f"unable to read recipe HTML {source_path}") from exc

    parser = _RecipeParser()
    try:
        parser.feed(source)
        parser.close()
    except ValueError as exc:
        raise RecipeIngestionError(f"unable to parse recipe HTML {source_path}") from exc

    missing = sorted(key for key in _REQUIRED_METADATA if not parser.metadata.get(key, "").strip())
    if missing:
        raise RecipeIngestionError(f"recipe is missing metadata: {', '.join(missing)}")
    title = _clean_text(parser.title_parts)
    if not title:
        raise RecipeIngestionError("recipe is missing an h1 title")
    if not parser.sections:
        raise RecipeIngestionError("recipe contains no logical sections")

    raw = parser.metadata
    try:
        metadata: dict[str, MetadataValue] = {
            "source_type": "recipe",
            "recipe_id": raw["recipe-id"].strip(),
            "updated_at": normalize_date(raw["updated-at"]),
            "store_id": normalize_store_id(raw["store-id"]),
            "sku": normalize_sku(raw["sku"]),
            "product_category": normalize_product_category(raw["product-category"]),
            "supplier": normalize_supplier(raw["supplier"]),
            "recipe_title": title,
        }
    except NormalizationError as exc:
        raise RecipeIngestionError(f"recipe has invalid metadata: {exc}") from exc

    chunks: list[DocumentChunk] = []
    for index, (heading, body) in enumerate(parser.sections):
        section_name = heading or f"Section {index + 1}"
        text = f"{title} — {section_name}: {body}" if body else f"{title} — {section_name}"
        chunks.append(
            DocumentChunk(
                document_id=f"recipe:{raw['recipe-id'].strip()}:{index}",
                text=text,
                metadata={**metadata, "section": section_name, "chunk_index": index},
            )
        )
    return chunks


def _clean_text(parts: list[str]) -> str:
    return " ".join(" ".join(parts).split())
