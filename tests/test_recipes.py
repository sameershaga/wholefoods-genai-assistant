from pathlib import Path

import pytest

from store_assistant.ingestion.recipes import RecipeIngestionError, ingest_recipe_html


def _recipe_html(*, sections: str | None = None, metadata: str | None = None) -> str:
    recipe_metadata = (
        metadata
        or """
      <meta name="recipe:recipe-id" content="REC-001">
      <meta name="recipe:updated-at" content="09/01/2026">
      <meta name="recipe:store-id" content="brooklyn 01">
      <meta name="recipe:sku" content="oat-001">
      <meta name="recipe:product-category" content="Dairy Alternatives">
      <meta name="recipe:supplier" content=" Green Valley   Foods ">
    """
    )
    recipe_sections = (
        sections
        if sections is not None
        else """
      <section><h2>Ingredients</h2><ul><li>Oats</li><li>Oat milk</li></ul></section>
      <section><h2>Method</h2><ol><li>Mix well.</li><li>Chill.</li></ol></section>
    """
    )
    return (
        f"<html><head>{recipe_metadata}</head><body>"
        f"<h1>Overnight Oats</h1>{recipe_sections}</body></html>"
    )


def test_recipe_chunks_follow_logical_sections_and_normalize_metadata(tmp_path: Path) -> None:
    source = tmp_path / "recipe.html"
    source.write_text(_recipe_html(), encoding="utf-8")

    chunks = ingest_recipe_html(source)

    assert [chunk.document_id for chunk in chunks] == ["recipe:REC-001:0", "recipe:REC-001:1"]
    assert [chunk.text for chunk in chunks] == [
        "Overnight Oats — Ingredients: Oats Oat milk",
        "Overnight Oats — Method: Mix well. Chill.",
    ]
    assert chunks[0].metadata == {
        "source_type": "recipe",
        "recipe_id": "REC-001",
        "updated_at": "2026-09-01T00:00:00Z",
        "store_id": "BROOKLYN-01",
        "sku": "OAT001",
        "product_category": "dairy_alternatives",
        "supplier": "green valley foods",
        "recipe_title": "Overnight Oats",
        "section": "Ingredients",
        "chunk_index": 0,
    }


@pytest.mark.parametrize(
    ("html", "error"),
    [
        ("<html><h1>Recipe</h1><section>Method</section></html>", "missing metadata"),
        (_recipe_html(sections=""), "no logical sections"),
        (_recipe_html().replace("<h1>Overnight Oats</h1>", ""), "missing an h1 title"),
        (
            _recipe_html().replace('content="09/01/2026"', 'content="not-a-date"'),
            "invalid metadata",
        ),
    ],
)
def test_rejects_unusable_recipe_html(tmp_path: Path, html: str, error: str) -> None:
    source = tmp_path / "recipe.html"
    source.write_text(html, encoding="utf-8")

    with pytest.raises(RecipeIngestionError, match=error):
        ingest_recipe_html(source)


def test_representative_recipe_fixture_is_ingestible() -> None:
    chunks = ingest_recipe_html(Path("data/recipes/oat_milk_overnight_oats.html"))

    assert len(chunks) == 2
    assert {chunk.metadata["store_id"] for chunk in chunks} == {"BROOKLYN-01"}
    assert {chunk.metadata["sku"] for chunk in chunks} == {"OAT001"}
