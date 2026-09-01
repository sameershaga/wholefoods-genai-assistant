"""Source ingestion and metadata normalization."""
"""Source-aware ingestion pipelines."""

from store_assistant.ingestion.contracts import ingest_supplier_contract
from store_assistant.ingestion.delivery_logs import ingest_delivery_logs
from store_assistant.ingestion.recipes import ingest_recipe_html

__all__ = ["ingest_delivery_logs", "ingest_recipe_html", "ingest_supplier_contract"]
