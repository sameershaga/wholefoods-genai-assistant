"""External-service provider boundaries and local implementations."""

from store_assistant.providers.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    LocalHashEmbeddingProvider,
)

__all__ = ["EmbeddingError", "EmbeddingProvider", "LocalHashEmbeddingProvider"]
