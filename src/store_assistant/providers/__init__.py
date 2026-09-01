"""External-service provider boundaries and local implementations."""

from store_assistant.providers.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    LocalHashEmbeddingProvider,
)
from store_assistant.providers.reranking import (
    LocalLexicalReranker,
    RerankCandidate,
    Reranker,
    RerankingError,
    RerankResult,
)
from store_assistant.providers.vector_store import (
    InMemoryVectorStore,
    VectorRecord,
    VectorSearchResult,
    VectorStore,
    VectorStoreError,
)

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "InMemoryVectorStore",
    "LocalHashEmbeddingProvider",
    "LocalLexicalReranker",
    "RerankCandidate",
    "Reranker",
    "RerankingError",
    "RerankResult",
    "VectorRecord",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreError",
]
