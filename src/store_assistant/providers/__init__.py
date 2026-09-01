"""External-service provider boundaries and local implementations."""

from store_assistant.providers.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    LocalHashEmbeddingProvider,
)
from store_assistant.providers.llm import (
    GeneratedAnswer,
    LLMError,
    LLMProvider,
    LocalExtractiveLLM,
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
    "GeneratedAnswer",
    "InMemoryVectorStore",
    "LocalHashEmbeddingProvider",
    "LLMError",
    "LLMProvider",
    "LocalExtractiveLLM",
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
