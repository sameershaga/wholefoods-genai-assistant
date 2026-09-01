"""External-service provider boundaries and local implementations."""

from store_assistant.providers.embeddings import (
    AmazonTitanEmbeddingProvider,
    EmbeddingError,
    EmbeddingProvider,
    LocalHashEmbeddingProvider,
)
from store_assistant.providers.llm import (
    AmazonBedrockLLM,
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
    PineconeVectorStore,
    VectorRecord,
    VectorSearchResult,
    VectorStore,
    VectorStoreError,
)

__all__ = [
    "AmazonBedrockLLM",
    "AmazonTitanEmbeddingProvider",
    "EmbeddingError",
    "EmbeddingProvider",
    "GeneratedAnswer",
    "InMemoryVectorStore",
    "PineconeVectorStore",
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
