"""Embedding provider contract and a deterministic, dependency-free local provider."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

Embedding = tuple[float, ...]
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class EmbeddingError(ValueError):
    """Raised when text cannot be embedded or provider configuration is invalid."""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Boundary implemented by local and production embedding services."""

    @property
    def dimension(self) -> int:
        """Return the fixed vector dimension emitted by this provider."""

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        """Embed a batch of document or query texts in input order."""


class BedrockRuntimeClient(Protocol):
    """Narrow subset of the Bedrock Runtime client used by Titan embeddings."""

    def invoke_model(self, **kwargs: object) -> dict[str, Any]:
        """Invoke a configured Bedrock model."""


@dataclass(frozen=True, slots=True)
class AmazonTitanEmbeddingProvider:
    """Amazon Titan Text Embeddings V2 adapter for a Bedrock Runtime client."""

    client: BedrockRuntimeClient
    model_id: str = "amazon.titan-embed-text-v2:0"
    dimensions: int = 1024
    normalize: bool = True

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise EmbeddingError("Bedrock model ID must be non-empty")
        if self.dimensions not in {256, 512, 1024}:
            raise EmbeddingError("Titan V2 dimensions must be 256, 512, or 1024")

    @property
    def dimension(self) -> int:
        return self.dimensions

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        vectors: list[Embedding] = []
        for position, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError(f"text at position {position} must be a non-empty string")
            request = {
                "inputText": text,
                "dimensions": self.dimensions,
                "normalize": self.normalize,
            }
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(request),
                    accept="application/json",
                    contentType="application/json",
                )
                payload = json.loads(response["body"].read())
                vector = tuple(float(value) for value in payload["embedding"])
            except Exception as exc:
                raise EmbeddingError(
                    f"Bedrock embedding failed for text at position {position}"
                ) from exc
            if len(vector) != self.dimensions or any(not math.isfinite(value) for value in vector):
                raise EmbeddingError(
                    f"Bedrock returned an invalid {self.dimensions}-dimension embedding"
                )
            vectors.append(vector)
        return vectors


@dataclass(frozen=True, slots=True)
class LocalHashEmbeddingProvider:
    """Feature-hashing embeddings for repeatable, offline development and tests.

    These vectors provide lightweight lexical similarity and are not intended to
    approximate the quality of a production semantic embedding model.
    """

    dimensions: int = 384

    def __post_init__(self) -> None:
        if self.dimensions < 8:
            raise EmbeddingError("embedding dimension must be at least 8")

    @property
    def dimension(self) -> int:
        return self.dimensions

    def embed(self, texts: Sequence[str]) -> list[Embedding]:
        vectors: list[Embedding] = []
        for position, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError(f"text at position {position} must be a non-empty string")
            vectors.append(self._embed_one(text))
        return vectors

    def _embed_one(self, text: str) -> Embedding:
        tokens = _TOKEN_PATTERN.findall(text.casefold())
        features = tokens + [
            f"{left}::{right}" for left, right in zip(tokens, tokens[1:], strict=False)
        ]
        values = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            encoded = int.from_bytes(digest)
            index = encoded % self.dimensions
            sign = 1.0 if encoded & (1 << 63) else -1.0
            values[index] += sign

        magnitude = math.sqrt(sum(value * value for value in values))
        if magnitude == 0:
            raise EmbeddingError("text must contain at least one alphanumeric token")
        return tuple(value / magnitude for value in values)
