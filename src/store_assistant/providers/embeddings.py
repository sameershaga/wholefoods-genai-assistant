"""Embedding provider contract and a deterministic, dependency-free local provider."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

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
        features = tokens + [f"{left}::{right}" for left, right in zip(tokens, tokens[1:])]
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
