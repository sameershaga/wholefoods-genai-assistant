"""Authentication boundary and locally runnable mock OIDC implementation."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

from store_assistant.ingestion.normalization import NormalizationError, normalize_store_id


class AuthenticationError(ValueError):
    """Raised when credentials or identity claims cannot be trusted."""


@dataclass(frozen=True, slots=True)
class UserContext:
    """Trusted identity and store scope propagated through a request."""

    user_id: str
    store_id: str
    display_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.user_id, str):
            raise AuthenticationError("user_id must be a string")
        user_id = self.user_id.strip()
        if not user_id:
            raise AuthenticationError("user_id must not be empty")
        if not isinstance(self.store_id, (str, int)):
            raise AuthenticationError("store_id must be a string or integer")
        try:
            store_id = normalize_store_id(self.store_id)
        except (AttributeError, NormalizationError) as exc:
            raise AuthenticationError("store_id is invalid") from exc
        display_name = self.display_name.strip() if self.display_name is not None else None
        object.__setattr__(self, "user_id", user_id)
        object.__setattr__(self, "store_id", store_id)
        object.__setattr__(self, "display_name", display_name or None)


@runtime_checkable
class AuthProvider(Protocol):
    """Provider boundary implemented by local mock and production OIDC adapters."""

    def authenticate(self, access_token: str) -> UserContext:
        """Validate a token and return trusted, normalized request context."""
        ...


class MockAuthProvider:
    """Deterministic token authenticator for local development and tests."""

    def __init__(self, users_by_token: Mapping[str, UserContext]) -> None:
        validated: dict[str, UserContext] = {}
        for token, context in users_by_token.items():
            clean_token = token.strip()
            if not clean_token:
                raise AuthenticationError("mock access tokens must not be empty")
            if clean_token in validated:
                raise AuthenticationError("mock access tokens must be unique")
            if not isinstance(context, UserContext):
                raise AuthenticationError("mock identities must be UserContext instances")
            validated[clean_token] = context
        if not validated:
            raise AuthenticationError("at least one mock identity is required")
        self._users_by_token = MappingProxyType(validated)

    def authenticate(self, access_token: str) -> UserContext:
        if not isinstance(access_token, str) or not access_token.strip():
            raise AuthenticationError("access token is required")
        try:
            return self._users_by_token[access_token.strip()]
        except KeyError as exc:
            raise AuthenticationError("access token is invalid") from exc
