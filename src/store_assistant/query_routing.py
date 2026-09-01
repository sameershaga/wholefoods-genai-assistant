"""Deterministic source routing for the local mixed-source index."""

from __future__ import annotations

import re
from typing import Final

_TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]+")
_SOURCE_TERMS: Final[tuple[tuple[str, frozenset[str]], ...]] = (
    (
        "supplier_contract",
        frozenset({"agreement", "contract", "renewal", "terms"}),
    ),
    (
        "recipe",
        frozenset({"cook", "ingredients", "make", "prepare", "recipe"}),
    ),
    (
        "delivery_log",
        frozenset(
            {
                "arrive",
                "arrived",
                "carton",
                "cartons",
                "delivered",
                "delivery",
                "have",
                "inventory",
                "stock",
            }
        ),
    ),
)


def infer_source_type(query: str) -> str | None:
    """Infer an unambiguous source filter, leaving uncertain queries unrestricted."""
    tokens = set(_TOKEN_PATTERN.findall(query.casefold()))
    matches = [source_type for source_type, terms in _SOURCE_TERMS if tokens & terms]
    return matches[0] if len(matches) == 1 else None
