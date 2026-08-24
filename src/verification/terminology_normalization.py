
"""Terminology normalization utilities for claim verification."""

from __future__ import annotations

import re
from collections.abc import Mapping


def normalize_terminology(
    text: str,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Normalize known aliases without changing the source text itself."""
    if not aliases:
        return text

    normalized = text

    ordered_aliases = sorted(
        aliases.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    )

    for alias, canonical in ordered_aliases:
        if not alias:
            continue

        pattern = re.compile(
            rf"(?<!\w){re.escape(alias)}(?!\w)",
            flags=re.IGNORECASE,
        )

        normalized = pattern.sub(
            canonical,
            normalized,
        )

    return normalized
