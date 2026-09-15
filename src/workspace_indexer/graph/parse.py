"""Parsing a file once, for every walker that needs it.

Two passes read a C# file's syntax tree -- imports, and the namespace
declarations those imports resolve against -- and parsing is the expensive
half. Kept here rather than in either scanner so neither owns the other, and
so a caller with no tree still gets one.
"""

from __future__ import annotations

import tree_sitter_language_pack as tslp
from structlog.stdlib import BoundLogger
from tree_sitter import Tree


def parse(text: str, language: str, *, log: BoundLogger) -> Tree | None:
    """The syntax tree, or None when the grammar is unavailable.

    A grammar cache miss with no network must cost the edges, never the file:
    the caller returns nothing rather than raising, and indexing continues.
    """
    try:
        return tslp.get_parser(language).parse(text.encode())  # pyright: ignore[reportArgumentType]
    except Exception as exc:
        log.debug("graph.parse_failed", language=language, error=str(exc))
        return None
