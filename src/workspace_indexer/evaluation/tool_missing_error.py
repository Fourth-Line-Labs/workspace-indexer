"""Raised when a lexical tool's binary is not installed."""

from __future__ import annotations


class ToolMissingError(RuntimeError):
    """The binary is not installed.

    Raised rather than scored as a miss: a missing tool is nothing like a tool
    that searched and found nothing, and recording it as zero recall would
    publish a number that defames it.
    """
