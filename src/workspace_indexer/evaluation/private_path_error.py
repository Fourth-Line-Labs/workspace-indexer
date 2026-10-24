"""Raised when redaction fails to remove a private path from a record."""

from __future__ import annotations


class PrivatePathError(RuntimeError):
    """A private path survived redaction, so the record is not written.

    The backstop, not the mechanism. Reaching this means the redaction has a
    bug, and writing the file anyway would publish exactly what the guard
    exists to prevent.
    """
