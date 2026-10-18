"""Raised when an eval case has no hand-written pattern for a lexical tool."""

from __future__ import annotations


class MissingPatternError(KeyError):
    """No `grep_pattern` for a case.

    Loud on purpose. The alternative -- falling back to the raw English query
    -- is the strawman baseline the comparison exists to avoid, and it would
    fail silently as a plausible-looking row of zeroes.
    """
