"""The resolution scope of a path: the repository it sits in.

Derived rather than stored, because it is exactly the first path segment and a
column would be a second source of truth for the same fact. Extracted to one
place because both the resolver and the origin classifier need it, and two
copies of a derivation drift the moment one of them learns about nested roots.
"""

from __future__ import annotations


def unit_of(rel_path: str) -> str:
    """The top-level directory of `rel_path`, or "" for a file at the root."""
    return rel_path.split("/")[0] if "/" in rel_path else ""
