"""The resolution scope derived from a path.

Shared by the resolver and the origin classifier, so its two cases are worth
pinning: a nested file belongs to its top-level directory, and a file sitting
directly in a root belongs to no repository at all.
"""

from __future__ import annotations

from workspace_indexer.graph.unit import unit_of


def test_a_nested_path_belongs_to_its_top_level_directory() -> None:
    assert unit_of("repo/src/lib/thing.py") == "repo"


def test_a_single_segment_path_has_no_unit() -> None:
    # A file directly in a workspace root is in no repository, and must not
    # be given one -- an empty unit is its own scope.
    assert unit_of("Program.cs") == ""


def test_a_two_segment_path_belongs_to_the_first() -> None:
    assert unit_of("repo/thing.py") == "repo"
