"""A gitignore pattern must not touch a file git already tracks.

Real repositories rather than mocks, because the whole defect was a
reimplementation of gitignore that matched patterns correctly and got the
specification wrong. Only git can settle what git would do.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import git_init, write
from workspace_indexer.discovery import IgnoreMatcher, SkipReason
from workspace_indexer.discovery.git_metadata import tracked_paths
from workspace_indexer.discovery.tracked_paths import TrackedPaths


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository where committed source matches a later .gitignore pattern.

    Built in that order on purpose, because it is how the state arises in the
    wild: source is committed, and a pattern that also matches it arrives
    afterwards — often from a language template for a *different* language in
    the same polyglot repository. `lib/` is real build output in Python and
    real source in TypeScript.
    """
    root = tmp_path / "repo"
    write(root / "src" / "lib" / "queryClient.ts", "export const q = 1\n")
    write(root / "src" / "app.ts", "import './lib/queryClient'\n")
    git_init(root)  # commits both

    write(root / ".gitignore", "lib/\nbuild/\n")
    # What the pattern legitimately catches: never committed.
    write(root / "src" / "lib" / "scratch.ts", "// untracked\n")
    write(root / "build" / "bundle.js", "// untracked\n")
    return root


def test_a_tracked_file_survives_a_pattern_that_matches_it(repo: Path) -> None:
    """The defect, stated as a test. git reports this file as not ignored."""
    matcher = IgnoreMatcher(repo, [], respect_gitignore=True)
    assert matcher.reason(repo / "src" / "lib" / "queryClient.ts") is None


def test_an_untracked_file_matching_the_same_pattern_is_still_skipped(repo: Path) -> None:
    """The fix must not become "ignore .gitignore"."""
    matcher = IgnoreMatcher(repo, [], respect_gitignore=True)
    assert matcher.reason(repo / "src" / "lib" / "scratch.ts") is SkipReason.GITIGNORED


def test_a_directory_holding_tracked_files_is_not_pruned(repo: Path) -> None:
    """Pruning happens before the files inside are ever seen, so the directory
    question has to be answered too — otherwise the file-level fix never runs."""
    matcher = IgnoreMatcher(repo, [], respect_gitignore=True)
    assert matcher.reason(repo / "src" / "lib", is_dir=True) is None


def test_a_directory_holding_nothing_tracked_is_still_pruned(repo: Path) -> None:
    matcher = IgnoreMatcher(repo, [], respect_gitignore=True)
    assert matcher.reason(repo / "build", is_dir=True) is SkipReason.GITIGNORED


def test_configured_excludes_still_win_over_tracked_status(repo: Path) -> None:
    """Tracked status overrides the gitignore rule and nothing else.

    A repository that has committed a log file or an eval artefact is exactly
    when the hardcoded protections matter most, so being tracked must not
    rescue one.
    """
    write(repo / "logs" / "workspace-indexer.jsonl", "{}\n")
    write(repo / "data" / "manifest.sqlite3", "x")
    git_init(repo)  # commits them, so they are genuinely tracked

    matcher = IgnoreMatcher(repo, ["logs/**", "**/*.sqlite3"], respect_gitignore=True)
    assert matcher.reason(repo / "logs" / "workspace-indexer.jsonl") is SkipReason.EXCLUDED
    assert matcher.reason(repo / "data" / "manifest.sqlite3") is SkipReason.EXCLUDED


def test_a_directory_outside_any_repository_keeps_its_current_behaviour(tmp_path: Path) -> None:
    """A .gitignore with no repository around it has no index to consult, so
    patterns apply on their own — which is what happened before this existed."""
    plain = tmp_path / "plain"
    write(plain / ".gitignore", "lib/\n")
    write(plain / "lib" / "thing.ts", "export const x = 1\n")

    matcher = IgnoreMatcher(plain, [], respect_gitignore=True)
    assert matcher.reason(plain / "lib" / "thing.ts") is SkipReason.GITIGNORED


def test_the_nearest_repository_answers_for_a_nested_one(tmp_path: Path) -> None:
    """A nested repo's own index decides its files, not its parent's.

    Walking up stops at the first `.git`, so a vendored or submodule-like
    checkout is answered with itself. Its parent has never heard of these
    files, and asking the parent would report every one of them as untracked.
    """
    outer = tmp_path / "outer"
    write(outer / "app.ts", "export const a = 1\n")
    git_init(outer)
    write(outer / ".gitignore", "lib/\n")

    inner = outer / "vendor" / "inner"
    write(inner / "lib" / "core.ts", "export const c = 1\n")
    git_init(inner)

    matcher = IgnoreMatcher(outer, [], respect_gitignore=True)
    assert matcher.reason(inner / "lib" / "core.ts") is None


def test_tracked_paths_is_none_outside_a_repository(tmp_path: Path) -> None:
    """None degrades to pattern-matching alone, never to indexing something a
    user asked to be ignored."""
    assert tracked_paths(tmp_path) is None


def test_tracked_paths_reads_the_index(repo: Path) -> None:
    paths = tracked_paths(repo)
    assert paths is not None
    assert "src/lib/queryClient.ts" in paths.files
    assert "src/lib/scratch.ts" not in paths.files


def test_every_ancestor_directory_counts_as_holding_the_file() -> None:
    """Pruning happens at the topmost matching directory, so an ancestor
    several levels up has to know it holds something."""
    paths = TrackedPaths.from_files(["a/b/c/thing.ts"])
    assert paths.holds("a", is_dir=True)
    assert paths.holds("a/b", is_dir=True)
    assert paths.holds("a/b/c", is_dir=True)
    assert not paths.holds("a/b/c", is_dir=False)
    assert paths.holds("a/b/c/thing.ts", is_dir=False)


def test_a_file_at_the_repository_root_contributes_no_directories() -> None:
    paths = TrackedPaths.from_files(["README.md"])
    assert paths.directories == frozenset()
    assert paths.holds("README.md", is_dir=False)


def test_a_sibling_directory_is_not_claimed() -> None:
    paths = TrackedPaths.from_files(["a/b/thing.ts"])
    assert not paths.holds("a/c", is_dir=True)
