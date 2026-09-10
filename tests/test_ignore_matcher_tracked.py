"""A gitignore pattern must not touch a file git already tracks.

Real repositories rather than mocks, because the whole defect was a
reimplementation of gitignore that matched patterns correctly and got the
specification wrong. Only git can settle what git would do.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.conftest import git_init, write
from workspace_indexer.discovery import IgnoreMatcher, SkipReason
from workspace_indexer.discovery.git_metadata import is_repo, repo_root, tracked_paths
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
    # And gitignore-matched, which the fixture's `lib/`+`build/` did not do.
    # Without this the gitignore branch never fires, the tracked override is
    # never consulted, and the ordering this test names cannot fail.
    write(repo / ".gitignore", "lib/\nbuild/\nlogs/\n*.sqlite3\n")

    both = IgnoreMatcher(repo, [], respect_gitignore=True)
    assert both.reason(repo / "logs" / "workspace-indexer.jsonl") is None, (
        "precondition: tracked + gitignored alone must be kept, or the test "
        "below proves nothing about excludes winning"
    )

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


def test_a_nested_repository_whose_own_root_matches_a_parent_pattern(tmp_path: Path) -> None:
    """A submodule or vendored checkout must not be pruned at its own root.

    When the path being judged *is* the repo root the walk found,
    `path.relative_to(repo)` is ".", and `TrackedPaths` derives directories
    from file paths so it never holds that entry. Answering False there pruned
    the whole nested repository, and because the walker prunes directories
    before reading what is inside them, every file under it went with it.

    Git's view is the opposite: the gitlink is tracked in the parent index, so
    the pattern has no effect on it.
    """
    outer = tmp_path / "outer"
    write(outer / "app.ts", "export const a = 1\n")
    inner = outer / "vendor" / "inner"
    write(inner / "core.ts", "export const c = 1\n")
    git_init(inner)
    git_init(outer)
    write(outer / ".gitignore", "vendor/\n")

    matcher = IgnoreMatcher(outer, [], respect_gitignore=True)
    assert matcher.reason(inner, is_dir=True) is None
    assert matcher.reason(inner / "core.ts") is None


def test_a_root_inside_a_repository_still_gets_the_override(tmp_path: Path) -> None:
    """One package of a monorepo, indexed as its own root.

    The upward `.git` walk stops at the matcher root, so a root that is a
    subdirectory of a repository used to find nothing and switch the override
    off -- while a `.gitignore` *at* that root still applied. That dropped
    tracked files for exactly those roots: issue #84 again, one level up.
    """
    repo = tmp_path / "mono"
    write(repo / "packages" / "app" / "src" / "lib" / "thing.ts", "export const t = 1\n")
    git_init(repo)
    write(repo / "packages" / "app" / ".gitignore", "lib/\n")
    write(repo / "packages" / "app" / "src" / "lib" / "scratch.ts", "// untracked\n")

    root = repo / "packages" / "app"
    matcher = IgnoreMatcher(root, [], respect_gitignore=True)
    assert matcher.reason(root / "src" / "lib" / "thing.ts") is None
    # Still discriminating: the untracked sibling is skipped.
    assert matcher.reason(root / "src" / "lib" / "scratch.ts") is SkipReason.GITIGNORED


# Marks a test whose *setup* cannot exist on Windows, not one that merely
# behaves differently there. Deliberately not a list with a count: this comment
# has gone stale twice already, once within the commit that wrote it, because a
# count needs editing every time a case is added. The kinds, not the tally:
#
# - a surrogate-escaped non-UTF-8 filename cannot be encoded into a UTF-16 name
# - control characters such as CR and LF are outright illegal in one
# - `os.fsdecode` on Windows is utf-8/surrogatepass, so it raises on a lone
#   non-UTF-8 byte rather than escaping it
#
# Each fails at file creation or at the decode, before any assertion runs, so
# the guard belongs on the test rather than inside it.
#
# CI is ubuntu-only, so nothing here would ever go red -- the suite is also run
# on Windows, and that is the reader this guard is for. Apply it to any new
# test that needs a filename the filesystem cannot spell.
_posix_filenames_only = pytest.mark.skipif(
    os.name == "nt",
    reason="these filenames cannot exist on Windows, so the test cannot be set up there",
)


@_posix_filenames_only
def test_a_filename_that_is_not_utf8_does_not_crash_the_walk(tmp_path: Path) -> None:
    """Filenames are bytes on this platform, and some are not valid UTF-8.

    Decoding `git ls-files -z` as text raised an uncaught UnicodeDecodeError,
    so a single such file aborted the run rather than degrading it.
    """
    repo = tmp_path / "weird"
    repo.mkdir()
    odd = os.fsdecode(b"caf\xe9.ts")
    (repo / odd).write_text("export const x = 1\n", encoding="utf-8")
    (repo / "normal.ts").write_text("export const y = 1\n", encoding="utf-8")
    git_init(repo)

    paths = tracked_paths(repo)
    assert paths is not None
    assert odd in paths.files
    assert "normal.ts" in paths.files


@_posix_filenames_only
def test_a_filename_containing_a_carriage_return_is_not_rewritten(tmp_path: Path) -> None:
    """`text=True` turns on universal newlines, which rewrote a CR inside a
    filename to LF -- so the lookup missed and a tracked file read as
    untracked. Exactly the silent loss this module exists to prevent."""
    repo = tmp_path / "cr"
    repo.mkdir()
    (repo / "we\rird.ts").write_text("export const z = 1\n", encoding="utf-8")
    git_init(repo)

    paths = tracked_paths(repo)
    assert paths is not None
    assert "we\rird.ts" in paths.files
    assert "we\nird.ts" not in paths.files


def test_a_root_inside_a_repository_is_still_known_to_be_one(tmp_path: Path) -> None:
    """The condition behind the missing-override warning.

    `_repo_root_for` falls back to the matcher root for a root that sits inside
    a repository, and such a root has no local `.git` -- so gating the warning
    on a filesystem check meant it could not fire for exactly the case that
    fallback introduced.

    Asserted through `is_repo`, which is what `_tracked_for` actually consults.
    An earlier version of this test asserted `repo_root` instead, which the
    matcher never calls -- so it pinned a different function and a regression
    in the gating would not have failed it.
    """
    repo = tmp_path / "mono"
    write(repo / "packages" / "app" / "src" / "thing.ts", "export const t = 1\n")
    git_init(repo)

    inside = repo / "packages" / "app"
    # The condition the old gate used, and why it was wrong: no local `.git`.
    assert not (inside / ".git").exists()
    # The condition the gate uses now.
    assert is_repo(inside)


@_posix_filenames_only
def test_a_toplevel_is_decoded_as_a_path_not_as_text(tmp_path: Path) -> None:
    """`repo_root` returns a path that exists, for a repository whose own
    directory name is not valid UTF-8.

    Decoding it with `errors="replace"` produced U+FFFD and a Path that does
    not exist, silently -- worse than the strict decode it replaced, which at
    least raised.
    """
    odd = tmp_path / os.fsdecode(b"rep\xf8")
    write(odd / "app.ts", "export const a = 1\n")
    git_init(odd)

    found = repo_root(odd)
    assert found is not None
    assert found.exists(), f"decoded to a path that does not exist: {found!r}"
    assert found.samefile(odd)


@_posix_filenames_only
def test_a_toplevel_containing_a_newline_is_not_truncated(tmp_path: Path) -> None:
    """A directory name may contain a newline, and git's output is newline-
    terminated, so splitting on newlines cut such a name at its first one and
    returned a path that does not exist.

    Stripping one trailing byte rather than splitting handles this and the
    trailing-newline case below, which is the one that looks ambiguous and is
    not.
    """
    odd = tmp_path / "rep\nname"
    write(odd / "app.ts", "export const a = 1\n")
    git_init(odd)

    found = repo_root(odd)
    assert found is not None
    assert found.exists(), f"truncated to a path that does not exist: {found!r}"
    assert found.samefile(odd)


@_posix_filenames_only
def test_a_toplevel_ending_in_a_newline_is_read_whole(tmp_path: Path) -> None:
    """The case that looks undecidable and is not.

    An earlier version of this module documented it as unreadable, reasoning
    that `--show-toplevel` has no `-z` form so the terminator and the name's
    last byte must be the same byte. They are not: git emits the name and then
    its own terminator, so the output ends with two newlines and stripping one
    recovers the name whole.

    Pinned because a limitation recorded but not real is worse than an unknown
    one -- it invites a workaround for a solved problem, or distrust of a
    correct answer.
    """
    odd = tmp_path / "trailing\n"
    write(odd / "app.ts", "export const a = 1\n")
    git_init(odd)

    found = repo_root(odd)
    assert found is not None
    assert found.exists()
    assert found.samefile(odd)
    assert found.name == "trailing\n", f"lost the trailing newline: {found.name!r}"
