"""Which directories the watcher places a watch on.

The Rust layer accepts no exclusion of any kind -- its whole configuration is
`watch_paths`, `debug`, `force_polling`, `poll_delay_ms`, `recursive` and
`ignore_permission_denied`. So an exclusion can only be expressed as *not
handing it the path*, which is what this decides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import structlog.testing

from tests.conftest import write
from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.watching import WatchScope


def config(root: Path, **index: Any) -> WorkspaceConfig:
    payload: dict[str, Any] = {"workspace": {"name": "w", "roots": [{"path": str(root)}]}}
    if index:
        payload["index"] = index
    return WorkspaceConfig.model_validate(payload)


def names(root: Path, **index: Any) -> set[str]:
    scope = WatchScope(config(root, **index))
    return {
        "." if d == root.resolve() else d.relative_to(root.resolve()).as_posix()
        for d in scope.directories()
    }


def test_the_root_itself_is_watched(tmp_path: Path) -> None:
    """A file created directly in the root must still wake a reindex."""
    assert "." in names(tmp_path)


def test_an_excluded_tree_is_never_handed_over(tmp_path: Path) -> None:
    """The crash this exists to prevent.

    A dangling symlink inside `.ralph` killed the whole watcher, and no
    configuration could avoid it: `index.exclude` already skipped the tree, but
    recursion descended into it before any filter ran. Not naming the directory
    is the only thing that keeps the watcher out of it.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / ".ralph" / "tasks").mkdir(parents=True)

    watched = names(tmp_path, exclude=["**/.ralph/**"])

    assert "src" in watched
    assert not any(name.startswith(".ralph") for name in watched)


def test_a_broken_symlink_inside_an_excluded_tree_is_never_visited(tmp_path: Path) -> None:
    """The exact reported shape, end to end."""
    (tmp_path / "src").mkdir()
    (tmp_path / ".ralph").mkdir()
    os.symlink(tmp_path / "nowhere", tmp_path / ".ralph" / "tasks")

    watched = names(tmp_path, exclude=["**/.ralph/**"])

    assert "src" in watched
    assert not any(".ralph" in name for name in watched)


def test_heavy_directories_are_left_out_even_without_excludes(tmp_path: Path) -> None:
    """`UNWATCHED_DIRS` is about the watch budget, not index tidiness.

    Exhausting the inotify limit stops the watcher working at all, which is
    worse than not noticing a change under `node_modules`.
    """
    (tmp_path / "src").mkdir()
    for n in range(5):
        (tmp_path / "node_modules" / f"pkg{n}").mkdir(parents=True)

    watched = names(tmp_path)

    assert "src" in watched
    assert not any("node_modules" in name for name in watched)


def test_git_internals_are_never_watched(tmp_path: Path) -> None:
    """`.git` churns on every command and is never indexed."""
    (tmp_path / ".git" / "objects").mkdir(parents=True)

    assert not any(".git" in name for name in names(tmp_path))


def test_an_unreadable_directory_is_skipped_and_said(tmp_path: Path) -> None:
    """Unreadable now is the condition that used to kill the watch later.

    Skipping it here means the Rust layer never receives it, and saying so
    means the operator knows that subtree will not trigger a reindex.
    """
    blocked = tmp_path / "blocked"
    (blocked / "inner").mkdir(parents=True)
    blocked.chmod(0o000)
    try:
        with structlog.testing.capture_logs() as logs:
            watched = names(tmp_path)
    finally:
        blocked.chmod(0o755)

    assert "blocked" in watched  # the directory itself is watchable
    assert "blocked/inner" not in watched  # its contents could not be read
    assert [e for e in logs if e["event"] == "watch.unreadable_directory"]


def test_a_nested_tree_is_enumerated_in_full(tmp_path: Path) -> None:
    """Every level needs its own watch: the watch is not recursive."""
    (tmp_path / "a" / "b" / "c").mkdir(parents=True)
    write(tmp_path / "a" / "b" / "c" / "f.py", "x = 1\n")

    assert {"a", "a/b", "a/b/c"} <= names(tmp_path)


# --- deciding when the watch has to be rebuilt ---------------------------


def test_a_new_directory_the_index_wants_is_covered(tmp_path: Path) -> None:
    (tmp_path / "src" / "feature").mkdir(parents=True)

    assert WatchScope(config(tmp_path)).covers(tmp_path / "src" / "feature")


def test_a_new_directory_the_index_ignores_is_not(tmp_path: Path) -> None:
    """Build output appears constantly. Rebuilding the watch for it would mean
    restarting the watcher every time a build runs."""
    (tmp_path / "obj" / "Debug").mkdir(parents=True)

    scope = WatchScope(config(tmp_path, exclude=["**/obj/**"]))

    assert not scope.covers(tmp_path / "obj" / "Debug")


def test_a_directory_outside_every_root_is_not_covered(tmp_path: Path) -> None:
    (tmp_path / "inside").mkdir()
    outside = tmp_path.parent / "outside-any-root"
    outside.mkdir(exist_ok=True)

    assert not WatchScope(config(tmp_path / "inside")).covers(outside)


@pytest.mark.parametrize("name", [".git", "node_modules", "__pycache__"])
def test_heavy_names_are_not_covered_at_any_depth(tmp_path: Path, name: str) -> None:
    deep = tmp_path / "src" / name / "inner"
    deep.mkdir(parents=True)

    assert not WatchScope(config(tmp_path)).covers(deep)
