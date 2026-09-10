"""Read git provenance for a root.

One subprocess batch per root, never per file. `git` is invoked rather than a
binding because it is guaranteed present on a dev box and correctly handles
worktrees, submodules, and detached HEADs without us reimplementing any of it.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from workspace_indexer.discovery.tracked_paths import TrackedPaths
from workspace_indexer.models import RepoInfo
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.discovery.git")

_TIMEOUT = 10


def _run(root: Path, args: tuple[str, ...]) -> bytes | None:
    """Raw stdout, or None when git could not answer."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("git.failed", args=args, error=str(exc))
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _git(root: Path, *args: str) -> str | None:
    """One line of git output, decoded.

    For commands whose output we control the shape of -- a branch name, a SHA,
    a remote URL. Never for paths: see `_paths`.
    """
    out = _run(root, args)
    return None if out is None else out.decode("utf-8", errors="replace").strip()


def _paths(root: Path, args: tuple[str, ...], *, separator: bytes) -> list[str] | None:
    """Paths from git, decoded the way the filesystem spells them.

    The separator is passed in rather than sniffed: `-z` output is NUL-
    separated, `--show-toplevel` is one newline-terminated line, and guessing
    from the bytes would pick wrong for a filename that legitimately contains
    a newline -- which is legal here.

    `os.fsdecode` rather than `str`, and bytes rather than `text=True`, because
    a path is not text on this platform -- it is bytes. Two ways that bites:

    - A filename that is not valid UTF-8 is legal on Linux, and decoding it
      strictly raises `UnicodeDecodeError`. That crashed the walk rather than
      degrading it, on any repository holding one.
    - `text=True` turns on universal newlines, which rewrites a CR inside a
      filename to LF. The lookup then misses and the file is treated as
      untracked, which is the silent loss this module exists to prevent.

    Both verified against a repository containing `caf\xe9.ts` and `we\rird.ts`.
    Nothing is stripped: a path may legitimately end in whitespace, and the
    trailing NUL is handled by dropping empty entries instead.
    """
    out = _run(root, args)
    if out is None:
        return None
    return [os.fsdecode(entry) for entry in out.split(separator) if entry]


def is_repo(root: Path) -> bool:
    # --git-dir gives a truthy answer for worktrees and submodules too, where a
    # bare `(root / ".git").is_dir()` check would say no.
    return _git(root, "rev-parse", "--git-dir") is not None


def tracked_paths(root: Path) -> TrackedPaths | None:
    """Everything in the repository's index, relative to its root.

    One subprocess per repository, which is what makes this affordable where
    `git check-ignore` per file is not: the answer is identical for every path
    in the repository, so it is fetched once and cached by the caller.

    None when the directory is not a repository, or git could not answer. Both
    degrade to pattern-matching alone -- the behaviour before this existed --
    rather than to indexing something a user asked to be ignored.

    Read through `_paths`, which decodes with `os.fsdecode` from bytes: a
    filename is not text on this platform, a non-UTF-8 name would otherwise
    crash the walk, and universal newlines would rewrite a CR inside a name
    and turn a tracked file into an untracked one.
    """
    paths = _paths(root, ("ls-files", "-z"), separator=b"\0")
    if paths is None:
        # Silent here on purpose. A workspace holds plain folders alongside
        # repositories and both get indexed, so "not a repository" is an
        # ordinary answer and warning about it would be noise on every such
        # root. Whether an absent answer is *surprising* is the caller's
        # question -- see IgnoreMatcher._tracked_for, which warns only for a
        # directory it has already confirmed holds a `.git`.
        return None
    return TrackedPaths.from_files(paths)


def is_linked_worktree(path: Path) -> bool:
    """Is `path` the root of a `git worktree add` checkout?

    Linked worktrees hold the same files as their main checkout at different
    paths, so indexing one duplicates a repository: two copies of every chunk
    competing in search, and -- worse, because it is silent -- route resolution
    finding two files for one endpoint and resolving to neither.

    Cheap by construction, because this is asked of every directory descended
    into. A worktree's `.git` is a *file* rather than a directory, so a missing
    or directory `.git` answers no without running git at all; only the handful
    of directories that could be one cost a subprocess.

    That file test is necessary but not sufficient: a submodule's `.git` is a
    file too, and excluding vendored submodule code would be a real loss. The
    distinguishing fact is that a worktree borrows its repository's object
    store, so its git-dir sits inside the common dir rather than being it --
    verified against a real worktree and a real submodule, because the
    tempting shorter test gets submodules wrong.
    """
    marker = path / ".git"
    if not marker.is_file():
        return False
    git_dir = _git(path, "rev-parse", "--git-dir")
    common = _git(path, "rev-parse", "--git-common-dir")
    if git_dir is None or common is None:
        return False
    return Path(git_dir).resolve() != Path(common).resolve()


def repo_root(path: Path) -> Path | None:
    """The repository `path` belongs to, or None if it is not in one.

    Asked of git rather than inferred by walking up looking for `.git`, because
    that directory is a *file* in a worktree and absent entirely in a submodule
    checkout -- both of which are ordinary states for a checked-out workspace.
    """
    # Through `_paths`, not `_git`: a toplevel is a filesystem path, and `_git`
    # says so in its own docstring. `errors="replace"` there would turn a
    # non-UTF-8 toplevel into U+FFFD and hand back a Path that does not exist,
    # and `.strip()` would eat a trailing space that is part of the name --
    # both silently, where the old strict decoding at least raised.
    #
    # One newline-terminated line, so the newline is the separator and the
    # trailing empty entry is dropped: one line in, one path out.
    found = _paths(
        path if path.is_dir() else path.parent,
        ("rev-parse", "--show-toplevel"),
        separator=b"\n",
    )
    return Path(found[0]) if found else None


def read_repo_info(root: Path) -> RepoInfo | None:
    """None when the directory is not a repository, which is expected — a
    workspace holds plain folders alongside repos and both get indexed."""
    if not is_repo(root):
        return None

    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    head = _git(root, "rev-parse", "HEAD")
    remote = _git(root, "config", "--get", "remote.origin.url")
    status = _git(root, "status", "--porcelain")

    return RepoInfo(
        name=root.name,
        remote_url=remote or None,
        branch=branch or None,
        head_sha=head or None,
        is_dirty=bool(status),
    )
