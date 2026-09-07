"""Read git provenance for a root.

One subprocess batch per root, never per file. `git` is invoked rather than a
binding because it is guaranteed present on a dev box and correctly handles
worktrees, submodules, and detached HEADs without us reimplementing any of it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from workspace_indexer.discovery.tracked_paths import TrackedPaths
from workspace_indexer.models import RepoInfo
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.discovery.git")

_TIMEOUT = 10


def _git(root: Path, *args: str, strip: bool = True) -> str | None:
    """`strip=False` for `-z` output: a path may legitimately end in
    whitespace, and trimming one while fixing silent file loss would be the
    same bug in a new place."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.debug("git.failed", args=args, error=str(exc))
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() if strip else result.stdout


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
    """
    out = _git(root, "ls-files", "-z", strip=False)
    if out is None:
        return None
    return TrackedPaths.from_files(path for path in out.split("\0") if path)


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
    top = _git(path if path.is_dir() else path.parent, "rev-parse", "--show-toplevel")
    return Path(top) if top else None


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
