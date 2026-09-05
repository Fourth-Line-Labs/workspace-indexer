"""The directories the watcher should actually place a watch on."""

from __future__ import annotations

import os
from pathlib import Path

from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.discovery.ignore_matcher import IgnoreMatcher
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.watching.scope")

# Directory names never worth a watch whatever the config says. Deliberately
# not the same list as the index excludes: this one is about the watch
# *budget*, and these are the trees that exhaust it. `.git` alone churns
# constantly and is never indexed.
#
# The divergence is real and worth knowing: index a tree named here without
# excluding it, and the index will hold it while the watcher will not notice it
# changing. That trade is deliberate -- exhausting the inotify budget stops the
# watcher working at all, which is worse than missing changes in `node_modules`.
UNWATCHED_DIRS = frozenset(
    {".git", "node_modules", ".venv", "venv", "__pycache__", "target", "dist", "build"}
)


class WatchScope:
    """Every directory the index would look in, and no others.

    The watcher used to hand the Rust layer one path per root with
    `recursive=True`, which is the only way to say "and everything under it" --
    and the Rust layer takes no exclusion of any kind. Its whole configuration
    is `watch_paths`, `debug`, `force_polling`, `poll_delay_ms`, `recursive`
    and `ignore_permission_denied`; there is nowhere to put an ignore rule.

    So the exclusions have to be expressed in *which paths are handed over*,
    which means enumerating the directories ourselves and watching each one
    without recursion. Two things follow, and both are the point:

    A path inside an excluded tree is never touched. That is what stops the
    watcher dying on a dangling symlink under a directory `index.exclude`
    already skips -- a failure no configuration could avoid, because the walk
    happened before any filter ran.

    And the watch gets much smaller. inotify watches directories, so the cost
    is per directory either way; recursion just meant every directory,
    including the ones we ignore. Measured on a real workspace: 1,517
    directories watched, 214 needed.
    """

    def __init__(self, config: WorkspaceConfig) -> None:
        self._config = config

    def directories(self) -> list[Path]:
        """Watchable directories across every root, roots included.

        Roots first and in configuration order, so the list is stable between
        rescopes and a diff of two of them reads sensibly.
        """
        found: list[Path] = []
        for root in self._config.workspace.roots:
            base = root.path.expanduser().resolve()
            if not base.is_dir():
                continue
            matcher = IgnoreMatcher(
                base, self._config.all_excludes, self._config.index.respect_gitignore
            )
            found.append(base)
            found.extend(self._below(base, matcher))
        return found

    def _below(self, base: Path, matcher: IgnoreMatcher) -> list[Path]:
        found: list[Path] = []
        stack = [base]
        while stack:
            current = stack.pop()
            try:
                entries = list(os.scandir(current))
            except OSError as exc:
                # Unreadable now is exactly the condition that used to kill the
                # watch later. Skipped and said, rather than handed to a layer
                # that treats it as fatal.
                log.warning(
                    "watch.unreadable_directory",
                    path=str(current),
                    error=str(exc),
                    detail="not watched; its contents will not trigger a reindex",
                )
                continue
            for entry in entries:
                try:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                except OSError:
                    continue
                if entry.name in UNWATCHED_DIRS:
                    continue
                path = Path(entry.path)
                if matcher.reason(path, is_dir=True) is not None:
                    continue
                found.append(path)
                stack.append(path)
        return found

    def covers(self, path: Path) -> bool:
        """Would a new directory at `path` be watched?

        Asked when something is created, to decide whether the watch has to be
        rebuilt. A directory the index would ignore is not worth restarting for
        -- and build output, which is what appears most often, is exactly that.
        """
        for root in self._config.workspace.roots:
            base = root.path.expanduser().resolve()
            if path == base or base in path.parents:
                if any(part in UNWATCHED_DIRS for part in path.relative_to(base).parts):
                    return False
                matcher = IgnoreMatcher(
                    base, self._config.all_excludes, self._config.index.respect_gitignore
                )
                return matcher.reason(path, is_dir=True) is None
        return False
