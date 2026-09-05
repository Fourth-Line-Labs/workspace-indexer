"""Filesystem traversal.

Uses os.scandir rather than Path.rglob because scandir returns the stat data
the kernel already fetched while listing the directory. rglob re-stats every
path, which doubles the syscall count on a tree this size.

The walker never opens a file. That is deliberate: the manifest's fast path
decides whether a file needs reading based on mtime and size alone, and reading
here would throw that saving away.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from workspace_indexer.config import RootConfig, WorkspaceConfig
from workspace_indexer.discovery.classify import classify, is_lockfile
from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.discovery.git_metadata import (
    is_linked_worktree,
    read_repo_info,
    repo_root,
)
from workspace_indexer.discovery.ignore_matcher import IgnoreMatcher
from workspace_indexer.discovery.skip_reason import SkipReason
from workspace_indexer.models import RepoInfo
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.discovery.walker")

# Always skipped, whatever the config says. `.git` is enormous, entirely
# machine-generated, and re-walking it would dwarf the real work.
_ALWAYS_SKIP_DIRS = frozenset({".git"})


class Walker:
    def __init__(self, config: WorkspaceConfig) -> None:
        self._config = config
        self._excluded_paths = config.excluded_paths
        # Files dropped, keyed by reason.
        self.skips: Counter[str] = Counter()
        # Directories not descended into. Tracked separately because one entry
        # here can stand for thousands of files, so folding it into `skips`
        # would make the file tally meaningless.
        self.pruned_dirs: Counter[str] = Counter()
        # Roots configured but not present on disk. Recorded rather than only
        # logged because a caller has to be able to act on it: a root we could
        # not read is not a root that turned out to be empty, and only one of
        # those two justifies deleting its index.
        self.unobservable_roots: set[str] = set()
        # Files that no .gitignore could have filtered, per root and unit: a
        # plain folder has no .gitignore to respect, so `respect_gitignore`
        # does nothing for it. Counted rather than merely flagged, because
        # "this unit contributed 8,000 files" is actionable and "check your
        # config" is not.
        self._unprotected: dict[str, dict[str, int]] = {}

    def walk(self, only_root: str | None = None) -> Iterator[FileCandidate]:
        for root in self._config.workspace.roots:
            if only_root and root.resolved_label != only_root:
                continue
            if not root.path.is_dir():
                self.unobservable_roots.add(root.resolved_label)
                log.warning(
                    "root.missing",
                    path=str(root.path),
                    label=root.resolved_label,
                    detail="its indexed files are left alone; a root that cannot be read "
                    "has not been shown to be empty",
                )
                continue
            yield from self._walk_root(root)
            self._warn_if_unfiltered(root)

    def _walk_root(self, root: RootConfig) -> Iterator[FileCandidate]:
        matcher = IgnoreMatcher(
            root.path,
            self._config.all_excludes,
            self._config.index.respect_gitignore,
        )
        # Repo metadata is read once per directory, and once per repository
        # behind that -- never per file.
        repo_cache: dict[str, RepoInfo | None] = {}
        self._unprotected.setdefault(root.resolved_label, {})
        follow = self._config.index.follow_symlinks
        max_bytes = self._config.index.max_file_bytes

        stack: list[Path] = [root.path]
        while stack:
            current = stack.pop()
            try:
                entries = list(os.scandir(current))
            except OSError as exc:
                self._skip(SkipReason.UNREADABLE, current)
                log.warning("scandir.failed", path=str(current), error=str(exc))
                continue

            for entry in entries:
                path = Path(entry.path)
                try:
                    is_dir = entry.is_dir(follow_symlinks=follow)
                except OSError:
                    self._skip(SkipReason.UNREADABLE, path)
                    continue

                if entry.is_symlink() and not follow:
                    self._skip(SkipReason.SYMLINK, path)
                    continue

                if is_dir:
                    # Note: hidden directories are NOT blanket-skipped. `.claude`
                    # is a primary target, so only `.git` is dropped outright.
                    if entry.name in _ALWAYS_SKIP_DIRS:
                        continue
                    prune = matcher.reason(path, is_dir=True)
                    if prune is not None:
                        self.pruned_dirs[prune.value] += 1
                        log.debug("discovery.prune", reason=prune.value, path=str(path))
                        continue
                    # Only reached for directories *discovered* under a root --
                    # the root itself is pushed straight onto the stack and
                    # never passes here. That is the escape hatch: a worktree
                    # someone deliberately configures as a root is indexed
                    # normally, while one that merely turns up beside its
                    # repository is not. Explicit beats inferred.
                    if is_linked_worktree(path):
                        self.pruned_dirs[SkipReason.WORKTREE.value] += 1
                        log.info(
                            "discovery.worktree_skipped",
                            path=str(path),
                            detail="a linked worktree holds a second copy of a repository "
                            "already indexed; add it as its own root to index it anyway",
                        )
                        continue
                    stack.append(path)
                    continue

                if not entry.is_file(follow_symlinks=follow):
                    continue

                candidate = self._consider(root, path, entry, matcher, repo_cache, max_bytes)
                if candidate is not None:
                    yield candidate

    def _consider(
        self,
        root: RootConfig,
        path: Path,
        entry: os.DirEntry[str],
        matcher: IgnoreMatcher,
        repo_cache: dict[str, RepoInfo | None],
        max_bytes: int,
    ) -> FileCandidate | None:
        reason = matcher.reason(path)
        if reason is not None:
            self._skip(reason, path)
            return None

        if path.resolve() in self._excluded_paths:
            # Our own operational files, by absolute path rather than pattern.
            self._skip(SkipReason.EXCLUDED, path)
            return None

        if is_lockfile(path):
            self._skip(SkipReason.LOCKFILE, path)
            return None

        try:
            stat = entry.stat(follow_symlinks=False)
        except OSError:
            self._skip(SkipReason.UNREADABLE, path)
            return None

        if stat.st_size == 0:
            self._skip(SkipReason.EMPTY, path)
            return None
        if stat.st_size > max_bytes:
            self._skip(SkipReason.TOO_LARGE, path)
            return None

        # OPAQUE files are still yielded: they are recorded in the manifest so
        # `status` can say the file is known and deliberately not embedded.
        # That is not a skip, so it is not counted as one.
        kind, language = classify(path)

        rel_path = path.relative_to(root.path).as_posix()
        unit = self._unit_for(root, rel_path)
        repo = self._repo_for(path.parent, repo_cache)
        if repo is None:
            # No repository above it, so `respect_gitignore` had nothing to
            # respect. Recorded per unit; reported once, at the end of the root.
            counts = self._unprotected.setdefault(root.resolved_label, {})
            counts[unit] = counts.get(unit, 0) + 1

        return FileCandidate(
            root_label=root.resolved_label,
            unit=unit,
            abs_path=path,
            rel_path=rel_path,
            kind=kind,
            language=language,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            repo=repo,
        )

    def _warn_if_unfiltered(self, root: RootConfig) -> None:
        """Say once that a root had files no ignore rule could reach.

        `respect_gitignore` does the filtering in practice, and it only works
        inside a git repository. Point a root at a vendored SDK or an unpacked
        sample and nothing filters it -- an `obj/` directory contributes
        thousands of generated `.cs` files that are real, parseable source and
        compete with the code on every query. The only symptom is worse
        results.

        Suppressed when `index.exclude` is configured, which is the issue's own
        rule and the right one: someone who has written exclude patterns has
        already met this problem, and a warning they cannot silence is a
        warning they learn to ignore.

        A warning, never a filter. Inventing a default exclude list for a plain
        folder would be the tool deciding what a user's directory contains.
        """
        if self._config.index.exclude:
            return
        counts = self._unprotected.get(root.resolved_label, {})
        if not counts:
            return
        named = ", ".join(
            f"{unit or '(root)'} ({files:,} files)"
            for unit, files in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        log.warning(
            "root.unfiltered",
            root=root.resolved_label,
            units=named,
            files=sum(counts.values()),
            detail="these files are in no git repository, so respect_gitignore had "
            "nothing to respect, and index.exclude is empty. Build output and "
            "dependencies will be indexed and will compete with real code on every "
            "query. Add exclude patterns for whatever this holds.",
        )

    def _repo_for(self, directory: Path, cache: dict[str, RepoInfo | None]) -> RepoInfo | None:
        """Which repository holds `directory`, memoised per directory.

        Read from the directory rather than from the unit, which is what this
        did before. A unit is the first path segment of a root, and that stops
        being the repository the moment a workspace nests one -- after which
        every file under it was attributed to no repository at all, losing
        `repo_name`, `repo_branch` and `repo_head_sha` from the payload and
        making the `repo` search filter unable to find them. Measured on this
        workspace: 309 files.

        Still never per file. `repo_root` is one git call per *directory*, and
        the repository's own metadata is read once per repository -- the same
        shape as before, one level lower.
        """
        key = str(directory)
        if key not in cache:
            top = repo_root(directory)
            if top is None:
                cache[key] = None
            else:
                top_key = f"\0{top}"
                if top_key not in cache:
                    cache[top_key] = read_repo_info(top)
                cache[key] = cache[top_key]
        return cache[key]

    @staticmethod
    def _unit_for(root: RootConfig, rel_path: str) -> str:
        """The top-level subdirectory a file belongs to.

        With recurse_into_children, a workspace root holds a mix of repos and
        plain folders; the unit is what makes "search only Repo2" expressible
        for both, where a repo-name filter would miss the plain folders.
        """
        if not root.recurse_into_children:
            return ""
        head, _, tail = rel_path.partition("/")
        return head if tail else ""

    def _skip(self, reason: SkipReason, path: Path) -> None:
        self.skips[reason.value] += 1
        log.debug("discovery.skip", reason=reason.value, path=str(path))
