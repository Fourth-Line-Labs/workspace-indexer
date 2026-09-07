"""Ignore-rule evaluation: config excludes plus the .gitignore chain.

Uses pathspec so we parse gitignore syntax in-process. Shelling out to
`git check-ignore` per path would be one subprocess per file, which on a
workspace-sized tree is the difference between seconds and many minutes.

GitIgnoreSpec rather than the generic PathSpec: it is the class built for
.gitignore semantics, where the last matching pattern wins and negation with
`!` has to override an earlier exclusion.

Patterns are not the whole of gitignore, though. A pattern has no effect on a
path git already tracks, so the index is consulted too -- once per repository,
which is what keeps the performance argument above intact. Without that, a
repository carrying a broad pattern from a language template it does not use
loses committed source with nothing logged: the files are never read, so no
skip is recorded against them and the run reports success.
"""

from __future__ import annotations

from pathlib import Path

import pathspec

from workspace_indexer.discovery.git_metadata import tracked_paths
from workspace_indexer.discovery.skip_reason import SkipReason
from workspace_indexer.discovery.tracked_paths import TrackedPaths

_GITIGNORE = ".gitignore"


class IgnoreMatcher:
    """Matches paths relative to one root.

    .gitignore files are discovered per directory as we descend and cached, so
    nested ignore files apply to their own subtree the way git does.
    """

    def __init__(self, root: Path, excludes: list[str], respect_gitignore: bool) -> None:
        self._root = root
        self._respect_gitignore = respect_gitignore
        self._excludes = pathspec.GitIgnoreSpec.from_lines(excludes)
        self._dir_specs: dict[Path, pathspec.GitIgnoreSpec | None] = {}
        # Both cached because both are asked per path: one subprocess per
        # repository and one upward walk per directory, never per file.
        self._tracked: dict[Path, TrackedPaths | None] = {}
        self._repo_roots: dict[Path, Path | None] = {}

    def _spec_for_dir(self, directory: Path) -> pathspec.GitIgnoreSpec | None:
        if directory in self._dir_specs:
            return self._dir_specs[directory]
        gitignore = directory / _GITIGNORE
        spec: pathspec.GitIgnoreSpec | None = None
        if gitignore.is_file():
            try:
                lines = gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
                spec = pathspec.GitIgnoreSpec.from_lines(lines)
            except OSError:
                spec = None
        self._dir_specs[directory] = spec
        return spec

    def _gitignored(self, path: Path, is_dir: bool) -> bool:
        """Walk up from the file's directory to the root, applying each
        .gitignore to the path relative to that ignore file's own directory."""
        directory = path.parent
        while True:
            spec = self._spec_for_dir(directory)
            if spec is not None:
                try:
                    relative = path.relative_to(directory).as_posix()
                except ValueError:  # pragma: no cover - defensive
                    break
                if is_dir:
                    relative += "/"
                if spec.match_file(relative):
                    return True
            if directory == self._root:
                break
            parent = directory.parent
            if parent == directory:
                break
            directory = parent
        return False

    def _repo_root_for(self, directory: Path) -> Path | None:
        """The repository holding `directory`, or None if it is in none.

        A `.git` entry marks the root, tested with `exists()` rather than
        `is_dir()` on purpose: a linked worktree and a submodule checkout both
        keep a `.git` *file* there, and both are ordinary states in a
        checked-out workspace. Walking up finds the nearest one, so a submodule
        is answered with itself rather than its parent.

        Stops at the matcher's root, matching where the .gitignore walk stops.
        A repository above the root is not looked for and does not need to be:
        no .gitignore above the root is applied either, so nothing up there can
        wrongly ignore anything.
        """
        if directory in self._repo_roots:
            return self._repo_roots[directory]
        found: Path | None = None
        current = directory
        while True:
            if (current / ".git").exists():
                found = current
                break
            if current == self._root:
                break
            parent = current.parent
            if parent == current:
                break
            current = parent
        self._repo_roots[directory] = found
        return found

    def _tracked_for(self, repo: Path) -> TrackedPaths | None:
        if repo not in self._tracked:
            self._tracked[repo] = tracked_paths(repo)
        return self._tracked[repo]

    def _is_tracked(self, path: Path, is_dir: bool) -> bool:
        """Whether git has this path in its index, or anything beneath it.

        A directory counts as tracked when it holds tracked files, because the
        walker prunes directories before reading what is inside them -- pruning
        one that holds committed source is how the files go missing without a
        single skip being recorded against them.
        """
        repo = self._repo_root_for(path if is_dir else path.parent)
        if repo is None:
            return False
        tracked = self._tracked_for(repo)
        if tracked is None:
            return False
        try:
            relative = path.relative_to(repo).as_posix()
        except ValueError:  # pragma: no cover - defensive
            return False
        return tracked.holds(relative, is_dir=is_dir)

    def reason(self, path: Path, is_dir: bool = False) -> SkipReason | None:
        """None means "keep it"."""
        try:
            relative = path.relative_to(self._root).as_posix()
        except ValueError:  # pragma: no cover - defensive
            return None
        probe = relative + "/" if is_dir else relative
        if self._excludes.match_file(probe):
            return SkipReason.EXCLUDED
        # Tracked paths override the *gitignore* rule and nothing else.
        # Deliberately after the configured excludes rather than before: those
        # are correctness rules about our own state, and a repository that has
        # committed a log file or an eval artefact is exactly when they matter
        # most. `_is_tracked` runs only once a pattern has already matched, so
        # the index is never consulted for a path nothing would have ignored.
        if (
            self._respect_gitignore
            and self._gitignored(path, is_dir)
            and not self._is_tracked(path, is_dir)
        ):
            return SkipReason.GITIGNORED
        return None
