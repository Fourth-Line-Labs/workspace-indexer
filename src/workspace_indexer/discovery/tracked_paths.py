"""What git has in the index for one repository.

A `.gitignore` pattern has no effect on a file git already tracks -- that is
part of the specification, not an optimisation -- so deciding whether a path is
ignored needs the index as well as the patterns. Matching patterns alone
reimplements half of gitignore and silently drops committed source whenever a
repository carries a broad pattern from a language template it does not use.

Held as files *and* their ancestor directories because the walker asks both
questions, and asks the directory one first: it prunes a directory before it
ever sees the files inside, so a directory holding tracked source has to be
descended into even when a pattern matches it.

Directories are derived from the file list rather than fetched separately. Git
does not track directories at all, so the only truthful definition of a tracked
directory is one with a tracked file somewhere beneath it.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel


class TrackedPaths(BaseModel):
    files: frozenset[str]
    directories: frozenset[str]

    @classmethod
    def from_files(cls, files: Iterable[str]) -> TrackedPaths:
        paths = frozenset(files)
        directories: set[str] = set()
        for path in paths:
            segments = path.split("/")[:-1]
            # Every ancestor, not just the immediate parent: pruning happens at
            # the topmost matching directory, so `a` has to be known to hold
            # something when only `a/b/c/file.ts` is tracked.
            for depth in range(1, len(segments) + 1):
                directories.add("/".join(segments[:depth]))
        return cls(files=paths, directories=frozenset(directories))

    def holds(self, relative: str, *, is_dir: bool) -> bool:
        """Whether git tracks this path, or anything beneath it."""
        return relative in (self.directories if is_dir else self.files)
