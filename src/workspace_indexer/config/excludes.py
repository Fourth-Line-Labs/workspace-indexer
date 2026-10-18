"""Exclusions that apply no matter what the user configures.

These are our own runtime state. A log file inside a watched root means the
watcher fires on our own writes, reindexes, writes more log lines, and spins
forever — not something a user should be able to configure themselves into.

Eval artefacts are here for the same reason in a different shape. A persisted
eval record quotes the query of every case verbatim, so indexing one makes it
a near-perfect lexical *and* semantic match for the very queries used to
measure retrieval quality — the measurement then scores our own output
instead of the workspace. A correctness rule, not a preference, so it does
not live in the user-editable exclude list.
"""

from __future__ import annotations

HARDCODED_EXCLUDES: tuple[str, ...] = (
    "**/.git/**",
    "logs/**",
    "**/logs/**",
    "data/**",
    "**/data/qdrant/**",
    "**/*.sqlite3",
    "**/*.sqlite3-journal",
    "**/workspace-indexer.jsonl*",
    # Quote every eval query verbatim — see the module docstring.
    "evals/**",
    "**/evals/**",
    "docs/eval-baselines.md",
    "**/docs/eval-baselines.md",
    # The write-up of the grep comparison quotes eval queries and the patterns
    # they are scored with, so it is an eval artefact in the same sense as a
    # record: a lexical tool allowed to read it matches the terms it is being
    # measured on.
    "docs/grep-comparison.md",
    "**/docs/grep-comparison.md",
)
