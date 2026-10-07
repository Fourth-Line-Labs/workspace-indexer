"""One list of the environment variables a context-building test must clear.

`Settings` reads `.env` from the working directory and the process environment,
so a test that builds a real `AppContext` picks up whatever the developer's
shell exports. The failures that causes are not loud: an exported
`VECTOR_STORE=mongodb` makes `build_vector_store` raise once `chdir` has moved
off the repo's `.env`, and an exported `SPARSE_MODEL` or `EMBEDDING_MODEL`
quietly changes what gets built -- so a test stops testing what it names, on
one machine.

Shared because the list reached four hand-maintained copies. A new `Settings`
field that matters would have to be edited into all of them, and the one that
got missed would fail only where it bites.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Everything that steers context construction. Credentials included: their
# absence is what keeps a test offline.
CONTEXT_ENV = (
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "SPARSE_MODEL",
    "VOYAGE_API_KEY",
    "VECTOR_STORE",
    "RERANK_ENABLED",
    "RERANK_MODEL",
    "QDRANT_MODE",
    "QDRANT_PATH",
    "STATE_DB",
)


def isolate_context_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Clear the inherited settings, then point the store and manifest at `tmp_path`.

    The chdir matters as much as the clearing: `Settings` reads `.env` from the
    working directory, and the repo's own would otherwise supply everything
    just cleared.
    """
    monkeypatch.chdir(tmp_path)
    for key in CONTEXT_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("QDRANT_MODE", "embedded")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setenv("STATE_DB", str(tmp_path / "manifest.sqlite3"))
