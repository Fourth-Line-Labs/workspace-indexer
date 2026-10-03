"""Several workspaces served from one process.

The registry exists because of a hard constraint, not an optimisation: an
embedded Qdrant locks its storage folder to a single client, so building one
context per workspace the obvious way fails outright on the default
configuration. Everything here is about sharing exactly what must be shared
while keeping separate what must stay separate.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from workspace_indexer.config import WorkspaceChoiceError, WorkspaceConfig
from workspace_indexer.workspace_registry import WorkspaceRegistry


@pytest.fixture(autouse=True)
def _isolate_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`Settings` reads `.env` from the working directory, which would
    otherwise point these at the developer's real Qdrant and credentials."""
    monkeypatch.chdir(tmp_path)
    for key in (
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
        "SPARSE_MODEL",
        "VOYAGE_API_KEY",
        "VECTOR_STORE",
        "QDRANT_MODE",
        "QDRANT_PATH",
        "STATE_DB",
        "RERANK_ENABLED",
        "RERANK_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("QDRANT_MODE", "embedded")
    monkeypatch.setenv("QDRANT_PATH", str(tmp_path / "qdrant"))


def _config(tmp_path: Path, *workspaces: dict[str, Any]) -> WorkspaceConfig:
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    return WorkspaceConfig.model_validate({"state_dir": str(state), "workspaces": list(workspaces)})


def _workspace(tmp_path: Path, name: str, **extra: Any) -> dict[str, Any]:
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    return {"name": name, "roots": [{"path": str(root)}], **extra}


@pytest.fixture
async def registry(tmp_path: Path) -> AsyncIterator[WorkspaceRegistry]:
    built = WorkspaceRegistry(
        _config(tmp_path, _workspace(tmp_path, "alpha"), _workspace(tmp_path, "beta"))
    )
    yield built
    await built.close()


# ---- the constraint this exists for ----------------------------------------


async def test_several_workspaces_build_against_one_embedded_store(
    registry: WorkspaceRegistry,
) -> None:
    """The regression. Building a context per workspace the obvious way raised
    "Storage folder is already accessed by another instance of Qdrant client"
    -- so multi-workspace serving did not work at all on the default
    configuration."""
    assert registry.names == ["alpha", "beta"]
    assert registry.context("alpha") is not registry.context("beta")


async def test_each_workspace_keeps_its_own_collection_and_manifest(
    registry: WorkspaceRegistry,
) -> None:
    """Sharing the client must not quietly share the data."""
    alpha, beta = registry.context("alpha"), registry.context("beta")

    assert alpha.store.collection_name(alpha.space) != beta.store.collection_name(beta.space)
    assert alpha.manifest is not beta.manifest


# ---- what is shared, and what decides it -----------------------------------


async def test_workspaces_with_the_same_embedding_share_one_backend(
    registry: WorkspaceRegistry,
) -> None:
    """The ordinary case -- every workspace inheriting from `.env` -- must load
    one dense and one sparse model however many workspaces exist, not one copy
    of the same model each."""
    alpha, beta = registry.context("alpha"), registry.context("beta")

    assert alpha.embeddings is beta.embeddings
    assert alpha.sparse is beta.sparse


async def test_a_workspace_that_overrides_its_embedding_gets_its_own_backend(
    tmp_path: Path,
) -> None:
    """Sharing is keyed on what the backend depends on, not on convenience: a
    workspace that overrode its model and still got the shared backend would
    be silently embedding with the wrong one."""
    registry = WorkspaceRegistry(
        _config(
            tmp_path,
            _workspace(tmp_path, "alpha"),
            _workspace(
                tmp_path,
                "beta",
                embedding={"model": "voyageai:voyage-code-4", "dimensions": 256},
            ),
        )
    )
    try:
        alpha, beta = registry.context("alpha"), registry.context("beta")

        assert alpha.embeddings is not beta.embeddings
        assert alpha.space != beta.space
        # The sparse model was not overridden, so that one is still shared.
        assert alpha.sparse is beta.sparse
    finally:
        await registry.close()


# ---- refusing to guess ------------------------------------------------------


async def test_asking_without_a_name_where_there_is_a_choice_raises(
    registry: WorkspaceRegistry,
) -> None:
    with pytest.raises(WorkspaceChoiceError, match="alpha, beta"):
        registry.context()


async def test_an_unknown_name_lists_the_configured_ones(
    registry: WorkspaceRegistry,
) -> None:
    with pytest.raises(WorkspaceChoiceError, match="alpha, beta"):
        registry.context("gamma")


async def test_a_lone_workspace_needs_no_name(tmp_path: Path) -> None:
    """A single-workspace config must behave exactly as it always has, with no
    parameter required anywhere."""
    registry = WorkspaceRegistry(_config(tmp_path, _workspace(tmp_path, "solo")))
    try:
        assert registry.context().config.workspace.name == "solo"
    finally:
        await registry.close()


# ---- what the agent is told -------------------------------------------------


async def test_describe_names_each_workspace_with_its_roots(tmp_path: Path) -> None:
    """Bare names say *that* two indexes exist but not which holds what, so the
    first call is a guess -- and a wrong guess returns a confident empty
    result, which reads as "this does not exist"."""
    alpha = _workspace(tmp_path, "alpha")
    for extra in ("second", "third"):
        (tmp_path / extra).mkdir(exist_ok=True)
        alpha["roots"].append({"path": str(tmp_path / extra)})

    registry = WorkspaceRegistry(_config(tmp_path, alpha, _workspace(tmp_path, "beta")))
    try:
        described = registry.describe()
    finally:
        await registry.close()

    assert described == ["alpha (alpha, second, third)", "beta (beta)"]


async def test_describe_caps_a_long_root_list(tmp_path: Path) -> None:
    """A workspace over a tree with `recurse_into_children` can hold dozens of
    roots, and the instructions are read in full on every session."""
    alpha = _workspace(tmp_path, "alpha")
    for index in range(6):
        extra = tmp_path / f"r{index}"
        extra.mkdir(exist_ok=True)
        alpha["roots"].append({"path": str(extra)})

    registry = WorkspaceRegistry(_config(tmp_path, alpha))
    try:
        described = registry.describe()
    finally:
        await registry.close()

    assert described == ["alpha (alpha, r0, r1, +4 more)"]


# ---- lifetime ---------------------------------------------------------------


async def test_closing_is_safe_when_the_client_is_shared(tmp_path: Path) -> None:
    """Each store was handed a client it does not own, so the registry closes
    it once at the end. A store that closed the shared client would take the
    connection away from every other workspace."""
    registry = WorkspaceRegistry(
        _config(tmp_path, _workspace(tmp_path, "alpha"), _workspace(tmp_path, "beta"))
    )
    alpha = registry.context("alpha")

    # Closing one context must leave the other usable, which is the behaviour
    # `owns_client=False` buys.
    await alpha.close()
    beta = registry.context("beta")
    assert await beta.store.count(beta.space) == 0

    await registry.close()
