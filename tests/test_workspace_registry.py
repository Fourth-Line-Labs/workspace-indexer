"""Several workspaces served from one process.

The registry exists because of a hard constraint, not an optimisation: an
embedded Qdrant locks its storage folder to a single client, so building one
context per workspace the obvious way fails outright on the default
configuration. Everything here is about sharing exactly what must be shared
while keeping separate what must stay separate.
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from workspace_indexer.app_context import AppContext
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
    # The *files*, not the objects. Two separately constructed `Manifest`s are
    # always distinct objects even over one sqlite file, so identity here
    # asserted nothing -- while the collision it claims to guard is exactly
    # two workspaces sharing one database.
    assert alpha.config.manifest_path(Path("/unused")) != beta.config.manifest_path(Path("/unused"))


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


# ---- failure paths ----------------------------------------------------------


async def test_a_failure_partway_through_the_build_releases_what_it_took(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing holds a reference to a constructor that raised.

    So the cleanup has to happen inside it or not at all. Asserted on the
    manifests, which are the resource that genuinely persists: a closed one
    raises when used, so the contexts built before the failure can be checked
    directly. The shared client is deliberately not asserted on -- the local
    Qdrant releases its lock when the object is collected, so a leak there is
    invisible to a test and would make this pass either way.
    """
    import workspace_indexer.workspace_registry as registry_module

    real_store = registry_module.build_vector_store
    real_context = registry_module.AppContext.from_config
    built: list[AppContext] = []
    calls: list[str] = []

    def record(*args: Any, **kwargs: Any) -> AppContext:
        context = real_context(*args, **kwargs)
        built.append(context)
        return context

    def fail_on_the_second(*args: Any, **kwargs: Any) -> Any:
        calls.append("x")
        if len(calls) > 1:
            raise RuntimeError("second store refused")
        return real_store(*args, **kwargs)

    monkeypatch.setattr(registry_module.AppContext, "from_config", record)
    monkeypatch.setattr(registry_module, "build_vector_store", fail_on_the_second)
    config = _config(tmp_path, _workspace(tmp_path, "alpha"), _workspace(tmp_path, "beta"))

    with pytest.raises(RuntimeError, match="second store refused"):
        WorkspaceRegistry(config)

    assert built, "nothing was built before the failure, so this asserts nothing"
    for context in built:
        # Already closed by the cleanup -- not closed here, which would make
        # the assertion below true whatever the registry did.
        with pytest.raises(sqlite3.ProgrammingError):
            context.manifest.file_count()


async def test_a_context_that_fails_to_close_does_not_strand_the_others(
    tmp_path: Path,
) -> None:
    """`close` promises the failures are reported, not swallowed, and that one
    bad context does not leak the rest. Its only failure-path logic, and it
    had no test."""
    import structlog.testing

    registry = WorkspaceRegistry(
        _config(tmp_path, _workspace(tmp_path, "alpha"), _workspace(tmp_path, "beta"))
    )

    async def refuse() -> None:
        raise RuntimeError("manifest is wedged")

    object.__setattr__(registry.context("alpha").store, "close", refuse)

    with structlog.testing.capture_logs() as logs:
        await registry.close()

    reported = [entry for entry in logs if entry["event"] == "registry.close_failed"]
    assert reported, "the failure was swallowed"
    assert any("manifest is wedged" in failure for failure in reported[0]["failures"])
    # The half the name promises, and the half the report alone cannot show:
    # stopping at alpha would produce exactly the same single-entry report.
    with pytest.raises(sqlite3.ProgrammingError):
        registry.context("beta").manifest.file_count()


async def test_a_failing_client_close_does_not_discard_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client close runs after the contexts, so an exception there used to
    escape before the collected failures were logged -- losing them, and in
    `serve`'s `finally` replacing the traceback of whatever ended the session.
    """
    import structlog.testing
    from qdrant_client import AsyncQdrantClient

    registry = WorkspaceRegistry(
        _config(tmp_path, _workspace(tmp_path, "alpha"), _workspace(tmp_path, "beta"))
    )

    async def refuse_store() -> None:
        raise RuntimeError("manifest is wedged")

    async def refuse_client(self: AsyncQdrantClient) -> None:
        raise RuntimeError("client will not close")

    object.__setattr__(registry.context("alpha").store, "close", refuse_store)
    monkeypatch.setattr(AsyncQdrantClient, "close", refuse_client)

    with structlog.testing.capture_logs() as logs:
        await registry.close()

    reported = [entry for entry in logs if entry["event"] == "registry.close_failed"]
    assert reported, "the client's failure discarded the whole report"
    failures = " ".join(reported[0]["failures"])
    assert "manifest is wedged" in failures
    assert "client will not close" in failures


def test_an_abandoned_build_closes_stores_as_well_as_manifests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifests are not the only thing a built context holds.

    Under `VECTOR_STORE=mongodb` nothing is shared and every store owns a live
    client, so a cleanup that closed only manifests would release nothing at
    all. Asserted here through a recording store rather than by configuring
    Mongo, since the behaviour under test is the registry's, not the driver's.
    """
    import workspace_indexer.workspace_registry as registry_module

    real_store = registry_module.build_vector_store
    closed: list[str] = []
    calls: list[str] = []

    def recording(settings: Any, name: str, **kwargs: Any) -> Any:
        calls.append(name)
        if len(calls) > 1:
            raise RuntimeError("second store refused")
        store = real_store(settings, name, **kwargs)
        original = store.close

        async def record() -> None:
            closed.append(name)
            await original()

        object.__setattr__(store, "close", record)
        return store

    monkeypatch.setattr(registry_module, "build_vector_store", recording)

    with pytest.raises(RuntimeError, match="second store refused"):
        WorkspaceRegistry(
            _config(tmp_path, _workspace(tmp_path, "alpha"), _workspace(tmp_path, "beta"))
        )

    assert closed == ["alpha"]


def test_a_cleanup_that_fails_does_not_replace_the_build_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_abandon` runs inside an `except` block, so an unguarded failure there
    escapes instead of the error that caused it -- handing the user a sqlite
    complaint about tidying up in place of "unknown rerank provider"."""
    import workspace_indexer.workspace_registry as registry_module

    real_store = registry_module.build_vector_store
    real_context = registry_module.AppContext.from_config
    calls: list[str] = []

    def wedge_the_manifest(*args: Any, **kwargs: Any) -> AppContext:
        context = real_context(*args, **kwargs)

        def refuse() -> None:
            raise RuntimeError("manifest will not close")

        object.__setattr__(context.manifest, "close", refuse)
        return context

    def fail_on_the_second(*args: Any, **kwargs: Any) -> Any:
        calls.append("x")
        if len(calls) > 1:
            raise RuntimeError("the real build error")
        return real_store(*args, **kwargs)

    monkeypatch.setattr(registry_module.AppContext, "from_config", wedge_the_manifest)
    monkeypatch.setattr(registry_module, "build_vector_store", fail_on_the_second)

    with pytest.raises(RuntimeError, match="the real build error"):
        WorkspaceRegistry(
            _config(tmp_path, _workspace(tmp_path, "alpha"), _workspace(tmp_path, "beta"))
        )
