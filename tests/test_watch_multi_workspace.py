"""`watch` over several workspaces: one watcher each, sharing what must be shared.

The design decision this encodes: rather than teaching the watching package
about workspaces, each workspace gets its own `Watcher` over its own
single-workspace config. Nothing in `watching/` knows workspaces exist -- and
where two workspaces cover the same tree, both watchers see the save and each
reindexes its own index, which is what has to happen because the file is in
both.

What that costs, and therefore what is asserted here: the inotify budget is
per *user*, so N independent checks would each report a fraction and none
would see the total; and N concurrent reindexes would mean N simultaneous
embedding calls on one save.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.isolated_env import isolate_context_env
from workspace_indexer.cli import app
from workspace_indexer.obs.logging import reset_for_tests


@pytest.fixture(autouse=True)
def _fresh_logging() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    reset_for_tests()
    yield
    reset_for_tests()


def _config(tmp_path: Path, *, shared: bool = False) -> Path:
    """Two workspaces. With `shared`, both also cover one common directory."""
    for name in ("alpha", "beta", "common", "state"):
        (tmp_path / name).mkdir(exist_ok=True)
    extra = f"      - path: {tmp_path / 'common'}\n" if shared else ""
    config = tmp_path / "workspace.yaml"
    config.write_text(
        "workspaces:\n"
        f"  - name: alpha\n    roots:\n      - path: {tmp_path / 'alpha'}\n{extra}"
        f"  - name: beta\n    roots:\n      - path: {tmp_path / 'beta'}\n{extra}"
        f"state_dir: {tmp_path / 'state'}\n"
        'logging:\n  console: "off"\n'
        f"  file:\n    path: {tmp_path / 'l.jsonl'}\n",
        encoding="utf-8",
    )
    return config


def _watchers_built(monkeypatch: pytest.MonkeyPatch, config: Path, *args: str) -> list[Any]:
    """Run `watch` with the watchers stubbed, and return the ones it built."""
    import workspace_indexer.watching.watcher as watcher_module

    built: list[Any] = []

    async def record(self: Any, stop: object = None) -> None:
        built.append(self)

    monkeypatch.setattr(watcher_module.Watcher, "run", record)
    result = CliRunner().invoke(app, ["watch", "--config", str(config), *args])
    # Checked here, so a failure during workspace selection or registry
    # construction says what went wrong instead of surfacing later as a
    # confusing assertion about how many watchers were built.
    assert result.exit_code == 0, result.output
    return built


def test_every_workspace_gets_its_own_watcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One per workspace, each over the single-workspace config the watching
    package already expects -- which is why that package needed no changes.

    Identified by `plan()`, which names the roots each watcher will cover:
    with a root per workspace, the plans say which workspace each watcher got
    without reading into it.
    """
    isolate_context_env(monkeypatch, tmp_path)

    built = _watchers_built(monkeypatch, _config(tmp_path))

    assert [sorted(w.plan()) for w in built] == [["alpha"], ["beta"]]


def test_naming_one_workspace_watches_only_that_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    isolate_context_env(monkeypatch, tmp_path)

    built = _watchers_built(monkeypatch, _config(tmp_path), "--workspace", "beta")

    assert [sorted(w.plan()) for w in built] == [["beta"]]


def test_a_shared_tree_is_watched_by_both_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fan-out, which falls out of the design rather than being written.

    Two workspaces covering one repository both hold its files in their own
    index, so a save there has to reindex both. Each watcher's scope covering
    the shared directory is what makes that happen.
    """
    isolate_context_env(monkeypatch, tmp_path)

    built = _watchers_built(monkeypatch, _config(tmp_path, shared=True))

    # Stated exactly rather than as a membership check: "common is in both
    # plans" also holds if both watchers wrongly covered every root.
    assert [sorted(w.plan()) for w in built] == [["alpha", "common"], ["beta", "common"]]


def test_the_inotify_budget_is_shared_across_workspaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """inotify's limit is per user, not per watch. Given a budget each, every
    watcher would report its own fraction and none would see the total -- so
    the one number anybody reads would always be too small.

    Asserted on the running total rather than on object identity: sharing the
    instance is only half of it, and a shared-but-stateless budget would pass
    an identity check while reporting exactly the fractions it was supposed to
    stop reporting.
    """
    from workspace_indexer.watching.inotify_budget import InotifyBudget

    shared = InotifyBudget(limit=1_000)
    shared.check(400)
    shared.check(400)

    assert shared.reserved == 800
    # Neither call alone crosses it; together they do, which is the state the
    # per-user limit makes real and a per-watcher budget cannot see.
    assert shared.check(400) is False


def test_the_cli_shares_one_lock_and_one_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both wirings, driven through the real `run()`.

    Neither was pinned before: the serialisation test built its own lock and
    called the private `_build_watcher`, and the budget test exercised
    `InotifyBudget` in isolation -- so moving `asyncio.Lock()` or
    `InotifyBudget.detect()` inside the per-watcher comprehension passed the
    entire suite.

    `Watcher` is replaced on the module the CLI resolves it from at call time,
    so this goes through the real command: real registry, real reindex
    closures, real `TaskGroup`.
    """
    import workspace_indexer.watching as watching_module
    from workspace_indexer.app_context import AppContext
    from workspace_indexer.config.watch_mode import WatchMode

    isolate_context_env(monkeypatch, tmp_path)

    running = 0
    overlapped = False
    budgets: list[Any] = []

    class Stats:
        files_changed = 0
        chunks_upserted = 0
        chunks_deleted = 0

    class SlowIndexer:
        async def run(self, only_root: str | None = None, **kwargs: Any) -> Stats:
            nonlocal running, overlapped
            running += 1
            overlapped = overlapped or running > 1
            await asyncio.sleep(0.05)
            running -= 1
            return Stats()

    class FakeWatcher:
        def __init__(self, _config: Any, **kwargs: Any) -> None:
            self._reindex = kwargs["reindex"]
            self._budget = kwargs["budget"]
            budgets.append(self._budget)

        def plan(self) -> dict[str, WatchMode]:
            return {"r": WatchMode.NATIVE}

        async def run(self, stop: object = None) -> None:
            # What a real watcher does on start, and then on one change.
            self._budget.check(2)
            await self._reindex("r")

    def slow_indexer(_self: AppContext) -> SlowIndexer:
        return SlowIndexer()

    monkeypatch.setattr(AppContext, "indexer", slow_indexer)
    monkeypatch.setattr(watching_module, "Watcher", FakeWatcher)

    result = CliRunner().invoke(app, ["watch", "--config", _config(tmp_path).as_posix()])

    assert result.exit_code == 0, result.output
    assert len(budgets) == 2
    # The lock: two watchers reindexing at once would have overlapped.
    assert not overlapped, "two workspaces reindexed at the same time"
    # The budget: a budget each would leave both reading 2.
    assert budgets[0].reserved == 4


def test_a_crash_in_one_watcher_names_it_and_cancels_the_others_before_closing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teardown with more than one watcher, which nothing covered.

    Two claims. The crash names the workspace, or an operator reading the JSONL
    of an unattended run has to recheck every one.

    And the siblings are cancelled *before* the registry closes. `gather`
    propagates the first failure and leaves its siblings running, so the close
    would happen underneath live watchers and their next reindex would fail
    against a closed manifest -- spurious `watch.reindex_failed` entries beside
    the one real crash. Asserted as an **order**, because that is the claim:
    merely observing that the sibling ended says nothing, since `asyncio.run`
    cancels whatever is left at the very end either way.
    """
    import json

    import workspace_indexer.watching as watching_module
    from workspace_indexer.config.watch_mode import WatchMode
    from workspace_indexer.workspace_registry import WorkspaceRegistry

    isolate_context_env(monkeypatch, tmp_path)

    order: list[str] = []
    real_close = WorkspaceRegistry.close

    async def recording_close(self: WorkspaceRegistry) -> None:
        order.append("registry closed")
        await real_close(self)

    class FakeWatcher:
        def __init__(self, config: Any, **kwargs: Any) -> None:
            self._name = config.workspace.name

        def plan(self) -> dict[str, WatchMode]:
            return {"r": WatchMode.NATIVE}

        async def run(self, stop: object = None) -> None:
            if self._name == "alpha":
                raise RuntimeError("the file cannot be accessed by the system")
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                order.append("sibling cancelled")
                raise

    monkeypatch.setattr(WorkspaceRegistry, "close", recording_close)
    monkeypatch.setattr(watching_module, "Watcher", FakeWatcher)

    result = CliRunner().invoke(app, ["watch", "--config", _config(tmp_path).as_posix()])

    assert result.exit_code != 0
    assert order == ["sibling cancelled", "registry closed"], order

    entries = [
        json.loads(line)
        for line in (tmp_path / "l-watch.jsonl").read_text().splitlines()
        if line.strip()
    ]
    crashed = [e for e in entries if e["event"] == "watch.crashed"]
    assert len(crashed) == 1
    assert crashed[0]["workspace"] == "alpha"
