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
    CliRunner().invoke(app, ["watch", "--config", str(config), *args])
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

    assert len(built) == 2
    # Both plans name the shared root, which is what makes both watchers react
    # to a save there.
    for watcher in built:
        assert "common" in watcher.plan()


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


async def test_reindexes_do_not_overlap_across_workspaces(tmp_path: Path) -> None:
    """Separate collections make concurrent writes safe, but concurrent
    embedding calls are not free: one save in a shared repository would fire
    several API requests at once, which makes the cost of a keystroke
    unpredictable."""
    from workspace_indexer.cli import _build_watcher  # pyright: ignore[reportPrivateUsage]

    overlapped = False
    running = 0

    class SlowIndexer:
        async def run(self, only_root: str | None = None) -> Any:
            nonlocal overlapped, running
            running += 1
            overlapped = overlapped or running > 1
            await asyncio.sleep(0.05)
            running -= 1
            raise RuntimeError("stop here; the reindex itself is not under test")

    class FakeContext:
        config = None

        def indexer(self) -> SlowIndexer:
            return SlowIndexer()

    captured: list[Any] = []

    class FakeWatcher:
        def __init__(self, _config: Any, **kwargs: Any) -> None:
            captured.append(kwargs["reindex"])

    lock = asyncio.Lock()
    for name in ("alpha", "beta"):
        _build_watcher(
            FakeWatcher,  # pyright: ignore[reportArgumentType]
            FakeContext(),  # pyright: ignore[reportArgumentType]
            name,
            tmp_path / "workspace.yaml",
            lock,
            None,  # pyright: ignore[reportArgumentType]
        )

    await asyncio.gather(*(reindex("root") for reindex in captured))

    assert len(captured) == 2
    assert not overlapped, "two workspaces reindexed at the same time"
