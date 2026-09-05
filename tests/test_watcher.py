"""The watcher: mode selection, and a real edit reaching a real reindex.

The mode-selection half is unit-tested against a fake /proc/mounts because the
interesting filesystems (9p, cifs) cannot be created in a test. The delivery
half runs against the real filesystem with a real inotify watch, because a
watcher that passes every unit test and never fires is exactly the failure
mode this feature exists to avoid.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import structlog.testing

# watchfiles ships no py.typed marker; scoped to the one symbol.
from watchfiles import Change  # pyright: ignore[reportUnknownVariableType]
from watchfiles._rust_notify import (  # pyright: ignore[reportUnknownVariableType]
    WatchfilesRustInternalError,
)

from workspace_indexer.config import WatchMode, WorkspaceConfig
from workspace_indexer.watching import ExcludeFilter, FilesystemProbe, InotifyBudget, Watcher

LOCAL_MOUNTS = "/dev/sda2 / ext4 rw 0 0\n/dev/sda3 /tmp ext4 rw 0 0\n"


def _config(root: Path, **watch: object) -> WorkspaceConfig:
    return WorkspaceConfig.model_validate(
        {
            "workspace": {"name": "w", "roots": [{"path": str(root), "label": "main"}]},
            "watch": watch,
        }
    )


def _noop(root: str | None) -> None:
    return None


def _watcher(
    config: WorkspaceConfig,
    *,
    mounts: str = LOCAL_MOUNTS,
    reindex: Callable[[str | None], object] | None = None,
    config_path: Path | None = None,
) -> Watcher:
    return Watcher(
        config,
        reindex=reindex or _noop,
        config_path=config_path,
        probe=FilesystemProbe(mounts),
        budget=InotifyBudget(limit=100_000),
    )


# --- choosing a mode -----------------------------------------------------


def test_a_local_root_is_watched_natively(tmp_path: Path) -> None:
    plan = _watcher(_config(tmp_path)).plan()
    assert plan == {"main": WatchMode.NATIVE}


def test_a_root_on_a_non_native_filesystem_is_polled(tmp_path: Path) -> None:
    """The whole point of the probe. inotify on this path would succeed and
    then never deliver an event."""
    mounts = f"drvfs {tmp_path} 9p rw 0 0\n"
    with structlog.testing.capture_logs() as logs:
        plan = _watcher(_config(tmp_path), mounts=mounts).plan()

    assert plan == {"main": WatchMode.POLL}
    warning = next(e for e in logs if e["event"] == "watch.polling")
    assert warning["filesystem"] == "9p"
    assert "never fire" in str(warning["detail"])


def test_an_explicit_mode_overrides_the_probe(tmp_path: Path) -> None:
    mounts = f"drvfs {tmp_path} 9p rw 0 0\n"
    config = _config(tmp_path, mode="native")
    assert _watcher(config, mounts=mounts).plan() == {"main": WatchMode.NATIVE}


def test_polling_can_be_forced_on_a_local_root(tmp_path: Path) -> None:
    config = _config(tmp_path, mode="poll")
    assert _watcher(config).plan() == {"main": WatchMode.POLL}


def test_the_watch_budget_is_reported(tmp_path: Path) -> None:
    (tmp_path / "a" / "b").mkdir(parents=True)
    with structlog.testing.capture_logs() as logs:
        _watcher(_config(tmp_path)).check_budget()

    entry = next(e for e in logs if e["event"] == "watch.budget")
    assert entry["needed"] == 3
    assert entry["limit"] == 100_000


def test_a_polled_root_needs_no_inotify_budget(tmp_path: Path) -> None:
    """Nothing is watched, so nothing is counted -- and a warning about a limit
    that does not apply is noise."""
    mounts = f"nfsserver:/x {tmp_path} nfs4 rw 0 0\n"
    with structlog.testing.capture_logs() as logs:
        _watcher(_config(tmp_path), mounts=mounts).check_budget()

    assert not [e for e in logs if e["event"].startswith("watch.budget")]


def test_heavy_directories_are_left_out_of_the_count(tmp_path: Path) -> None:
    """Ignore rules keep the watcher functional, not just the index tidy."""
    (tmp_path / "src").mkdir()
    for i in range(20):
        (tmp_path / "node_modules" / f"pkg{i}").mkdir(parents=True)

    with structlog.testing.capture_logs() as logs:
        _watcher(_config(tmp_path)).check_budget()

    assert next(e for e in logs if e["event"] == "watch.budget")["needed"] == 2


# --- actually delivering ------------------------------------------------


async def test_a_real_edit_triggers_a_real_reindex(tmp_path: Path) -> None:
    """End to end against the filesystem, with a real inotify watch.

    Everything above would pass just as happily if `awatch` were never wired
    up. This is the test that fails if the watcher looks healthy and does
    nothing, which is the failure this feature exists to prevent.
    """
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("x = 1\n", encoding="utf-8")

    reindexed: list[str | None] = []

    async def reindex(root: str | None) -> None:
        reindexed.append(root)

    config = _config(tmp_path, debounce_ms=100)
    watcher = _watcher(config, reindex=reindex)
    stop = asyncio.Event()
    task = asyncio.create_task(watcher.run(stop))

    # Let the watch establish before touching anything.
    await asyncio.sleep(0.6)
    (tmp_path / "app" / "main.py").write_text("x = 2\n", encoding="utf-8")

    for _ in range(60):
        if reindexed:
            break
        await asyncio.sleep(0.1)

    stop.set()
    await asyncio.wait_for(task, timeout=10)
    assert reindexed == ["main"]


async def test_rapid_edits_coalesce_into_one_reindex(tmp_path: Path) -> None:
    """A formatter run touches forty files in two seconds. Forty reindexes of
    the same root would embed the same content over and over."""
    (tmp_path / "app").mkdir()
    reindexed: list[str | None] = []

    async def reindex(root: str | None) -> None:
        reindexed.append(root)
        await asyncio.sleep(0)

    config = _config(tmp_path, debounce_ms=400)
    watcher = _watcher(config, reindex=reindex)
    stop = asyncio.Event()
    task = asyncio.create_task(watcher.run(stop))

    await asyncio.sleep(0.6)
    for i in range(25):
        (tmp_path / "app" / f"f{i}.py").write_text(f"x = {i}\n", encoding="utf-8")
        await asyncio.sleep(0.005)

    for _ in range(60):
        if reindexed:
            break
        await asyncio.sleep(0.1)
    await asyncio.sleep(0.5)

    stop.set()
    await asyncio.wait_for(task, timeout=10)
    # 25 files, one root, and far fewer than 25 reindexes.
    assert reindexed
    assert len(reindexed) <= 3
    assert set(reindexed) == {"main"}


async def test_editing_the_config_reloads_it_without_a_restart(tmp_path: Path) -> None:
    """Ignore rules and settings go live on save.

    The honest limit is a *new root*: `awatch` cannot be told about another
    path mid-iteration, so adding one still needs a restart. The watcher says
    so rather than leaving it to be discovered.
    """
    root = tmp_path / "root"
    root.mkdir()
    config_file = tmp_path / "workspace.yaml"
    config_file.write_text("initial", encoding="utf-8")

    reloads: list[int] = []

    def reload() -> WorkspaceConfig:
        reloads.append(1)
        return _config(root, debounce_ms=100)

    watcher = Watcher(
        _config(root, debounce_ms=100),
        reindex=_noop,
        config_path=config_file,
        probe=FilesystemProbe(LOCAL_MOUNTS),
        budget=InotifyBudget(limit=100_000),
        reload_config=reload,
    )
    stop = asyncio.Event()
    with structlog.testing.capture_logs() as logs:
        task = asyncio.create_task(watcher.run(stop))
        await asyncio.sleep(0.6)
        config_file.write_text("changed", encoding="utf-8")

        for _ in range(60):
            if reloads:
                break
            await asyncio.sleep(0.1)
        stop.set()
        await asyncio.wait_for(task, timeout=10)

    assert reloads
    notice = next(e for e in logs if e["event"] == "watch.config_reloaded")
    assert "restart" in str(notice["detail"])


async def test_a_half_written_config_does_not_kill_the_watcher(tmp_path: Path) -> None:
    """Observing a YAML file mid-write is normal. Keeping the old config beats
    dying on a transient parse error."""
    root = tmp_path / "root"
    root.mkdir()

    def reload() -> WorkspaceConfig:
        raise ValueError("mapping values are not allowed here")

    watcher = Watcher(
        _config(root),
        reindex=_noop,
        config_path=tmp_path / "workspace.yaml",
        probe=FilesystemProbe(LOCAL_MOUNTS),
        budget=InotifyBudget(limit=100_000),
        reload_config=reload,
    )

    with structlog.testing.capture_logs() as logs:
        # Driven directly: reproducing a torn write through the real watch
        # loop is a race, and the behaviour under test is the recovery.
        await watcher.handle_changes({(Change.modified, str(tmp_path / "workspace.yaml"))})

    assert [e for e in logs if e["event"] == "watch.config_invalid"]


# --- surviving a walk the OS refuses ------------------------------------


async def test_a_rust_walk_failure_is_logged_with_something_to_act_on(tmp_path: Path) -> None:
    """The reported Windows crash, made diagnosable.

    A dangling symlink inside an `index.exclude`d tree killed the watcher with
    a raw traceback. It cannot be prevented -- watchfiles filters changes the
    Rust watcher has already produced, so recursion happens first -- so the
    obligation is to say what happened and what to do, and to leave a record
    after the terminal is gone.
    """
    watcher = _watcher(_config(tmp_path))

    async def explode(*_: object, **__: object) -> None:
        raise WatchfilesRustInternalError(
            "error in underlying watcher: IO error for operation on "
            r"C:\x\.ralph\tasks: The file cannot be accessed by the system. (os error 1920)"
        )

    with (
        patch.object(Watcher, "_watch", explode),
        structlog.testing.capture_logs() as logs,
        pytest.raises(WatchfilesRustInternalError),
    ):
        await watcher.run()

    failed = [entry for entry in logs if entry["event"] == "watch.walk_failed"]
    assert len(failed) == 1
    assert "os error 1920" in failed[0]["error"]
    # The path is in the error, and the detail says why config cannot help.
    assert "index.exclude" in failed[0]["detail"]


async def test_the_watch_filter_and_permission_flag_are_passed_to_awatch(
    tmp_path: Path,
) -> None:
    """Both are easy to drop in a refactor and neither has a visible symptom.

    Without the filter, excluded trees quietly wake reindexes again; without
    the flag, an unreadable directory is fatal where it need not be.
    """
    watcher = _watcher(_config(tmp_path))
    seen: dict[str, object] = {}

    async def capture(*paths: object, **kwargs: object):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return
        yield  # pragma: no cover - makes this an async generator

    with patch("workspace_indexer.watching.watcher.awatch", capture):
        await watcher.run()

    assert isinstance(seen["watch_filter"], ExcludeFilter)
    assert seen["ignore_permission_denied"] is True


# --- issue #62: watching only what the index looks at --------------------


async def test_the_watch_is_scoped_and_not_recursive(tmp_path: Path) -> None:
    """Recursion is what descends into excluded trees.

    The Rust layer accepts no exclusion, so the only way to stay out of a tree
    is to not name it and to not recurse into its parent.
    """
    (tmp_path / "src").mkdir()
    (tmp_path / ".ralph" / "tasks").mkdir(parents=True)
    config = WorkspaceConfig.model_validate(
        {
            "workspace": {"name": "w", "roots": [{"path": str(tmp_path), "label": "main"}]},
            "index": {"exclude": ["**/.ralph/**"]},
        }
    )
    watcher = _watcher(config)
    seen: dict[str, Any] = {}

    async def capture(*paths: object, **kwargs: object):  # type: ignore[no-untyped-def]
        seen["paths"] = [str(p) for p in paths]
        seen.update(kwargs)
        return
        yield  # pragma: no cover - makes this an async generator

    with patch("workspace_indexer.watching.watcher.awatch", capture):
        await watcher.run()

    assert seen["recursive"] is False
    assert any(p.endswith("src") for p in seen["paths"])
    assert not any(".ralph" in p for p in seen["paths"])


async def test_a_new_indexed_directory_rebuilds_the_watch(tmp_path: Path) -> None:
    """A watch cannot be extended while it runs.

    Without a rebuild the new tree would simply never be watched, and edits in
    it would never reindex -- silently, which is the worst version.
    """
    (tmp_path / "src").mkdir()
    watcher = _watcher(_config(tmp_path))
    scopes: list[int] = []

    async def capture(*paths: object, **kwargs: object):  # type: ignore[no-untyped-def]
        scopes.append(len(paths))
        if len(scopes) == 1:
            (tmp_path / "src" / "feature").mkdir()
            yield {(Change.added, str(tmp_path / "src" / "feature"))}
        return

    with patch("workspace_indexer.watching.watcher.awatch", capture):
        await watcher.run()

    assert len(scopes) == 2, "expected exactly one rebuild"
    assert scopes[1] == scopes[0] + 1


async def test_a_new_excluded_directory_does_not(tmp_path: Path) -> None:
    """Build output appears constantly. Rebuilding for it would restart the
    watcher every time a build runs."""
    (tmp_path / "src").mkdir()
    config = WorkspaceConfig.model_validate(
        {
            "workspace": {"name": "w", "roots": [{"path": str(tmp_path), "label": "main"}]},
            "index": {"exclude": ["**/obj/**"]},
        }
    )
    watcher = _watcher(config)
    scopes: list[int] = []

    async def capture(*paths: object, **kwargs: object):  # type: ignore[no-untyped-def]
        scopes.append(len(paths))
        if len(scopes) == 1:
            (tmp_path / "obj").mkdir()
            yield {(Change.added, str(tmp_path / "obj"))}
        return

    with patch("workspace_indexer.watching.watcher.awatch", capture):
        await watcher.run()

    assert len(scopes) == 1, "an ignored directory must not rebuild the watch"


async def test_a_new_file_does_not_rebuild_the_watch(tmp_path: Path) -> None:
    """Its directory is already watched; only directories need new watches."""
    (tmp_path / "src").mkdir()
    watcher = _watcher(_config(tmp_path))
    scopes: list[int] = []

    async def capture(*paths: object, **kwargs: object):  # type: ignore[no-untyped-def]
        scopes.append(len(paths))
        if len(scopes) == 1:
            new = tmp_path / "src" / "app.py"
            new.write_text("x = 1\n", encoding="utf-8")
            yield {(Change.added, str(new))}
        return

    with patch("workspace_indexer.watching.watcher.awatch", capture):
        await watcher.run()

    assert len(scopes) == 1
