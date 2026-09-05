"""Keeping the index fresh without a manual run.

A trigger, not a second indexing path. Every change ends up going through the
same `Indexer.run(only_root=...)` the CLI calls, so the watcher cannot develop
its own opinion about what counts as changed -- and the decision ladder, the
orphan pruning and the root scoping are all already tested there.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

# watchfiles ships no py.typed marker, so strict mode cannot see through to
# awatch's signature. Scoped to the one symbol rather than the module.
from watchfiles import Change, awatch  # pyright: ignore[reportUnknownVariableType]
from watchfiles._rust_notify import (  # pyright: ignore[reportUnknownVariableType]
    WatchfilesRustInternalError,
)

from workspace_indexer.config import WatchMode, WatchSection, WorkspaceConfig
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.watching.change_debouncer import ChangeDebouncer
from workspace_indexer.watching.exclude_filter import ExcludeFilter
from workspace_indexer.watching.filesystem_probe import FilesystemProbe
from workspace_indexer.watching.inotify_budget import InotifyBudget
from workspace_indexer.watching.watch_scope import UNWATCHED_DIRS, WatchScope

log = get_logger("workspace_indexer.watching")

# Re-exported from its new home so `from ...watching.watcher import
# UNWATCHED_DIRS` keeps working; it lives with the code that applies it.
__all__ = ["UNWATCHED_DIRS", "Watcher"]


class Watcher:
    def __init__(
        self,
        config: WorkspaceConfig,
        *,
        reindex: Callable[[str | None], object],
        config_path: Path | None = None,
        probe: FilesystemProbe | None = None,
        budget: InotifyBudget | None = None,
        reload_config: Callable[[], WorkspaceConfig] | None = None,
    ) -> None:
        self._config = config
        self._reindex = reindex
        self._config_path = config_path.resolve() if config_path else None
        self._probe = probe or FilesystemProbe()
        self._budget = budget or InotifyBudget.detect()
        self._reload_config = reload_config
        self._debouncer = ChangeDebouncer(config, self._config_path)
        self._scope = WatchScope(config)
        # Set when a directory appears that the index would look in. The
        # watch cannot be extended while it runs, so this asks run() to
        # rebuild rather than silently leaving the new tree unwatched.
        self._rescope = False

    @property
    def settings(self) -> WatchSection:
        return self._config.watch

    def plan(self) -> dict[str, WatchMode]:
        """Which mode each root will be watched in, decided before starting.

        Computed up front so `watch` can report it, rather than leaving the
        answer to be inferred from whether anything ever happens.
        """
        configured = self._config.watch.mode
        decided: dict[str, WatchMode] = {}
        for root in self._config.workspace.roots:
            path = root.path.expanduser().resolve()
            if configured is not WatchMode.AUTO:
                decided[root.resolved_label] = configured
                continue
            native = self._probe.supports_inotify(path)
            decided[root.resolved_label] = WatchMode.NATIVE if native else WatchMode.POLL
            if not native:
                log.warning(
                    "watch.polling",
                    root=root.resolved_label,
                    path=str(path),
                    filesystem=self._probe.filesystem_for(path),
                    detail="this filesystem delivers no change notifications, so "
                    "inotify would succeed and then never fire. Polling instead.",
                )
        return decided

    def check_budget(self) -> None:
        """Report the inotify headroom for the roots that will use it."""
        plan = self.plan()
        native = [
            root
            for root in self._config.workspace.roots
            if plan.get(root.resolved_label) is WatchMode.NATIVE
        ]
        if not native:
            return
        # Counted from the scope rather than from a coarse name list: the
        # watch is placed on exactly these directories, so this is the real
        # number rather than an estimate that was always too high.
        watched = set(self._scope.directories())
        native_roots = {root.path.expanduser().resolve() for root in native}
        needed = sum(
            1
            for directory in watched
            if any(directory == base or base in directory.parents for base in native_roots)
        )
        self._budget.check(needed)

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Watch until cancelled, reindexing each root whose files changed."""
        plan = self.plan()
        self.check_budget()

        # One poll interval covers the whole watch, so any polled root forces
        # polling for all of them. watchfiles offers no per-path mode, and
        # mixing would mean two concurrent watches to keep in step.
        force_polling = any(mode is WatchMode.POLL for mode in plan.values())
        log.info(
            "watch.start",
            roots={label: mode.value for label, mode in plan.items()},
            polling=force_polling,
            debounce_ms=self._config.watch.debounce_ms,
        )

        while True:
            paths = [str(p) for p in self._scope.directories()]
            if self._config_path is not None and self._config.watch.reload_config:
                # Watched explicitly: workspace.yaml usually sits outside every
                # root, so nothing else would notice it change.
                paths.append(str(self._config_path))
            log.info("watch.scoped", directories=len(paths))

            self._rescope = False
            try:
                await self._watch(paths, stop, force_polling)
            except WatchfilesRustInternalError as exc:
                self._report_walk_failure(exc)
                raise
            if not self._rescope or (stop is not None and stop.is_set()):
                return
            # A directory the index cares about appeared. Nothing can be added
            # to a running watch, so the only way to cover it is to rebuild --
            # rare in practice, because build output is excluded and excluded
            # directories do not trigger this.
            log.info(
                "watch.rescoping",
                detail="a new directory inside a watched tree needs its own watch, "
                "which cannot be added to a running watcher",
            )

    def _is_new_directory(self, path: Path) -> bool:
        """Is this a directory the index would look in?

        Both halves matter. A new *file* needs no new watch -- its directory is
        already watched. And a new directory the index ignores is not worth a
        rebuild, which is what keeps build output from restarting the watcher
        every time it runs.
        """
        try:
            if not path.is_dir():
                return False
        except OSError:
            return False
        return self._scope.covers(path)

    def _report_walk_failure(self, exc: WatchfilesRustInternalError) -> None:
        """The Rust watcher failed on a path, rather than reporting a change.

        Much rarer than it was: the watch is now scoped to the directories the
        index would look in, so a broken symlink inside an excluded tree is
        never touched at all. What is left is a path we *do* watch becoming
        unreadable while the watcher holds it -- which no configuration avoids,
        and which a raw traceback gives an operator nothing to act on.
        """
        log.error(
            "watch.walk_failed",
            error=str(exc),
            detail="the filesystem watcher could not read a path it was watching. "
            "The error names it: remove or repair it, then restart watch. If it "
            "sits in a tree you do not index, adding it to index.exclude will keep "
            "the watcher out of it as well.",
        )

    async def _watch(
        self, paths: list[str], stop: asyncio.Event | None, force_polling: bool
    ) -> None:
        async for batch in awatch(
            *paths,
            stop_event=stop,
            debounce=self._config.watch.debounce_ms,
            step=min(50, self._config.watch.debounce_ms),
            force_polling=force_polling,
            poll_delay_ms=self._config.watch.poll_interval_ms,
            # Not recursive: recursion is what descends into excluded trees,
            # and the Rust layer accepts no exclusion of its own. Every
            # directory worth watching is named explicitly instead.
            recursive=False,
            # Still worth having with a scoped watch: it drops editor
            # scratch files and `.pyc` beside a file we do watch, which no
            # amount of choosing directories can exclude.
            watch_filter=ExcludeFilter(self._config),
            # A directory we cannot read is not a reason to kill the watch.
            ignore_permission_denied=True,
        ):
            await self.handle_changes(batch)

    async def handle_changes(self, batch: set[tuple[Change, str]]) -> None:
        """Process one settled batch of events.

        Public because it is the whole behaviour of this class minus the event
        loop, and driving a torn config write through the real watch is a race
        rather than a test.
        """
        for change, raw in batch:
            path = Path(raw)
            if change is Change.added and not self._rescope and self._is_new_directory(path):
                self._rescope = True
            self._debouncer.add(path)

        roots, config_changed = self._debouncer.drain()

        if config_changed:
            await self._reload()
            # A reloaded config can add roots, and `awatch` cannot be told
            # about a new path mid-iteration. Say so rather than pretending.
            log.warning(
                "watch.config_reloaded",
                detail="settings and ignore rules are live; a newly added *root* "
                "needs a restart of `watch` to be observed",
            )

        for label in sorted(roots):
            log.info("watch.reindex", root=label)
            result = self._reindex(label)
            if asyncio.isfuture(result) or asyncio.iscoroutine(result):
                await result

    async def _reload(self) -> None:
        if self._reload_config is None:
            return
        try:
            self._config = self._reload_config()
        except Exception as exc:
            # A half-saved YAML file is a normal thing to observe mid-write.
            # Keeping the old config beats dying on a transient parse error.
            log.error("watch.config_invalid", error=str(exc))
            return
        self._debouncer = ChangeDebouncer(self._config, self._config_path)
        self._scope = WatchScope(self._config)
