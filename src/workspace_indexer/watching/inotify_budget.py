"""How many inotify watches a tree needs, against how many the kernel allows.

`/proc/sys/fs/inotify/max_user_watches` caps watches per *user*, not per
process -- often 65536, sometimes 8192, and shared with every editor, language
server and file manager already running. A workspace containing `node_modules`
will exhaust it.

This is the other reason our ignore rules matter. They are not only about index
quality: they are what keeps the watcher functional at all. Exhausting the
limit surfaces as `OSError: [Errno 28] No space left on device` from
`inotify_add_watch`, which is one of the least helpful error messages in Linux
-- it has nothing to do with disk space.
"""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.watching.budget")

MAX_USER_WATCHES = Path("/proc/sys/fs/inotify/max_user_watches")

# Warn above this share of the limit. Well below 1.0 because the limit is
# shared: being at 80% ourselves means the next editor to open cannot watch.
_WARN_AT = 0.8


class InotifyBudget:
    """Counts the directories a watch would need and reports the headroom."""

    def __init__(self, limit: int | None) -> None:
        """`limit` is taken literally, None included.

        Explicit rather than defaulting to a probe, so that None can mean "the
        limit is unknown" -- otherwise the one state this class must handle is
        the one its constructor cannot express.
        """
        self._limit = limit
        self._reserved = 0

    @classmethod
    def detect(cls) -> InotifyBudget:
        """Read the live limit, or unknown where there is no /proc."""
        return cls(_read_limit())

    @property
    def limit(self) -> int | None:
        """None when the limit cannot be read: not Linux, or no /proc."""
        return self._limit

    def count_directories(self, root: Path, excluded: set[str] | None = None) -> int:
        """Directories under `root`, which is what inotify watches.

        inotify watches directories, not files -- a watch on a directory
        reports changes to its immediate children. So the cost of watching a
        tree is its directory count, and `node_modules` is expensive because it
        is deep and wide, not because it is large.
        """
        skip = excluded or set()
        total = 1
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                entries = list(current.iterdir())
            except OSError:
                continue
            for entry in entries:
                if not entry.is_dir() or entry.is_symlink():
                    continue
                if entry.name in skip:
                    continue
                total += 1
                stack.append(entry)
        return total

    @property
    def reserved(self) -> int:
        """Directories accounted for so far, across every `check` on this object."""
        return self._reserved

    def check(self, needed: int, label: str | None = None) -> bool:
        """Log the headroom. Returns False when the watch will not fit.

        Reported rather than enforced: a watcher that refuses to start is worse
        than one that starts and says it is short, because the second at least
        tells you which directory to exclude.

        **Cumulative.** The kernel limit is per *user*, not per watch, so what
        matters is everything this process will watch -- and `watch` runs one
        watcher per workspace. Each call adds to a running total and the
        thresholds are judged against that, so the last caller sees the real
        figure and any caller that crosses the line says so. With a single
        watcher, which is every other caller, the total is its own number and
        nothing changes.

        Both numbers are logged, and `label` names the caller. Reporting only
        the running total under the name `needed` would read as "this watcher
        needs 4" on the second of two workspaces -- a figure the reader would
        then act on by excluding trees from the wrong one.

        One trap for later: a re-check after a rescope or a config reload would
        double-count, because this never forgets. Nothing re-checks today;
        anything that starts to must pass the delta.
        """
        self._reserved += needed
        total = self._reserved
        if self._limit is None:
            log.debug("watch.budget_unknown", needed=needed, reserved=total, label=label)
            return True

        share = total / self._limit if self._limit else 1.0
        if total > self._limit:
            log.error(
                "watch.budget_exceeded",
                needed=needed,
                reserved=total,
                label=label,
                limit=self._limit,
                detail="more directories than inotify can watch; the watch will fail "
                "with 'No space left on device', which is not about disk space. "
                "Exclude large trees such as node_modules, or raise "
                "fs.inotify.max_user_watches.",
            )
            return False
        if share >= _WARN_AT:
            log.warning(
                "watch.budget_tight",
                needed=needed,
                reserved=total,
                label=label,
                limit=self._limit,
                used_share=round(share, 2),
                detail="the limit is per user and shared with editors and language "
                "servers already running",
            )
        else:
            log.info("watch.budget", needed=needed, reserved=total, label=label, limit=self._limit)
        return True


def _read_limit() -> int | None:
    try:
        return int(MAX_USER_WATCHES.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
