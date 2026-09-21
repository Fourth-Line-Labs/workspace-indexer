"""Top-level workspace.yaml model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr, model_validator

from workspace_indexer.config.chunking_section import ChunkingSection
from workspace_indexer.config.eval_section import EvalSection
from workspace_indexer.config.excludes import HARDCODED_EXCLUDES
from workspace_indexer.config.graph_section import GraphSection
from workspace_indexer.config.index_section import IndexSection
from workspace_indexer.config.logging_config import LoggingConfig
from workspace_indexer.config.root_config import RootConfig
from workspace_indexer.config.search_section import SearchSection
from workspace_indexer.config.strict import Strict
from workspace_indexer.config.watch_section import WatchSection
from workspace_indexer.config.workspace_choice_error import WorkspaceChoiceError
from workspace_indexer.config.workspace_section import WorkspaceSection


class WorkspaceConfig(Strict):
    """One file, one or many indexes.

    `workspace:` (one) and `workspaces:` (a list) are the same thing written two
    ways -- the singular form normalises into a one-element list, so every
    config written before this existed keeps working unchanged, and a
    single-workspace setup behaves exactly as it did.

    Everything outside `workspaces` is shared. Two workspaces on one machine
    have the same excludes, the same chunking and the same search settings in
    practice, and the cost of saying so twice is that they drift apart.
    """

    workspaces: list[WorkspaceSection] = Field(min_length=1)
    # Where per-workspace manifests live. Only needed once there is more than
    # one workspace: a single workspace keeps using STATE_DB, so no existing
    # index has to move. Config rather than an environment variable because
    # STATE_DB being an env var is precisely why two workspaces needed two
    # environments -- `--config` cannot redirect it.
    state_dir: Path | None = None
    index: IndexSection = Field(default_factory=IndexSection)
    chunking: ChunkingSection = Field(default_factory=ChunkingSection)
    search: SearchSection = Field(default_factory=SearchSection)
    watch: WatchSection = Field(default_factory=WatchSection)
    graph: GraphSection = Field(default_factory=GraphSection)
    eval: EvalSection = Field(default_factory=EvalSection)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    # Every configured dataset, captured before `select` narrows the list.
    # Private because it is derived, not written: a key here would let a
    # config claim a dataset it does not have.
    _dataset_paths: set[Path] = PrivateAttr(default_factory=lambda: set[Path]())

    @model_validator(mode="before")
    @classmethod
    def _accept_either_spelling(cls, data: Any) -> Any:
        """`workspace:` is `workspaces:` with one entry.

        Done before field validation so that `extra="forbid"` never sees the
        singular key. Both at once is refused rather than merged: a config
        holding two answers to "what do I index" has no obviously right
        reading, and guessing one would index something nobody asked for.
        """
        if not isinstance(data, dict):
            return data
        raw: dict[str, Any] = dict(data)  # pyright: ignore[reportUnknownArgumentType]
        if "workspace" in raw:
            if "workspaces" in raw:
                raise ValueError(
                    "use either `workspace:` for one or `workspaces:` for several, not both"
                )
            raw["workspaces"] = [raw.pop("workspace")]
        return raw

    @model_validator(mode="after")
    def _workspaces_are_distinct(self) -> WorkspaceConfig:
        names = [w.name for w in self.workspaces]
        dupes = {name for name in names if names.count(name) > 1}
        if dupes:
            raise ValueError(
                f"duplicate workspace names {sorted(dupes)}; the name keys the "
                "collection and the manifest, so two workspaces sharing one would "
                "overwrite each other"
            )
        self._dataset_paths = {w.eval.dataset for w in self.workspaces if w.eval} | {
            self.eval.dataset
        }
        if len(self.workspaces) > 1 and self.state_dir is None:
            raise ValueError(
                "several workspaces need `state_dir:` — each keeps its own manifest, "
                "and STATE_DB names a single file. Set it to a directory; the "
                "manifests are named after the workspaces."
            )
        return self

    @property
    def workspace(self) -> WorkspaceSection:
        """The one workspace this config describes.

        Raises where there is a choice to make, rather than picking. Everything
        downstream of `select()` sees a config with exactly one workspace, so
        this is the accessor the whole indexing path uses and it is correct for
        all of them.
        """
        if len(self.workspaces) != 1:
            raise WorkspaceChoiceError(None, self.workspace_names)
        return self.workspaces[0]

    @property
    def workspace_names(self) -> list[str]:
        return [w.name for w in self.workspaces]

    def select(self, name: str | None = None) -> WorkspaceConfig:
        """This config as the named workspace sees it.

        Returns a config holding exactly one workspace, with that workspace's
        overrides folded into the shared sections. The point is that nothing
        downstream has to know overrides exist: the indexer, the walker and the
        watcher each receive what looks like the single-workspace config they
        have always received.

        `name` may be omitted when there is only one, so a single-workspace
        config needs no `--workspace` anywhere.
        """
        if name is None:
            if len(self.workspaces) != 1:
                raise WorkspaceChoiceError(None, self.workspace_names)
            chosen = self.workspaces[0]
        else:
            found = next((w for w in self.workspaces if w.name == name), None)
            if found is None:
                raise WorkspaceChoiceError(name, self.workspace_names)
            chosen = found
        narrowed = self.model_copy(
            update={"workspaces": [chosen], "eval": chosen.eval or self.eval}
        )
        # Carried across explicitly. `model_copy` keeps private state, but the
        # set has to outlive the narrowing either way: after selecting one
        # workspace the others are gone, and their datasets must still never be
        # indexed.
        narrowed._dataset_paths = set(self._dataset_paths)
        return narrowed

    def manifest_path(self, state_db: Path) -> Path:
        """Where this workspace's manifest lives.

        `state_db` is the STATE_DB setting, used unchanged when no `state_dir`
        is configured -- so a single workspace keeps the exact file it already
        has and nothing orphans. With `state_dir` set, each workspace gets its
        own file named after it, because the manifest has no workspace column
        and two workspaces sharing one would collide on any root label they
        happen to share.
        """
        if self.state_dir is None:
            return state_db
        return self.state_dir.expanduser() / f"{self.workspace.name}.sqlite3"

    @property
    def all_excludes(self) -> list[str]:
        return [*HARDCODED_EXCLUDES, *self.index.exclude]

    @property
    def excluded_paths(self) -> set[Path]:
        """Specific files that must never be indexed, whatever the patterns say.

        The eval dataset contains the query text of every case, which makes it a
        perfect lexical *and* semantic match for its own queries. Indexing it
        does not merely add noise: it puts the dataset at the top of its own
        results and corrupts every measurement taken afterwards.

        Derived rather than hardcoded because the path is configurable, but not
        user-overridable for the same reason `logs/` is not: it is a correctness
        rule, not a preference.

        Every workspace's dataset, not only the selected one. They cost
        nothing to exclude and a dataset sitting inside another workspace's
        tree is exactly the case a narrower rule would miss.
        """
        return {path.expanduser().resolve() for path in self._dataset_paths | {self.eval.dataset}}

    def root_containing(self, path: Path) -> RootConfig | None:
        """The configured root that holds `path`, or None if no root does.

        Longest match wins, so a root nested inside another is attributed to
        the more specific one. Lives here rather than in the watcher because
        two callers now need the same answer -- the debouncer deciding which
        root to reindex, and the watch filter deciding whether to care at all
        -- and two copies of a prefix comparison is two chances to disagree
        about which root owns a file.
        """
        best: tuple[int, RootConfig] | None = None
        for root in self.workspace.roots:
            base = root.path.expanduser().resolve()
            if path == base or base in path.parents:
                depth = len(base.parts)
                if best is None or depth > best[0]:
                    best = (depth, root)
        return best[1] if best else None

    def root_by_label(self, label: str) -> RootConfig:
        for root in self.workspace.roots:
            if root.resolved_label == label:
                return root
        known = ", ".join(r.resolved_label for r in self.workspace.roots)
        raise KeyError(f"no root labelled {label!r}; known roots: {known}")
