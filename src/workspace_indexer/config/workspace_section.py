"""The `workspace:` block."""

from __future__ import annotations

from pydantic import Field, model_validator

from workspace_indexer.config.embedding_section import EmbeddingSection
from workspace_indexer.config.eval_section import EvalSection
from workspace_indexer.config.root_config import RootConfig
from workspace_indexer.config.strict import Strict


class WorkspaceSection(Strict):
    """One index: a name, what it covers, and anything it does differently.

    Only the three things that genuinely vary per workspace live here. The
    excludes, the chunking, the search settings and the rest are shared,
    because in practice they are identical across workspaces on one machine and
    duplicating them is how they drift apart.

    `eval` and `embedding` are None when the workspace uses the shared
    settings, which is the ordinary case -- see `WorkspaceConfig.select`, which
    folds an override in so that nothing downstream has to know an override was
    possible.
    """

    name: str
    roots: list[RootConfig] = Field(min_length=1)
    eval: EvalSection | None = None
    embedding: EmbeddingSection | None = None

    @model_validator(mode="after")
    def _unique_labels(self) -> WorkspaceSection:
        labels = [r.resolved_label for r in self.roots]
        dupes = {label for label in labels if labels.count(label) > 1}
        if dupes:
            raise ValueError(
                f"duplicate root labels {sorted(dupes)}; labels key the manifest and the "
                "payload filter, so they must be unique — set `label:` explicitly"
            )
        return self
