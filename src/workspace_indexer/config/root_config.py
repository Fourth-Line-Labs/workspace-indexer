"""One configured directory to index."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator, model_validator

from workspace_indexer.config.strict import Strict


class RootConfig(Strict):
    path: Path
    label: str | None = None
    recurse_into_children: bool = False

    @field_validator("path")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("label")
    @classmethod
    def _single_component(cls, v: str | None) -> str | None:
        """A label identifies one root inside a recorded path.

        `LexicalRetriever._display` prefixes a hit with it and `PublicPathGuard`
        matches the allowlist against that form, so a label containing a
        separator makes one root's prefix a prefix of another's: labels
        `public` and `public/private` are distinct, pass every uniqueness
        check, and allowlisting the first publishes the second's
        `public/private/secret.py` verbatim. Rejected here, where there is one
        place to do it, rather than encoded at each point that renders a path.
        """
        if v is not None and ("/" in v or "\\" in v):
            raise ValueError(
                f"root label {v!r} contains a path separator. A label names one root in a "
                "recorded path, so it has to be a single component."
            )
        return v

    @model_validator(mode="after")
    def _label_is_usable(self) -> RootConfig:
        """A root has to have a name, and some paths do not supply one.

        `resolved_label` falls back to `path.name`, which pathlib gives as the
        empty string for `.`, `""` and `/`. An empty label is unique against
        every other label, so it passes every duplicate check -- and it is
        also the prefix that matches every recorded path, so allowlisting such
        a root in a multi-root eval publishes all of them. The same hole the
        separator check above closes, reached by a different spelling.
        """
        if not self.resolved_label:
            raise ValueError(
                f"root {str(self.path)!r} has no usable name, so nothing can identify it in a "
                "recorded path. Give it an explicit `label:`."
            )
        return self

    @property
    def resolved_label(self) -> str:
        return self.label or self.path.name
