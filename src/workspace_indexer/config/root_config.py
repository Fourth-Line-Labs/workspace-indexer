"""One configured directory to index."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator

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

    @property
    def resolved_label(self) -> str:
        return self.label or self.path.name
