"""Keeping paths that are nobody else's business out of a public repository.

`evals/` is committed. The corpus an eval runs over is a whole workspace, and a
retriever broad enough to match across the tree will put whatever is in that
workspace into a recorded run and from there into public git history.

An allowlist, not a denylist, and the reason is a leak rather than a
preference. The first version of this named the roots to *hide*; it was
configured with the two known client trees, the records were scanned for those
two names, and the scan came back clean -- because a denylist scan can only
find what it was already told about. Two other private trees in the same
workspace, a named employer and a directory of job applications, were recorded
and pushed. Naming what may be published inverts the failure: a tree nobody
thought about is redacted by default instead of published by default.

Redaction rather than refusal, because refusal does not work. A lexical pattern
matches outside the allowlist on most runs, so a guard that rejected such a
record would reject every record and the measurement could not be taken at all.
Nothing the eval reports depends on the identity of those files: no expectation
names one, so recall and MRR are unchanged, and breadth is a count.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from workspace_indexer.evaluation.eval_record import EvalRecord
from workspace_indexer.evaluation.eval_result import EvalResult
from workspace_indexer.evaluation.private_path_error import PrivatePathError
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.evaluation")

# Colon-separated absolute paths, like PATH. Read from the environment rather
# than from workspace.yaml so the layout of a private tree never enters a
# committed file.
ENV_VAR = "EVAL_PUBLIC_ROOTS"

# The namespace redacted paths land in. Never compared against the allowlist --
# see `apply` -- so a real directory of the same name cannot turn every
# redaction into a reported leak.
REDACTED_PREFIX = "redacted/"


class PublicPathGuard:
    """Replaces every recorded path outside the allowlist with a digest."""

    def __init__(
        self,
        public_roots: list[Path],
        workspace_roots: list[Path],
        *,
        root_qualified_paths: bool = False,
    ) -> None:
        # Resolved on both sides. The rest of the codebase resolves roots, and
        # a guard comparing an unresolved alias against a resolved spelling --
        # symlinked /home against /Users, /var against /private/var -- would
        # match nothing and silently redact nothing. A privacy control whose
        # misconfiguration degrades to zero redaction has to canonicalise.
        self._workspace_roots = [root.expanduser().resolve() for root in workspace_roots]
        self._roots = [root.expanduser().resolve() for root in public_roots]
        # Whether a recorded path says which root it came from. Only a
        # multi-root run records one: `LexicalRetriever._display` prefixes the
        # root's name there, while `SearchRetriever` returns `hit.rel_path`,
        # which the walker computes relative to the owning root with no name.
        # Defaults to False because the failure direction of a wrong answer
        # here is a leak: a caller that forgets gets more redaction, never
        # less.
        self._qualified = root_qualified_paths
        names = [root.name for root in self._workspace_roots]
        # Two roots with the same basename render identically even when the
        # name is recorded, so the prefix no longer identifies one root.
        self._names_unique = len(set(names)) == len(names)
        self._prefixes: list[str] = []
        for public in self._roots:
            self._prefixes.append(str(public))
            relative = self._relative_forms(public)
            if not relative and self._under_a_workspace_root(public):
                # Under a root, yet with no form that can match: the recorded
                # paths are root-ambiguous, so nothing relative is matched at
                # all and this root's contents redact with everything else.
                # Accepting the bare relative form instead would publish any
                # path from a *different* root that happened to reproduce this
                # root's spelling, which is the fail-open again.
                log.warning(
                    "eval.root_ambiguous_recorded_paths",
                    detail=(
                        "more than one workspace root and the recorded paths do not identify "
                        "which one each came from, so no path can be shown to lie under the "
                        "allowlist and every one is redacted"
                    ),
                )
            elif not relative:
                # Cannot match a root-relative recorded path, which is the form
                # both retrieval surfaces actually emit -- so a root configured
                # this way protects nothing. Said out loud rather than left to
                # look like a clean run.
                log.warning(
                    "eval.public_root_outside_workspace",
                    detail=(
                        "configured public root lies under no workspace root, so it cannot "
                        "match the root-relative paths a run records"
                    ),
                )
            self._prefixes.extend(relative)

    def _under_a_workspace_root(self, public: Path) -> bool:
        return any(
            public == workspace or workspace in public.parents
            for workspace in self._workspace_roots
        )

    def _relative_forms(self, public: Path) -> list[str]:
        """How this root appears in a root-relative recorded path.

        Root-aware, because the recorded form depends on how many workspace
        roots there are. With one root, every recorded path is relative to it,
        and a public root equal to it yields `""` -- everything recorded is
        then public. Returned as `""` rather than `Path('.')`, whose string
        form matches no recorded path and which is how the denylist version of
        this silently allowlisted nothing.

        With more than one root, `""` must *not* be produced: `is_public`
        treats it as matching every path, so allowlisting one root would
        publish every other root with it -- the exact fail-open the allowlist
        exists to close, and one a single-root test cannot reach.

        A relative form is produced there only when it identifies exactly one
        root: the run has to record the root's name (`root_qualified_paths`),
        and the basenames have to be distinct. Otherwise nothing is returned,
        because a prefix that two roots can both produce publishes the one that
        was not allowlisted. Narrow -- it needs a directory inside a private
        root named after a public one -- but narrow fail-open is still
        fail-open, and this control is the one that already leaked once.
        """
        forms: list[str] = []
        if len(self._workspace_roots) == 1:
            workspace = self._workspace_roots[0]
            if public == workspace:
                forms.append("")
            elif workspace in public.parents:
                forms.append(public.relative_to(workspace).as_posix())
            return forms
        if not (self._qualified and self._names_unique):
            return forms
        for workspace in self._workspace_roots:
            if public == workspace:
                forms.append(workspace.name)
            elif workspace in public.parents:
                relative = public.relative_to(workspace).as_posix()
                forms.append(f"{workspace.name}/{relative}")
        return forms

    @classmethod
    def parse(
        cls,
        raw: str,
        workspace_roots: list[Path],
        *,
        root_qualified_paths: bool = False,
    ) -> PublicPathGuard:
        """From a colon-separated list, like PATH."""
        roots = [Path(part) for part in raw.split(os.pathsep) if part.strip()]
        return cls(roots, workspace_roots, root_qualified_paths=root_qualified_paths)

    @classmethod
    def from_env(
        cls, workspace_roots: list[Path], *, root_qualified_paths: bool = False
    ) -> PublicPathGuard:
        """Straight from the process environment.

        The CLI reads the value through `Settings` instead, because the
        configured list lives in `.env`, which pydantic-settings loads into the
        settings object and *not* into `os.environ` -- a guard built this way
        inside the CLI would find nothing and redact nothing, silently.
        """
        return cls.parse(
            os.environ.get(ENV_VAR, ""),
            workspace_roots,
            root_qualified_paths=root_qualified_paths,
        )

    @property
    def active(self) -> bool:
        """Unset means no redaction, so a workspace with nothing to hide is not
        forced to enumerate itself. The CLI says so on every recorded run: an
        inactive privacy control that announces itself is the one thing that
        would have caught the leak this class was rewritten after."""
        return bool(self._prefixes)

    def is_public(self, path: str) -> bool:
        if not self.active:
            return True
        return any(
            # `""` is produced only for a single-root workspace, where every
            # recorded path is relative to that one root. `_relative_forms`
            # never yields it for a multi-root workspace, where it would
            # publish every root at once.
            prefix == "" or path == prefix or path.startswith(prefix + "/")
            # Segment-aware, so a sibling sharing a name prefix is not treated
            # as inside: `work-notes` is not under `work`.
            for prefix in self._prefixes
        )

    def redact(self, path: str) -> str:
        # Stable across runs, so the same file is recognisably the same hit
        # twice, and one-way, so the name cannot be recovered. 16 hex rather
        # than 8: collisions are unlikely either way, but there is no reason to
        # spend the identity. Stability is per textual form and per machine,
        # which costs nothing -- `comparable_to` keys on a config hash that
        # includes the resolved roots, so records from two layouts are never
        # compared in the first place.
        digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
        return f"{REDACTED_PREFIX}{digest}"

    def apply(self, record: EvalRecord) -> EvalRecord:
        """A redacted copy of the record, verified before it is returned."""
        if not self.active:
            return record
        leaked: list[str] = []
        results: list[EvalResult] = []
        for result in record.results:
            rewritten: list[str] = []
            for path in result.found:
                if self.is_public(path):
                    rewritten.append(path)
                    continue
                replacement = self.redact(path)
                # Verified against what was private *before* redaction rather
                # than by re-testing the output: the redacted form is a path
                # too, and re-testing it would report every redaction as a leak
                # for anyone whose allowlist happens to exclude `redacted/`.
                if replacement == path:
                    leaked.append(path)
                rewritten.append(replacement)
            results.append(result.model_copy(update={"found": rewritten}))
        if leaked:
            raise PrivatePathError(
                f"{len(leaked)} path(s) outside the allowlist survived redaction; "
                "refusing to write the record"
            )
        return record.model_copy(update={"results": results})
