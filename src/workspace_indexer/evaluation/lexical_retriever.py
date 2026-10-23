"""Scoring a command-line search tool on the same dataset as the index.

The comparison this exists for is not "is semantic search better than grep".
On a query naming an identifier, grep is faster, exact and free, and this
harness should say so. The question is narrower: when a query describes what
code *does* rather than what it is called, does the index find the file where
a competent pattern returns nothing or an unusable flood?

Which makes two things load-bearing. The pattern, because a tool handed the
raw English question matches nothing and a dataset of such runs would be a
sham baseline dressed as a comparison -- so every case carries a pattern
written by hand to win, and a case without one is an error rather than a
silent zero. And the corpus, because a lexical tool allowed to search the
harness's own records and dataset finds the queries quoted back at it and
scores them above the workspace. The exclusions come from the index's own
effective set for that reason.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

from workspace_indexer.evaluation.lexical_tool import LexicalTool
from workspace_indexer.evaluation.missing_pattern_error import MissingPatternError
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.evaluation")

# A tool that cannot answer in this long has not answered. Generous: the
# slowest honest run measured here is under two seconds.
_TIMEOUT_S = 60.0


class LexicalRetriever:
    """Runs one pattern per query and ranks the files it matched.

    Ranking is by match count, descending. grep returns an unordered set, so
    some order has to be imposed before MRR means anything, and this is the
    most favourable honest one: a file mentioning the term eight times is a
    better guess than one mentioning it once. Filesystem order would be the
    literal truth and would understate the tool.
    """

    def __init__(
        self,
        tool: LexicalTool,
        roots: list[tuple[str, Path]],
        patterns: Mapping[str, str],
        exclude_dirs: tuple[str, ...] = (),
        exclude_files: tuple[str, ...] = (),
    ) -> None:
        # Labelled, not bare paths. A recorded hit is prefixed with the root it
        # came from when there is more than one, and the basename is not an
        # identifier: `RootConfig.resolved_label` is, and the configuration
        # allows `/x/api` and `/y/api` to carry distinct ones. Prefixing with
        # the basename made two different files the same recorded string --
        # and because `PublicPathGuard.redact` digests that string, the same
        # digest after redaction too.
        labels = [label for label, _ in roots]
        if len(set(labels)) != len(labels):
            # Refused rather than disambiguated here: a label is what the rest
            # of the system already keys roots by -- `config_hash`, the
            # manifest, `--unit` -- so a duplicate is a configuration error to
            # fix once, not a collision for each consumer to invent a rule for.
            raise ValueError(
                "two workspace roots resolve to the same label, so a recorded path "
                "cannot say which one it came from. Give them distinct `label:` values."
            )
        self._tool = tool
        self._roots = roots
        self._patterns = patterns
        self._exclude_dirs = exclude_dirs
        self._exclude_files = exclude_files
        self.name = tool.name
        # How many files matched before the rank cutoff. The harness reads it
        # after each call, because recall@k cannot express either failure that
        # matters here: an empty result and a 400-file flood both leave an
        # agent with nowhere to go, and recall scores the flood as a win when
        # the right file happens to be somewhere inside it.
        self.last_breadth: int | None = None

    async def retrieve(self, query: str, limit: int) -> list[str]:
        pattern = self._patterns.get(query)
        if not pattern:
            raise MissingPatternError(
                f"no grep_pattern for {query!r}. Every case needs one written to "
                "win, or the comparison is against a strawman."
            )

        # Keyed by root as well as path. Two roots can each hold a `src/main.py`
        # -- the multi-repo layout the example config encourages -- and merging
        # them on the relative form alone would rank two distinct files as one
        # and undercount breadth. Where one root nests inside another, the same
        # file arrives under two keys and is counted twice; that is visible in
        # the output rather than silently summed.
        counts: dict[tuple[str, str], int] = {}
        for label, root in self._roots:
            for rel, count in (await self._run(pattern, root)).items():
                key = (label, rel)
                counts[key] = counts.get(key, 0) + count

        self.last_breadth = len(counts)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0][1], item[0][0]))
        log.info(
            "eval.lexical",
            tool=self._tool.name,
            pattern=pattern,
            breadth=len(counts),
            returned=min(len(ranked), limit),
        )
        return [self._display(label, rel) for (label, rel), _ in ranked[:limit]]

    def _display(self, label: str, rel: str) -> str:
        """Relative to the root, as the index reports it -- and prefixed with
        the root's label when there is more than one, so two roots holding the
        same relative path stay distinguishable in the record. The label rather
        than the basename, because two roots can share a basename and the
        prefix has to identify exactly one of them: `PublicPathGuard` matches
        the allowlist against this form, so an ambiguous prefix publishes the
        root that was not allowlisted. Expectations match as substrings, so the
        prefix costs nothing."""
        return rel if len(self._roots) == 1 else f"{label}/{rel}"

    async def _run(self, pattern: str, root: Path) -> dict[str, int]:
        argv = self._tool.argv(pattern, root, self._exclude_dirs, self._exclude_files)
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=_TIMEOUT_S)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise

        # 1 is "no matches", which is a result and not a failure. Anything
        # above that is a broken pattern or an unreadable root, and scoring it
        # as an honest zero would hide a bug in the experiment itself.
        if process.returncode not in (0, 1):
            detail = stderr.decode("utf-8", "replace").strip().splitlines()
            raise RuntimeError(
                f"{self._tool.name} exited {process.returncode} on {pattern!r}: "
                f"{detail[0] if detail else 'no output'}"
            )
        return _counts(stdout.decode("utf-8", "replace"), root)


def _counts(output: str, root: Path) -> dict[str, int]:
    """Parse `path:count` lines, dropping the zeroes and relativising.

    GNU grep's -c reports every file it walked, matched or not, while
    ripgrep's reports only the files that matched. Filtering here rather than
    in the flags is what lets one parser serve both -- and a zero kept by
    accident would count towards breadth, inflating the flood number for grep
    alone.
    """
    counts: dict[str, int] = {}
    for line in output.splitlines():
        path, _, raw = line.rpartition(":")
        if not path or not raw.isdigit():
            continue
        count = int(raw)
        if count:
            counts[_relative(path, root)] = count
    return counts


def _relative(path: str, root: Path) -> str:
    """Paths as the index reports them: relative to the root they were found
    under. Uniform output is what lets one set of expectations score both, and
    it keeps an absolute home directory out of a committed record."""
    try:
        return str(Path(path).relative_to(root))
    except ValueError:
        return path
