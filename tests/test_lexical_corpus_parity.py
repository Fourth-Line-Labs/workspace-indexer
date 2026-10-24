"""The corpus the lexical tools search, against the one the index holds.

Every claim in `docs/grep-comparison.md` rests on the three tools reading the
same files, and that has now been false three times: the harness's own records
and dataset, lockfiles, and hidden files that ripgrep skips by default. Each
was a rule `Walker` enforces in code rather than as a pattern, so no reading of
the exclusion list could have found it, and each was caught by a reviewer
rather than by this suite.

Enumerating the rules one at a time is what produced that run of three. These
tests assert the property instead: whatever the walker decides, the tools agree
with it. A fourth rule added to `Walker` fails here rather than quietly
inflating a published number.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from workspace_indexer.cli import build_lexical_retriever
from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.discovery import Walker
from workspace_indexer.evaluation import GREP, RIPGREP, EvalCase, LexicalTool

# Matches any line with a character on it, so the "query" selects the whole
# corpus and the comparison is between two file sets rather than two rankings.
EVERYTHING = "."

# CI installs ripgrep precisely so this does not skip there -- one of the three
# confounds this file exists to catch was a ripgrep default, and a guard that
# quietly skips half of itself on the only machine that runs it unattended is
# not a guard. The marker is for a contributor's laptop, where a skip that says
# why beats a failure that looks like a bug in the code.
needs_ripgrep = pytest.mark.skipif(
    shutil.which("rg") is None, reason="ripgrep is not installed; CI installs it"
)

# Marked per parameter rather than per test, so a laptop without ripgrep still
# runs the grep half instead of losing both.
TOOLS = [pytest.param(GREP, id="grep"), pytest.param(RIPGREP, marks=needs_ripgrep, id="ripgrep")]
_IDENTITY = ("-c", "user.email=t@example.com", "-c", "user.name=T")


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """One of everything the walker treats specially, and no `.gitignore`.

    Deliberately ungitignored: grep does not read `.gitignore` and ripgrep
    does, and that difference is real, deliberate and reported rather than
    flattened. Keeping it out of this fixture is what lets the two tools be
    compared to each other as well as to the walker.
    """
    root = tmp_path / "corpus"
    _write(root / "plain" / "notes.md", "ordinary content\n")
    # Hidden, and a primary target: ripgrep skips these by default.
    _write(root / ".claude" / "commands" / "deploy.md", "deploy instructions\n")
    # Rejected by `is_lockfile`, not by any pattern.
    _write(root / "poetry.lock", "generated content\n")
    # Over `max_file_bytes` below.
    _write(root / "huge.py", "x = 1\n" * 4000)
    # Excluded by a trailing-slash pattern, which is how a user writes a
    # directory in `index.exclude`.
    _write(root / "vendor" / "dep.py", "vendored\n")
    # A linked worktree holds a second copy of a repository already indexed.
    repo = root / "product"
    _write(repo / "auth.py", "def authenticate(): ...\n")
    subprocess.run(["git", *_IDENTITY, "init", "-q"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", *_IDENTITY, "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", *_IDENTITY, "commit", "-qm", "init"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", *_IDENTITY, "worktree", "add", "-q", str(root / "product-wt"), "-b", "wt"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return root


def _config(root: Path) -> WorkspaceConfig:
    payload: dict[str, Any] = {
        "workspace": {
            "name": "test",
            "roots": [{"path": str(root), "recurse_into_children": True}],
        },
        "index": {"exclude": ["vendor/"], "max_file_bytes": 5_000},
    }
    return WorkspaceConfig.model_validate(payload)


def _indexed(config: WorkspaceConfig) -> set[str]:
    return {candidate.rel_path for candidate in Walker(config).walk()}


async def _searched(tool: LexicalTool, config: WorkspaceConfig) -> set[str]:
    """Through the CLI's own assembly, not a copy of it.

    Rebuilding the exclusion set here would test this file against itself: the
    thing that has been wrong three times is which exclusions reach the tools,
    and that decision lives in `_lexical_retriever`.
    """
    case = EvalCase(query="q", expect=["plain/notes.md"], grep_pattern=EVERYTHING)
    retriever = build_lexical_retriever(
        config,
        tool=tool.name,
        cases=[case],
        dataset=config.eval.dataset,
    )
    # Far above the corpus size, so the cutoff cannot hide a disagreement.
    return set(await retriever.retrieve("q", 10_000))


@pytest.mark.parametrize("tool", TOOLS)
async def test_a_lexical_tool_searches_exactly_what_the_index_holds(
    tool: LexicalTool, corpus: Path
) -> None:
    """Equality, not containment. Over-inclusion is how the lockfiles and the
    answer key got scored; under-inclusion is how ripgrep scored a hidden file
    as absent while grep and the index both found it. Both directions are
    failures of the same claim."""
    config = _config(corpus)

    assert await _searched(tool, config) == _indexed(config)


@needs_ripgrep
async def test_grep_and_ripgrep_read_the_same_files(corpus: Path) -> None:
    """The document'"'"'s stated evidence that the corpus is shared is that the two
    binaries score alike. They scored alike while reading different corpora:
    grep returned ten hits under hidden directories and ripgrep none, a 109
    file difference in total breadth, and the agreement in MRR hid it."""
    config = _config(corpus)

    assert await _searched(GREP, config) == await _searched(RIPGREP, config)


@needs_ripgrep
async def test_a_hidden_file_is_searched_by_both(corpus: Path) -> None:
    """`.claude/` is a primary target rather than something to skip, and
    ripgrep skips hidden paths unless told otherwise."""
    config = _config(corpus)
    hidden = ".claude/commands/deploy.md"

    assert hidden in _indexed(config)
    assert hidden in await _searched(RIPGREP, config)
    assert hidden in await _searched(GREP, config)


@pytest.mark.parametrize("tool", TOOLS)
async def test_a_file_over_the_size_limit_is_not_a_candidate(
    tool: LexicalTool, corpus: Path
) -> None:
    """`max_file_bytes` is a number compared against a stat, and grep has no
    size flag at all -- so this one cannot be pushed into the command."""
    config = _config(corpus)

    assert "huge.py" not in _indexed(config)
    assert "huge.py" not in await _searched(tool, config)


@pytest.mark.parametrize("tool", TOOLS)
async def test_a_linked_worktree_is_not_a_candidate(tool: LexicalTool, corpus: Path) -> None:
    """A linked worktree is a second copy of a repository already indexed, so
    every file in it would be counted twice and rank against itself."""
    config = _config(corpus)

    assert not [path for path in _indexed(config) if path.startswith("product-wt/")]
    assert not [path for path in await _searched(tool, config) if path.startswith("product-wt/")]
