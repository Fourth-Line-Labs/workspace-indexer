"""Scoring grep and ripgrep on the same dataset as the index.

Real files and real binaries rather than a stubbed subprocess: what is being
measured is how these tools actually behave -- which files they walk, how they
report counts, what they do with no match -- and a fake would only confirm the
assumptions this comparison exists to check.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from workspace_indexer.evaluation.lexical_retriever import LexicalRetriever
from workspace_indexer.evaluation.lexical_tool import (
    GREP,
    RIPGREP,
    LexicalTool,
    translate_excludes,
)
from workspace_indexer.evaluation.missing_pattern_error import MissingPatternError
from workspace_indexer.evaluation.tool_missing_error import ToolMissingError

requires_rg = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """One term, three files, deliberately uneven counts -- the ranking is the
    only thing standing between an unordered match set and an MRR."""
    (tmp_path / "many.py").write_text("needle\nneedle\nneedle\n", encoding="utf-8")
    (tmp_path / "few.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "none.py").write_text("haystack\n", encoding="utf-8")
    vendored = tmp_path / "node_modules"
    vendored.mkdir()
    (vendored / "dep.js").write_text("needle\nneedle\nneedle\nneedle\n", encoding="utf-8")
    return tmp_path


async def test_files_rank_by_match_count(corpus: Path) -> None:
    """grep returns an unordered set, so an order has to be imposed before a
    rank means anything. Count descending is the most favourable honest one."""
    retriever = LexicalRetriever(GREP, [corpus], {"q": "needle"}, ("node_modules",))

    assert await retriever.retrieve("q", 10) == ["many.py", "few.py"]


async def test_a_file_without_the_term_is_not_a_candidate(corpus: Path) -> None:
    """GNU grep's -c prints every file it walked, matched or not. A zero kept
    by accident counts towards breadth and would inflate the flood rate for
    grep alone, while ripgrep -- which prints only matches -- stayed honest."""
    retriever = LexicalRetriever(GREP, [corpus], {"q": "needle"}, ("node_modules",))

    assert "none.py" not in await retriever.retrieve("q", 10)


async def test_breadth_counts_every_match_not_only_the_returned_ones(corpus: Path) -> None:
    """The flood is the point. Truncating to the cutoff and reporting that
    number would say two files matched when the agent's problem is that two
    hundred did."""
    retriever = LexicalRetriever(GREP, [corpus], {"q": "needle"}, ("node_modules",))

    returned = await retriever.retrieve("q", 1)

    assert len(returned) == 1
    assert retriever.last_breadth == 2


async def test_an_excluded_directory_is_not_searched(corpus: Path) -> None:
    """Vendored code the index never saw. Without this the tools answer a
    question about a different corpus, and node_modules would win on count."""
    retriever = LexicalRetriever(GREP, [corpus], {"q": "needle"}, ("node_modules",))

    assert await retriever.retrieve("q", 10) == ["many.py", "few.py"]

    unfiltered = LexicalRetriever(GREP, [corpus], {"q": "needle"}, ())
    assert any("node_modules" in path for path in await unfiltered.retrieve("q", 10))


async def test_no_match_is_an_empty_result_not_a_failure(corpus: Path) -> None:
    """grep exits 1 when nothing matched. Treating a non-zero exit as an error
    would turn the single most interesting lexical outcome into a crash."""
    retriever = LexicalRetriever(GREP, [corpus], {"q": "absent-token"}, ())

    assert await retriever.retrieve("q", 10) == []
    assert retriever.last_breadth == 0


async def test_a_case_without_a_pattern_is_an_error(corpus: Path) -> None:
    """The strawman guard. Falling back to the raw English question would match
    nothing and record a plausible-looking row of zeroes."""
    retriever = LexicalRetriever(GREP, [corpus], {})

    with pytest.raises(MissingPatternError):
        await retriever.retrieve("how does reindexing decide what to skip", 10)


async def test_paths_are_relative_to_the_root(corpus: Path) -> None:
    """Uniform with the index, so one set of expectations scores both -- and so
    an absolute home directory never reaches a committed record."""
    retriever = LexicalRetriever(GREP, [corpus], {"q": "needle"}, ("node_modules",))

    assert all(not Path(path).is_absolute() for path in await retriever.retrieve("q", 10))


@requires_rg
async def test_ripgrep_ranks_the_same_corpus_the_same_way(corpus: Path) -> None:
    """Different binary, different count syntax, same answer. A divergence here
    is a parsing bug rather than a finding about either tool."""
    retriever = LexicalRetriever(RIPGREP, [corpus], {"q": "needle"}, ("node_modules",))

    assert await retriever.retrieve("q", 10) == ["many.py", "few.py"]


def test_a_missing_binary_is_raised_not_scored() -> None:
    """A tool that is not installed has not been beaten. Recording it as zero
    recall would publish a number that defames it."""
    absent = LexicalTool(
        name="nope",
        binary="definitely-not-a-binary",
        flags=(),
        exclude_dir_template="",
        exclude_file_template="",
    )

    with pytest.raises(ToolMissingError):
        absent.argv("x", Path("/tmp"))


def test_exclusions_translate_to_directories_and_file_globs() -> None:
    """Both shapes have to reach the tools. `logs/**` is rooted rather than
    `**/`-prefixed and was previously untranslatable, which is how the harness
    ended up searching its own logs and records -- the files that quote every
    eval query verbatim."""
    dirs, files = translate_excludes(
        [
            "**/node_modules/**",
            "logs/**",
            "**/logs/**",
            "evals/**",
            "docs/eval-baselines.md",
            "**/*.sqlite3",
        ]
    )

    assert dirs == ("node_modules", "logs", "evals")
    assert files == ("eval-baselines.md", "*.sqlite3")


def test_a_directory_name_with_glob_characters_is_refused() -> None:
    """`--exclude-dir` is a glob and ripgrep's `--glob` is another dialect, so
    a name carrying `[` would exclude the wrong directories in one and make
    ripgrep exit 2 in the other. There is no escape meaning the same thing in
    both, and silently excluding something else is how a corpus diverges."""
    with pytest.raises(ValueError, match="glob characters"):
        translate_excludes(["**/bu[ld]/**"])


async def test_an_excluded_file_is_not_searched(corpus: Path) -> None:
    """The eval dataset quotes every query and every pattern, so a tool allowed
    to read it finds the answer key. It is excluded by name, not by directory."""
    (corpus / "answers.yaml").write_text("needle\nneedle\nneedle\nneedle\n", encoding="utf-8")
    retriever = LexicalRetriever(GREP, [corpus], {"q": "needle"}, ("node_modules",), ())
    assert "answers.yaml" in await retriever.retrieve("q", 10)

    filtered = LexicalRetriever(
        GREP, [corpus], {"q": "needle"}, ("node_modules",), ("answers.yaml",)
    )
    assert "answers.yaml" not in await filtered.retrieve("q", 10)


async def test_two_roots_holding_the_same_relative_path_stay_distinct(tmp_path: Path) -> None:
    """The multi-repo layout the example config encourages. Keying on the
    relative form alone summed two different files into one entry, ranked them
    as one and undercounted breadth."""
    for name in ("alpha", "beta"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "main.py").write_text("needle\n", encoding="utf-8")

    retriever = LexicalRetriever(GREP, [tmp_path / "alpha", tmp_path / "beta"], {"q": "needle"})

    found = await retriever.retrieve("q", 10)

    assert retriever.last_breadth == 2
    assert len(set(found)) == 2


async def test_a_tool_error_is_raised_not_scored_as_an_empty_result(corpus: Path) -> None:
    """Exit 1 is "no matches"; anything above it is a broken pattern or an
    unreadable root. Scoring that as an honest zero would hide a bug in the
    experiment itself behind a plausible number."""
    broken = LexicalTool(
        name="broken",
        binary="grep",
        flags=("-r", "-c", "--definitely-not-a-flag"),
        exclude_dir_template="--exclude-dir={name}",
        exclude_file_template="--exclude={name}",
    )
    retriever = LexicalRetriever(broken, [corpus], {"q": "needle"})

    with pytest.raises(RuntimeError, match="exited"):
        await retriever.retrieve("q", 10)


async def test_a_tool_that_never_returns_is_killed(
    corpus: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The timeout path. A hung binary would otherwise stall the harness
    indefinitely with no record of why."""
    import workspace_indexer.evaluation.lexical_retriever as module

    # `sh -c` rather than `sleep`: argv always ends with `-e <pattern> <root>`,
    # which sleep rejects as an operand and exits 1 -- indistinguishable from
    # "no matches", so the test passed without ever reaching the timeout.
    sleeper = LexicalTool(
        name="sleeper",
        binary="sh",
        flags=("-c", "sleep 30"),
        exclude_dir_template="--ignored={name}",
        exclude_file_template="--ignored={name}",
    )
    monkeypatch.setattr(module, "_TIMEOUT_S", 0.3)
    retriever = LexicalRetriever(sleeper, [corpus], {"q": "needle"})

    with pytest.raises(TimeoutError):
        await retriever.retrieve("q", 10)
