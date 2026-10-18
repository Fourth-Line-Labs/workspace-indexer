"""Retrieval-quality scoring.

The arithmetic matters: a harness that reports the wrong number is worse than
none, because decisions get made on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.evaluation import EvalCase, EvalReport, EvalResult, load_cases


def _result(expected: list[str], found: list[str]) -> EvalResult:
    rank = next((i for i, path in enumerate(found, 1) if any(w in path for w in expected)), None)
    return EvalResult(query="q", expected=expected, found=found, first_hit_rank=rank)


def test_recall_is_the_fraction_of_expected_files_found() -> None:
    result = _result(["a.py", "b.py"], ["src/a.py", "src/z.py"])
    assert result.recall == pytest.approx(0.5)


def test_recall_of_a_complete_miss_is_zero() -> None:
    assert _result(["a.py"], ["src/z.py"]).recall == 0.0


def test_expected_paths_match_as_substrings() -> None:
    """So a case survives a file moving within a directory without the dataset
    needing a rewrite."""
    assert _result(["widget.py"], ["src/pkg/widget.py"]).recall == 1.0


def test_reciprocal_rank_rewards_ranking_first() -> None:
    """Recall alone cannot see the difference between rank 1 and rank 10."""
    assert _result(["a.py"], ["a.py", "b.py"]).reciprocal_rank == 1.0
    assert _result(["a.py"], ["b.py", "a.py"]).reciprocal_rank == pytest.approx(0.5)


def test_reciprocal_rank_of_a_miss_is_zero() -> None:
    assert _result(["a.py"], ["z.py"]).reciprocal_rank == 0.0


def test_report_averages_across_cases() -> None:
    report = EvalReport(
        label="t",
        limit=10,
        results=[_result(["a.py"], ["a.py"]), _result(["b.py"], ["z.py"])],
    )
    assert report.recall_at_k == pytest.approx(0.5)
    assert report.mrr_at_k == pytest.approx(0.5)


def test_empty_report_does_not_divide_by_zero() -> None:
    report = EvalReport(label="t", limit=10, results=[])
    assert report.recall_at_k == 0.0
    assert report.mrr_at_k == 0.0


def test_misses_are_listed_for_inspection() -> None:
    """A score without the failing queries is not actionable."""
    report = EvalReport(
        label="t",
        limit=10,
        results=[_result(["a.py"], ["a.py"]), _result(["b.py"], ["z.py"])],
    )
    assert [m.expected for m in report.misses] == [["b.py"]]


def test_loading_a_dataset(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text(
        "- query: how does the watcher pick a backend\n"
        "  expect: [watcher.py]\n"
        "- query: where is the IDF modifier set\n"
        "  expect: [qdrant_store.py]\n"
        "  note: the easiest thing here to get wrong\n",
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert len(cases) == 2
    assert cases[0].expect == ["watcher.py"]
    assert cases[1].note


def test_missing_dataset_explains_what_to_write(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="real queries"):
        load_cases(tmp_path / "nope.yaml")


def test_empty_dataset_is_not_an_error(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("\n", encoding="utf-8")
    assert load_cases(path) == []


def test_malformed_case_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "eval.yaml"
    path.write_text("- query: missing the expect key\n", encoding="utf-8")
    with pytest.raises(Exception, match="expect"):
        load_cases(path)


def test_case_model_requires_a_query() -> None:
    with pytest.raises(Exception, match="query"):
        EvalCase.model_validate({"expect": ["a.py"]})


def test_expect_matching_is_case_insensitive() -> None:
    """A dataset saying CONTRIBUTING.md must match docs/contributing.md.
    Scoring a successful retrieval as a miss makes the measurement lie."""
    assert _result(["CONTRIBUTING.md"], ["ralph/docs/contributing.md"]).recall == 1.0


def test_first_hit_rank_is_also_case_insensitive() -> None:
    """recall and rank iterate in opposite directions; both have to agree, or a
    case gets full recall and a reciprocal rank of zero."""
    result = _result(["CONTRIBUTING.md"], ["a.md", "docs/CONTRIBUTING.md"])
    assert result.recall == 1.0
    assert result.first_hit_rank == 2


def test_match_direction_is_not_reversible() -> None:
    """The expectation is a substring of the path, never the other way round.
    A (str, list[str]) signature would let the arguments be swapped silently,
    because both orderings typecheck."""
    from workspace_indexer.evaluation.eval_result import path_matches

    assert path_matches("manifest.py", "src/state/manifest.py")
    assert not path_matches("src/state/manifest.py", "manifest.py")


def _breadth(breadths: list[int | None], limit: int = 10) -> EvalReport:
    return EvalReport(
        label="t",
        limit=limit,
        results=[
            EvalResult(query="q", expected=["a.py"], found=["a.py"], breadth=b) for b in breadths
        ],
    )


def test_an_empty_result_is_counted_separately_from_a_miss() -> None:
    """An agent can act on a wrong answer: it reads the file and moves on. It
    cannot act on silence, and recall scores the two the same."""
    assert _breadth([0, 0, 3, 5]).empty_rate == pytest.approx(0.5)


def test_a_flood_is_counted_although_recall_calls_it_a_success() -> None:
    """The failure recall@k cannot see. Four hundred files containing the right
    one scores a perfect recall while leaving the agent a haystack, and the
    ranking that chose which ten it sees is match count, not relevance."""
    report = _breadth([400, 2], limit=10)

    assert report.results[0].recall == 1.0  # recall is blind to it
    assert report.flood_rate == pytest.approx(0.5)


def test_the_flood_threshold_scales_with_the_cutoff() -> None:
    """Thirty files is a flood at -n 10 and an ordinary page at -n 50, so a
    flat number would mean different things at different limits."""
    assert _breadth([40], limit=10).flood_rate == pytest.approx(1.0)
    assert _breadth([40], limit=50).flood_rate == pytest.approx(0.0)


def test_a_retriever_that_reports_no_breadth_is_excluded_not_counted_as_zero() -> None:
    """A dense retriever returns `limit` hits by construction and has no
    breadth. Treating its None as zero would report it as silent on every
    query -- a 100% empty rate for the tool that is never empty."""
    report = _breadth([None, None, 0])

    assert report.empty_rate == pytest.approx(1.0)  # one scored case, and it was empty
    assert report.median_breadth == pytest.approx(0.0)
    assert _breadth([None, None]).empty_rate == pytest.approx(0.0)


async def test_the_harness_records_breadth_only_from_a_retriever_that_reports_it() -> None:
    """Asked after the call through an optional capability, so the protocol
    every retriever implements keeps its contract that `limit` is a cap. A
    retriever that cannot exceed the cutoff is left at None rather than being
    given a number that only restates the cutoff."""
    from workspace_indexer.evaluation import EvalHarness

    class Wide:
        name = "wide"

        def __init__(self) -> None:
            # Starts unset and is assigned inside `retrieve`, exactly as the
            # real retriever does. As a class-level constant the assertion
            # below passed whether the harness read the attribute before or
            # after the await, so it pinned the value and not the ordering --
            # and a regression that moved the read above the call would have
            # broken every real run with this test still green.
            self.last_breadth: int | None = None

        async def retrieve(self, query: str, limit: int) -> list[str]:
            self.last_breadth = 42
            return ["a.py"]

    class Plain:
        name = "plain"

        async def retrieve(self, query: str, limit: int) -> list[str]:
            return ["a.py"]

    cases = [EvalCase(query="q", expect=["a.py"])]

    wide = await EvalHarness(Wide()).run(cases, limit=10)
    plain = await EvalHarness(Plain()).run(cases, limit=10)

    assert wide.results[0].breadth == 42
    assert plain.results[0].breadth is None
