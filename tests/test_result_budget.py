"""Packing hits into a token budget without lying about what was cut."""

from __future__ import annotations

from workspace_indexer.mcp import ResultBudget
from workspace_indexer.models import DocumentType, SearchHit


def _hit(index: int, tokens: int, text: str | None = None) -> SearchHit:
    return SearchHit(
        chunk_id=f"id-{index}",
        score=1.0 - index / 100,
        rel_path=f"src/file{index}.py",
        root_label="r",
        doc_type=DocumentType.IMPLEMENTATION,
        start_line=index * 10 + 1,
        end_line=index * 10 + 9,
        source_text=text if text is not None else "x" * (tokens * 4),
        token_count=tokens,
    )


def test_everything_fits_when_the_budget_allows() -> None:
    results, dropped = ResultBudget(1000).pack([_hit(i, 100) for i in range(5)])
    assert len(results) == 5
    assert dropped == 0
    assert all(not r.text_truncated for r in results)


def test_overflow_drops_whole_hits_and_counts_them() -> None:
    """Three complete chunks beat ten fragments, so whole hits go rather than
    every hit being shaved to uselessness."""
    results, dropped = ResultBudget(250, min_chunk_tokens=64).pack([_hit(i, 100) for i in range(5)])
    assert len(results) + dropped == 5
    assert dropped > 0
    # The ones kept are the highest ranked, in order.
    assert [r.rel_path for r in results] == [f"src/file{i}.py" for i in range(len(results))]


def test_a_single_oversized_chunk_is_truncated_and_flagged() -> None:
    """An agent that thinks it read a whole function will confidently describe
    the half it got."""
    results, _ = ResultBudget(100).pack([_hit(0, 5000)])
    assert len(results) == 1
    assert results[0].text_truncated
    assert len(results[0].text) < 5000 * 4
    assert "truncated" in results[0].text


def test_truncation_cuts_on_a_line_boundary() -> None:
    body = "\n".join(f"line {i} of the function body" for i in range(200))
    results, _ = ResultBudget(40).pack([_hit(0, 2000, text=body)])
    kept = results[0].text.split("\n... ")[0]
    assert kept
    # Every retained line is whole, so the result is still readable code.
    assert all(line in body.splitlines() for line in kept.splitlines())


def test_location_is_always_anchored() -> None:
    results, _ = ResultBudget(1000).pack([_hit(3, 10)])
    assert results[0].location == "src/file3.py:31-39"


def test_staleness_survives_packing() -> None:
    """An agent editing from stale text writes a patch that will not apply, so
    this flag must never be dropped in formatting."""
    hit = _hit(0, 10)
    hit.stale = True
    results, _ = ResultBudget(1000).pack([hit])
    assert results[0].stale is True


def test_rerank_score_wins_when_present() -> None:
    """The displayed score should be the one that decided the order."""
    hit = _hit(0, 10)
    hit.rerank_score = 0.42
    results, _ = ResultBudget(1000).pack([hit])
    assert results[0].score == 0.42


def test_missing_token_count_is_estimated_not_treated_as_free() -> None:
    """A zero token_count on every hit would make the budget unbounded."""
    hit = _hit(0, 0, text="y" * 8000)
    results, dropped = ResultBudget(100).pack([hit, _hit(1, 50)])
    assert results[0].text_truncated
    assert dropped == 1


def test_empty_input_is_not_an_error() -> None:
    assert ResultBudget(100).pack([]) == ([], 0)


def test_the_top_hit_survives_a_budget_too_small_to_hold_it() -> None:
    """Returning nothing is indistinguishable from "no matches", and sends the
    agent looking elsewhere for something we actually found."""
    results, dropped = ResultBudget(10, min_chunk_tokens=64).pack([_hit(i, 500) for i in range(3)])
    assert len(results) == 1
    assert results[0].text_truncated
    assert dropped == 2


# ---- locations only (#71) ---------------------------------------------------


def test_omitting_bodies_keeps_the_anchor_and_drops_the_code() -> None:
    """The anchor is the point: `location` and the line range are what the
    agent needs to read the file itself."""
    results, dropped = ResultBudget(1000).pack([_hit(0, 100)], include_text=False)

    assert dropped == 0
    assert results[0].text == ""
    assert results[0].text_omitted is True
    assert results[0].rel_path == "src/file0.py"
    assert results[0].start_line == 1
    assert results[0].end_line == 9


def test_a_withheld_body_is_distinguishable_from_an_empty_chunk() -> None:
    """Both have `text == ""`. Without the flag an agent reads the first as the
    second and concludes the file has no content."""
    withheld, _ = ResultBudget(1000).pack([_hit(0, 100)], include_text=False)
    genuinely_empty, _ = ResultBudget(1000).pack([_hit(0, 1, text="")])

    assert withheld[0].text == genuinely_empty[0].text == ""
    assert withheld[0].text_omitted is True
    assert genuinely_empty[0].text_omitted is False


def test_dropping_bodies_fits_far_more_hits_in_the_same_budget() -> None:
    """The whole reason the mode exists. Measured on real tool calls, every
    response that overflowed did so with bodies included."""
    hits = [_hit(i, 500) for i in range(50)]

    with_text, dropped_with_text = ResultBudget(6000).pack(hits)
    without_text, dropped_without = ResultBudget(6000).pack(hits, include_text=False)

    assert dropped_with_text > 0, "the text case must overflow or this compares nothing"
    assert dropped_without == 0
    assert len(without_text) == 50
    assert len(without_text) > len(with_text) * 3


def test_body_size_stops_mattering_once_bodies_are_omitted() -> None:
    """Cost is measured off the serialized result, not off the hit, so a chunk
    that would have dominated the budget costs the same as a tiny one."""
    huge, _ = ResultBudget(6000).pack([_hit(0, 100_000)], include_text=False)
    small, _ = ResultBudget(6000).pack([_hit(0, 1)], include_text=False)

    assert huge[0].text == small[0].text == ""
    assert not huge[0].text_truncated


def test_an_omitted_body_is_never_marked_truncated() -> None:
    """`text_truncated` means "you have part of the chunk". Withholding the
    body gives you none of it, and conflating the two would have the agent
    believe it holds a prefix it does not."""
    results, _ = ResultBudget(50).pack([_hit(i, 5000) for i in range(3)], include_text=False)
    assert results
    assert all(not r.text_truncated for r in results)


def test_the_small_chunk_floor_does_not_apply_without_bodies() -> None:
    """The floor exists because half a chunk is useless, and it is set near
    what a body-less hit costs. A location has no partial form -- it fits or it
    does not -- so honouring the floor would stop packing with room still left.

    Asserted as a comparison rather than a count: the exact number of hits that
    fit depends on how wide the metadata happens to be, which is not the claim.
    """
    hits = [_hit(i, 500) for i in range(40)]

    tight, _ = ResultBudget(600, min_chunk_tokens=64).pack(hits, include_text=False)
    loose, _ = ResultBudget(600, min_chunk_tokens=1).pack(hits, include_text=False)

    assert tight, "nothing was packed, so this compares nothing"
    assert len(tight) == len(loose)


def test_omitting_bodies_does_not_change_the_default_path() -> None:
    """The default must produce exactly what it produced before the flag
    existed, or a normal search silently changes."""
    hits = [_hit(i, 100) for i in range(5)]
    explicit, _ = ResultBudget(1000).pack(hits, include_text=True)
    default, _ = ResultBudget(1000).pack(hits)

    assert [r.model_dump() for r in explicit] == [r.model_dump() for r in default]
    assert all(r.text and not r.text_omitted for r in default)
