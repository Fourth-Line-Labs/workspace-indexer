"""The three nested denominators, and the one case that must not be a number.

`non_framework` is derived by subtraction and `first_party_resolution` refuses
to answer when there is nothing to answer about. Both are decisions rather than
arithmetic, so both are pinned here.
"""

from __future__ import annotations

from workspace_indexer.graph.origin_coverage import OriginCoverage


def _coverage(**kwargs: int) -> OriginCoverage:
    fields: dict[str, int] = {
        "total": 0,
        "first_party": 0,
        "first_party_resolved": 0,
        "declared_dependency": 0,
        "framework": 0,
        "unclassified": 0,
        "unrecorded": 0,
    }
    fields.update(kwargs)
    return OriginCoverage(**fields)


def test_non_framework_drops_only_the_standard_library() -> None:
    coverage = _coverage(total=100, framework=30, first_party=40, declared_dependency=30)
    assert coverage.non_framework == 70


def test_non_framework_is_derived_by_subtraction_not_by_summing_buckets() -> None:
    # An origin added later must be counted here without this property
    # learning its name, so the total minus framework is the definition.
    coverage = _coverage(total=100, framework=10, first_party=5)
    assert coverage.non_framework == 90


def test_the_gate_is_the_resolved_share_of_first_party_edges() -> None:
    coverage = _coverage(total=500, first_party=200, first_party_resolved=150, framework=300)
    assert coverage.first_party_resolution == 0.75


def test_a_fully_resolved_language_gates_at_one() -> None:
    coverage = _coverage(total=10, first_party=4, first_party_resolved=4, framework=6)
    assert coverage.first_party_resolution == 1.0


def test_no_first_party_edges_is_neither_a_pass_nor_a_failure() -> None:
    # None rather than 0.0 or 1.0: reporting either would make a corpus with
    # nothing to resolve look like a result.
    coverage = _coverage(total=80, framework=80)
    assert coverage.first_party_resolution is None


def test_an_empty_language_reports_nothing_rather_than_dividing_by_zero() -> None:
    assert _coverage().first_party_resolution is None


def test_unrecorded_is_not_folded_into_unclassified() -> None:
    # "classification never ran" and "it ran and no rule claimed this" call
    # for opposite next moves: re-index, versus write a rule.
    coverage = _coverage(total=10, unrecorded=10)
    assert coverage.unrecorded == 10
    assert coverage.unclassified == 0


def test_the_gate_percent_is_floored_not_rounded() -> None:
    """1,234 of 1,240 is 99%, not 100%.

    The original display formatted the raw rate with `.0%`, which rounds 0.995
    up — so a language six edges short printed "100%". This is the one column
    where that number has to mean complete.
    """
    coverage = _coverage(total=1_240, first_party=1_240, first_party_resolved=1_234)
    assert coverage.first_party_resolution_percent == 99


def test_the_gate_percent_reaches_a_hundred_only_when_every_edge_resolved() -> None:
    whole = _coverage(total=10, first_party=10, first_party_resolved=10)
    assert whole.first_party_resolution_percent == 100
    one_short = _coverage(total=10, first_party=10, first_party_resolved=9)
    assert one_short.first_party_resolution_percent == 90


def test_a_single_missing_edge_in_a_large_corpus_still_reads_below_a_hundred() -> None:
    """The case a float would lose: 9,999 of 10,000 is 99.99%."""
    coverage = _coverage(total=10_000, first_party=10_000, first_party_resolved=9_999)
    assert coverage.first_party_resolution_percent == 99


def test_no_first_party_edges_has_no_gate_percent() -> None:
    assert _coverage(total=80, framework=80).first_party_resolution_percent is None


def test_the_gate_percent_never_exceeds_a_hundred() -> None:
    """Defensive: resolved should never outrun first_party, but if a future
    bug made it, the column must not print 103%."""
    odd = _coverage(total=10, first_party=5, first_party_resolved=7)
    assert odd.first_party_resolution_percent == 100
