"""The failure message the gate prints when a baseline moves.

Tested apart from the corpus because it is the acceptance criterion, not a
convenience: a gate whose output sends someone spelunking gets disabled, and
then there is neither the gate nor the confidence. The counts can be right
while the message that reports them is unusable, and nothing else here would
notice -- every other test in this suite exercises `describe` only on the
equal path, where it returns an empty string.
"""

from __future__ import annotations

from tests.language_baseline import LanguageBaseline
from tests.language_fixtures import describe


def _row(**overrides: int) -> LanguageBaseline:
    fields = {
        "files_indexed": 5,
        "files_with_imports": 4,
        "import_edges": 6,
        "namespace_declarations": 0,
        "first_party": 4,
        "first_party_resolved": 4,
        "declared_dependency": 0,
        "framework": 1,
        "unclassified": 1,
        "declined": 0,
    }
    fields.update(overrides)
    return LanguageBaseline.model_validate(fields)


def test_nothing_moved_says_nothing() -> None:
    """An empty string is what the assertion tests, so it has to mean equal."""
    assert describe({"python": _row()}, {"python": _row()}) == ""


def test_a_loss_names_the_language_the_field_and_the_direction() -> None:
    """The four things someone needs to act: which language, which number,
    which way it went, and where the fixtures are."""
    message = describe({"python": _row(first_party_resolved=1)}, {"python": _row()})
    assert message == (
        "python.first_party_resolved: 4 -> 1 (-3), see tests/fixtures/languages/python/"
    )


def test_a_gain_is_signed_too() -> None:
    """A count going up is as much a change as one going down -- a fixture
    added on purpose reads the same way here as a resolver that started
    over-claiming, and the sign is what tells them apart at a glance."""
    message = describe({"csharp": _row(unclassified=4)}, {"csharp": _row()})
    assert "csharp.unclassified: 1 -> 4 (+3)" in message


def test_every_moved_field_is_reported_not_just_the_first() -> None:
    """Disabling the C# resolver moves three numbers at once. Reporting one
    would send someone chasing a symptom."""
    measured = {"csharp": _row(first_party=0, first_party_resolved=0, unclassified=4)}
    message = describe(measured, {"csharp": _row()})
    assert message.count("\n") == 2
    assert "csharp.first_party: 4 -> 0 (-4)" in message
    assert "csharp.first_party_resolved: 4 -> 0 (-4)" in message
    assert "csharp.unclassified: 1 -> 4 (+3)" in message


def test_a_new_language_says_to_record_it_rather_than_reporting_deltas() -> None:
    """Adding a fixture tree is not a regression, and a diff of every field
    against nothing would read like one."""
    message = describe({"python": _row(), "go": _row()}, {"python": _row()})
    assert message == "go: no baseline row (fixtures/languages/go/ is new)"


def test_a_baseline_with_no_fixtures_is_named_as_that() -> None:
    """The opposite mistake: a row left behind by a deleted tree. Without this
    branch it would print a delta against zero for every field."""
    message = describe({"python": _row()}, {"python": _row(), "go": _row()})
    assert message == "go: baseline row with no fixture directory"
