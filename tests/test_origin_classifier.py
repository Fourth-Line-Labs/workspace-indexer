"""Precedence in the origin classifier, which is where its behaviour lives.

Each test names the rule that must win and the one it must beat, because the
ordering is the design and a silent reordering would move coverage numbers
without failing anything else.
"""

from __future__ import annotations

import pytest

from workspace_indexer.graph.import_origin import ImportOrigin
from workspace_indexer.graph.origin_classifier import OriginClassifier


def _classify(
    classifier: OriginClassifier,
    module: str,
    *,
    language: str = "csharp",
    is_relative: bool = False,
    resolved: str | None = None,
    root_label: str = "src",
    from_path: str = "repo/lib/thing.cs",
) -> ImportOrigin:
    return classifier.classify(
        module,
        language=language,
        is_relative=is_relative,
        resolved=resolved,
        root_label=root_label,
        from_path=from_path,
    )


@pytest.fixture
def bare() -> OriginClassifier:
    return OriginClassifier()


def test_a_resolved_edge_is_first_party_whatever_it_is_called(bare: OriginClassifier) -> None:
    # Beats the framework rule: if it resolved to an indexed file then it is
    # ours, and the name is irrelevant.
    assert (
        _classify(bare, "System.Text", resolved="repo/lib/system_text.cs")
        is ImportOrigin.FIRST_PARTY
    )


def test_a_relative_edge_is_first_party_even_unresolved(bare: OriginClassifier) -> None:
    # This is the bucket that makes resolver bugs visible: a relative
    # specifier names a neighbour, so failing to resolve it is ours to fix.
    assert (
        _classify(bare, "./sibling", language="typescript", is_relative=True)
        is ImportOrigin.FIRST_PARTY
    )


def test_framework_when_nothing_is_declared(bare: OriginClassifier) -> None:
    assert _classify(bare, "System.Text.Json") is ImportOrigin.FRAMEWORK


def test_an_unclaimed_module_is_unclassified_not_framework(bare: OriginClassifier) -> None:
    assert _classify(bare, "Microsoft.Extensions.AI") is ImportOrigin.UNCLASSIFIED


def test_a_declared_dependency_beats_the_framework_rule() -> None:
    # `System.Text.Json` genuinely ships as a package. A project that declares
    # it really does get it from there, so declared data wins over the prefix.
    classifier = OriginClassifier({("src", "repo"): frozenset({"System.Text.Json"})})
    assert _classify(classifier, "System.Text.Json") is ImportOrigin.DECLARED_DEPENDENCY


def test_a_declared_id_claims_its_child_namespaces() -> None:
    classifier = OriginClassifier({("src", "repo"): frozenset({"Azure.Messaging"})})
    assert _classify(classifier, "Azure.Messaging.ServiceBus") is ImportOrigin.DECLARED_DEPENDENCY


def test_a_declared_id_claims_a_node_subpath() -> None:
    classifier = OriginClassifier({("src", "repo"): frozenset({"lodash"})})
    assert (
        _classify(classifier, "lodash/debounce", language="typescript", from_path="repo/a.ts")
        is ImportOrigin.DECLARED_DEPENDENCY
    )


def test_a_declared_id_does_not_claim_a_longer_name() -> None:
    # The separator is required, or `lodash` swallows `lodashy`.
    classifier = OriginClassifier({("src", "repo"): frozenset({"lodash"})})
    assert (
        _classify(classifier, "lodashy", language="typescript", from_path="repo/a.ts")
        is ImportOrigin.UNCLASSIFIED
    )


def test_dependencies_are_scoped_to_their_unit() -> None:
    # One repository's manifest says nothing about another's imports.
    classifier = OriginClassifier({("src", "repo-a"): frozenset({"Vendor.Thing"})})
    assert (
        _classify(classifier, "Vendor.Thing", from_path="repo-a/x.cs")
        is ImportOrigin.DECLARED_DEPENDENCY
    )
    assert (
        _classify(classifier, "Vendor.Thing", from_path="repo-b/x.cs") is ImportOrigin.UNCLASSIFIED
    )


def test_dependencies_are_scoped_to_their_root() -> None:
    # Two roots can hold repositories of the same name.
    classifier = OriginClassifier({("src", "repo"): frozenset({"Vendor.Thing"})})
    assert (
        _classify(classifier, "Vendor.Thing", root_label="other", from_path="repo/x.cs")
        is ImportOrigin.UNCLASSIFIED
    )


def test_a_file_at_the_root_of_a_root_has_an_empty_unit() -> None:
    classifier = OriginClassifier({("src", ""): frozenset({"Vendor.Thing"})})
    assert (
        _classify(classifier, "Vendor.Thing", from_path="Program.cs")
        is ImportOrigin.DECLARED_DEPENDENCY
    )


def test_an_empty_dependency_set_is_not_an_error(bare: OriginClassifier) -> None:
    # Nothing read for this unit yet, so third-party imports report as
    # unclassified rather than being asserted to be anything.
    classifier = OriginClassifier({("src", "repo"): frozenset()})
    assert _classify(classifier, "Vendor.Thing") is ImportOrigin.UNCLASSIFIED
    assert _classify(bare, "Vendor.Thing") is ImportOrigin.UNCLASSIFIED
