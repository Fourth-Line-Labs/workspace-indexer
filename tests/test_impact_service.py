"""The dependency graph, as an agent sees it.

Every test here runs against a real SQLite manifest holding a real graph.
The queries are SQL and the joins are the thing that can be wrong, so a mocked
manifest would assert our idea of the query rather than the query.

The failure this file is mostly guarding against is not a wrong number. It is
a *plausible* empty answer: an agent told "nothing imports this" about a file
in an unscanned language, or about the wrong file, will delete working code.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import make_source
from workspace_indexer.classification import Classification
from workspace_indexer.graph import ImportEdge
from workspace_indexer.graph.import_origin import ImportOrigin
from workspace_indexer.mcp.impact_service import ImpactService
from workspace_indexer.models import DocumentType, FileKind
from workspace_indexer.state import Manifest

ROOT = "workspace"


@pytest.fixture
def manifest(tmp_path: Path) -> Iterator[Manifest]:
    with Manifest(tmp_path / "manifest.sqlite3") as m:
        yield m


def add_file(
    manifest: Manifest,
    rel_path: str,
    *,
    doc_type: DocumentType = DocumentType.IMPLEMENTATION,
    language: str | None = "python",
) -> None:
    source = make_source(
        "x",
        kind=FileKind.CODE,
        language=language,
        rel_path=rel_path,
        root_label=ROOT,
        unit=rel_path.split("/")[0],
    )
    manifest.record_file(
        source,
        chunker="code",
        chunker_version=1,
        classification=Classification(doc_type=doc_type, confidence=1.0, reason="fixture"),
        classifier_version=1,
    )


def add_import(
    manifest: Manifest,
    importer: str,
    module: str,
    *,
    line: int = 1,
    resolved: str | None = None,
) -> None:
    existing = manifest.imports_of(ROOT, importer)
    manifest.record_imports(
        ROOT,
        importer,
        [*existing, ImportEdge(module=module, kind="import", is_relative=True, line=line)],
    )
    if resolved:
        manifest.set_resolved_path(ROOT, importer, module, resolved)


@pytest.fixture
def graph(manifest: Manifest) -> Manifest:
    """One target imported by a caller and two tests, plus a decoy.

    The decoy (`legacy/helper.py`) exists so that suffix matching has something
    to be wrong about.
    """
    add_file(manifest, "app/src/helper.py")
    add_file(manifest, "app/src/service.py")
    add_file(manifest, "app/tests/test_helper.py", doc_type=DocumentType.TEST)
    add_file(manifest, "app/tests/test_service.py", doc_type=DocumentType.TEST)
    add_file(manifest, "legacy/helper.py")

    add_import(manifest, "app/src/service.py", "./helper", line=3, resolved="app/src/helper.py")
    add_import(
        manifest, "app/tests/test_helper.py", "app.src.helper", line=5, resolved="app/src/helper.py"
    )
    add_import(
        manifest,
        "app/tests/test_service.py",
        "app.src.helper",
        line=7,
        resolved="app/src/helper.py",
    )
    # Two edges we cannot follow, for opposite reasons. `httpx` is a declared
    # package with no file in this workspace, so leaving it unresolved is
    # correct. `app.src.missing` is first-party and unresolved, which is a gap
    # in the graph -- the distinction the note has to keep.
    add_import(manifest, "app/src/helper.py", "httpx", line=1)
    add_import(manifest, "app/src/helper.py", "./service", line=2, resolved="app/src/service.py")

    # Origins last, in one call: record_imports replaces every row for a file,
    # so anything set between add_import calls would be wiped by the next one.
    first = ImportOrigin.FIRST_PARTY.value
    manifest.record_origins(
        [
            (ROOT, "app/src/service.py", "./helper", first),
            (ROOT, "app/tests/test_helper.py", "app.src.helper", first),
            (ROOT, "app/tests/test_service.py", "app.src.helper", first),
            (ROOT, "app/src/helper.py", "./service", first),
            (ROOT, "app/src/helper.py", "httpx", ImportOrigin.DECLARED_DEPENDENCY.value),
        ]
    )
    return manifest


def test_used_by_spans_the_whole_workspace(graph: Manifest) -> None:
    report = ImpactService(graph).impact_of("app/src/helper.py")

    assert report.used_by_total == 3
    assert [d.location for d in report.used_by] == [
        "app/src/service.py:3",
        "app/tests/test_helper.py:5",
        "app/tests/test_service.py:7",
    ]


def test_dependents_are_counted_by_document_type(graph: Manifest) -> None:
    """The synthesis the graph alone cannot give: "one caller and two tests".

    This is the line an agent acts on without reading the list.
    """
    report = ImpactService(graph).impact_of("app/src/helper.py")
    assert report.used_by_by_type == {"test": 2, "implementation": 1}


def test_callers_are_kept_ahead_of_tests_when_truncating(graph: Manifest) -> None:
    """What breaks matters more than what notices it broke."""
    report = ImpactService(graph).impact_of("app/src/helper.py", limit=1)

    assert [d.rel_path for d in report.used_by] == ["app/src/service.py"]
    assert report.dropped_used_by == 2
    # ...and the counts still describe all three, not the one that fitted.
    assert report.used_by_by_type == {"test": 2, "implementation": 1}
    assert report.note is not None and "omitted" in report.note


def test_an_unfollowable_import_is_listed_not_dropped(graph: Manifest) -> None:
    """Reporting a file as importing less than it does is the worse lie."""
    report = ImpactService(graph).impact_of("app/src/helper.py")

    modules = {d.module: d for d in report.depends_on}
    assert modules["httpx"].resolved is False
    assert modules["httpx"].rel_path is None
    assert modules["./service"].resolved is True
    assert modules["./service"].rel_path == "app/src/service.py"
    # The note has to say *why* it could not be followed. A declared package
    # is outside this workspace by nature; calling that the same thing as an
    # unresolved first-party edge would send an agent to read a package that
    # does not exist.
    assert report.note is not None
    assert "outside this workspace by nature" in report.note
    assert "first-party but unresolved" not in report.note


def test_an_ambiguous_path_is_not_guessed(graph: Manifest) -> None:
    """Two files end with `helper.py`. Answering for one of them would tell the
    agent that nothing imports a file it never asked about."""
    report = ImpactService(graph).impact_of("helper.py")

    assert report.rel_path == ""
    assert report.used_by == []
    assert sorted(report.candidates) == ["app/src/helper.py", "legacy/helper.py"]
    assert report.note is not None and "Nothing was guessed" in report.note


def test_a_full_path_wins_over_being_a_suffix_of_another(manifest: Manifest) -> None:
    add_file(manifest, "helper.py")
    add_file(manifest, "app/helper.py")
    add_import(manifest, "app/helper.py", "./x", line=1)

    report = ImpactService(manifest).impact_of("helper.py")
    assert report.rel_path == "helper.py"
    assert report.candidates == []


def test_a_suffix_only_matches_on_a_directory_boundary(manifest: Manifest) -> None:
    """`store.py` must not match `my_store.py`. A silently wrong file is worse
    than no answer."""
    add_file(manifest, "app/my_store.py")

    report = ImpactService(manifest).impact_of("store.py")
    assert report.rel_path == ""
    assert report.candidates == []
    assert report.note is not None and "No indexed file matches" in report.note


def test_an_unscanned_language_says_so_rather_than_reporting_zero(
    manifest: Manifest,
) -> None:
    """The whole reason this tool carries a note.

    A .bicep file has no import scanner, so both lists are empty for a reason
    that has nothing to do with the file. Read as "nothing depends on this",
    that is a licence to delete it.
    """
    add_file(manifest, "infra/main.bicep", language="bicep")

    report = ImpactService(manifest).impact_of("infra/main.bicep")

    assert report.used_by_total == 0
    assert report.note is not None
    assert "not scanned for bicep" in report.note
    assert "says nothing about whether anything depends" in report.note


def test_a_scanned_language_with_no_dependents_explains_the_gap(
    manifest: Manifest,
) -> None:
    """Zero here is a real measurement, but still not proof of no dependants:
    an import spelled as a package name is recorded and never resolved."""
    add_file(manifest, "app/orphan.py")

    report = ImpactService(manifest).impact_of("app/orphan.py")

    assert report.used_by_total == 0
    assert report.note is not None
    assert "No indexed file resolves an import to this one" in report.note
    assert "not scanned" not in report.note


def test_the_call_is_recorded_for_harvesting(graph: Manifest) -> None:
    """Same contract as the search tools: a call that disappointed is an eval
    case waiting to be written, and finding it must be a query."""
    from workspace_indexer.mcp.tool_call_recorder import ToolCallRecorder

    ImpactService(graph, recorder=ToolCallRecorder(graph)).impact_of("app/src/helper.py")

    calls = graph.tool_calls()
    assert [c.tool for c in calls] == ["impact_of"]
    assert calls[0].query == "app/src/helper.py"
    assert "app/src/service.py" in calls[0].returned_paths


def test_deleting_a_file_removes_its_edges(graph: Manifest) -> None:
    """The cascade is what keeps "who imports this" correct with no separate
    invalidation step."""
    service = ImpactService(graph)
    assert service.impact_of("app/src/helper.py").used_by_total == 3

    graph.forget_file(ROOT, "app/tests/test_helper.py")

    report = service.impact_of("app/src/helper.py")
    assert report.used_by_total == 2
    assert report.used_by_by_type == {"implementation": 1, "test": 1}


def test_a_re_export_importer_is_flagged_as_a_hop(manifest: Manifest) -> None:
    """Measured against the live index, not imagined.

    `result_budget.py` reported two dependents while `test_result_budget.py`
    plainly uses it -- because that test imports the package, so the edge lands
    on `__init__.py`. The graph is right and the number is misleading, which is
    the one combination a note has to cover.
    """
    add_file(manifest, "app/src/thing.py")
    add_file(manifest, "app/src/__init__.py")
    add_import(manifest, "app/src/__init__.py", "./thing", line=1, resolved="app/src/thing.py")

    report = ImpactService(manifest).impact_of("app/src/thing.py")

    assert report.used_by_total == 1
    assert report.note is not None
    assert "re-export files" in report.note
    assert "app/src/__init__.py" in report.note
    assert "the real number of callers is higher" in report.note


def test_a_plain_importer_is_not_flagged_as_a_hop(graph: Manifest) -> None:
    """The warning has to be absent when it does not apply, or it becomes
    noise the agent learns to skip."""
    report = ImpactService(graph).impact_of("app/src/helper.py")
    assert report.note is not None and "re-export" not in report.note


def add_route_call(
    manifest: Manifest,
    caller: str,
    url: str,
    *,
    line: int = 1,
    exact: bool = True,
) -> None:
    from workspace_indexer.graph.route_call import RouteCall

    manifest.record_routes(ROOT, caller, [], [RouteCall(target=url, line=line, exact=exact)])


def add_route_declaration(manifest: Manifest, path: str, template: str) -> None:
    from workspace_indexer.graph.route_declaration import RouteDeclaration

    manifest.record_routes(
        ROOT,
        path,
        [RouteDeclaration(template=template, method="GET", line=1, kind="controller")],
        [],
    )


def test_an_http_caller_is_reported_and_kept_apart_from_importers(
    manifest: Manifest,
) -> None:
    """An importer breaks at compile time; a caller over HTTP breaks at run
    time, in another repository, possibly deployed separately. Merging them
    would hide the distinction that makes the question worth asking."""
    from workspace_indexer.graph.route_target import RouteTarget

    add_file(manifest, "api/Api/RemittanceController.cs", language="csharp")
    add_file(manifest, "web/app/page.ts", language="typescript")
    add_route_declaration(manifest, "api/Api/RemittanceController.cs", "api/Remittance")
    add_route_call(manifest, "web/app/page.ts", "/api/Remittance")
    manifest.set_route_resolution(
        ROOT,
        "web/app/page.ts",
        "/api/Remittance",
        1,
        RouteTarget(
            root_label=ROOT,
            rel_path="api/Api/RemittanceController.cs",
            template="api/Remittance",
        ),
    )

    report = ImpactService(manifest).impact_of("api/Api/RemittanceController.cs")

    assert [c.rel_path for c in report.called_by] == ["web/app/page.ts"]
    assert report.called_by_total == 1
    # Not folded into the import relationship.
    assert report.used_by == []
    assert report.note is not None and "over HTTP" in report.note


def test_the_calling_side_lists_what_it_reaches(manifest: Manifest) -> None:
    add_file(manifest, "web/app/page.ts", language="typescript")
    add_route_call(manifest, "web/app/page.ts", "/api/Remittance")

    report = ImpactService(manifest).impact_of("web/app/page.ts")

    assert [c.module for c in report.calls] == ["/api/Remittance"]
    assert report.calls_total == 1
    # Unresolved, and saying so rather than implying the endpoint is missing.
    assert report.calls[0].resolved is False


def test_the_note_reports_the_first_party_rate_not_the_all_edges_rate(graph: Manifest) -> None:
    """The number an agent sees has to be the one that means something.

    Against every edge this workspace reads 4 of 5 (80%), because one edge is a
    declared package that was never reachable. Against first-party edges it is
    4 of 4 -- and that is the figure worth acting on, because first-party is
    the only origin where an unresolved edge is a defect.
    """
    report = ImpactService(graph).impact_of("app/src/helper.py")

    assert report.note is not None
    assert "4 of 4 first-party python import edges resolve" in report.note
    assert "(100%)" in report.note
    # The remainder is accounted for rather than quietly dropped from view.
    assert "1 edge(s) name the standard library or a declared dependency" in report.note
    # And the old, misleading denominator is gone.
    assert "4 of 5" not in report.note


def test_an_unresolved_first_party_edge_is_called_a_gap_not_a_package(
    graph: Manifest,
) -> None:
    """A first-party edge names a file that is indexed, so failing to reach it
    is a hole in the graph. Reporting it as an external dependency sends an
    agent off to read a package that does not exist."""
    # On the decoy, not on helper.py. `record_imports` replaces every row for
    # a file, so adding an edge to helper.py here would delete and re-insert
    # its others -- losing `./service`'s resolution and `httpx`'s origin, and
    # quietly testing against a graph the fixture did not intend. legacy/
    # helper.py has no imports of its own, so there is nothing to wipe.
    add_import(graph, "legacy/helper.py", "app.src.absent", line=1)
    graph.record_origins(
        [(ROOT, "legacy/helper.py", "app.src.absent", ImportOrigin.FIRST_PARTY.value)]
    )

    report = ImpactService(graph).impact_of("legacy/helper.py")

    assert report.note is not None
    assert "first-party but unresolved" in report.note
    assert "does exist and is indexed" in report.note


def test_a_language_with_no_first_party_edges_reports_no_rate(manifest: Manifest) -> None:
    """The C# position today: edges extracted, none identified as first-party
    because that needs the namespace declarations table. No rate is not the
    same as a rate of zero, and must not read as one."""
    add_file(manifest, "app/Thing.cs", language="csharp")
    add_import(manifest, "app/Thing.cs", "System.Text", line=1)
    manifest.record_origins([(ROOT, "app/Thing.cs", "System.Text", ImportOrigin.FRAMEWORK.value)])

    report = ImpactService(manifest).impact_of("app/Thing.cs")

    assert report.note is not None
    assert "no csharp import edge has been identified as first-party" in report.note
    assert "%" not in report.note


def test_adding_an_edge_to_the_decoy_leaves_the_fixture_intact(graph: Manifest) -> None:
    """Guards the trap the previous test walks around.

    `record_imports` deletes and re-inserts every row for a file, so an
    `add_import` against an already-classified file silently strips the
    resolutions and origins the fixture set. Pinned here so a future test that
    reaches for the convenient file finds out from a failure rather than from a
    passing assertion over a degraded graph.
    """
    add_import(graph, "legacy/helper.py", "app.src.absent", line=1)

    helper = {d.module: d for d in graph.dependencies_of(ROOT, "app/src/helper.py")}
    assert helper["./service"].rel_path == "app/src/service.py"
    assert helper["httpx"].origin == ImportOrigin.DECLARED_DEPENDENCY.value


def test_an_unclassified_index_is_not_described_as_a_classified_one(
    manifest: Manifest,
) -> None:
    """ "We never looked" must not read as "we looked and found nothing".

    An index predating the origin column has every edge unrecorded. Reporting
    that as "all N edges are framework, declared dependencies or unclassified"
    asserts a classification result that never happened.
    """
    add_file(manifest, "app/src/thing.py")
    add_import(manifest, "app/src/thing.py", "httpx", line=1)

    report = ImpactService(manifest).impact_of("app/src/thing.py")

    assert report.note is not None
    assert "has an origin recorded" in report.note
    assert "predates origin classification" in report.note
    assert "are framework, declared dependencies or unclassified" not in report.note


def test_the_workspace_note_accounts_for_every_edge(graph: Manifest) -> None:
    """The listed counts have to sum to the total, or an agent cannot tell a
    withheld number from unreachable code."""
    add_import(graph, "legacy/helper.py", "some.unruled.thing", line=1)

    report = ImpactService(graph).impact_of("app/src/helper.py")
    coverage = graph.origin_coverage()["python"]

    assert (
        coverage.first_party
        + coverage.framework
        + coverage.declared_dependency
        + coverage.unclassified
        + coverage.unrecorded
        == coverage.total
    )
    assert report.note is not None
    # The unrecorded edge just added must be spoken for, not dropped.
    assert coverage.unrecorded
    assert "predate the last classification run" in report.note
