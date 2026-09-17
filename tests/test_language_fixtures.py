"""The offline half of the eval: extraction, resolution and classification.

The retrieval eval needs embeddings, a key, a store, and is nondeterministic.
This is not a degraded version of it -- it measures a different half, the one
CI has been blind to. Import extraction, namespace extraction, resolution and
origin classification are pure functions of the file bytes: tree-sitter parses
locally, the manifest is a file, resolution is a query. Chunk boundaries are
*not* on that list and are deliberately unmeasured here -- they move with the
branch name, which is #93. So it runs on a clean runner with no credentials, in seconds, and can
therefore gate a pull request rather than be reported after the fact.

The fixtures are authored, so the counts are exact. A band would hide the
change it exists to catch.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

from tests.indexer_harness import Harness
from tests.language_fixtures import (
    FIXTURE_ROOT,
    describe,
    fixture_languages,
    load_baselines,
    measure,
    source_files,
)
from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.graph import SUPPORTED as EXTRACTOR_LANGUAGES
from workspace_indexer.state import Manifest
from workspace_indexer.storage.qdrant_store import QdrantStore

ROOT_LABEL = "languages"


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def indexed(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[Manifest]:
    """The fixture corpus, indexed once for the whole module, with no API calls.

    Through the real `Indexer` rather than by calling the scanners directly:
    a gate that reimplements the pipeline measures the reimplementation. Only
    the paid backend is faked, which is what makes the run free.

    Module-scoped because every test here only reads: at function scope the
    corpus was re-indexed four times, and "indexed once" was true within a test
    and false across the file. The loop scope has to match the fixture scope or
    the store is created on a loop that is closed before the last test uses it.
    """
    tmp_path = tmp_path_factory.mktemp("language-fixtures")
    config = WorkspaceConfig.model_validate(
        {
            "workspace": {
                "name": "language-fixtures",
                "roots": [{"path": str(FIXTURE_ROOT), "label": ROOT_LABEL}],
            },
            "index": {"exclude": ["**/node_modules/**"]},
        }
    )
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="language-fixtures", payload_indexes=False)
    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        await Harness(config, store, manifest, tmp_path).indexer().run()
        yield manifest
    await client.close()


def test_every_extractable_language_has_a_fixture_tree() -> None:
    """A language gains an extractor and gains a fixture, or the build fails.

    The same rule the rest of the guard suite encodes: if a list appears twice,
    one of them is derived or a test proves they match. Without this, adding a
    sixth language to `SUPPORTED` would leave it measured by nothing, and the
    gate would go on passing while covering less than it claims.
    """
    missing = sorted(EXTRACTOR_LANGUAGES - set(fixture_languages()))
    assert not missing, (
        f"languages with an import extractor and no fixture tree: {missing}. "
        f"Add tests/fixtures/languages/<language>/ with at least one source file, "
        f"and a README saying what each file exercises."
    )


def test_every_fixture_tree_holds_something_to_measure() -> None:
    """A directory is not coverage.

    `source_files()` skips READMEs, so a language directory containing only a
    README measures nothing -- and an all-zero baseline row matches those
    nothing-counts exactly, the accounting identity holds at 0 + 0 = 0, and
    every other test in this file passes. The guard above would report the
    language as covered.

    Its failure message used to suggest precisely that: "add a directory with a
    README". So this is not a hypothetical someone would have to go out of
    their way to hit -- it was the documented next step.
    """
    empty = sorted(
        language
        for language in fixture_languages()
        if not any(rel.split("/")[0] == language for rel in source_files())
    )
    assert not empty, f"fixture trees with no source file, so nothing is measured for them: {empty}"


def test_every_fixture_tree_has_a_baseline_row() -> None:
    """And the other direction: a fixture nobody measures is decoration."""
    expected, _ = load_baselines()
    missing = sorted(set(fixture_languages()) - set(expected))
    assert not missing, f"fixture trees with no baseline row: {missing}"


def test_every_language_matches_its_baseline(indexed: Manifest) -> None:
    """Exact counts per language per bucket.

    A failure here is not necessarily a regression -- a fixture added on
    purpose moves these too. It is a claim that the numbers changed and nobody
    said so, which is the thing that has been invisible.
    """
    expected, _ = load_baselines()
    measured, _ = measure(indexed, root_label=ROOT_LABEL)

    differences = describe(measured, expected)
    assert not differences, (
        f"the fixture corpus no longer measures what was recorded:\n{differences}"
    )


def test_the_first_party_gate_is_whole_for_every_language(indexed: Manifest) -> None:
    """The one column where a shortfall is a defect.

    Stated separately from the baseline comparison because the baselines would
    happily record a drop: 4 of 5 resolved is a perfectly stable number. Only a
    first-party edge can reach a file in this corpus, so anything less than all
    of them is a resolver that stopped following something it was following.

    It does not cover C#, and cannot. There, first-party membership is *decided*
    by resolution -- a using is known to be ours because a declared namespace
    matched it -- so a C# resolver that stops working moves those edges into
    `unclassified` rather than leaving them first-party and unresolved, and this
    ratio stays 0 of 0. Verified by disabling `NamespaceResolver`: this test
    passed and the baseline comparison failed with
    `csharp.first_party: 3 -> 0`. For C# the baseline is the guard; for
    everything else the two are independent.
    """
    measured, _ = measure(indexed, root_label=ROOT_LABEL)
    short = {
        language: (row.first_party_resolved, row.first_party)
        for language, row in measured.items()
        if row.first_party_resolved != row.first_party
    }
    assert not short, f"first-party edges that no longer reach a file: {short}"


def test_only_the_intended_files_are_withheld(indexed: Manifest) -> None:
    """Which files the secret scanner kept out of the index.

    The direction that matters is the false positive: a withheld file leaves no
    row anywhere, so nothing reports it and retrieval silently stops being able
    to answer from it. `python/config_values.py` exists to be indexed *despite*
    holding a token limit, a connection-string template and a credential's
    name; `python/credentials.py` exists to be withheld. Both directions fail
    here rather than being noticed months later.
    """
    _, expected = load_baselines()
    _, withheld = measure(indexed, root_label=ROOT_LABEL)

    unexpected = sorted(set(withheld) - set(expected))
    missing = sorted(set(expected) - set(withheld))
    assert not unexpected, (
        f"files kept out of the index that should be in it: {unexpected}. "
        f"A false positive here is silent data loss."
    )
    assert not missing, f"files indexed that hold a credential: {missing}"


def test_every_fixture_file_is_accounted_for(indexed: Manifest) -> None:
    """No fixture is silently skipped.

    A file the walk never reached is indistinguishable, in every count above,
    from a file that produced nothing -- so the corpus could shrink without a
    single number moving.
    """
    _, expected_withheld = load_baselines()
    measured, withheld = measure(indexed, root_label=ROOT_LABEL)

    indexed_count = sum(row.files_indexed for row in measured.values())
    assert indexed_count + len(withheld) == len(source_files())
    assert sorted(withheld) == sorted(expected_withheld)
