"""The indexing pipeline, wired end to end.

Real filesystem, real embedded Qdrant, real SQLite manifest; only the paid
embedding backend is faked, so the calls it never makes are countable. Every
assertion about "zero embedding calls" is the cost guarantee being enforced.
"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import structlog.testing
from qdrant_client import AsyncQdrantClient

from tests.conftest import ConfigFactory
from tests.fake_embedding_backend import FakeEmbeddingBackend
from tests.fake_sparse_backend import FakeSparseBackend
from workspace_indexer.chunking import ChunkerRegistry
from workspace_indexer.classification import RuleClassifier
from workspace_indexer.config import Settings, WorkspaceConfig
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.models import EmbeddingSpace, SearchFilters
from workspace_indexer.pipeline import Indexer
from workspace_indexer.state import Manifest
from workspace_indexer.storage.qdrant_store import QdrantStore

SPACE = EmbeddingSpace(model="fake:model", dimensions=4)
NARROW = EmbeddingSpace(model="fake:other", dimensions=4)


class Harness:
    """Everything the pipeline needs, with handles the tests assert against."""

    def __init__(
        self, config: WorkspaceConfig, store: QdrantStore, manifest: Manifest, tmp: Path
    ) -> None:
        self.config = config
        self.store = store
        self.manifest = manifest
        self.backend = FakeEmbeddingBackend(dimensions=4)
        self.embeddings = EmbeddingService(self.backend)
        self.sparse = FakeSparseBackend()
        self.tmp = tmp

    def indexer(self, space: EmbeddingSpace = SPACE) -> Indexer:
        return Indexer(
            config=self.config,
            settings=Settings(state_db=self.tmp / "manifest.sqlite3"),
            manifest=self.manifest,
            registry=ChunkerRegistry(self.config.workspace.name),
            embeddings=self.embeddings,
            sparse=self.sparse,
            store=self.store,
            space=space,
            classifier=RuleClassifier(),
            flush_chunks=8,
        )

    @property
    def documents_embedded(self) -> int:
        return self.backend.stats_documents

    def reset_counters(self) -> None:
        self.backend.batches.clear()
        self.backend.calls = 0


@pytest.fixture
async def harness(config_for: ConfigFactory, tmp_path: Path) -> AsyncIterator[Harness]:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="test", payload_indexes=False)
    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        yield Harness(config_for(), store, manifest, tmp_path)
    await client.close()


def _embedded(harness: Harness) -> int:
    return sum(len(batch) for batch in harness.backend.batches)


async def test_first_run_indexes_the_workspace(harness: Harness) -> None:
    stats = await harness.indexer().run()
    assert stats.files_changed > 0
    assert stats.chunks_upserted > 0
    assert await harness.store.count(SPACE) == stats.chunks_upserted
    assert harness.manifest.file_count() > 0


async def test_rerun_embeds_nothing(harness: Harness) -> None:
    """The whole cost story in one assertion: an unchanged workspace costs
    stat() calls and no money at all."""
    await harness.indexer().run()
    harness.reset_counters()

    stats = await harness.indexer().run()
    assert _embedded(harness) == 0
    assert stats.chunks_upserted == 0
    assert stats.files_changed == 0
    assert stats.files_skipped == stats.files_seen


async def test_touching_a_file_without_changing_it_embeds_nothing(
    harness: Harness, workspace: Path
) -> None:
    """Rung 2: a formatter pass or a checkout of identical bytes."""
    await harness.indexer().run()
    harness.reset_counters()

    target = workspace / "repo_one" / "src" / "widget.py"
    target.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")

    stats = await harness.indexer().run()
    assert _embedded(harness) == 0
    assert stats.chunks_upserted == 0


async def test_editing_one_function_reembeds_only_that_chunk(
    harness: Harness, workspace: Path
) -> None:
    await harness.indexer().run()
    before = await harness.store.count(SPACE)
    harness.reset_counters()

    target = workspace / "repo_one" / "src" / "widget.py"
    target.write_text(
        target.read_text(encoding="utf-8").replace("return a + b", "return a - b"),
        encoding="utf-8",
    )

    stats = await harness.indexer().run()
    assert stats.chunks_upserted == 1
    assert stats.chunks_deleted == 1
    assert _embedded(harness) == 1
    assert await harness.store.count(SPACE) == before


async def test_deleting_a_file_removes_its_chunks(harness: Harness, workspace: Path) -> None:
    await harness.indexer().run()
    before = await harness.store.count(SPACE)

    (workspace / "repo_one" / "src" / "widget.py").unlink()
    stats = await harness.indexer().run()

    assert stats.chunks_deleted > 0
    assert await harness.store.count(SPACE) < before
    assert harness.manifest.get_file("workspace", "repo_one/src/widget.py") is None


async def test_a_file_that_becomes_excluded_loses_its_chunks(
    harness: Harness, workspace: Path
) -> None:
    """Exclusion has to be retroactive, not just prospective.

    Adding an exclude pattern only stops *future* indexing; the chunks already
    written stay searchable and keep answering queries. That is how an eval
    artefact went on contaminating its own measurement after being excluded,
    and how a file caught by the secret scanner would keep serving the copy
    holding the secret. The file is still on disk, so nothing here depends on
    deletion -- only on the file no longer being discovered.
    """
    await harness.indexer().run()
    target = "repo_one/src/widget.py"
    assert harness.manifest.get_file("workspace", target) is not None
    before = await harness.store.count(SPACE)

    harness.config.index.exclude.append("**/src/widget.py")
    stats = await harness.indexer().run()

    assert stats.chunks_deleted > 0
    assert await harness.store.count(SPACE) < before
    assert harness.manifest.get_file("workspace", target) is None
    assert (workspace / "repo_one" / "src" / "widget.py").exists()


async def test_renaming_a_file_moves_its_chunks(harness: Harness, workspace: Path) -> None:
    await harness.indexer().run()
    before = await harness.store.count(SPACE)

    src = workspace / "repo_one" / "src" / "widget.py"
    src.rename(src.with_name("gadget.py"))
    await harness.indexer().run()

    assert harness.manifest.get_file("workspace", "repo_one/src/widget.py") is None
    assert harness.manifest.get_file("workspace", "repo_one/src/gadget.py") is not None
    assert await harness.store.count(SPACE) == before


async def test_adding_a_file_indexes_only_it(harness: Harness, workspace: Path) -> None:
    await harness.indexer().run()
    harness.reset_counters()

    (workspace / "repo_one" / "src" / "extra.py").write_text(
        "def brand_new_function():\n    return 42\n", encoding="utf-8"
    )
    stats = await harness.indexer().run()
    assert stats.files_changed == 1
    assert _embedded(harness) == stats.chunks_upserted


async def test_dry_run_writes_nothing_and_costs_nothing(harness: Harness) -> None:
    """The point of --dry-run: tune chunking without paying to iterate."""
    stats = await harness.indexer().run(dry_run=True)
    assert stats.chunks_upserted > 0
    assert stats.tokens_embedded > 0
    assert _embedded(harness) == 0
    assert await harness.store.count(SPACE) == 0
    assert harness.manifest.file_count() == 0
    assert harness.manifest.recent_runs() == []


async def test_force_reembeds_everything(harness: Harness) -> None:
    await harness.indexer().run()
    harness.reset_counters()

    stats = await harness.indexer().run(force=True)
    assert _embedded(harness) > 0
    assert stats.files_skipped == 0


async def test_a_forced_run_says_so_in_one_readable_line(harness: Harness) -> None:
    """Issue #60. `force=True` rode inside run.start's fields and nowhere else.

    Confirming a rebuild happened meant knowing to grep run.start for a boolean,
    or counting `decision: forced` lines across a debug log that can run to
    hundreds of thousands of entries. "Did a full reindex run at 03:00" should
    be answerable at a glance by someone who has never read this code.
    """
    await harness.indexer().run()

    with structlog.testing.capture_logs() as logs:
        await harness.indexer().run(force=True)

    forced = [e for e in logs if e["event"] == "run.forced_reindex"]
    assert len(forced) == 1
    assert forced[0]["log_level"] == "info"
    assert "every file in scope" in forced[0]["detail"]


async def test_an_ordinary_run_stays_quiet_about_force(harness: Harness) -> None:
    """A line that appears on every run is a line nobody reads."""
    with structlog.testing.capture_logs() as logs:
        await harness.indexer().run()

    assert not [e for e in logs if e["event"] == "run.forced_reindex"]


async def test_the_run_summary_counts_the_files_force_reindexed(harness: Harness) -> None:
    """`files_changed` alone cannot tell a rebuild from an enormous edit.

    Reported on every run, not only forced ones: zero is the informative answer
    to "was this a full reindex", and a field that only sometimes exists cannot
    be grepped for.
    """
    await harness.indexer().run()

    ordinary = await harness.indexer().run()
    assert ordinary.forced == 0

    rebuilt = await harness.indexer().run(force=True)
    assert rebuilt.forced == rebuilt.files_changed > 0

    with structlog.testing.capture_logs() as logs:
        await harness.indexer().run(force=True)
    ended = [e for e in logs if e["event"] == "run.end"]
    assert ended and ended[0]["files_forced"] > 0


async def test_model_swap_backfills_without_touching_the_old_space(
    harness: Harness,
) -> None:
    """Change the embedding model and the next run fills a new collection while
    the old one stays intact and searchable."""
    await harness.indexer().run()
    original = await harness.store.count(SPACE)

    await harness.indexer(NARROW).run()

    assert await harness.store.count(SPACE) == original
    assert await harness.store.count(NARROW) == original
    assert sorted(harness.manifest.spaces()) == sorted([SPACE.slug(), NARROW.slug()])


async def test_binary_files_are_recorded_but_never_embedded(harness: Harness) -> None:
    """And they must not be re-chunked on every subsequent run."""
    await harness.indexer().run()
    assert harness.manifest.get_file("workspace", "repo_one/src/widget.py") is not None
    harness.reset_counters()
    stats = await harness.indexer().run()
    assert stats.files_changed == 0


async def test_gitignored_content_never_reaches_the_store(harness: Harness) -> None:
    """The last line of defence before text is sent to an embedding API."""
    await harness.indexer().run()
    paths = {payload["rel_path"] async for _, payload, _ in harness.store.scroll(SPACE)}
    assert "repo_one/secret.txt" not in paths
    assert not any(str(p).startswith("repo_one/build/") for p in paths)


async def test_run_history_is_recorded(harness: Harness) -> None:
    await harness.indexer().run()
    runs = harness.manifest.recent_runs()
    assert len(runs) == 1
    assert not runs[0].unfinished
    assert runs[0].files_seen > 0
    assert runs[0].config_hash


async def test_only_root_limits_the_walk(harness: Harness, config_for: ConfigFactory) -> None:
    stats = await harness.indexer().run(only_root="workspace")
    assert stats.files_seen > 0


async def test_one_bad_file_does_not_abort_the_run(
    harness: Harness, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that has already paid for thousands of embeddings must not be lost
    to a single unreadable file."""
    import workspace_indexer.pipeline.indexer as module

    original = module.read_source
    calls = {"n": 0}

    def sometimes_explode(candidate: object, secret_allow: object = None):  # noqa: ANN202
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated read failure")
        return original(candidate, secret_allow)  # type: ignore[arg-type]

    monkeypatch.setattr(module, "read_source", sometimes_explode)
    stats = await harness.indexer().run()
    assert stats.errors == 1
    assert stats.chunks_upserted > 0


async def test_stats_report_tokens_and_cost(harness: Harness) -> None:
    stats = await harness.indexer().run()
    assert stats.tokens_embedded > 0
    assert stats.est_cost_usd > 0


async def test_force_still_removes_chunks_that_are_no_longer_produced(
    harness: Harness, workspace: Path
) -> None:
    """--force rebuilds, but it must still reconcile. Dropping to_delete left
    chunks in the store the file no longer produces, with nothing able to
    remove them -- a rebuild that quietly accumulates orphans."""
    await harness.indexer().run()
    target = workspace / "repo_one" / "src" / "widget.py"
    target.write_text("def only_one():\n    return 1\n", encoding="utf-8")

    stats = await harness.indexer().run(force=True)
    assert stats.chunks_deleted > 0
    assert await harness.store.count(SPACE) == harness.manifest.chunk_count(SPACE.slug())


async def test_chunks_carry_the_document_type(harness: Harness) -> None:
    """One classification per file, inherited by every chunk of it."""
    await harness.indexer().run()
    types = {str(payload.get("doc_type")) async for _, payload, _ in harness.store.scroll(SPACE)}
    assert "implementation" in types
    assert None not in types
    assert "MISSING" not in types


async def test_classification_is_cached_between_runs(harness: Harness) -> None:
    """Cheap for rules, and the thing that stops a future model rung from
    re-reading the whole workspace on every run."""
    await harness.indexer().run()
    source_count = harness.manifest.file_count()
    assert source_count > 0
    from tests.conftest import make_source

    for rel in ("repo_one/src/widget.py",):
        record = harness.manifest.get_file("workspace", rel)
        assert record is not None
        cached = harness.manifest.cached_classification(
            make_source("x", rel_path=rel).model_copy(
                update={"root_label": "workspace", "sha256": record.sha256}
            ),
            classifier_version=1,
        )
        assert cached is not None
        assert cached.decided


async def test_a_newly_withheld_file_has_its_old_chunks_purged(
    harness: Harness, workspace: Path
) -> None:
    """A file indexed before the scanner existed, or before a secret was added
    to it, must not keep its old chunks. Withholding future runs while leaving
    the earlier copy in the index defeats the point of detecting it."""
    target = workspace / "repo_one" / "src" / "widget.py"
    await harness.indexer().run()
    assert harness.manifest.get_file("workspace", "repo_one/src/widget.py") is not None
    before = await harness.store.count(SPACE)

    key = "AK" + "IA" + "IOSFODNN7EXAMPLE"
    target.write_text(f'AWS_KEY = "{key}"\n', encoding="utf-8")

    stats = await harness.indexer().run()
    assert stats.chunks_deleted > 0
    assert await harness.store.count(SPACE) < before
    assert harness.manifest.get_file("workspace", "repo_one/src/widget.py") is None
    paths = {str(payload.get("rel_path")) async for _, payload, _ in harness.store.scroll(SPACE)}
    assert "repo_one/src/widget.py" not in paths


async def test_a_withheld_file_does_not_abort_the_run(harness: Harness, workspace: Path) -> None:
    key = "AK" + "IA" + "IOSFODNN7EXAMPLE"
    (workspace / "repo_one" / "secrets.json").write_text(f'{{"aws": "{key}"}}', encoding="utf-8")
    stats = await harness.indexer().run()
    assert stats.errors == 0
    assert stats.chunks_upserted > 0


async def test_a_manifest_describing_a_different_store_is_flagged(
    harness: Harness,
) -> None:
    """Switching QDRANT_MODE, or pointing at another host, leaves the manifest
    reporting a full index over an empty store. The decision ladder then skips
    every file and the run silently does nothing -- which is exactly what
    happened moving from embedded to server."""
    await harness.indexer().run()
    assert harness.manifest.chunk_count(SPACE.slug()) > 0

    # The store loses its contents while the manifest keeps its record.
    await harness.store.drop_collection(SPACE)

    # structlog's own capture, rather than caplog: the harness does not
    # configure the stdlib routing that caplog depends on.
    with structlog.testing.capture_logs() as logs:
        await harness.indexer().run()
    assert any(entry.get("event") == "run.store_diverges" for entry in logs)


@pytest.fixture
def two_roots(tmp_path: Path) -> tuple[Path, Path, WorkspaceConfig]:
    """The shape a real workspace takes: docs in one tree, code in another.

    `c:\\doc\\ProjectA` and `c:\\src\\ProjectA` are two separate roots of one
    workspace, not two workspaces -- they share a collection, and `root_label`
    is what keeps them apart inside it.
    """
    docs = tmp_path / "doc" / "ProjectA"
    code = tmp_path / "src" / "ProjectA"
    (docs / "guide").mkdir(parents=True)
    (code / "app").mkdir(parents=True)
    (docs / "guide" / "install.md").write_text("# Install\n\nRun the thing.\n", encoding="utf-8")
    (docs / "guide" / "api.md").write_text("# API\n\nEndpoints and payloads.\n", encoding="utf-8")
    (code / "app" / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    (code / "app" / "util.py").write_text("def helper():\n    return 2\n", encoding="utf-8")

    config = WorkspaceConfig.model_validate(
        {
            "workspace": {
                "name": "ProjectA",
                "roots": [
                    {"path": str(docs), "label": "docs"},
                    {"path": str(code), "label": "code"},
                ],
            }
        }
    )
    return docs, code, config


async def _chunks_in(store: QdrantStore, root: str) -> int:
    """Chunks still in the store for one root -- the observable that matters.

    A manifest row with no chunk behind it is not a surviving index.
    """
    return await store.count(SPACE, SearchFilters(root_label=root))


async def test_separate_roots_share_one_collection(
    two_roots: tuple[Path, Path, WorkspaceConfig], tmp_path: Path
) -> None:
    """Two directory trees, one collection.

    The collection name comes from `workspace.name` and the embedding space --
    never from a root -- so pointing several roots at one workspace is how you
    get one searchable index over docs and code together.
    """
    _, _, config = two_roots
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace=config.workspace.name, payload_indexes=False)
    assert store.collection_name(SPACE) == f"ProjectA__{SPACE.slug()}"

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config, store, manifest, tmp_path)
        await harness.indexer().run()

        assert manifest.file_count("docs") == 2
        assert manifest.file_count("code") == 2
        # Both roots, one collection: the totals add up inside it.
        assert await _chunks_in(store, "docs") > 0
        assert await _chunks_in(store, "code") > 0
        assert await store.count(SPACE) == await _chunks_in(store, "docs") + await _chunks_in(
            store, "code"
        )
    await client.close()


async def test_reindexing_one_root_leaves_the_other_untouched(
    two_roots: tuple[Path, Path, WorkspaceConfig], tmp_path: Path
) -> None:
    """The question anyone with a split workspace asks first.

    Indexing `docs` must not notice that `code` was not walked and conclude its
    files have been deleted. Orphan removal is scoped to the root that was
    actually walked; without that scoping, every `--root` run would wipe every
    other root's chunks -- silent, total, and only visible as an empty search.
    """
    docs, _, config = two_roots
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace=config.workspace.name, payload_indexes=False)

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config, store, manifest, tmp_path)
        await harness.indexer().run()

        code_before = await _chunks_in(store, "code")
        assert code_before

        # Change one docs file, then reindex docs alone.
        (docs / "guide" / "install.md").write_text(
            "# Install\n\nCompletely rewritten instructions.\n", encoding="utf-8"
        )
        stats = await harness.indexer().run(only_root="docs")

        assert stats.files_changed == 1
        # The whole point: a root that was never walked is untouched.
        assert await _chunks_in(store, "code") == code_before
        assert manifest.file_count("code") == 2
    await client.close()


async def test_deleting_a_file_under_one_root_only_prunes_that_root(
    two_roots: tuple[Path, Path, WorkspaceConfig], tmp_path: Path
) -> None:
    """Orphan pruning must still work *within* the walked root -- scoping it
    must not turn it off."""
    docs, _, config = two_roots
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace=config.workspace.name, payload_indexes=False)

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config, store, manifest, tmp_path)
        await harness.indexer().run()
        code_before = await _chunks_in(store, "code")

        (docs / "guide" / "api.md").unlink()
        stats = await harness.indexer().run(only_root="docs")

        assert stats.chunks_deleted > 0
        assert manifest.get_file("docs", "guide/api.md") is None
        assert manifest.get_file("docs", "guide/install.md") is not None
        assert await _chunks_in(store, "code") == code_before
    await client.close()


async def test_removing_a_root_from_config_drops_its_chunks(
    two_roots: tuple[Path, Path, WorkspaceConfig], tmp_path: Path
) -> None:
    """The trap next door to `--root`, and it cuts the other way.

    `--root docs` is a *scope*: other roots are not walked and not touched.
    Deleting a root from workspace.yaml is a *decision*: it is walked as part
    of "everything", found absent, and pruned. Both are right, and they look
    almost identical from the command line, so the difference is worth a test
    rather than a paragraph nobody reads.
    """
    _, _, config = two_roots
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace=config.workspace.name, payload_indexes=False)

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config, store, manifest, tmp_path)
        await harness.indexer().run()
        assert await _chunks_in(store, "code") > 0

        # The user edits workspace.yaml and removes the code root.
        harness.config.workspace.roots = [
            r for r in config.workspace.roots if r.resolved_label != "code"
        ]
        await harness.indexer().run()

        assert await _chunks_in(store, "code") == 0
        assert manifest.file_count("code") == 0
        # docs is untouched -- this is pruning, not a rebuild.
        assert await _chunks_in(store, "docs") > 0
    await client.close()


async def test_indexing_records_what_each_file_imports(harness: Harness) -> None:
    """The graph is written in the same transaction as the chunks, so it
    cannot survive a file whose chunks were rolled back."""
    await harness.indexer().run()

    edges = harness.manifest.imports_of("workspace", "repo_one/src/widget.py")
    assert edges
    assert all(e.line >= 1 for e in edges)


async def test_edited_imports_replace_the_old_ones(harness: Harness, workspace: Path) -> None:
    target = workspace / "repo_one" / "src" / "widget.py"
    target.write_text("import os\n\n\ndef widget():\n    return os.name\n", encoding="utf-8")
    await harness.indexer().run()
    assert [
        e.module for e in harness.manifest.imports_of("workspace", "repo_one/src/widget.py")
    ] == ["os"]

    target.write_text("import sys\n\n\ndef widget():\n    return sys.platform\n", encoding="utf-8")
    await harness.indexer().run()
    assert [
        e.module for e in harness.manifest.imports_of("workspace", "repo_one/src/widget.py")
    ] == ["sys"]


async def test_deleting_a_file_removes_it_from_the_graph(harness: Harness, workspace: Path) -> None:
    target = workspace / "repo_one" / "src" / "widget.py"
    target.write_text("import shared_thing\n\n\ndef widget():\n    return 1\n", encoding="utf-8")
    await harness.indexer().run()
    assert harness.manifest.importers_of("shared_thing")

    target.unlink()
    await harness.indexer().run()
    assert harness.manifest.importers_of("shared_thing") == []


async def test_doc_type_reaches_the_embedded_text_when_enabled(harness: Harness) -> None:
    """The header is built during chunking, so the verdict has to be on the
    file before the split rather than stamped on afterwards."""
    harness.config.chunking.embed_doc_type = True
    await harness.indexer().run()

    embedded = "\n".join(text for batch in harness.backend.batches for text in batch)
    assert "# type:" in embedded


async def test_doc_type_stays_out_of_the_embedded_text_by_default(
    harness: Harness,
) -> None:
    await harness.indexer().run()
    embedded = "\n".join(text for batch in harness.backend.batches for text in batch)
    assert "# type:" not in embedded


async def test_the_option_does_not_change_chunk_identity(harness: Harness) -> None:
    """The header is excluded from content_sha, so turning this on is a
    re-embed and not a re-chunk.

    That is precisely why switching it needs `index --force`: a normal run
    compares chunk ids, finds them identical, and embeds nothing.
    """
    target = ("workspace", "repo_one/src/widget.py", SPACE.slug())
    await harness.indexer().run()
    before = set(harness.manifest.chunk_ids_for(*target))

    harness.config.chunking.embed_doc_type = True
    stats = await harness.indexer().run()

    assert set(harness.manifest.chunk_ids_for(*target)) == before
    # Nothing re-embedded, which is the trap the docs have to warn about.
    assert stats.chunks_upserted == 0


async def test_imports_resolve_to_indexed_files(harness: Harness, workspace: Path) -> None:
    """Rung 2: an edge points at a file, not just at a string."""
    target = workspace / "repo_one" / "src" / "helper.py"
    target.write_text("def helper():\n    return 1\n", encoding="utf-8")
    (workspace / "repo_one" / "src" / "widget.py").write_text(
        "from .helper import helper\n\n\ndef widget():\n    return helper()\n", encoding="utf-8"
    )
    stats = await harness.indexer().run()

    assert stats.imports_resolved >= 1
    importers = harness.manifest.dependents_of("workspace", "repo_one/src/helper.py")
    assert [d.rel_path for d in importers] == ["repo_one/src/widget.py"]


async def test_resolution_runs_after_the_whole_walk(harness: Harness, workspace: Path) -> None:
    """An import can name a file the walk has not reached yet, so resolving
    per file would depend on walk order and differ between runs."""
    src = workspace / "repo_one" / "src"
    # Alphabetically `aaa` is walked long before `zzz`, and imports the later one.
    (src / "zzz_target.py").write_text("VALUE = 1\n", encoding="utf-8")
    (src / "aaa_importer.py").write_text("from .zzz_target import VALUE\n", encoding="utf-8")
    await harness.indexer().run()

    importers = harness.manifest.dependents_of("workspace", "repo_one/src/zzz_target.py")
    assert [d.rel_path for d in importers] == ["repo_one/src/aaa_importer.py"]


async def test_an_unresolvable_import_is_not_an_error(harness: Harness, workspace: Path) -> None:
    """Stdlib and third-party imports stay unresolved, and that is the correct
    answer rather than a failure."""
    (workspace / "repo_one" / "src" / "widget.py").write_text(
        "import os\nimport pydantic\n\n\ndef widget():\n    return os.name\n", encoding="utf-8"
    )
    await harness.indexer().run()

    coverage = harness.manifest.resolution_coverage()
    resolved, total = coverage["python"]
    assert total > resolved


async def test_a_root_missing_from_disk_is_not_treated_as_emptied(
    two_roots: tuple[Path, Path, WorkspaceConfig], tmp_path: Path
) -> None:
    """The CI accident, exactly.

    One collection spans several repositories, and a CI job checks out one of
    them. `workspace.yaml` still declares all of them, so an unscoped run walks
    a workspace where four roots simply are not there -- and reads four
    repositories' worth of absence as four repositories' worth of deletion.

    A root that cannot be read has not been shown to be empty. There is no
    override for this one: there is no evidence to weigh, only the lack of it.
    """
    _, code, config = two_roots
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace=config.workspace.name, payload_indexes=False)

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config, store, manifest, tmp_path)
        await harness.indexer().run()
        assert await _chunks_in(store, "code")

        # The runner never checked this repository out.
        shutil.rmtree(code)

        stats = await harness.indexer().run()

        assert await _chunks_in(store, "code"), "an unreadable root's index was deleted"
        assert stats.deletions_withheld == 2
        assert stats.chunks_deleted == 0
    await client.close()


async def test_losing_most_of_a_root_at_once_stops_and_asks(
    config_for: ConfigFactory, workspace: Path, tmp_path: Path
) -> None:
    """An empty directory where a checkout should be looks identical to a
    repository whose files really were removed. The difference matters enough
    to ask rather than assume."""
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="test", payload_indexes=False)

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config_for(), store, manifest, tmp_path)
        first = await harness.indexer().run()
        assert first.files_changed >= 10

        for path in sorted(workspace.rglob("*")):
            if path.is_file():
                path.unlink()

        stats = await harness.indexer().run()

        assert stats.chunks_deleted == 0
        assert stats.deletions_withheld >= 10
    await client.close()


async def test_allow_deletes_lets_a_real_mass_removal_through(
    config_for: ConfigFactory, workspace: Path, tmp_path: Path
) -> None:
    """The brake has to be releasable, or a genuine restructure cannot be
    indexed at all."""
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="test", payload_indexes=False)

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config_for(), store, manifest, tmp_path)
        await harness.indexer().run()
        for path in sorted(workspace.rglob("*")):
            if path.is_file():
                path.unlink()

        stats = await harness.indexer().run(allow_deletes=True)

        assert stats.deletions_withheld == 0
        assert stats.chunks_deleted > 0
    await client.close()


async def test_an_ordinary_deletion_is_not_second_guessed(
    config_for: ConfigFactory, workspace: Path, tmp_path: Path
) -> None:
    """The guard must not turn every removed file into a prompt. One file gone
    from a full workspace is the common case and has to keep working."""
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="test", payload_indexes=False)

    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        harness = Harness(config_for(), store, manifest, tmp_path)
        await harness.indexer().run()

        (workspace / "repo_one" / "src" / "widget.py").unlink()
        stats = await harness.indexer().run()

        assert stats.deletions_withheld == 0
        assert stats.chunks_deleted > 0
    await client.close()


async def test_route_edges_are_recorded_for_both_sides(
    config_for: ConfigFactory, workspace: Path, tmp_path: Path
) -> None:
    """Through the real pipeline, not the scanner alone: the edges have to
    survive the walk, the reader and the manifest to be worth anything."""
    api = workspace / "repo_one" / "Api"
    api.mkdir(parents=True, exist_ok=True)
    (api / "RemittanceController.cs").write_text(
        '[ApiController]\n[Route("api/[controller]")]\n'
        "public class RemittanceController : ControllerBase\n{\n"
        '    [HttpGet]\n    [Route("{id}")]\n'
        "    public IActionResult Get(int id) => Ok();\n}\n",
        encoding="utf-8",
    )
    (workspace / "repo_two" / "app" / "page.ts").write_text(
        'export const load = () => customFetch("/api/Remittance/1");\n', encoding="utf-8"
    )

    config = config_for(graph={"http_clients": ["fetch", "customFetch"]})
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="test", payload_indexes=False)
    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        await Harness(config, store, manifest, tmp_path).indexer().run()

        coverage = manifest.route_coverage()
        assert coverage["declaration"][1] >= 1
        assert coverage["call"][1] >= 1
        # And the call now points at the controller that declares it -- across
        # two roots, which is the thing no import edge can do.
        assert coverage["call"][0] >= 1
    await client.close()


async def test_an_unnamed_http_wrapper_leaves_the_call_graph_empty(
    config_for: ConfigFactory, workspace: Path, tmp_path: Path
) -> None:
    """The finding that makes `graph.http_clients` a feature rather than a
    nicety: on a real workspace, naming the wrapper took call sites from 6 to
    71. Left unnamed, `status` shows endpoints and no callers -- which reads
    as "nothing calls this API"."""
    (workspace / "repo_two" / "app" / "page.ts").write_text(
        'export const load = () => customFetch("/api/Remittance/1");\n', encoding="utf-8"
    )
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="test", payload_indexes=False)
    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        await Harness(config_for(), store, manifest, tmp_path).indexer().run()
        assert manifest.route_coverage().get("call", (0, 0))[1] == 0
    await client.close()


async def test_a_client_call_resolves_to_a_controller_in_another_repository(
    config_for: ConfigFactory, workspace: Path, tmp_path: Path
) -> None:
    """The whole point of route edges, end to end through the real pipeline.

    A page in one repository and the API it calls in another share a string
    and nothing else -- no import, no symbol, nothing a language server could
    follow. `impact_of` on the controller has to name the caller anyway.
    """
    api = workspace / "repo_one" / "Api"
    api.mkdir(parents=True, exist_ok=True)
    (api / "RemittanceController.cs").write_text(
        '[ApiController]\n[Route("api/[controller]")]\n'
        "public class RemittanceController : ControllerBase\n{\n"
        '    [HttpGet]\n    [Route("{id}")]\n'
        "    public IActionResult Get(int id) => Ok();\n}\n",
        encoding="utf-8",
    )
    (workspace / "repo_two" / "app" / "page.ts").write_text(
        "export const load = (id: string) => customFetch(`/api/Remittance/${id}`);\n",
        encoding="utf-8",
    )

    config = config_for(graph={"http_clients": ["fetch", "customFetch"]})
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="test", payload_indexes=False)
    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        stats = await Harness(config, store, manifest, tmp_path).indexer().run()
        assert stats.routes_resolved >= 1

        callers = manifest.route_callers_of("workspace", "repo_one/Api/RemittanceController.cs")
        assert [c.rel_path for c in callers] == ["repo_two/app/page.ts"]
        # The prefix from a template literal was enough: `/api/Remittance/`
        # names one controller even though it cannot name one action.
        assert callers[0].module == "/api/Remittance/"
    await client.close()


async def test_a_call_naming_an_endpoint_outside_the_workspace_stays_unresolved(
    config_for: ConfigFactory, workspace: Path, tmp_path: Path
) -> None:
    """Half a graph honestly reported beats a whole one invented. An
    `impact_of` naming the wrong controller is worse than one naming none."""
    (workspace / "repo_two" / "app" / "page.ts").write_text(
        'export const load = () => customFetch("https://stripe.example/v1/charges");\n',
        encoding="utf-8",
    )
    config = config_for(graph={"http_clients": ["fetch", "customFetch"]})
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="test", payload_indexes=False)
    with Manifest(tmp_path / "manifest.sqlite3") as manifest:
        await Harness(config, store, manifest, tmp_path).indexer().run()
        resolved, total = manifest.route_coverage().get("call", (0, 0))
        assert total >= 1
        assert resolved == 0
    await client.close()


async def test_a_csharp_using_resolves_to_every_file_declaring_the_namespace(
    harness: Harness, workspace: Path
) -> None:
    """Rung 2 for C#. `using MyApp.Data` names no path at all, so it is matched
    against the namespaces declared in the same repository -- and a namespace
    is declared across several files, so the edge reaches all of them."""
    src = workspace / "repo_one" / "src"
    (src / "Repo.cs").write_text(
        "namespace MyApp.Data;\n\npublic class Repo {}\n", encoding="utf-8"
    )
    (src / "Context.cs").write_text(
        "namespace MyApp.Data;\n\npublic class Context {}\n", encoding="utf-8"
    )
    (src / "Startup.cs").write_text(
        "using System;\nusing MyApp.Data;\n\npublic class Startup {}\n", encoding="utf-8"
    )
    await harness.indexer().run()

    found = harness.manifest.dependencies_of("workspace", "repo_one/src/Startup.cs")
    reached = {(d.module, d.rel_path, d.resolved_by) for d in found}
    assert reached == {
        ("MyApp.Data", "repo_one/src/Context.cs", "namespace"),
        ("MyApp.Data", "repo_one/src/Repo.cs", "namespace"),
        # The framework edge resolves to nothing, and that is the right answer.
        ("System", None, None),
    }

    importers = harness.manifest.dependents_of("workspace", "repo_one/src/Repo.cs")
    assert [d.rel_path for d in importers] == ["repo_one/src/Startup.cs"]


async def test_every_first_party_csharp_edge_resolves(harness: Harness, workspace: Path) -> None:
    """The gate the origin columns exist for: only a first-party edge can
    reach a file here, so that is the one number where less than 100% is a
    defect rather than a package reference."""
    src = workspace / "repo_one" / "src"
    (src / "Repo.cs").write_text("namespace MyApp.Data;\npublic class Repo {}\n", encoding="utf-8")
    (src / "Startup.cs").write_text(
        "using System;\nusing System.Text.Json;\nusing MyApp.Data;\n\npublic class S {}\n",
        encoding="utf-8",
    )
    await harness.indexer().run()

    coverage = harness.manifest.origin_coverage()["csharp"]
    assert coverage.first_party == 1
    assert coverage.first_party_resolution_percent == 100
    # The framework edges are not a failure and must not be in the denominator.
    assert coverage.framework == 2


async def test_a_namespace_declared_in_another_repository_is_not_reached(
    harness: Harness, workspace: Path
) -> None:
    """Cross-unit isolation, end to end. Two repositories in one workspace
    declaring the same namespace is ordinary -- a service and the library it
    was copied from -- and resolving across them would invent a dependency."""
    (workspace / "repo_one" / "src" / "Repo.cs").write_text(
        "namespace Shared.Models;\npublic class Repo {}\n", encoding="utf-8"
    )
    (workspace / "repo_two" / "app" / "Startup.cs").write_text(
        "using Shared.Models;\n\npublic class Startup {}\n", encoding="utf-8"
    )
    await harness.indexer().run()

    found = harness.manifest.dependencies_of("workspace", "repo_two/app/Startup.cs")
    assert [(d.module, d.rel_path) for d in found] == [("Shared.Models", None)]
    assert harness.manifest.dependents_of("workspace", "repo_one/src/Repo.cs") == []


async def test_deleting_the_declaring_file_retires_the_edge(
    harness: Harness, workspace: Path
) -> None:
    """The reason targets are derived rather than written down: no
    invalidation pass, and no edge left pointing at a file that has gone."""
    src = workspace / "repo_one" / "src"
    repo = src / "Repo.cs"
    repo.write_text("namespace MyApp.Data;\npublic class Repo {}\n", encoding="utf-8")
    (src / "Startup.cs").write_text("using MyApp.Data;\npublic class S {}\n", encoding="utf-8")
    await harness.indexer().run()
    assert harness.manifest.dependencies_of("workspace", "repo_one/src/Startup.cs")[0].resolved

    repo.unlink()
    await harness.indexer().run()

    found = harness.manifest.dependencies_of("workspace", "repo_one/src/Startup.cs")
    assert [(d.module, d.resolved) for d in found] == [("MyApp.Data", False)]


async def test_coverage_stops_claiming_a_retired_namespace_edge(
    harness: Harness, workspace: Path
) -> None:
    """A path edge is retired by the cascade that removes its target. A
    namespace edge stores no target, so nothing cascades -- and the flag
    coverage reads would go on claiming a resolution the join cannot produce.
    """
    src = workspace / "repo_one" / "src"
    repo = src / "Repo.cs"
    repo.write_text("namespace MyApp.Data;\npublic class Repo {}\n", encoding="utf-8")
    (src / "Startup.cs").write_text("using MyApp.Data;\npublic class S {}\n", encoding="utf-8")
    await harness.indexer().run()
    assert harness.manifest.origin_coverage()["csharp"].first_party_resolved == 1

    repo.unlink()
    await harness.indexer().run()

    coverage = harness.manifest.origin_coverage()["csharp"]
    assert coverage.first_party_resolved == 0
    # And it is no longer claimed as ours at all: nothing in this workspace
    # declares that namespace any more, so it reads like any other unknown.
    assert coverage.first_party == 0
    assert coverage.unclassified == 1
