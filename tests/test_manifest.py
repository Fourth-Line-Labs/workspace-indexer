"""The manifest and its decision ladder.

This is where the incremental-reindex guarantee lives, so the tests are about
cost as much as correctness: each rung has to be reachable, and the expensive
rungs must not be reached when a cheaper one would answer.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.conftest import make_source
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.classification import Classification
from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.graph import ImportEdge
from workspace_indexer.models import Chunk, DocumentType, FileKind, RunStats, SourceFile
from workspace_indexer.state.index_decision import IndexDecision
from workspace_indexer.state.manifest import Manifest

SPACE = "voyageai_voyage-code-4_2048"
OTHER_SPACE = "voyageai_voyage-code-4_1024"
VERSION = 1


@pytest.fixture
def manifest(tmp_path: Path) -> Iterator[Manifest]:
    with Manifest(tmp_path / "manifest.sqlite3") as m:
        yield m


def _candidate(
    rel_path: str = "src/widget.py", *, mtime_ns: int = 1000, size: int = 200
) -> FileCandidate:
    return FileCandidate(
        root_label="repo_one",
        unit="repo_one",
        abs_path=Path("/tmp") / rel_path,
        rel_path=rel_path,
        kind=FileKind.CODE,
        language="python",
        size=size,
        mtime_ns=mtime_ns,
    )


def _source(
    rel_path: str = "src/widget.py", text: str = "x = 1", *, mtime_ns: int = 1000
) -> SourceFile:
    source = make_source(text, rel_path=rel_path)
    return source.model_copy(update={"mtime_ns": mtime_ns, "size": len(text.encode())})


def _chunk(source: SourceFile, body: str, index: int = 0) -> Chunk:
    return build_chunk(
        source,
        "labbox",
        source_text=body,
        start_line=1,
        end_line=1,
        chunker="code",
        version=VERSION,
        chunk_index=index,
        symbol_path=f"sym{index}",
    )


def _register(manifest: Manifest, source: SourceFile, chunks: list[Chunk]) -> None:
    """Record a file as fully indexed, the way the pipeline will."""
    manifest.record_file(source, chunker="code", chunker_version=VERSION)
    manifest.record_chunks(chunks, SPACE)
    manifest.record_space(source.root_label, source.rel_path, SPACE, len(chunks))


def _decide(manifest: Manifest, candidate: FileCandidate, *, force: bool = False) -> IndexDecision:
    return manifest.decide_from_stat(
        candidate, space_slug=SPACE, chunker_version=VERSION, force=force
    )


# ---- the ladder --------------------------------------------------------


def test_unknown_file_is_new(manifest: Manifest) -> None:
    assert _decide(manifest, _candidate()) is IndexDecision.NEW


def test_unchanged_file_skips_without_a_read(manifest: Manifest) -> None:
    """Rung 1, the common case on every rerun: one stat(), zero reads."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = _decide(manifest, _candidate(mtime_ns=source.mtime_ns, size=source.size))
    assert decision is IndexDecision.SKIP_UNCHANGED
    assert not decision.needs_read


def test_changed_mtime_requires_a_read(manifest: Manifest) -> None:
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = _decide(manifest, _candidate(mtime_ns=9999, size=source.size))
    assert decision is IndexDecision.REINDEX
    assert decision.needs_read


def test_changed_size_alone_requires_a_read(manifest: Manifest) -> None:
    """Some tools preserve mtime while rewriting content, so size is checked
    too rather than trusted away."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = _decide(manifest, _candidate(mtime_ns=source.mtime_ns, size=source.size + 1))
    assert decision is IndexDecision.REINDEX


def test_identical_content_after_a_touch_costs_no_embedding(manifest: Manifest) -> None:
    """Rung 2: a formatter pass or a `git checkout` of the same bytes."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    rewritten = _source(mtime_ns=9999)
    decision = manifest.decide_from_hash(rewritten)
    assert decision is IndexDecision.SKIP_SAME_CONTENT
    assert not decision.needs_embedding


def test_different_content_reindexes(manifest: Manifest) -> None:
    source = _source(text="x = 1")
    _register(manifest, source, [_chunk(source, "body")])
    decision = manifest.decide_from_hash(_source(text="x = 2", mtime_ns=9999))
    assert decision is IndexDecision.REINDEX
    assert decision.needs_embedding


def test_hash_decision_on_an_unknown_file_is_new(manifest: Manifest) -> None:
    assert manifest.decide_from_hash(_source("never/seen.py")) is IndexDecision.NEW


def test_touch_makes_the_next_run_take_the_cheap_path(manifest: Manifest) -> None:
    """After rung 2 decides the content is identical, the new mtime has to be
    recorded or every subsequent run pays the read again."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    rewritten = _source(mtime_ns=9999)
    manifest.touch(rewritten)
    assert _decide(manifest, _candidate(mtime_ns=9999, size=rewritten.size)) is (
        IndexDecision.SKIP_UNCHANGED
    )


def test_chunker_version_bump_forces_a_rechunk(manifest: Manifest) -> None:
    """Rung 4: we changed a strategy, so content hashes say nothing useful."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = manifest.decide_from_stat(
        _candidate(mtime_ns=source.mtime_ns, size=source.size),
        space_slug=SPACE,
        chunker_version=VERSION + 1,
    )
    assert decision is IndexDecision.RECHUNK_STRATEGY
    assert decision.needs_embedding


def test_new_space_backfills_an_unchanged_file(manifest: Manifest) -> None:
    """Rung 6, the model-swap path: change EMBEDDING_MODEL and the next run
    fills a new collection without discarding the old one."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = manifest.decide_from_stat(
        _candidate(mtime_ns=source.mtime_ns, size=source.size),
        space_slug=OTHER_SPACE,
        chunker_version=VERSION,
    )
    assert decision is IndexDecision.BACKFILL_SPACE


def test_force_skips_straight_past_every_shortcut(manifest: Manifest) -> None:
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    decision = _decide(manifest, _candidate(mtime_ns=source.mtime_ns, size=source.size), force=True)
    assert decision is IndexDecision.FORCED
    assert decision.needs_embedding


def test_a_file_that_produces_no_chunks_is_not_retried_forever(manifest: Manifest) -> None:
    """The bug this table exists to prevent: an image legitimately produces
    zero chunks, and without recording that it looks identical to a file whose
    chunks are missing — so every run would re-chunk every binary."""
    source = make_source("", kind=FileKind.IMAGE, language=None, rel_path="docs/logo.png")
    manifest.record_file(source, chunker="opaque", chunker_version=VERSION)
    manifest.record_chunks([], SPACE)
    manifest.record_space(source.root_label, source.rel_path, SPACE, 0)

    decision = manifest.decide_from_stat(
        _candidate("docs/logo.png", mtime_ns=source.mtime_ns, size=source.size),
        space_slug=SPACE,
        chunker_version=VERSION,
    )
    assert decision is IndexDecision.SKIP_UNCHANGED


# ---- chunk-level diffing ----------------------------------------------


def test_editing_one_function_touches_one_chunk(manifest: Manifest) -> None:
    """The heart of the cost story: forty functions in a file, one edited, one
    chunk re-embedded."""
    source = _source()
    original = [_chunk(source, f"body {i}", i) for i in range(40)]
    _register(manifest, source, original)

    edited = list(original)
    edited[7] = _chunk(source, "body 7 CHANGED", 7)
    delta = manifest.diff_chunks(
        source.root_label, source.rel_path, SPACE, [c.chunk_id for c in edited]
    )
    assert len(delta.to_upsert) == 1
    assert len(delta.to_delete) == 1
    assert len(delta.unchanged) == 39


def test_unchanged_file_diffs_to_nothing(manifest: Manifest) -> None:
    source = _source()
    chunks = [_chunk(source, f"body {i}", i) for i in range(5)]
    _register(manifest, source, chunks)
    delta = manifest.diff_chunks(
        source.root_label, source.rel_path, SPACE, [c.chunk_id for c in chunks]
    )
    assert delta.is_noop


def test_shrinking_a_file_deletes_the_surplus(manifest: Manifest) -> None:
    source = _source()
    chunks = [_chunk(source, f"body {i}", i) for i in range(5)]
    _register(manifest, source, chunks)
    delta = manifest.diff_chunks(
        source.root_label, source.rel_path, SPACE, [c.chunk_id for c in chunks[:2]]
    )
    assert delta.to_upsert == []
    assert len(delta.to_delete) == 3


def test_diff_is_scoped_to_the_space(manifest: Manifest) -> None:
    """A chunk present in the old space must still count as new in a fresh one,
    or a backfill would write nothing."""
    source = _source()
    chunks = [_chunk(source, "body", 0)]
    _register(manifest, source, chunks)
    delta = manifest.diff_chunks(
        source.root_label, source.rel_path, OTHER_SPACE, [c.chunk_id for c in chunks]
    )
    assert delta.to_upsert == [chunks[0].chunk_id]


def test_duplicate_produced_ids_are_collapsed(manifest: Manifest) -> None:
    """Two identical chunks in one file share an id; upserting it twice would
    double-count the cost."""
    source = _source()
    chunk = _chunk(source, "body", 0)
    delta = manifest.diff_chunks(
        source.root_label, source.rel_path, SPACE, [chunk.chunk_id, chunk.chunk_id]
    )
    assert delta.to_upsert == [chunk.chunk_id]


def test_forget_chunks_is_space_scoped(manifest: Manifest) -> None:
    source = _source()
    chunk = _chunk(source, "body", 0)
    _register(manifest, source, [chunk])
    manifest.record_chunks([chunk], OTHER_SPACE)
    manifest.forget_chunks([chunk.chunk_id], SPACE)
    assert manifest.chunk_count(SPACE) == 0
    assert manifest.chunk_count(OTHER_SPACE) == 1


# ---- deletion and rename ----------------------------------------------


def test_deleted_file_is_reported_as_an_orphan(manifest: Manifest) -> None:
    """Rung 5: a row with no file on disk."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    assert manifest.orphans(seen=set()) == [("repo_one", "src/widget.py")]


def test_a_file_still_on_disk_is_not_an_orphan(manifest: Manifest) -> None:
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    assert manifest.orphans(seen={("repo_one", "src/widget.py")}) == []


def test_orphan_detection_is_scoped_to_the_root_that_was_walked(manifest: Manifest) -> None:
    """Indexing one root must not delete every other root's index."""
    a = _source("a.py")
    b = _source("b.py").model_copy(update={"root_label": "repo_two"})
    _register(manifest, a, [_chunk(a, "body")])
    _register(manifest, b, [_chunk(b, "body")])
    assert manifest.orphans(seen=set(), root_label="repo_one") == [("repo_one", "a.py")]


def test_forgetting_a_file_cascades_to_its_chunks(manifest: Manifest) -> None:
    source = _source()
    _register(manifest, source, [_chunk(source, f"body {i}", i) for i in range(3)])
    assert manifest.chunk_count(SPACE) == 3
    manifest.forget_file(source.root_label, source.rel_path)
    assert manifest.chunk_count(SPACE) == 0
    assert manifest.file_count() == 0


def test_rename_removes_the_old_path_and_adds_the_new(manifest: Manifest) -> None:
    old = _source("src/old.py")
    _register(manifest, old, [_chunk(old, "body")])
    old_ids = manifest.chunk_ids_for(old.root_label, old.rel_path, SPACE)

    new = _source("src/new.py")
    _register(manifest, new, [_chunk(new, "body")])
    manifest.forget_file(old.root_label, old.rel_path)

    assert manifest.chunk_ids_for(old.root_label, old.rel_path, SPACE) == []
    new_ids = manifest.chunk_ids_for(new.root_label, new.rel_path, SPACE)
    assert new_ids and new_ids != old_ids


def test_chunk_ids_for_a_missing_file_is_empty_not_an_error(manifest: Manifest) -> None:
    assert manifest.chunk_ids_for("repo_one", "nope.py", SPACE) == []


# ---- reads and history ------------------------------------------------


def test_record_file_is_idempotent(manifest: Manifest) -> None:
    source = _source()
    for _ in range(3):
        manifest.record_file(source, chunker="code", chunker_version=VERSION)
    assert manifest.file_count() == 1


def test_get_file_round_trips(manifest: Manifest) -> None:
    source = _source()
    manifest.record_file(source, chunker="code", chunker_version=VERSION)
    record = manifest.get_file(source.root_label, source.rel_path)
    assert record is not None
    assert record.sha256 == source.sha256
    assert record.chunker == "code"
    assert record.language == "python"


def test_get_missing_file_is_none(manifest: Manifest) -> None:
    assert manifest.get_file("repo_one", "nope.py") is None


def test_counts_by_root_and_space(manifest: Manifest) -> None:
    a = _source("a.py")
    b = _source("b.py").model_copy(update={"root_label": "repo_two"})
    _register(manifest, a, [_chunk(a, "body")])
    _register(manifest, b, [_chunk(b, "body"), _chunk(b, "other", 1)])
    assert manifest.counts_by_root() == {"repo_one": 1, "repo_two": 1}
    assert manifest.chunk_count(SPACE) == 3
    assert manifest.chunk_count() == 3


def test_spaces_are_listed(manifest: Manifest) -> None:
    """`status` reports which collections exist, including after a model swap."""
    source = _source()
    chunk = _chunk(source, "body")
    _register(manifest, source, [chunk])
    manifest.record_chunks([chunk], OTHER_SPACE)
    assert manifest.spaces() == sorted([SPACE, OTHER_SPACE])


def test_run_history_records_cost(manifest: Manifest) -> None:
    """So "why did this cost $40" is answerable from the manifest rather than
    by scraping logs."""
    stats = RunStats(
        run_id="run1", started_at=datetime.now(UTC), mode="index", config_hash="abc123"
    )
    manifest.start_run(stats)
    stats.finished_at = datetime.now(UTC)
    stats.files_seen = 100
    stats.chunks_upserted = 42
    stats.tokens_embedded = 5000
    stats.est_cost_usd = 0.12
    manifest.finish_run(stats)

    runs = manifest.recent_runs()
    assert len(runs) == 1
    assert runs[0].chunks_upserted == 42
    assert runs[0].est_cost_usd == pytest.approx(0.12)
    assert runs[0].config_hash == "abc123"
    assert runs[0].finished_at is not None


def test_an_interrupted_run_stays_unfinished(manifest: Manifest) -> None:
    """A crashed run should be visible as such, not silently absent."""
    manifest.start_run(RunStats(run_id="run1", started_at=datetime.now(UTC)))
    assert manifest.recent_runs()[0].unfinished


# ---- durability -------------------------------------------------------


def test_state_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "manifest.sqlite3"
    source = _source()
    with Manifest(path) as first:
        _register(first, source, [_chunk(source, "body")])
    with Manifest(path) as second:
        assert second.file_count() == 1
        assert second.chunk_count(SPACE) == 1


def test_wal_mode_is_enabled(tmp_path: Path) -> None:
    """So an MCP server can read while the indexer writes; without WAL SQLite
    takes a global write lock."""
    import sqlite3

    path = tmp_path / "manifest.sqlite3"
    with Manifest(path):
        pass
    db = sqlite3.connect(path)
    mode = db.execute("PRAGMA journal_mode").fetchone()[0]
    db.close()
    assert mode.lower() == "wal"


def test_parent_directory_is_created(tmp_path: Path) -> None:
    """Otherwise the failure lands at the first write, deep into a run."""
    with Manifest(tmp_path / "nested" / "deeper" / "manifest.sqlite3") as m:
        assert m.file_count() == 0


def test_schema_file_ships_with_the_package() -> None:
    """It is read at runtime relative to the module, so a packaging change that
    drops non-Python files would break every fresh install."""
    from workspace_indexer.state import manifest as module

    schema = Path(module.__file__).with_name("schema.sql")
    assert schema.is_file()
    assert "CREATE TABLE IF NOT EXISTS files" in schema.read_text(encoding="utf-8")


def test_transaction_can_be_rolled_back(manifest: Manifest) -> None:
    """Batched writes must not leave a half-indexed file behind on failure."""
    source = _source()
    manifest.begin()
    manifest.record_file(source, chunker="code", chunker_version=VERSION)
    manifest.rollback()
    assert manifest.file_count() == 0


def test_transaction_commits(manifest: Manifest) -> None:
    source = _source()
    manifest.begin()
    manifest.record_file(source, chunker="code", chunker_version=VERSION)
    manifest.commit()
    assert manifest.file_count() == 1


@pytest.mark.integration
def test_schema_sql_is_included_in_a_built_wheel(tmp_path: Path) -> None:
    """schema.sql is read at runtime, so a packaging change that drops
    non-Python files would break every fresh install while leaving the editable
    dev install working. Marked integration because it builds a wheel.
    """
    import subprocess
    import sys
    import zipfile

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "wheel"
    # sys.executable, not "python": the interpreter running the tests is not
    # guaranteed to be on PATH under that name.
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(root), "-w", str(out), "--no-deps", "-q"],
        check=True,
        capture_output=True,
    )
    wheels = list(out.glob("*.whl"))
    assert wheels, "no wheel produced"
    names = zipfile.ZipFile(wheels[0]).namelist()
    assert "workspace_indexer/state/schema.sql" in names


def test_forget_space_removes_only_that_space(manifest: Manifest) -> None:
    """Deleting a collection must clear the manifest too, or status reports
    collections that no longer exist and the backfill rung believes those files
    are already complete."""
    source = _source()
    chunk = _chunk(source, "body")
    _register(manifest, source, [chunk])
    manifest.record_chunks([chunk], OTHER_SPACE)
    manifest.record_space(source.root_label, source.rel_path, OTHER_SPACE, 1)

    removed = manifest.forget_space(SPACE)
    assert removed == 1
    assert manifest.spaces() == [OTHER_SPACE]
    assert manifest.chunk_count(SPACE) == 0
    assert manifest.chunk_count(OTHER_SPACE) == 1


def test_forget_space_keeps_the_files(manifest: Manifest) -> None:
    """A file may well be indexed in another space; only its chunks go."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    manifest.forget_space(SPACE)
    assert manifest.get_file(source.root_label, source.rel_path) is not None


def test_forgotten_space_is_reindexed_rather_than_skipped(manifest: Manifest) -> None:
    """The point of clearing file_spaces: the next run must backfill, not
    decide the file is already done."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    manifest.forget_space(SPACE)
    decision = _decide(manifest, _candidate(mtime_ns=source.mtime_ns, size=source.size))
    assert decision is IndexDecision.BACKFILL_SPACE


def test_forget_an_unknown_space_is_a_no_op(manifest: Manifest) -> None:
    assert manifest.forget_space("never-existed") == 0


def test_copy_space_duplicates_the_rows_under_a_new_slug(manifest: Manifest) -> None:
    """Reprojection derives a narrower collection from vectors already paid
    for, preserving chunk ids. The manifest has to learn about the new space or
    a later index cannot tell which chunks exist."""
    source = _source()
    chunks = [_chunk(source, f"body {i}", i) for i in range(4)]
    _register(manifest, source, chunks)

    copied = manifest.copy_space(SPACE, OTHER_SPACE)
    assert copied == 4
    assert manifest.chunk_count(SPACE) == 4
    assert manifest.chunk_count(OTHER_SPACE) == 4
    assert sorted(manifest.spaces()) == sorted([SPACE, OTHER_SPACE])


def test_copy_space_preserves_chunk_ids(manifest: Manifest) -> None:
    """Ids are deliberately identical across spaces; that is what lets a
    reprojection reuse them."""
    source = _source()
    chunks = [_chunk(source, "body")]
    _register(manifest, source, chunks)
    manifest.copy_space(SPACE, OTHER_SPACE)
    assert manifest.chunk_ids_for(
        source.root_label, source.rel_path, OTHER_SPACE
    ) == manifest.chunk_ids_for(source.root_label, source.rel_path, SPACE)


def test_copy_space_marks_the_target_complete(manifest: Manifest) -> None:
    """Without the file_spaces rows the next run treats every file as needing
    a backfill and re-embeds a collection that is already full."""
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    manifest.copy_space(SPACE, OTHER_SPACE)
    decision = manifest.decide_from_stat(
        _candidate(mtime_ns=source.mtime_ns, size=source.size),
        space_slug=OTHER_SPACE,
        chunker_version=VERSION,
    )
    assert decision is IndexDecision.SKIP_UNCHANGED


def test_copy_space_is_idempotent(manifest: Manifest) -> None:
    source = _source()
    _register(manifest, source, [_chunk(source, "body")])
    manifest.copy_space(SPACE, OTHER_SPACE)
    assert manifest.copy_space(SPACE, OTHER_SPACE) == 1


def test_copying_an_untracked_space_copies_nothing(manifest: Manifest) -> None:
    """Reprojector warns on this: the source was never tracked, so the target
    cannot be either, and the derived collection would be invisible to orphan
    cleanup."""
    assert manifest.copy_space("never-existed", OTHER_SPACE) == 0


# ---- classification caching -------------------------------------------


def _verdict(doc_type: DocumentType = DocumentType.NORMATIVE) -> Classification:
    return Classification(doc_type=doc_type, confidence=0.9, reason="path: under adr/")


def test_classification_round_trips(manifest: Manifest) -> None:
    source = _source()
    manifest.record_file(
        source,
        chunker="text",
        chunker_version=VERSION,
        classification=_verdict(),
        classifier_version=1,
    )
    cached = manifest.cached_classification(source, classifier_version=1)
    assert cached is not None
    assert cached.doc_type is DocumentType.NORMATIVE
    assert cached.confidence == pytest.approx(0.9)
    assert cached.reason == "path: under adr/"


def test_changed_content_invalidates_the_cached_verdict(manifest: Manifest) -> None:
    """Keyed on the content hash: a file whose bytes changed must be
    reclassified, since its type may well have changed with them."""
    source = _source(text="original")
    manifest.record_file(
        source,
        chunker="text",
        chunker_version=VERSION,
        classification=_verdict(),
        classifier_version=1,
    )
    edited = _source(text="completely different content")
    assert manifest.cached_classification(edited, classifier_version=1) is None


def test_a_moved_file_is_not_reclassified(manifest: Manifest) -> None:
    """The cache is per path *and* hash, so a rename looks new -- which is
    correct, because path is itself a classification signal."""
    source = _source("src/a.py")
    manifest.record_file(
        source,
        chunker="text",
        chunker_version=VERSION,
        classification=_verdict(),
        classifier_version=1,
    )
    assert manifest.cached_classification(_source("src/b.py"), 1) is None


def test_bumping_the_classifier_version_invalidates(manifest: Manifest) -> None:
    """The rules changed, so a stored verdict comes from a ruleset we no
    longer trust."""
    source = _source()
    manifest.record_file(
        source,
        chunker="text",
        chunker_version=VERSION,
        classification=_verdict(),
        classifier_version=1,
    )
    assert manifest.cached_classification(source, classifier_version=1) is not None
    assert manifest.cached_classification(source, classifier_version=2) is None


def test_unknown_file_has_no_cached_verdict(manifest: Manifest) -> None:
    assert manifest.cached_classification(_source("never/seen.py"), 1) is None


def test_recording_without_a_classification_is_allowed(manifest: Manifest) -> None:
    """Older rows and the no-classifier path both land here."""
    source = _source()
    manifest.record_file(source, chunker="text", chunker_version=VERSION)
    cached = manifest.cached_classification(source, classifier_version=0)
    assert cached is not None
    assert cached.doc_type is DocumentType.UNKNOWN


def test_migration_adds_columns_to_an_older_database(tmp_path: Path) -> None:
    """A schema change must not force a rebuild. A full index costs real time
    and money, so an existing database has to be upgraded in place."""
    import sqlite3

    path = tmp_path / "old.sqlite3"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        """
        CREATE TABLE files (
            root_label TEXT NOT NULL, rel_path TEXT NOT NULL, abs_path TEXT NOT NULL,
            mtime_ns INTEGER NOT NULL, size INTEGER NOT NULL, sha256 TEXT NOT NULL,
            kind TEXT NOT NULL, language TEXT, chunker TEXT,
            chunker_version INTEGER NOT NULL DEFAULT 0, indexed_at TEXT NOT NULL,
            PRIMARY KEY (root_label, rel_path));
        INSERT INTO files VALUES
            ('r', 'a.py', '/a.py', 1, 1, 'abc', 'code', 'python', 'code', 1, 'then');
        """
    )
    legacy.commit()
    legacy.close()

    with Manifest(path) as m:
        assert m.file_count() == 1
        record = m.get_file("r", "a.py")
        assert record is not None
        assert record.sha256 == "abc"

    columns = {r[1] for r in sqlite3.connect(path).execute("PRAGMA table_info(files)")}
    assert {"doc_type", "doc_confidence", "doc_reason", "classifier_version"} <= columns


def test_migration_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite3"
    for _ in range(3):
        with Manifest(path) as m:
            m.file_count()


def test_price_fields_survive_the_round_trip_to_the_runs_table(tmp_path: Path) -> None:
    """`unpriced_requests` used to die between EmbeddingStats and the database.

    The runs table exists so "why did this cost $40" is answerable. It cannot
    answer that if the only thing it stores is a number that reads as zero
    whether the run was free or unmeasured.
    """
    with Manifest(tmp_path / "m.sqlite3") as manifest:
        stats = RunStats(
            run_id="r1",
            started_at=datetime.now(UTC),
            tokens_embedded=3_190_000,
            est_cost_usd=0.0,
            unpriced_requests=7,
        )
        manifest.start_run(stats)
        stats.finished_at = datetime.now(UTC)
        manifest.finish_run(stats)

        row = manifest.recent_runs()[0]
        assert row.unpriced_requests == 7
        assert row.cost_is_estimate is False
        assert row.cost_display == "unpriced (7)"


def test_an_estimated_cost_is_marked_as_one(tmp_path: Path) -> None:
    with Manifest(tmp_path / "m.sqlite3") as manifest:
        stats = RunStats(
            run_id="r2",
            started_at=datetime.now(UTC),
            est_cost_usd=0.3828,
            cost_is_estimate=True,
        )
        manifest.start_run(stats)
        stats.finished_at = datetime.now(UTC)
        manifest.finish_run(stats)

        row = manifest.recent_runs()[0]
        assert row.cost_is_estimate is True
        assert row.cost_display == "~$0.3828"


def test_a_genuinely_free_run_still_shows_a_cost(tmp_path: Path) -> None:
    """A local model costs nothing, and $0.0000 is the right answer for it."""
    with Manifest(tmp_path / "m.sqlite3") as manifest:
        stats = RunStats(run_id="r3", started_at=datetime.now(UTC))
        manifest.start_run(stats)
        stats.finished_at = datetime.now(UTC)
        manifest.finish_run(stats)

        assert manifest.recent_runs()[0].cost_display == "$0.0000"


def test_total_tokens_sums_every_run(tmp_path: Path) -> None:
    with Manifest(tmp_path / "m.sqlite3") as manifest:
        assert manifest.total_tokens_embedded() == 0
        for index, tokens in enumerate([3_190_000, 1_600]):
            stats = RunStats(
                run_id=f"run{index}",
                started_at=datetime.now(UTC),
                tokens_embedded=tokens,
            )
            manifest.start_run(stats)
            stats.finished_at = datetime.now(UTC)
            manifest.finish_run(stats)

        assert manifest.total_tokens_embedded() == 3_191_600


def test_a_database_predating_the_price_columns_is_migrated(tmp_path: Path) -> None:
    """A full index costs real time and money, so a schema change must not mean
    "rebuild". The old rows read back as priced-at-zero, which is optimistic
    for the API runs that predate the fix and correct for local ones -- either
    way it is a readable row rather than a query error.
    """
    path = tmp_path / "old.sqlite3"
    with Manifest(path) as manifest:
        stats = RunStats(run_id="old", started_at=datetime.now(UTC), tokens_embedded=99)
        manifest.start_run(stats)
        stats.finished_at = datetime.now(UTC)
        manifest.finish_run(stats)

    # Drop the columns the way an older version would never have had them.
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE runs DROP COLUMN unpriced_requests")
        db.execute("ALTER TABLE runs DROP COLUMN cost_is_estimate")

    with Manifest(path) as reopened:
        row = reopened.recent_runs()[0]
        assert row.run_id == "old"
        assert row.unpriced_requests == 0
        assert row.tokens_embedded == 99


def _edge(module: str, line: int = 1, kind: str = "import") -> ImportEdge:
    return ImportEdge(module=module, kind=kind, is_relative=module.startswith("."), line=line)


def _lang_source(rel_path: str, language: str) -> SourceFile:
    return make_source("x = 1", rel_path=rel_path, language=language)


def test_imports_round_trip(manifest: Manifest) -> None:
    source = _source("repo/a.py")
    manifest.record_file(source, chunker="code", chunker_version=1)
    manifest.record_imports(
        source.root_label, source.rel_path, [_edge("os"), _edge(".sib", line=2)]
    )

    edges = manifest.imports_of(source.root_label, source.rel_path)
    assert [(e.module, e.line, e.is_relative) for e in edges] == [
        ("os", 1, False),
        (".sib", 2, True),
    ]


def test_recording_imports_replaces_rather_than_merges(manifest: Manifest) -> None:
    """An import deleted from the source has to disappear from the graph."""
    source = _source("repo/a.py")
    manifest.record_file(source, chunker="code", chunker_version=1)
    manifest.record_imports(source.root_label, source.rel_path, [_edge("os"), _edge("sys", 2)])
    manifest.record_imports(source.root_label, source.rel_path, [_edge("os")])

    assert [e.module for e in manifest.imports_of(source.root_label, source.rel_path)] == ["os"]


def test_the_reverse_edge_finds_every_importer(manifest: Manifest) -> None:
    """The half a per-project language server structurally cannot answer:
    which files across the whole workspace reference this."""
    for name in ("repo/a.py", "repo/b.py", "other/c.py"):
        source = _source(name)
        manifest.record_file(source, chunker="code", chunker_version=1)
        module = "shared.util" if name != "other/c.py" else "unrelated"
        manifest.record_imports(source.root_label, source.rel_path, [_edge(module)])

    importers = manifest.importers_of("shared.util")
    assert [rel for _, rel, _ in importers] == ["repo/a.py", "repo/b.py"]
    assert manifest.importers_of("nobody.imports.this") == []


def test_deleting_a_file_removes_its_edges(manifest: Manifest) -> None:
    """Reverse edges are a query rather than stored state, so a deleted file
    stops being an importer with no separate invalidation step."""
    source = _source("repo/a.py")
    manifest.record_file(source, chunker="code", chunker_version=1)
    manifest.record_imports(source.root_label, source.rel_path, [_edge("shared.util")])
    assert manifest.importers_of("shared.util")

    manifest.forget_file(source.root_label, source.rel_path)
    assert manifest.importers_of("shared.util") == []


def test_coverage_separates_no_edges_from_no_extractor(manifest: Manifest) -> None:
    """An empty graph answer is ambiguous, and the readings call for opposite
    next moves: "nothing imports this" versus "nobody looked"."""
    importing = _lang_source("repo/a.py", "python")
    manifest.record_file(importing, chunker="code", chunker_version=1)
    manifest.record_imports(importing.root_label, importing.rel_path, [_edge("os")])

    standalone = _lang_source("repo/b.py", "python")
    manifest.record_file(standalone, chunker="code", chunker_version=1)

    styles = _lang_source("repo/a.scss", "scss")
    manifest.record_file(styles, chunker="code", chunker_version=1)

    coverage = manifest.import_coverage()
    assert coverage["python"] == (1, 2)
    # Present, at zero -- the caller decides what that means from the
    # supported-language list, which is why both numbers are reported.
    assert coverage["scss"] == (0, 1)


def test_origins_round_trip_and_group_by_language(manifest: Manifest) -> None:
    source = _lang_source("repo/a.py", "python")
    manifest.record_file(source, chunker="code", chunker_version=1)
    manifest.record_imports(source.root_label, source.rel_path, [_edge("os"), _edge("mine")])

    manifest.record_origins(
        [
            (source.root_label, source.rel_path, "os", "framework"),
            (source.root_label, source.rel_path, "mine", "first_party"),
        ]
    )

    coverage = manifest.origin_coverage()["python"]
    assert coverage.total == 2
    assert coverage.framework == 1
    assert coverage.first_party == 1
    assert coverage.unrecorded == 0


def test_an_unclassified_origin_stays_distinguishable_from_an_unrecorded_one(
    manifest: Manifest,
) -> None:
    """The distinction the whole bucket scheme exists to preserve.

    "Classification never ran" means re-index; "it ran and no rule claimed
    this" means write a rule. One number for both loses the difference.
    """
    source = _lang_source("repo/a.py", "python")
    manifest.record_file(source, chunker="code", chunker_version=1)
    manifest.record_imports(source.root_label, source.rel_path, [_edge("a"), _edge("b", line=2)])
    manifest.record_origins([(source.root_label, source.rel_path, "a", "unclassified")])

    coverage = manifest.origin_coverage()["python"]
    assert coverage.unclassified == 1
    assert coverage.unrecorded == 1


def test_imports_for_origin_reports_the_resolution_state(manifest: Manifest) -> None:
    """Classification needs to know what resolved, because a resolved edge is
    first-party by construction whatever it is called."""
    source = _lang_source("repo/a.py", "python")
    manifest.record_file(source, chunker="code", chunker_version=1)
    manifest.record_imports(source.root_label, source.rel_path, [_edge("os"), _edge(".sib", 2)])
    manifest.set_resolved_path(source.root_label, source.rel_path, ".sib", "repo/sib.py")

    by_module = {module: resolved for _, _, module, _, _, resolved in manifest.imports_for_origin()}
    assert by_module == {"os": None, ".sib": "repo/sib.py"}


def test_the_gate_counts_only_resolved_first_party_edges(manifest: Manifest) -> None:
    source = _lang_source("repo/a.py", "python")
    manifest.record_file(source, chunker="code", chunker_version=1)
    manifest.record_imports(
        source.root_label,
        source.rel_path,
        [_edge(".hit"), _edge(".miss", line=2), _edge("os", line=3)],
    )
    manifest.set_resolved_path(source.root_label, source.rel_path, ".hit", "repo/hit.py")
    manifest.record_origins(
        [
            (source.root_label, source.rel_path, ".hit", "first_party"),
            (source.root_label, source.rel_path, ".miss", "first_party"),
            (source.root_label, source.rel_path, "os", "framework"),
        ]
    )

    coverage = manifest.origin_coverage()["python"]
    # The framework edge is excluded from the denominator, which is the point:
    # against all three edges this would read 33%.
    assert coverage.first_party == 2
    assert coverage.first_party_resolved == 1
    assert coverage.first_party_resolution == 0.5


def test_origins_are_dropped_with_their_file(manifest: Manifest) -> None:
    source = _lang_source("repo/a.py", "python")
    manifest.record_file(source, chunker="code", chunker_version=1)
    manifest.record_imports(source.root_label, source.rel_path, [_edge("os")])
    manifest.record_origins([(source.root_label, source.rel_path, "os", "framework")])

    manifest.forget_file(source.root_label, source.rel_path)

    assert manifest.origin_coverage() == {}
