"""The SQLite manifest that drives incremental reindexing.

Deliberately synchronous. SQLite here is local single-file I/O measured in
microseconds, and wrapping each of tens of thousands of tiny queries in
asyncio.to_thread would cost more in executor hand-off than the queries
themselves. Writes are batched into transactions instead.

Not stored in Qdrant because the hot path is millions of cheap existence checks
against a local file, and because this has to survive swapping the vector store
out entirely.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

from workspace_indexer.classification import Classification
from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.graph.dependency import Dependency
from workspace_indexer.graph.dependent import Dependent
from workspace_indexer.graph.import_edge import ImportEdge
from workspace_indexer.graph.origin_coverage import OriginCoverage
from workspace_indexer.graph.route_call import RouteCall
from workspace_indexer.graph.route_declaration import RouteDeclaration
from workspace_indexer.graph.route_target import RouteTarget
from workspace_indexer.graph.unit import unit_of
from workspace_indexer.models import Chunk, DocumentType, RunStats, SourceFile, ToolCall
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.state.chunk_delta import ChunkDelta
from workspace_indexer.state.file_record import FileRecord
from workspace_indexer.state.index_decision import IndexDecision
from workspace_indexer.state.run_record import RunRecord

log = get_logger("workspace_indexer.state.manifest")

_SCHEMA = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Manifest:
    def __init__(self, path: Path) -> None:
        self._path = path.expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self._path, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        # WAL so a reader (an MCP server) is not blocked by the indexer's
        # writes. Without it SQLite takes a global write lock.
        self._db.executescript(_SCHEMA.read_text(encoding="utf-8"))
        self._migrate()

    def _migrate(self) -> None:
        """Add columns that a database created by an older version lacks.

        CREATE TABLE IF NOT EXISTS silently does nothing to an existing table,
        so a schema change would otherwise surface as a query error against a
        live index rather than at startup. Rebuilding is not an acceptable
        answer when a full index costs real time and money.
        """
        additions = {
            "files": {
                "doc_type": "TEXT NOT NULL DEFAULT 'unknown'",
                "doc_confidence": "REAL NOT NULL DEFAULT 0.0",
                "doc_reason": "TEXT NOT NULL DEFAULT ''",
                "classifier_version": "INTEGER NOT NULL DEFAULT 0",
            },
            "runs": {
                "unpriced_requests": "INTEGER NOT NULL DEFAULT 0",
                "cost_is_estimate": "INTEGER NOT NULL DEFAULT 0",
            },
            # Nullable on purpose: NULL means classification has never run
            # over this row, which must stay distinguishable from "it ran
            # and no rule claimed this edge".
            "imports": {"resolved_path": "TEXT", "origin": "TEXT"},
            "route_edges": {"resolved_root": "TEXT"},
        }
        for table, columns in additions.items():
            existing = {str(row["name"]) for row in self._db.execute(f"PRAGMA table_info({table})")}
            for column, definition in columns.items():
                if column not in existing:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                    log.info("state.migrated", table=table, column=column)

        # After the columns exist, never before: schema.sql runs first, so an
        # index over an added column cannot live there -- it would fail to
        # create on every database predating the column and take the open with
        # it. The reverse edge everyone wants is "which files import this one".
        self._db.execute("CREATE INDEX IF NOT EXISTS imports_by_target ON imports (resolved_path)")

    def __enter__(self) -> Manifest:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._db.close()

    # ---- the decision ladder -------------------------------------------

    def decide_from_stat(
        self,
        candidate: FileCandidate,
        *,
        space_slug: str,
        chunker_version: int,
        force: bool = False,
    ) -> IndexDecision:
        """Rungs 1, 4 and 6 — everything answerable without opening the file."""
        if force:
            return IndexDecision.FORCED

        row = self._db.execute(
            "SELECT mtime_ns, size, chunker_version FROM files "
            "WHERE root_label = ? AND rel_path = ?",
            (candidate.root_label, candidate.rel_path),
        ).fetchone()
        if row is None:
            return IndexDecision.NEW

        if row["chunker_version"] != chunker_version:
            # We changed a strategy, so content hashes say nothing useful.
            return IndexDecision.RECHUNK_STRATEGY

        if not self._has_space(candidate.root_label, candidate.rel_path, space_slug):
            # The model-swap path: unchanged file, new vector space to fill.
            return IndexDecision.BACKFILL_SPACE

        if row["mtime_ns"] == candidate.mtime_ns and row["size"] == candidate.size:
            return IndexDecision.SKIP_UNCHANGED

        return IndexDecision.REINDEX

    def decide_from_hash(self, source: SourceFile) -> IndexDecision:
        """Rung 2 — the file looked changed, but is it?

        Catches the very common case of identical bytes rewritten: a formatter
        run, a `git checkout` of the same content, a `touch`. Costs one read
        and zero embedding calls.
        """
        row = self._db.execute(
            "SELECT sha256 FROM files WHERE root_label = ? AND rel_path = ?",
            (source.root_label, source.rel_path),
        ).fetchone()
        if row is None:
            return IndexDecision.NEW
        if row["sha256"] == source.sha256:
            return IndexDecision.SKIP_SAME_CONTENT
        return IndexDecision.REINDEX

    def touch(self, source: SourceFile) -> None:
        """Record the new mtime for a file whose content turned out identical,
        so the next run gets the cheap answer from rung 1."""
        self._db.execute(
            "UPDATE files SET mtime_ns = ?, size = ?, indexed_at = ? "
            "WHERE root_label = ? AND rel_path = ?",
            (source.mtime_ns, source.size, _now(), source.root_label, source.rel_path),
        )

    # ---- recording -----------------------------------------------------

    def cached_classification(
        self, source: SourceFile, classifier_version: int
    ) -> Classification | None:
        """A previous verdict for these exact bytes, or None.

        Keyed on the content hash rather than the path, so a file that moved is
        not reclassified and one that changed always is. This is what keeps a
        future model-based rung from re-reading the whole workspace on every
        run.
        """
        row = self._db.execute(
            "SELECT sha256, doc_type, doc_confidence, doc_reason, classifier_version "
            "FROM files WHERE root_label = ? AND rel_path = ?",
            (source.root_label, source.rel_path),
        ).fetchone()
        if row is None:
            return None
        if row["sha256"] != source.sha256:
            return None
        if row["classifier_version"] != classifier_version:
            # The rules changed, so the stored verdict is from a ruleset we no
            # longer trust.
            return None
        return Classification(
            doc_type=DocumentType(row["doc_type"]),
            confidence=float(row["doc_confidence"]),
            reason=str(row["doc_reason"]),
        )

    def record_file(
        self,
        source: SourceFile,
        *,
        chunker: str,
        chunker_version: int,
        classification: Classification | None = None,
        classifier_version: int = 0,
    ) -> None:
        self._db.execute(
            "INSERT INTO files (root_label, rel_path, abs_path, mtime_ns, size, sha256, "
            "kind, language, chunker, chunker_version, doc_type, doc_confidence, "
            "doc_reason, classifier_version, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (root_label, rel_path) DO UPDATE SET "
            "abs_path = excluded.abs_path, mtime_ns = excluded.mtime_ns, "
            "size = excluded.size, sha256 = excluded.sha256, kind = excluded.kind, "
            "language = excluded.language, chunker = excluded.chunker, "
            "chunker_version = excluded.chunker_version, doc_type = excluded.doc_type, "
            "doc_confidence = excluded.doc_confidence, doc_reason = excluded.doc_reason, "
            "classifier_version = excluded.classifier_version, "
            "indexed_at = excluded.indexed_at",
            (
                source.root_label,
                source.rel_path,
                str(source.abs_path),
                source.mtime_ns,
                source.size,
                source.sha256,
                source.kind.value,
                source.language,
                chunker,
                chunker_version,
                (classification.doc_type if classification else DocumentType.UNKNOWN).value,
                classification.confidence if classification else 0.0,
                classification.reason if classification else "",
                classifier_version,
                _now(),
            ),
        )

    def diff_chunks(
        self, root_label: str, rel_path: str, space_slug: str, chunk_ids: Sequence[str]
    ) -> ChunkDelta:
        """Compare produced ids against stored ids for one file and space."""
        stored = {
            row["chunk_id"]
            for row in self._db.execute(
                "SELECT chunk_id FROM chunks "
                "WHERE root_label = ? AND rel_path = ? AND space_slug = ?",
                (root_label, rel_path, space_slug),
            )
        }
        produced = list(dict.fromkeys(chunk_ids))
        produced_set = set(produced)
        return ChunkDelta(
            to_upsert=[cid for cid in produced if cid not in stored],
            to_delete=sorted(stored - produced_set),
            unchanged=[cid for cid in produced if cid in stored],
        )

    def record_chunks(self, chunks: Iterable[Chunk], space_slug: str) -> None:
        rows = [
            (
                chunk.chunk_id,
                space_slug,
                chunk.meta.root_label,
                chunk.meta.rel_path,
                chunk.meta.content_sha,
                chunk.meta.token_estimate,
                _now(),
            )
            for chunk in chunks
        ]
        if not rows:
            return
        self._db.executemany(
            "INSERT INTO chunks (chunk_id, space_slug, root_label, rel_path, content_sha, "
            "token_count, embedded_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (chunk_id, space_slug) DO UPDATE SET "
            "content_sha = excluded.content_sha, token_count = excluded.token_count, "
            "embedded_at = excluded.embedded_at",
            rows,
        )

    def forget_chunks(self, chunk_ids: Sequence[str], space_slug: str) -> None:
        if not chunk_ids:
            return
        self._db.executemany(
            "DELETE FROM chunks WHERE chunk_id = ? AND space_slug = ?",
            [(cid, space_slug) for cid in chunk_ids],
        )

    def record_space(
        self, root_label: str, rel_path: str, space_slug: str, chunk_count: int
    ) -> None:
        """Mark a file complete for a space, including when it produced nothing.

        Without the zero case, every image and binary would look forever like a
        file whose chunks are missing, and re-chunk on every single run.
        """
        self._db.execute(
            "INSERT INTO file_spaces (root_label, rel_path, space_slug, chunk_count, "
            "embedded_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (root_label, rel_path, space_slug) DO UPDATE SET "
            "chunk_count = excluded.chunk_count, embedded_at = excluded.embedded_at",
            (root_label, rel_path, space_slug, chunk_count, _now()),
        )

    # ---- deletion ------------------------------------------------------

    def chunk_ids_for(self, root_label: str, rel_path: str, space_slug: str) -> list[str]:
        return [
            row["chunk_id"]
            for row in self._db.execute(
                "SELECT chunk_id FROM chunks "
                "WHERE root_label = ? AND rel_path = ? AND space_slug = ?",
                (root_label, rel_path, space_slug),
            )
        ]

    def record_imports(self, root_label: str, rel_path: str, edges: Iterable[ImportEdge]) -> None:
        """Replace everything this file imports.

        Replace rather than merge: an import deleted from the source has to
        disappear from the graph, and comparing edge-by-edge to discover that
        costs more than rewriting a handful of rows.
        """
        self._db.execute(
            "DELETE FROM imports WHERE root_label = ? AND rel_path = ?",
            (root_label, rel_path),
        )
        self._db.executemany(
            "INSERT OR REPLACE INTO imports "
            "(root_label, rel_path, module, kind, is_relative, line) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(root_label, rel_path, e.module, e.kind, int(e.is_relative), e.line) for e in edges],
        )

    def imports_of(self, root_label: str, rel_path: str) -> list[ImportEdge]:
        rows = self._db.execute(
            "SELECT module, kind, is_relative, line FROM imports "
            "WHERE root_label = ? AND rel_path = ? ORDER BY line",
            (root_label, rel_path),
        )
        return [
            ImportEdge(
                module=str(r["module"]),
                kind=str(r["kind"]),
                is_relative=bool(r["is_relative"]),
                line=int(r["line"]),
            )
            for r in rows
        ]

    def unresolved_imports(self) -> list[tuple[str, str, str, str, bool]]:
        """Every edge with no target yet, as (root_label, rel_path, module,
        language, is_relative).

        Resolution needs the whole file set, so it runs after the walk rather
        than per file -- an import can name a file that has not been reached
        yet.
        """
        rows = self._db.execute(
            "SELECT i.root_label, i.rel_path, i.module, i.is_relative, f.language "
            "FROM imports i JOIN files f "
            "ON f.root_label = i.root_label AND f.rel_path = i.rel_path "
            "WHERE i.resolved_path IS NULL"
        )
        return [
            (
                str(r["root_label"]),
                str(r["rel_path"]),
                str(r["module"]),
                str(r["language"] or ""),
                bool(r["is_relative"]),
            )
            for r in rows
        ]

    def imports_for_origin(self) -> list[tuple[str, str, str, str, bool, str | None]]:
        """Every edge as (root_label, rel_path, module, language, is_relative,
        resolved_path), for classification.

        All of them, not just the unclassified ones: an edge that resolves is
        first-party by construction, so a resolution that lands this run
        changes an origin decided last run. Re-deciding all of them is one
        SELECT and one executemany, which is cheaper than tracking which rows
        a resolution pass invalidated.

        DISTINCT because `line` is not selected: a module imported on three
        lines of one file is three rows and one classification, and
        `record_origins` keys without `line` too, so each duplicate would
        re-stamp every row the first one already did.
        """
        rows = self._db.execute(
            "SELECT DISTINCT i.root_label, i.rel_path, i.module, i.is_relative, "
            "i.resolved_path, f.language "
            "FROM imports i JOIN files f "
            "ON f.root_label = i.root_label AND f.rel_path = i.rel_path"
        )
        return [
            (
                str(r["root_label"]),
                str(r["rel_path"]),
                str(r["module"]),
                str(r["language"] or ""),
                bool(r["is_relative"]),
                None if r["resolved_path"] is None else str(r["resolved_path"]),
            )
            for r in rows
        ]

    def record_origins(self, origins: Iterable[tuple[str, str, str, str]]) -> None:
        """Set the origin for each (root_label, rel_path, module, origin).

        Keyed without `line`, like `set_resolved_path`: a file importing the
        same module twice imports it from the same place both times.
        """
        self._db.executemany(
            "UPDATE imports SET origin = ? WHERE root_label = ? AND rel_path = ? AND module = ?",
            [(origin, root, rel, module) for root, rel, module, origin in origins],
        )

    def set_resolved_path(self, root_label: str, rel_path: str, module: str, resolved: str) -> None:
        self._db.execute(
            "UPDATE imports SET resolved_path = ? "
            "WHERE root_label = ? AND rel_path = ? AND module = ?",
            (resolved, root_label, rel_path, module),
        )

    def dependents_of(self, root_label: str, rel_path: str) -> list[Dependent]:
        """Which indexed files import this one.

        The reverse edge, resolved. This is the half a per-project language
        server structurally cannot answer, because it spans every repository in
        the workspace rather than one project.

        Joined to `files` rather than returning bare paths so the caller gets
        each importer's doc_type in the same query. That join is what turns a
        list of paths into "three tests and one route exercise this", which is
        the answer an agent can act on.
        """
        rows = self._db.execute(
            "SELECT i.rel_path AS rel_path, i.line AS line, i.module AS module, "
            "f.doc_type AS doc_type, f.language AS language "
            "FROM imports i JOIN files f "
            "ON f.root_label = i.root_label AND f.rel_path = i.rel_path "
            "WHERE i.root_label = ? AND i.resolved_path = ? "
            "ORDER BY i.rel_path, i.line",
            (root_label, rel_path),
        )
        return [
            Dependent(
                rel_path=str(r["rel_path"]),
                line=int(r["line"]),
                module=str(r["module"]),
                doc_type=str(r["doc_type"]),
                language=str(r["language"]) if r["language"] else None,
            )
            for r in rows
        ]

    def dependencies_of(self, root_label: str, rel_path: str) -> list[Dependency]:
        """What this file imports, with the target's metadata where resolved.

        A LEFT JOIN, not an inner one: an edge naming a package or a stdlib
        module has no row in `files` and must still come back. Dropping it
        would report a file as importing less than it does, which is a worse
        lie than admitting we cannot follow the edge.
        """
        rows = self._db.execute(
            "SELECT i.module AS module, i.line AS line, i.resolved_path AS resolved_path, "
            "i.origin AS origin, f.doc_type AS doc_type, f.language AS language "
            "FROM imports i LEFT JOIN files f "
            "ON f.root_label = i.root_label AND f.rel_path = i.resolved_path "
            "WHERE i.root_label = ? AND i.rel_path = ? "
            "ORDER BY i.line",
            (root_label, rel_path),
        )
        return [
            Dependency(
                module=str(r["module"]),
                line=int(r["line"]),
                rel_path=str(r["resolved_path"]) if r["resolved_path"] else None,
                origin=str(r["origin"]) if r["origin"] else None,
                doc_type=str(r["doc_type"]) if r["doc_type"] else None,
                language=str(r["language"]) if r["language"] else None,
            )
            for r in rows
        ]

    def record_routes(
        self,
        root_label: str,
        rel_path: str,
        declarations: Iterable[RouteDeclaration],
        calls: Iterable[RouteCall],
    ) -> None:
        """Replace every route edge this file owns.

        Replace rather than merge, for the same reason imports do: a route
        deleted from the source has to disappear, and comparing edge by edge to
        discover that costs more than rewriting a handful of rows.
        """
        self._db.execute(
            "DELETE FROM route_edges WHERE root_label = ? AND rel_path = ?",
            (root_label, rel_path),
        )
        rows = [
            (root_label, rel_path, "declaration", d.template, d.method, d.line, 1)
            for d in declarations
        ]
        rows += [
            (root_label, rel_path, "call", c.target, None, c.line, int(c.exact)) for c in calls
        ]
        self._db.executemany(
            "INSERT OR REPLACE INTO route_edges "
            "(root_label, rel_path, kind, template, method, line, exact) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    def route_targets(self) -> list[RouteTarget]:
        """Every declared endpoint, across every root.

        Every root deliberately: this is the one resolver in the codebase that
        must look outside the file's own repository, because a page calling an
        API is the case it exists for.
        """
        rows = self._db.execute(
            "SELECT root_label, rel_path, template FROM route_edges WHERE kind = 'declaration'"
        )
        return [
            RouteTarget(
                root_label=str(r["root_label"]),
                rel_path=str(r["rel_path"]),
                template=str(r["template"]),
            )
            for r in rows
        ]

    def unresolved_calls(self) -> list[tuple[str, str, str, int, bool]]:
        """Every call with no target yet, as (root, path, template, line, exact).

        After the walk rather than during it, for the same reason imports
        resolve late: a call can name an endpoint in a repository the walk has
        not reached, so resolving per file would depend on walk order.
        """
        rows = self._db.execute(
            "SELECT root_label, rel_path, template, line, exact FROM route_edges "
            "WHERE kind = 'call' AND resolved_path IS NULL"
        )
        return [
            (
                str(r["root_label"]),
                str(r["rel_path"]),
                str(r["template"]),
                int(r["line"]),
                bool(r["exact"]),
            )
            for r in rows
        ]

    def set_route_resolution(
        self,
        root_label: str,
        rel_path: str,
        template: str,
        line: int,
        target: RouteTarget,
    ) -> None:
        self._db.execute(
            "UPDATE route_edges SET resolved_path = ?, resolved_root = ? "
            "WHERE root_label = ? AND rel_path = ? AND kind = 'call' "
            "AND template = ? AND line = ?",
            (target.rel_path, target.root_label, root_label, rel_path, template, line),
        )

    def route_callers_of(self, root_label: str, rel_path: str) -> list[Dependent]:
        """Which files call an endpoint this file declares.

        The reverse edge, and the one nothing else in this codebase can
        answer: it crosses repositories, and the two files share a string
        rather than a symbol.
        """
        rows = self._db.execute(
            "SELECT r.root_label AS root_label, r.rel_path AS rel_path, r.line AS line, "
            "r.template AS template, f.doc_type AS doc_type, f.language AS language "
            "FROM route_edges r JOIN files f "
            "ON f.root_label = r.root_label AND f.rel_path = r.rel_path "
            "WHERE r.kind = 'call' AND r.resolved_root = ? AND r.resolved_path = ? "
            "ORDER BY r.root_label, r.rel_path, r.line",
            (root_label, rel_path),
        )
        return [
            Dependent(
                rel_path=str(r["rel_path"]),
                line=int(r["line"]),
                module=str(r["template"]),
                doc_type=str(r["doc_type"]),
                language=str(r["language"]) if r["language"] else None,
            )
            for r in rows
        ]

    def route_calls_from(self, root_label: str, rel_path: str) -> list[Dependency]:
        """Which endpoints this file calls, resolved where possible."""
        rows = self._db.execute(
            "SELECT r.template AS template, r.line AS line, r.resolved_path AS resolved_path, "
            "f.doc_type AS doc_type, f.language AS language "
            "FROM route_edges r LEFT JOIN files f "
            "ON f.root_label = r.resolved_root AND f.rel_path = r.resolved_path "
            "WHERE r.kind = 'call' AND r.root_label = ? AND r.rel_path = ? "
            "ORDER BY r.line",
            (root_label, rel_path),
        )
        return [
            Dependency(
                module=str(r["template"]),
                line=int(r["line"]),
                rel_path=str(r["resolved_path"]) if r["resolved_path"] else None,
                doc_type=str(r["doc_type"]) if r["doc_type"] else None,
                language=str(r["language"]) if r["language"] else None,
            )
            for r in rows
        ]

    def route_coverage(self) -> dict[str, tuple[int, int]]:
        """Per edge kind: how many resolve to a file, and how many exist.

        Both halves, as everywhere else here. A call that resolved to nothing
        is not an endpoint with no server -- it is one we could not match yet,
        and reporting only the first number would turn that into a claim.
        """
        rows = self._db.execute(
            "SELECT kind, COUNT(*) AS total, "
            "SUM(CASE WHEN resolved_path IS NOT NULL THEN 1 ELSE 0 END) AS resolved "
            "FROM route_edges GROUP BY kind"
        )
        return {str(r["kind"]): (int(r["resolved"] or 0), int(r["total"])) for r in rows}

    def route_precision(self) -> tuple[int, int]:
        """Calls whose whole URL was recoverable, and calls in total.

        The rest gave up only a static prefix, which names an endpoint but
        cannot separate two routes that share it. Worth reporting on its own:
        it is the ceiling on how precise any matching can be.
        """
        row = self._db.execute(
            "SELECT COUNT(*) AS total, SUM(exact) AS exact FROM route_edges WHERE kind = 'call'"
        ).fetchone()
        return (int(row["exact"] or 0), int(row["total"] or 0))

    def find_paths(self, needle: str, limit: int = 8) -> list[tuple[str, str]]:
        """Indexed files whose path is, or ends with, `needle`.

        Suffix matching because every path an agent holds came from a search
        result or a traceback, and neither is guaranteed to be the full path
        from the root. Matching on a `/` boundary so `store.py` cannot match
        `my_store.py` -- a silent wrong-file answer is worse than no answer.

        Shortest first: given `store.py` and `legacy/store.py`, the shorter
        path is the one a person naming a file that briefly usually means.
        """
        escaped = needle.replace("\\", r"\\").replace("%", r"\%").replace("_", r"\_")
        rows = self._db.execute(
            "SELECT root_label, rel_path FROM files "
            r"WHERE rel_path = ? OR rel_path LIKE ? ESCAPE '\' "
            "ORDER BY LENGTH(rel_path), root_label, rel_path LIMIT ?",
            (needle, f"%/{escaped}", limit),
        )
        return [(str(r["root_label"]), str(r["rel_path"])) for r in rows]

    def resolution_coverage(self) -> dict[str, tuple[int, int]]:
        """Per language: edges resolved to a file, and total edges.

        Both halves, for the same reason coverage is reported at all: an
        unresolved edge is not a missing dependency, it is one we cannot follow
        yet, and those call for opposite conclusions.
        """
        rows = self._db.execute(
            "SELECT f.language AS language, COUNT(*) AS total, "
            "SUM(CASE WHEN i.resolved_path IS NOT NULL THEN 1 ELSE 0 END) AS resolved "
            "FROM imports i JOIN files f "
            "ON f.root_label = i.root_label AND f.rel_path = i.rel_path "
            "WHERE f.language IS NOT NULL GROUP BY f.language"
        )
        return {str(r["language"]): (int(r["resolved"] or 0), int(r["total"])) for r in rows}

    def origin_coverage(self) -> dict[str, OriginCoverage]:
        """Per language: edges split by where the import points.

        Counted in SQL rather than by loading the edges, because the only
        question asked of this is how many, and the table is the one thing that
        already knows.
        """
        rows = self._db.execute(
            """
            SELECT f.language AS language,
                   COUNT(*) AS total,
                   SUM(i.origin = 'first_party') AS first_party,
                   SUM(i.origin = 'first_party'
                       AND i.resolved_path IS NOT NULL) AS first_party_resolved,
                   SUM(i.origin = 'declared_dependency') AS declared_dependency,
                   SUM(i.origin = 'framework') AS framework,
                   SUM(i.origin = 'unclassified') AS unclassified,
                   SUM(i.origin IS NULL) AS unrecorded
            FROM imports i JOIN files f
            ON f.root_label = i.root_label AND f.rel_path = i.rel_path
            WHERE f.language IS NOT NULL GROUP BY f.language
            """
        )
        return {
            str(r["language"]): OriginCoverage(
                total=int(r["total"]),
                first_party=int(r["first_party"] or 0),
                first_party_resolved=int(r["first_party_resolved"] or 0),
                declared_dependency=int(r["declared_dependency"] or 0),
                framework=int(r["framework"] or 0),
                unclassified=int(r["unclassified"] or 0),
                unrecorded=int(r["unrecorded"] or 0),
            )
            for r in rows
        }

    def indexed_documents(self) -> list[tuple[str, str, str]]:
        """Every indexed file as (abs_path, rel_path, doc_type).

        Returned raw rather than grouped because the only grouping worth having
        -- by repository -- cannot be derived from a path. A workspace root can
        hold a repository nested two directories down, and files above it that
        are in no repository at all, and the two are indistinguishable until
        git is asked.
        """
        return [
            (str(row["abs_path"]), str(row["rel_path"]), str(row["doc_type"]))
            for row in self._db.execute("SELECT abs_path, rel_path, doc_type FROM files")
        ]

    def files_by_unit(self) -> dict[tuple[str, str], set[str]]:
        """Indexed rel_paths grouped by (root_label, unit).

        The resolver's search space. A unit is the top-level directory of a
        root -- a repository, by construction -- and is derived here rather
        than stored, because it is exactly the first path segment and a column
        would be a second source of truth for the same fact.

        Keyed by root as well, so two roots holding a repo of the same name
        cannot resolve into each other.
        """
        grouped: dict[tuple[str, str], set[str]] = {}
        for row in self._db.execute("SELECT root_label, rel_path FROM files"):
            rel = str(row["rel_path"])
            unit = unit_of(rel)
            grouped.setdefault((str(row["root_label"]), unit), set()).add(rel)
        return grouped

    def importers_of(self, module: str) -> list[tuple[str, str, int]]:
        """Every file importing `module`, as (root_label, rel_path, line).

        The reverse edge. Exact string match, because this rung stores what the
        source wrote and has not resolved anything -- `.models` and
        `workspace_indexer.models` are two different strings here even when
        they name the same file. Resolution is what would join them, and
        whether that is worth building is what the coverage numbers decide.
        """
        rows = self._db.execute(
            "SELECT root_label, rel_path, line FROM imports WHERE module = ? "
            "ORDER BY root_label, rel_path, line",
            (module,),
        )
        return [(str(r["root_label"]), str(r["rel_path"]), int(r["line"])) for r in rows]

    def import_coverage(self) -> dict[str, tuple[int, int]]:
        """Per language: files with at least one edge, and total files.

        Both halves matter. A language with no resolver reports (0, n), which
        has to stay distinguishable from a language that genuinely imports
        nothing -- otherwise an empty answer from the graph reads as "nothing
        depends on this" when it means "we never looked".
        """
        rows = self._db.execute(
            """
            SELECT f.language AS language,
                   COUNT(*) AS files,
                   SUM(CASE WHEN i.n > 0 THEN 1 ELSE 0 END) AS with_edges
            FROM files f
            LEFT JOIN (
                SELECT root_label, rel_path, COUNT(*) AS n
                FROM imports GROUP BY root_label, rel_path
            ) i ON i.root_label = f.root_label AND i.rel_path = f.rel_path
            WHERE f.language IS NOT NULL
            GROUP BY f.language
            """
        )
        return {str(r["language"]): (int(r["with_edges"] or 0), int(r["files"])) for r in rows}

    def forget_file(self, root_label: str, rel_path: str) -> None:
        """Drops the file and, by cascade, its chunk and space rows."""
        self._db.execute(
            "DELETE FROM files WHERE root_label = ? AND rel_path = ?", (root_label, rel_path)
        )

    def copy_space(self, source_slug: str, target_slug: str) -> int:
        """Record that every chunk in one space now also exists in another.

        Reprojection derives a narrower collection from vectors already paid
        for, deliberately preserving chunk ids. Without copying these rows the
        manifest has no record of the target space at all, so a later index
        cannot tell which chunks are already present and orphan cleanup cannot
        see them — which is how stale content ends up stranded in a live
        collection.

        Returns the number of chunk rows written.
        """
        self._db.execute(
            "INSERT INTO chunks (chunk_id, space_slug, root_label, rel_path, "
            "content_sha, token_count, embedded_at) "
            "SELECT chunk_id, ?, root_label, rel_path, content_sha, token_count, ? "
            "FROM chunks WHERE space_slug = ? "
            "ON CONFLICT (chunk_id, space_slug) DO UPDATE SET "
            "content_sha = excluded.content_sha, token_count = excluded.token_count, "
            "embedded_at = excluded.embedded_at",
            (target_slug, _now(), source_slug),
        )
        self._db.execute(
            "INSERT INTO file_spaces (root_label, rel_path, space_slug, chunk_count, "
            "embedded_at) "
            "SELECT root_label, rel_path, ?, chunk_count, ? "
            "FROM file_spaces WHERE space_slug = ? "
            "ON CONFLICT (root_label, rel_path, space_slug) DO UPDATE SET "
            "chunk_count = excluded.chunk_count, embedded_at = excluded.embedded_at",
            (target_slug, _now(), source_slug),
        )
        copied = self.chunk_count(target_slug)
        log.info("state.space_copied", source=source_slug, target=target_slug, chunks=copied)
        return copied

    def forget_space(self, space_slug: str) -> int:
        """Drop every record of one embedding space.

        Used when a collection is deleted: without this the manifest still
        believes the space exists, `status` reports collections that are gone,
        and the backfill rung thinks those files are already complete.

        The files themselves stay -- they may well be indexed in another space.
        Returns the number of chunk rows removed.
        """
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE space_slug = ?", (space_slug,)
        ).fetchone()
        removed = int(row["n"])
        self._db.execute("DELETE FROM chunks WHERE space_slug = ?", (space_slug,))
        self._db.execute("DELETE FROM file_spaces WHERE space_slug = ?", (space_slug,))
        log.info("state.space_forgotten", space=space_slug, chunks=removed)
        return removed

    def orphans(
        self, seen: set[tuple[str, str]], root_label: str | None = None
    ) -> list[tuple[str, str]]:
        """Rung 5: rows with no corresponding file on disk.

        Scoped to a root when only one was walked, so indexing a single root
        does not delete every other root's index.
        """
        if root_label is None:
            rows = self._db.execute("SELECT root_label, rel_path FROM files")
        else:
            rows = self._db.execute(
                "SELECT root_label, rel_path FROM files WHERE root_label = ?", (root_label,)
            )
        return [
            (row["root_label"], row["rel_path"])
            for row in rows
            if (row["root_label"], row["rel_path"]) not in seen
        ]

    # ---- reads ---------------------------------------------------------

    def get_file(self, root_label: str, rel_path: str) -> FileRecord | None:
        row = self._db.execute(
            "SELECT * FROM files WHERE root_label = ? AND rel_path = ?", (root_label, rel_path)
        ).fetchone()
        return None if row is None else FileRecord(**dict(row))

    def file_count(self, root_label: str | None = None) -> int:
        if root_label is None:
            row = self._db.execute("SELECT COUNT(*) AS n FROM files").fetchone()
        else:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM files WHERE root_label = ?", (root_label,)
            ).fetchone()
        return int(row["n"])

    def chunk_count(self, space_slug: str | None = None) -> int:
        if space_slug is None:
            row = self._db.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        else:
            row = self._db.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE space_slug = ?", (space_slug,)
            ).fetchone()
        return int(row["n"])

    def spaces(self) -> list[str]:
        return [
            row["space_slug"]
            for row in self._db.execute(
                "SELECT DISTINCT space_slug FROM chunks ORDER BY space_slug"
            )
        ]

    def counts_by_root(self) -> dict[str, int]:
        return {
            row["root_label"]: int(row["n"])
            for row in self._db.execute(
                "SELECT root_label, COUNT(*) AS n FROM files GROUP BY root_label"
            )
        }

    def _has_space(self, root_label: str, rel_path: str, space_slug: str) -> bool:
        row = self._db.execute(
            "SELECT 1 FROM file_spaces WHERE root_label = ? AND rel_path = ? AND space_slug = ?",
            (root_label, rel_path, space_slug),
        ).fetchone()
        return row is not None

    # ---- run history ---------------------------------------------------

    def start_run(self, stats: RunStats) -> None:
        self._db.execute(
            "INSERT INTO runs (run_id, started_at, mode, config_hash) VALUES (?, ?, ?, ?)",
            (stats.run_id, stats.started_at.isoformat(), stats.mode, stats.config_hash),
        )

    def finish_run(self, stats: RunStats) -> None:
        self._db.execute(
            "UPDATE runs SET finished_at = ?, files_seen = ?, files_skipped = ?, "
            "files_changed = ?, chunks_upserted = ?, chunks_deleted = ?, "
            "tokens_embedded = ?, est_cost_usd = ?, unpriced_requests = ?, "
            "cost_is_estimate = ?, errors = ? WHERE run_id = ?",
            (
                (stats.finished_at or datetime.now(UTC)).isoformat(),
                stats.files_seen,
                stats.files_skipped,
                stats.files_changed,
                stats.chunks_upserted,
                stats.chunks_deleted,
                stats.tokens_embedded,
                stats.est_cost_usd,
                stats.unpriced_requests,
                int(stats.cost_is_estimate),
                stats.errors,
                stats.run_id,
            ),
        )

    def record_tool_call(self, call: ToolCall) -> None:
        """Append one MCP tool call. Never updates -- each call is an event."""
        self._db.execute(
            "INSERT INTO mcp_calls (called_at, tool, query, parameters, returned, "
            "returned_paths, total_matches, dropped_for_budget, note, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(UTC).isoformat(),
                call.tool,
                call.query,
                json.dumps(call.parameters, sort_keys=True),
                call.returned,
                json.dumps(call.returned_paths),
                call.total_matches,
                call.dropped_for_budget,
                call.note,
                call.duration_ms,
            ),
        )
        self._db.commit()

    def tool_calls(self, limit: int = 50, *, disappointing_only: bool = False) -> list[ToolCall]:
        """Recent calls, newest first.

        `disappointing_only` is the harvesting filter: calls that returned
        nothing, or had to drop results to fit. Those are the ones worth
        turning into eval cases.
        """
        where = "WHERE returned = 0 OR dropped_for_budget > 0" if disappointing_only else ""
        rows = self._db.execute(
            f"SELECT * FROM mcp_calls {where} ORDER BY called_at DESC LIMIT ?",  # noqa: S608
            (limit,),
        )
        return [
            ToolCall(
                tool=str(r["tool"]),
                query=str(r["query"]),
                parameters=json.loads(str(r["parameters"])),
                returned_paths=json.loads(str(r["returned_paths"])),
                total_matches=int(r["total_matches"]),
                dropped_for_budget=int(r["dropped_for_budget"]),
                note=r["note"],
                duration_ms=float(r["duration_ms"]),
            )
            for r in rows
        ]

    def tool_call_stats(self) -> dict[str, tuple[int, int]]:
        """Per tool: total calls, and how many returned nothing."""
        rows = self._db.execute(
            "SELECT tool, COUNT(*) AS n, "
            "SUM(CASE WHEN returned = 0 THEN 1 ELSE 0 END) AS empty "
            "FROM mcp_calls GROUP BY tool"
        )
        return {str(r["tool"]): (int(r["n"]), int(r["empty"] or 0)) for r in rows}

    def total_tokens_embedded(self) -> int:
        """Every token this manifest has paid to embed, across all runs.

        A floor on account usage rather than a measure of it: work embedded
        with the same key outside this index is invisible here, and our own
        figure is soft wherever the provider reported no count.
        """
        row = self._db.execute("SELECT COALESCE(SUM(tokens_embedded), 0) AS n FROM runs").fetchone()
        return int(row["n"])

    def recent_runs(self, limit: int = 10) -> list[RunRecord]:
        return [
            RunRecord(**dict(row))
            for row in self._db.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            )
        ]

    # ---- transactions --------------------------------------------------

    def begin(self) -> None:
        """Batch writes. Per-statement autocommit would fsync tens of
        thousands of times on a full index."""
        self._db.execute("BEGIN")

    def commit(self) -> None:
        self._db.execute("COMMIT")

    def rollback(self) -> None:
        self._db.execute("ROLLBACK")
