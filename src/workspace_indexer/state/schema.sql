-- Manifest schema.
--
-- Local, single-file, and deliberately not the vector store: the hot path is
-- millions of cheap existence checks, and this survives swapping Qdrant out
-- entirely.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

-- One row per file we have seen. mtime_ns and size drive the fast path; sha256
-- is only ever compared after those say something changed.
CREATE TABLE IF NOT EXISTS files (
    root_label      TEXT    NOT NULL,
    rel_path        TEXT    NOT NULL,
    abs_path        TEXT    NOT NULL,
    mtime_ns        INTEGER NOT NULL,
    size            INTEGER NOT NULL,
    sha256          TEXT    NOT NULL,
    kind            TEXT    NOT NULL,
    language        TEXT,
    chunker         TEXT,
    chunker_version INTEGER NOT NULL DEFAULT 0,
    -- What role this document plays, cached against the content hash above so
    -- unchanged bytes are never reclassified. classifier_version invalidates
    -- precisely the files a rule change should affect.
    doc_type        TEXT    NOT NULL DEFAULT 'unknown',
    doc_confidence  REAL    NOT NULL DEFAULT 0.0,
    doc_reason      TEXT    NOT NULL DEFAULT '',
    classifier_version INTEGER NOT NULL DEFAULT 0,
    indexed_at      TEXT    NOT NULL,
    PRIMARY KEY (root_label, rel_path)
);

-- Which embedding spaces a given file has been embedded into, and how many
-- chunks it produced. chunk_count exists so a file that legitimately produces
-- nothing -- an image, a binary -- is recorded as complete rather than looking
-- forever like a file whose chunks are missing.
CREATE TABLE IF NOT EXISTS file_spaces (
    root_label  TEXT    NOT NULL,
    rel_path    TEXT    NOT NULL,
    space_slug  TEXT    NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    embedded_at TEXT    NOT NULL,
    PRIMARY KEY (root_label, rel_path, space_slug),
    FOREIGN KEY (root_label, rel_path) REFERENCES files (root_label, rel_path)
        ON DELETE CASCADE
);

-- One row per chunk per space. The primary key is the pair, not the id alone:
-- the same chunk exists in two spaces during a model swap, which is the whole
-- point of being able to backfill a new collection without discarding the old.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT    NOT NULL,
    space_slug  TEXT    NOT NULL,
    root_label  TEXT    NOT NULL,
    rel_path    TEXT    NOT NULL,
    content_sha TEXT    NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    embedded_at TEXT    NOT NULL,
    PRIMARY KEY (chunk_id, space_slug),
    FOREIGN KEY (root_label, rel_path) REFERENCES files (root_label, rel_path)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS chunks_by_file
    ON chunks (root_label, rel_path, space_slug);

-- What each file imports, as the source wrote it. Unresolved on purpose:
-- turning "@/hooks/useThing" into a file needs tsconfig paths and barrel
-- resolution, which is a per-language project of its own.
--
-- Deliberately not in the vector payload. Reverse lookup -- "who imports this"
-- -- is a relational question, and it is the half of a dependency graph that
-- semantic search and a per-project language server both structurally cannot
-- answer.
CREATE TABLE IF NOT EXISTS imports (
    root_label  TEXT    NOT NULL,
    rel_path    TEXT    NOT NULL,
    module      TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    is_relative INTEGER NOT NULL DEFAULT 0,
    line        INTEGER NOT NULL,
    -- The indexed file this edge points at, once resolved. NULL means either
    -- "not resolvable yet" (a package, a tsconfig alias, a C# namespace) or
    -- "ambiguous", and those must stay distinguishable from "no edge" -- see
    -- resolution_coverage().
    resolved_path TEXT,
    PRIMARY KEY (root_label, rel_path, module, line),
    FOREIGN KEY (root_label, rel_path) REFERENCES files (root_label, rel_path)
        ON DELETE CASCADE
);

-- The reverse edge is the whole point, so the module column is indexed rather
-- than scanned. Deleting a file cascades its rows away, which means "who
-- imports X" stays correct without any separate invalidation step.
CREATE INDEX IF NOT EXISTS imports_by_module ON imports (module);

-- The index on resolved_path is created in _migrate() rather than here.
-- executescript runs before the migration, so on a database predating that
-- column this statement would fail and take the whole open with it.

-- Which namespaces each file declares: the other end of a C# `using`.
--
-- A separate table rather than a column on `files` because a file may declare
-- more than one namespace, and because the relation this exists to serve runs
-- the other way -- symbol to files, many to many.
--
-- This is why a C# using is resolved by join rather than by writing
-- `imports.resolved_path`: that column holds one path, `imports` is keyed per
-- using *site*, and a namespace is declared across N files. There is
-- physically nowhere to put N targets. Deriving them also inherits the
-- property the import index already claims -- deleting a file cascades its
-- declarations away, so "which files does this using reach" stays correct with
-- no separate invalidation step and no dependence on walk order.
CREATE TABLE IF NOT EXISTS namespace_declarations (
    root_label  TEXT    NOT NULL,
    rel_path    TEXT    NOT NULL,
    -- Fully qualified, as a `using` would have to spell it: a namespace
    -- nested inside another is stored joined.
    symbol      TEXT    NOT NULL,
    -- `namespace` today. Type declarations can share this table later to
    -- narrow a using to the single file declaring what it references.
    kind        TEXT    NOT NULL DEFAULT 'namespace',
    line        INTEGER NOT NULL,
    PRIMARY KEY (root_label, rel_path, symbol, line),
    FOREIGN KEY (root_label, rel_path) REFERENCES files (root_label, rel_path)
        ON DELETE CASCADE
);

-- The join goes symbol -> files, so that is the index -- and every selective
-- query pins the root as well, so the root is in it. A namespace declared in
-- two repositories of one workspace is the ordinary case this exists to keep
-- apart, and a symbol-only index would read both before filtering.
CREATE INDEX IF NOT EXISTS namespace_declarations_by_symbol
    ON namespace_declarations (symbol, root_label);

-- Endpoints a file exposes, and endpoints a file calls.
--
-- One table for both sides because they are two halves of one question and a
-- join across two tables would be the only way to ask it. `kind` says which:
-- `declaration` is a thing that can be pointed at, `call` is a pointer.
--
-- Separate from `imports` because the resolution rules are opposite. An import
-- resolves within its own repository, deliberately; a route edge is worth
-- having precisely because it crosses one. Sharing a table would mean one
-- column meaning two different things depending on a neighbouring column,
-- which is how a query comes to be quietly wrong.
CREATE TABLE IF NOT EXISTS route_edges (
    root_label  TEXT    NOT NULL,
    rel_path    TEXT    NOT NULL,
    kind        TEXT    NOT NULL,
    -- The route template for a declaration, the called URL for a call. Stored
    -- as written; normalisation is a matching decision and belongs to whatever
    -- does the matching, not to the record of what the code says.
    template    TEXT    NOT NULL,
    method      TEXT,
    line        INTEGER NOT NULL,
    -- 0 when only the static prefix of the URL was recoverable, because the
    -- rest was interpolated. A prefix names an endpoint but cannot tell two
    -- routes sharing it apart, and a resolver that forgot this would report a
    -- precision it does not have.
    exact       INTEGER NOT NULL DEFAULT 1,
    -- The declaring file this call reaches, once matched. NULL means "not
    -- matched yet or not matchable", which must stay distinguishable from "no
    -- such call" for the same reason unresolved imports do.
    resolved_path TEXT,
    -- The target's root as well as its path. The import graph needs no such
    -- column because it resolves inside one repository by design; a route edge
    -- is worth having precisely because it crosses one, and `rel_path` alone
    -- is ambiguous the moment two repositories both hold Api/HomeController.cs.
    resolved_root TEXT,
    PRIMARY KEY (root_label, rel_path, kind, template, line),
    FOREIGN KEY (root_label, rel_path) REFERENCES files (root_label, rel_path)
        ON DELETE CASCADE
);

-- The reverse edge is the whole point here too: "who calls this endpoint".
CREATE INDEX IF NOT EXISTS route_edges_by_template ON route_edges (template);
CREATE INDEX IF NOT EXISTS route_edges_by_target ON route_edges (resolved_path);

-- What an agent asked through the MCP tools, and what it got back.
--
-- Here rather than only in the log because the useful question is relational:
-- "which calls returned nothing" is a WHERE clause and a log scrape. Those
-- calls are eval cases waiting to be written, which turns the queries an agent
-- actually asks into the dataset instead of sixteen someone invented.
--
-- Paths only, never source_text: recording bodies would duplicate the index
-- into the manifest, and paths plus ranks are what an eval scores.
--
-- NOTE: this table contains query text verbatim, which has corrupted a
-- measurement three times before. It lives in data/, which is hardcoded-
-- excluded from indexing, and a test asserts that stays true.
CREATE TABLE IF NOT EXISTS mcp_calls (
    called_at          TEXT    NOT NULL,
    tool               TEXT    NOT NULL,
    query              TEXT    NOT NULL,
    parameters         TEXT    NOT NULL DEFAULT '{}',
    returned           INTEGER NOT NULL DEFAULT 0,
    returned_paths     TEXT    NOT NULL DEFAULT '[]',
    total_matches      INTEGER NOT NULL DEFAULT 0,
    dropped_for_budget INTEGER NOT NULL DEFAULT 0,
    note               TEXT,
    duration_ms        REAL    NOT NULL DEFAULT 0.0
);

-- The harvesting query is "calls that disappointed", so that is the index.
CREATE INDEX IF NOT EXISTS mcp_calls_by_outcome ON mcp_calls (returned, called_at);

-- Run history, so "why did this cost $40" is answerable from the manifest
-- rather than by scraping logs, and a cost regression is visible over time.
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    mode            TEXT    NOT NULL,
    files_seen      INTEGER NOT NULL DEFAULT 0,
    files_skipped   INTEGER NOT NULL DEFAULT 0,
    files_changed   INTEGER NOT NULL DEFAULT 0,
    chunks_upserted INTEGER NOT NULL DEFAULT 0,
    chunks_deleted  INTEGER NOT NULL DEFAULT 0,
    tokens_embedded INTEGER NOT NULL DEFAULT 0,
    est_cost_usd    REAL    NOT NULL DEFAULT 0.0,
    unpriced_requests INTEGER NOT NULL DEFAULT 0,
    cost_is_estimate  INTEGER NOT NULL DEFAULT 0,
    errors          INTEGER NOT NULL DEFAULT 0,
    config_hash     TEXT
);
