# Reference

Every command, every configuration option, every MCP tool. `README.md` is the
quickstart and `docs/deployment.md` is how to run it somewhere else; this is
the page you look things up in.

Defaults here are the ones in code, not the ones in the example config —
`config/workspace.example.yaml` sets some values explicitly for illustration.
Where a default deserves an explanation, it has one.

---

## 1. Commands

All commands accept `--config PATH` (default `config/workspace.yaml`). An MCP
client or a systemd unit starts from its own working directory, so pass an
absolute path there.

### `index`

Walk the configured roots and index what changed.

| flag | default | |
|---|---|---|
| `--root LABEL` | all roots | Index only this root. **Scopes orphan pruning too**, so other roots are neither walked nor touched. Removing a root from `workspace.yaml` is the opposite: it *is* pruned. |
| `--dry-run` | off | Chunk plan and token estimate, no API calls and nothing stored. How to tune chunking without paying to iterate. |
| `--force` | off | Ignore the mtime and hash shortcuts and re-embed everything. Needed after changing anything that alters *what is embedded* without altering chunk identity — `chunking.embed_doc_type` is the example. |


**Deleting from an absence.** Files gone from disk are removed from the index.
That decision is made from an *absence*, which is also what a partial checkout,
an unmounted volume and a failed clone produce. Two brakes:

- **A root that is not on disk is never pruned.** It has not been shown to be
  empty; it has not been read. No override, because there is no evidence to
  weigh. The run reports the count and exits **3**.
- **Losing more than half of a root at once** (and at least 10 files) stops and
  says so rather than proceeding. `--allow-deletes` is how you say the files
  really are gone — a repository restructure legitimately trips this.

This matters most with **one collection spanning several repositories**, which
is the arrangement that makes a cross-repo search work at all. A CI job checks
out one repo while `workspace.yaml` still declares all of them, so an unscoped
run sees several repositories' worth of absence. Pass `--root <label>` in CI so
the run only judges what it can see; the brakes are what catch it when nobody
did.

### `search`

| flag | default | |
|---|---|---|
| `--limit N` | `search.default_limit` | |
| `--unit NAME` | all | Restrict to one top-level directory of a root. |
| `--lang NAME` | all | `python`, `csharp`, `tsx`… |
| `--kind KIND` | all | `code`, `markdown`, `pdf`, `text`, `image`, `opaque`. |
| `--path PREFIX` | all | Restrict to a directory. |
| `--fusion MODE` | `search.fusion` | `rrf`, `dense_only`, `sparse_only`. A debugging tool: when a query returns junk, the first question is which branch produced it. |
| `--rerank / --no-rerank` | config | Override reranking for one call. |
| `--full` | off | Print whole chunks instead of a preview. |

### `status`

What is indexed, in which spaces, what recent runs cost, and import-graph
coverage per language. Reports `inconsistent` when the manifest and the store
disagree about how many chunks exist — usually a store rebuilt without the
manifest, which `index --force` fixes.

A `--force` run announces itself. `run.forced_reindex` is logged once at info
with what it means, and `run.end` carries `files_forced` on **every** run —
zero being the informative answer to "was this a full rebuild". Previously
`force` rode inside `run.start`'s fields, findable only by someone who already
knew to look, and the per-file `decision: forced` lines are debug and can run
to hundreds of thousands. Note that `.env`'s `LOG_LEVEL` overrides
`logging.level`, so a `LOG_LEVEL=WARNING` keeps this out of the console — it is
always in the JSONL, which is always DEBUG.

### `grounding`

Per repository, whether the index can answer **why** the code is the way it is.
Measures four sources — design docs, normative docs, commit rationale, and
`WHY:`/`DECISION:`/`HACK:` markers in tracked code — and grades each `absent`,
`thin` or `present`.

The verdict is the *best* source, not the average: thorough design documents
answer "why" whether or not the commits also do.

Reads what is **indexed**, not what is on disk, because the question it answers
is whether an empty search result means "not written down" or "not retrieved".
Commit and marker evidence comes from `git log` and `git grep` on the
repository each file actually belongs to — resolved from git rather than from
the path, since a root can hold a repository nested below it and loose files
above that repository. A unit in no repository reports those sources as
unavailable rather than as zero.

Three findings are called out separately because they change what to do next:

- **Commits cite tickets but state no reasons** — the rationale exists, in an
  issue tracker this index does not read. The fix is an integration, not better
  commit discipline.
- **Most commit subjects were written by a merge tool** — history that looks
  full by any length-based measure and reads as nothing.
- **The indexed files are no longer at their recorded paths** — said alone,
  replacing every other finding for that repository. An unreadable path and an
  undocumented repository look identical to git and mean opposite things: the
  first is a fact about the index, and it invalidates the row. Reindex first.

Where every source is `absent`, the report says so plainly: generated prose
would be invention rather than retrieval, and it would sit in the same
collection as the real architecture docs, indistinguishable from them.

Thresholds are judgment calls, stated in the output so they can be argued with:
one document per hundred code files, 15% of commits carrying a stated reason,
and at least 20 commits before a share counts as a habit. All three are pinned
by tests.

### `explain PATH`

Dump the chunks one file produces, with symbols and line ranges. The
chunk-quality debugging tool; no API calls.

### `mirror --to qdrant|mongodb`

Copies the current collection into the other backend. **No re-embedding** --
vectors are the expensive part of an index and they are backend neutral, so the
same floats mean the same thing in either store.

Resumes by default: points are keyed by chunk id, so re-running after an
interruption replaces rather than duplicates. `--overwrite` drops the target
collection first, which is the only way to make the target *match* the source
rather than merely include it -- a point deleted from the source is not deleted
from the target by a plain mirror, because nothing scrolls past it to say so.

On Atlas the search indexes build asynchronously, so a mirror that has finished
writing is not yet one that can answer. The command says so; `status` reports
when they are queryable.

A mirrored collection holds two search indexes for as long as it exists. That
matters less than the *lazy release*: Atlas frees a dropped collection's
indexes minutes after the drop, so a full test run — which creates and drops
several pairs — can transiently exceed the budget even well under it. The
fixtures retry rather than fail, so a run self-heals; dropping a resident
mirror first simply makes it quicker. Re-mirroring costs one command and no
embedding tokens.

### `reproject --dimensions N`

Derive a narrower collection by Matryoshka truncation, with no re-embedding.
The first N components of a Voyage vector are a valid N-dimensional embedding,
so going 2048 → 1024 is free. Going the other way is a full re-embed, which is
why indexing wide and narrowing later is the cheaper order.

### `eval`

Score retrieval against a dataset of query/expected-file pairs.

| flag | default | |
|---|---|---|
| `--dataset PATH` | `eval.dataset` | |
| `--limit N` | `10` | The k in recall@k and MRR@k. |
| `--tool NAME` | `search` | `search`, `search_code` or `find_guidance`. Scores an **MCP tool** rather than the raw search path — the only way to measure what a document-type filter is worth, since the filter is the hypothesis. |
| `--group NAME` | `all` | `all`, `retrieval` or `guidance`. |
| `--fusion MODE` | config | |
| `--rerank / --no-rerank` | config | Is the reranker earning its latency? |
| `--dimensions N` | active space | Evaluate a reprojected collection. |
| `--save / --no-save` | save | Record the run under `evals/`. |
| `--compare / --no-compare` | compare | Diff against the last **comparable** run. Two runs are comparable only when config hash, fusion, reranker, tool, case group and case count all match — see §2.4. |

### `serve`

Run the MCP server over stdio. See §4.

### `watch`

Watch the roots and reindex as files change. A trigger, not a second indexing
path: every change goes through the same `index --root` the CLI performs.

**The watch is placed on exactly the directories the index looks in.** The
Rust watcher underneath accepts no exclusion of any kind — its whole
configuration is `watch_paths`, `debug`, `force_polling`, `poll_delay_ms`,
`recursive` and `ignore_permission_denied` — so an exclusion can only be
expressed by *not handing it the path*. Every watchable directory is therefore
enumerated up front and watched without recursion.

Two things follow, and both are the point.

**An excluded tree is never touched at all.** That is what stops the watcher
dying on a dangling symlink or broken reparse point inside a directory
`index.exclude` already skips — a failure no configuration could avoid before,
because recursion descended before any filter ran.

**The watch is far smaller.** inotify watches directories, so the cost is per
directory either way; recursion just meant *every* directory. Measured on a
1,154-file workspace: **1,517 watched before, 215 after — an 86% reduction**,
enumerated in 0.06s. `watch.scoped` logs the count at startup.

A new directory the index cares about cannot be added to a running watch, so
one appearing rebuilds it (`watch.rescoping`). A new *file* does not — its
directory is already watched — and neither does a new directory the index
ignores, which is what keeps a build from restarting the watcher every time it
runs.

`UNWATCHED_DIRS` (`.git`, `node_modules`, `.venv`, `target`, `dist`, `build`,
`__pycache__`) is applied on top, and is deliberately **not** the same list as
`index.exclude`: it is about the watch budget. Index one of those trees without
excluding it and the index will hold it while the watcher will not notice it
changing — a real divergence, accepted because exhausting the inotify limit
stops the watcher working at all.

`ExcludeFilter` still applies to events, dropping editor scratch files
(`.swp`, `~`, `.pyc`) beside a file that *is* watched — which choosing
directories cannot do.

Failures are logged as well as printed. A watcher runs unattended, so a
console-only report is a report to nobody:

| event | when |
|---|---|
| `watch.reindex_failed` | one root's reindex raised; the watcher keeps going |
| `watch.crashed` | anything raised out of the watch loop; the process ends |
| `watch.stopped` | clean exit, with `reason` of `completed` or `interrupted` |

The terminator matters: a log ending at `watch.start` used to be ambiguous
between crashed, killed and still running.

---

## 2. `workspace.yaml`

*What* to index. Safe to commit; its `roots` are paths, so prefer `~`-relative
ones or keep a per-host file.

### 2.1 `workspace`

| key | default | |
|---|---|---|
| `name` | **required** | Names the Qdrant collection: `{name}__{embedding-space}`. Two workspaces with different names get separate collections, which is how you keep a test corpus from shifting your baselines. |
| `roots` | **required** | List of `{path, label?, recurse_into_children?}`. `label` defaults to the directory name and keys the manifest and the payload filter, so a collision would silently merge two roots. |

### 2.2 `index`

| key | default | |
|---|---|---|
| `respect_gitignore` | `true` | Honours `.gitignore` per directory. **Only works inside a git repo** — a plain folder has none. |
| `follow_symlinks` | `false` | |
| `max_file_bytes` | `1048576` | |
**Linked worktrees are never indexed by discovery.** A `git worktree add`
checkout holds the same files as its main checkout at a second path, so
indexing one duplicates a repository: two chunks of every file competing in
search, and — worse, because it is silent — route resolution finding two files
for one endpoint and therefore resolving to *neither*, so cross-repo edges
disappear rather than duplicate. Measured on this repository: one worktree
beside its checkout took the walk from 1,136 files to 1,425.

Detection is `git rev-parse --git-dir` against `--git-common-dir`; a worktree
borrows its repository's object store, so the two differ. The shorter test —
"`.git` is a file" — is true of **submodules** as well, and would drop vendored
code from the index; both cases are pinned by tests against real repositories.

The rule applies to *discovered* directories only. A worktree named directly as
a root in `workspace.yaml` is indexed normally — explicit beats inferred, and
that is the escape hatch for working inside one. The watcher drops worktree
changes for the same reason, so an agent saving into a worktree does not
re-walk the whole root to discover nothing.

| `exclude` | `[]` | gitignore syntax. Excluded directories are *pruned*, not filtered, so `**/node_modules/**` costs one `stat()` rather than a walk. Exclusion is retroactive: adding a pattern removes chunks already indexed. |
| `secret_allow` | `[]` | Files where the content secret-scan is expected to false-positive — token fixtures, checksum lists. Scoped by glob, so allowing one file does not disable the check elsewhere. |

Some exclusions are not configurable at all: `logs/`, `data/`, `.git/`,
`*.sqlite3`, `evals/` and `docs/eval-baselines.md`. A log file inside a watched
root is an infinite reindex loop, and an eval artefact quotes every query
verbatim so indexing one corrupts the measurement. Those are correctness
rules, not preferences.

### 2.3 `chunking`

| key | default | |
|---|---|---|
| `code.max_tokens` | `512` | Applies to header + source, not source alone. |
| `code.min_tokens` | `24` | Below this, adjacent definitions merge. |
| `code.include_context_header` | `true` | Prefix repo, path, language and symbol before embedding. A method lifted from its class is meaningless without it. |
| `markdown.max_tokens` | `512` | |
| `markdown.split_on_heading_depth` | `3` | |
| `text.max_tokens` | `512` | |
| `text.overlap_paragraphs` | `1` | |
| `opaque.mode` | `metadata_only` | Binary and image files: recorded, not embedded. |
| — | — | **YAML, TOML, INI and `.properties` split on unindented lines** as well as blank ones. A `nav:` tree has no blank lines in it, so the prose splitter saw one paragraph the length of the file. |
| — | — | **PDFs have no settings of their own** and use `markdown.max_tokens`. They are prose, and a second knob would be one more thing to tune with no evidence it wants a different value. |
| `embed_doc_type` | `false` | Also write `# type: normative` into the embedded text. **Off on measured evidence**: it dropped recall@10 from 0.875 to 0.812, and worst on the guidance cases it was meant to help. `doc_type` has nine values across thousands of chunks, so the line carries almost no discriminating signal. Changing it needs `index --force` — the header is excluded from `content_sha`, so a normal run finds identical chunk ids and re-embeds nothing. |
| `overrides` | `{}` | Pin an extension to a chunker: `{".mdx": "markdown"}`. The extension point for a file type whose detected language routes it somewhere unhelpful. |

**Nothing silently loses its tail.** A block that overflows the budget is cut
at line boundaries — except a fenced code block, which is kept whole because
cutting a fence embeds badly and displays worse. That exception is now *said*
rather than left to be discovered: `chunk.indivisible_block` logs at info with
the file, the line range and the reason.

Measured on a 1,153-file workspace: 31 chunks are indivisible, all of them
markdown code fences, and **none exceeds the model's 32K input limit** — so
nothing actually truncates against `voyage-code-4`. Before this, structured
config files accounted for a further 12, and those were silently losing their
tails.

`embed.truncated` says which of the two it is:

| `cause` | level | meaning |
|---|---|---|
| `indivisible_block` | info | one block could not be made smaller; the tradeoff worked as designed |
| `chunker_overshoot` | **warning** | a chunk exceeded the limit without being marked indivisible — the chunker produced something larger than its own budget, which is a defect |

A warning that fires on a known, accepted tradeoff is a warning that gets
filtered out, and it takes the real ones with it.

### 2.4 `graph`

| key | default | |
|---|---|---|
| `http_clients` | `[fetch, axios, ky, superagent]` | Functions that make an HTTP request. **Name whatever your codebase wraps `fetch` in.** A bare name matches `name(...)`; an object name matches its request methods, so `axios` covers `axios.get`. |
| `razor_pages_dir` | `Pages` | Directory whose contents map to routes by file path. |
| `client_base_paths` | `[api]` | Segments the client prepends that the server never declares — added by a reverse proxy, a dev-server rewrite, or `UsePathBase`. Tried **only as a fallback**, after the URL as written fails. |

`http_clients` is the setting that decides whether the route graph sees
anything. Measured on a real React workspace: the default list found **6** call
sites, and adding the project's own wrapper found **71**. A codebase that wraps
`fetch` and does not say so here gets a route graph showing endpoints and no
callers — which reads as "nothing calls this API".

`status` reports that shape explicitly when it sees it.

`client_base_paths` matters more than it looks. Measured on a real React +
minimal-API codebase, **every** client call began `/api/` and **no** endpoint
did, because a proxy adds the segment — resolution went from 4% to 23% once the
fallback existed. It is a fallback rather than a rewrite so that a workspace
whose routes genuinely begin with `api` (an ASP.NET controller app does) keeps
its own answer.

Both ASP.NET styles are extracted, and measured on two real projects they were
**mutually exclusive**: one had 66 `[Route]` attributes and no minimal APIs,
the other 1037 C# files with zero `[Route]` and ~140 `MapGet`/`MapGroup` calls.
A group's prefix is joined to every route registered on it, including groups
configured on the line they are created (`app.MapGroup("/x").WithTags(...)`) and
nested groups. A route whose builder this cannot follow — one arriving as a
method parameter, whose prefix the caller decides — is skipped rather than
emitted without its prefix, because a route missing its prefix is not
incomplete, it is wrong.

Razor Pages routes come from file location, not the directive: measured on the
same workspace, **all twenty** `@page` directives were bare. An explicit
`@page "/template"` still wins where one exists, and a `.cshtml` with no `@page`
at all is a layout or partial and declares nothing.

### 2.5 `search`

| key | default | |
|---|---|---|
| `fusion` | `rrf` | `rrf` fuses dense and sparse on *rank*, which is right because cosine and BM25 scores are not on a comparable scale. |
| `prefetch_limit` | `50` | Candidates per branch before fusion. Cheaper than more dimensions if recall is short. |
| `default_limit` | `10` | |
| `check_staleness` | `true` | Verify each hit against the file on disk. Needs read access to the source: **set false where the MCP server runs next to Qdrant rather than next to the code**, or every hit comes back flagged and the flag stops meaning anything. |
| `rerank.enabled` | `true` | Off resolves to a `NoopReranker` — a different object, not a branch in the search path. |
| `rerank.model` | `voyageai:rerank-2.5-lite` | |
| `rerank.candidates` | `50` | Fused hits fed to the reranker. |
| `rerank.top_n` | `10` | Returned after reranking. |
| `rerank.rerank_text` | `embed_text` | Rerank the context-headed text, not bare source. |
| `rerank.instruction` | none | Prepended to the query — the API has no instruction parameter. |
| `rerank.on_error` | `degrade` | `degrade` falls back to fusion order and logs; `fail` raises, which the eval harness wants so a silent degradation cannot corrupt a measurement. |

### 2.6 `watch`

| key | default | |
|---|---|---|
| `mode` | `auto` | `auto` decides per root from `/proc/mounts`. inotify on a 9p, CIFS or NFS mount **succeeds and then never fires**, so those are polled. Unknown filesystems are polled too: guessing "native" wrong costs a watcher that silently never fires. |
| `debounce_ms` | `1500` | One editor save is several events; a formatter run is hundreds. |
| `poll_interval_ms` | `5000` | For roots that cannot use inotify. |
| `reload_config` | `true` | Reload this file on change. Settings and ignore rules go live; **a newly added root still needs a restart**. |

### 2.7 `eval` and `logging`

| key | default | |
|---|---|---|
| `eval.dataset` | `config/eval.yaml` | Never indexed — it quotes every query. |
| `eval.metrics` | `[recall@10, mrr@10]` | |
| `logging.level` | `INFO` | Console level. `.env`'s `LOG_LEVEL` overrides it. |
| `logging.console` | `pretty` | `pretty`, `json` or `off`. Writes to **stderr**, which is what keeps stdio MCP usable. |
| `logging.file.path` | `logs/workspace-indexer.jsonl` | Always DEBUG regardless of console level — you cannot retroactively raise a log level after the failure you needed to see. **The command's name is inserted before the suffix**, so `serve` writes `workspace-indexer-serve.jsonl` — see below. |
| `logging.file.max_bytes` | `20971520` | Size-based rotation, because indexer output is bursty. |
| `logging.file.backup_count` | `10` | |
| `logging.logfire.enabled` | `false` | |
| `logging.logfire.send_to_cloud` | `false` | **pydantic-ai instrumentation captures call inputs, and for an embedding call the input is your source code.** Never a default. |

**Each command logs to its own file.** `logging.file.path` names the shape;
the command is spliced in before the suffix, so `index` writes
`workspace-indexer-index.jsonl` and `serve` writes `workspace-indexer-serve.jsonl`.

This is not tidiness, it is a correctness fix. The stdlib
`RotatingFileHandler` *renames* the live file on rollover, and **Windows
refuses to rename a file another process holds open** — so a long-running
`serve` and a concurrent reindex collide with `WinError 32` and one of them
dies.

Roles are not the whole story, because one role can have several live
processes: every Claude Code session in a workspace spawns its own `serve`. The
file sink is therefore `ConcurrentRotatingFileHandler`, which locks around
appends and rollover.

**This is not a Windows-only problem**, which is what it looked like at first.
Measured on Linux with three processes writing one role's file through the
stdlib handler: **346 of 360 lines survived**, and rollover raised
`FileNotFoundError` as two processes raced to rename the same file. Windows
adds a fatal `WinError 32`; POSIX just loses lines quietly, which is the worse
half. A test pins the line count, because every weaker assertion — intact JSON,
every writer represented, rollover happened — passed with the broken handler.

Putting the PID in the file name was the alternative and was rejected:
`backup_count` would then mean that many backups *per process*, so the ceiling
this config documents would stop being true and grow with session count.
Sessions stay tellable apart by the `run_id` every line already carries.

It also keeps both readable. A full reindex would otherwise bury a session's
worth of MCP queries in the same file, and the MCP half is the one you want
when a search misbehaves mid-session.

Files open on first write, so a command that logs nothing — or aborts on a
config error before logging — leaves nothing behind.

**A duplicate key in `workspace.yaml` is an error, not a silent overwrite.**
YAML keeps the last of two identical keys and says nothing, so a whole block,
comments included, can be edited and have no effect while the file still parses
and the program still runs. The loader rejects it and names the key and line.

---

## 3. `.env`

*How* to index. Never committed. Real environment variables win over the file,
which is what makes the `.mcp.json` `env` block work.

| variable | default | |
|---|---|---|
| `EMBEDDING_MODEL` | `voyageai:voyage-code-4` | `provider:model`. The whole provider abstraction. |
| `EMBEDDING_DIMENSIONS` | `2048` | Part of the collection name, so changing it addresses a different collection rather than migrating one. |
| `EMBEDDING_QUANTIZATION` | `float32` | Accepted, currently unused. |
| `EMBEDDING_BATCH_SIZE` | `64` | |
| `EMBEDDING_MAX_CONCURRENCY` | `4` | |
| `EMBEDDING_MAX_BATCH_TOKENS` | `100000` | Providers cap tokens per request, not just document count. |
| `EMBEDDING_PRICE_PER_MTOK` | none | Used only when the provider reports no price. `genai-prices` has no entry for `voyage-code-4`, so without this a run records as *unpriced* rather than as free. A configured rate goes stale silently, so anything priced this way displays as `~$0.1234`. |
| `EMBEDDING_FREE_TIER_TOKENS` | none | `status` shows drawdown against it. A floor on usage, never an authority: the allowance is per account and spent by anything using the key. |
| `VOYAGE_API_KEY` | none | Exported to `os.environ` at startup, because provider SDKs read it from there. |
| `SPARSE_MODEL` | `Qdrant/bm25` | Local, free, no API call. |
| `RERANK_ENABLED` | from yaml | Overrides `search.rerank.enabled`, and only when actually set — the default here does not silently beat a `workspace.yaml` that configured reranking. |
| `RERANK_MODEL` | from yaml | Overrides `search.rerank.model`, same rule as above. A missing API key resolves to a no-op reranker and logs once — a search never fails because an optional enhancement is unconfigured. |
| `VECTOR_STORE` | `qdrant` | `qdrant` or `mongodb`. The `QDRANT_*` keys are read only for the first, the `MONGODB_*` keys only for the second. |
| `QDRANT_MODE` | `embedded` | `embedded` is single-process and **ignores payload indexes**, so `doc_type` filtering scans. `server` is required for the MCP server and the watcher. |
| `QDRANT_PATH` | `data/qdrant` | Embedded mode only. Relative — a process started elsewhere resolves it elsewhere. |
| `QDRANT_URL` | `http://localhost:6333` | |
| `QDRANT_API_KEY` | none | **Qdrant has no authentication by default, and the payload contains your source text.** Set this whenever it is not bound to loopback. |
| `QDRANT_ON_DISK_PAYLOAD` | `true` | |
| `MONGODB_CONNECTION_STRING` | none | Atlas: Connect → Drivers. Carries the password inline, so it belongs here and never in `workspace.yaml`. Needs `poetry install --extras mongo`. |
| `MONGODB_DATABASE` | `workspace_indexer` | Collections inside it are named exactly as Qdrant's are. |
| `MONGODB_VECTOR_DTYPE` | `float32` | `float32` or `int8`, both stored as BSON `binData`. See §7. |
| `STATE_DB` | `data/manifest.sqlite3` | Give a second workspace its own, or both share one and the divergence check misfires. |
| `LOG_LEVEL` | from yaml | |
| `LOGFIRE_ENABLED` / `LOGFIRE_SEND_TO_CLOUD` | none | Override the `logging.logfire` block in `workspace.yaml`. |
| `LOGFIRE_TOKEN` | none | Read from the environment by the logfire SDK itself, not by this code. Declared here so it is documented and so an unknown key is not rejected. |

---

## 4. MCP server

`workspace-indexer serve` speaks MCP over stdio: the client starts the process
and talks down a pipe. No port, nothing left running. See
`docs/deployment.md` §4 for the `.mcp.json` recipe and why every path in it
must be absolute.

On Qdrant this requires `QDRANT_MODE=server` — the server holds the index open
for the whole session, which embedded mode's exclusive lock forbids. Mongo has
no equivalent restriction; Atlas is a server by construction.

### Tools

Separate tools rather than one tool with an intent argument, because the agent
already knows its intent and declaring it is far more reliable than inferring
it from query text.

**`search_code`** — implementation and documentation, by meaning.

| parameter | type | default | |
|---|---|---|---|
| `query` | string | required | |
| `limit` | integer 1–50 | `8` | |
| `repo` | string | none | Restrict to one repository. |
| `language` | string | none | |
| `path_prefix` | string | none | |
| `include_tests` | boolean | `false` | Tests and generated files are excluded by default: a test naming a symbol twenty times otherwise outranks the file defining it. |

**`find_guidance`** — specifications, design documents and guides only.

| parameter | type | default | |
|---|---|---|---|
| `query` | string | required | |
| `limit` | integer 1–50 | `8` | |
| `repo` | string | none | |
| `doc_type` | string | none | Narrow to one type. Accepts aliases (`spec`, `adr`, `architecture`, `readme`…). **An unrecognised value returns an error naming the valid types — never an empty result.** |

Guidance covers `normative`, `design` **and `guide`**. The last is there on
evidence: normative + design alone scored no better than plain search over the
eight guidance cases, because filtering out `guide` lost `CONTRIBUTING.md`
entirely. Adding it took recall from 0.812 to 0.938.

**`get_file_context`** — every indexed chunk of one file, in order.

| parameter | type | default | |
|---|---|---|---|
| `rel_path` | string | required | A path from a search result, or a trailing portion of one. |
| `limit` | integer 1–100 | `20` | |

**`list_document_types`** — the taxonomy, with counts and example paths. No
parameters. A type reported at **count 0** genuinely has none in this
workspace: if `normative` is 0, read the implementation instead of hunting for
specs.

**`impact_of`** — what one file imports, and what imports it. Answers from the
manifest alone: no embedding call, no vector search.

| parameter | type | default | |
|---|---|---|---|
| `rel_path` | string | required | A path from a search result, or a trailing portion of one. |
| `limit` | integer 1–200 | `25` | Maximum edges **per direction**. |

`called_by` and `calls` are the HTTP half: files that reach this one over a
URL rather than by importing it, and the endpoints this one calls. They are
kept apart from `used_by`/`depends_on` deliberately — **an importer breaks at
compile time, an HTTP caller breaks at run time, in another repository, and no
compiler will warn you.** Merging them would hide the distinction that makes
the question worth asking.

This is the only edge in the system that crosses repositories, and it is the
reason one shared collection is worth its costs. A React page and the C#
controller it calls share a *string*, not a symbol — there is no import for a
language server to follow.

`used_by` is the half that is expensive to get any other way — it spans every
repository in the workspace — and each entry is anchored as `path:line` at the
import statement. `used_by_by_type` counts every dependent by document type
over the whole result rather than the page that fitted, so `{"test": 3,
"implementation": 1}` tells you a signature change breaks one caller and three
tests without reading the list.

An ambiguous path is **never guessed**. Two files ending in `store.py` come
back as `candidates` with an empty report, because answering for the wrong one
tells an agent that nothing imports a file it never asked about.

Read `note` before concluding anything from a small or empty result. It
distinguishes the three ways this tool can be silent for reasons that have
nothing to do with the file:

- The language has no import scanner (`bicep`, `powershell`, HTML, CSS). Both
  lists are then empty by construction.
- Every edge naming the file is spelled as a package, a build alias or a C#
  namespace, none of which resolve to a path.
- An importer is a **re-export file** (`__init__.py`, `index.ts`). That edge is
  a hop, not a destination: whatever imports the symbol *through* the barrel is
  not counted, so the real caller count is higher. Run `impact_of` on the
  barrel to follow the next hop. This project's own one-class-per-file mandate
  guarantees the case, and TypeScript barrels behave identically.

**`grounding`** — whether a repository records *why* it is the way it is.
Answers from the manifest plus `git log` and `git grep`; no embedding call, no
LLM.

| parameter | type | default | |
|---|---|---|---|
| `repo` | string | all | Restrict to one repository, as named in a search result. |

**Call this when `find_guidance` returns nothing, before concluding anything
from that.** An empty search has two causes — the index missed it, or nobody
wrote it down — and they call for opposite next moves. Nothing else in the
index can tell them apart, because both look like zero hits.

Four sources per repository (design docs, normative docs, commit rationale,
`WHY:`/`DECISION:`/`HACK:` markers), each `absent`, `thin` or `present`. A
repository's own `verdict` is the **best** of its sources — they are
alternatives, and thorough design documents answer "why" whether or not the
commits also do. The report-level `note` inverts that and follows the
**weakest** repository, because across repositories they are separate
codebases and an agent told "well covered" on the strength of one would trust
the others.

Where the verdict is `absent`, the rationale genuinely was not recorded. Say
so rather than inferring a plausible one: generated rationale would be
indistinguishable from a retrieved one, and it is the failure this tool exists
to prevent.

Read `notes`. They carry the findings that change what to do — most usefully
that a repository's reasons live in an issue tracker this index cannot read,
which turns "no rationale" into "wrong system".

An unrecognised `repo` is an error naming the indexed ones, never an empty
result — empty here reads as "records no reasons", the strongest claim the
tool can make, and a typo must not manufacture it.
#### Working in a worktree

`search_code`, `find_guidance` and `get_file_context` take an optional
`worktree`. **It is required only once a repository actually has worktrees** —
a workspace that has never run `git worktree add` never sees the parameter, the
error, or the cost.

| `worktree` | behaviour |
|---|---|
| omitted, no worktrees exist | normal search |
| omitted, worktrees exist | **error** listing them and offering `"none"` |
| `"none"` | main checkout; every worktree ignored |
| a name or absolute path | that worktree only |

The refusal is deliberate, because neither default is safe. Answering from the
index serves a developer working in a worktree the wrong checkout, silently.
Reporting divergence across *all* worktrees flags a file because some other
agent is mid-edit on a branch this caller has never heard of — and abandoned
branches make that noise permanent, so the flag stops being read. One round
trip, once per session, is the same trade `find_guidance` makes for an
unrecognised document type.

When scoped, hits the worktree has changed are marked `stale`, and **every**
hit's `abs_path` points at that worktree's copy — changed or not — so the agent
can always open the path it was handed without working out which checkout it is
in. Divergence is measured against the *main checkout's* HEAD, so a change the
agent has already committed in its worktree still counts.

`abs_path` is on every result, not only scoped ones (~190 tokens per 8-hit
response). `rel_path` remains the index's name for the file and is unchanged by
scoping — it has to mean the same thing whoever is asking.

**The limit worth knowing:** this reports that a file differs, never what it
now says. The index holds the main checkout's text, so a file heavily rewritten
in a worktree returns its *old* content with a flag, and a file **created** in
a worktree has no hit to flag at all. Search is a pointer for code you are
actively editing, not a source.

`list_document_types`, `grounding` and `impact_of` take no `worktree`: the
first two answer about the index and the repository rather than about a working
copy, and taxing every graph query with a checkout question buys nothing.

### Resource

`workspace-indexer://taxonomy` serves the same taxonomy as JSON. Both surfaces
exist because clients differ in how reliably a model sees a resource the user
has not attached, whereas a tool is always in context.

### Route graph

Client call sites matched to the endpoints they reach. `status` reports the
counts; `impact_of` reports the edges.

Resolution is to a **file**, not to an endpoint, and that is what makes the
prefix-only calls usable. `` fetch(`/api/remit/${id}`) `` cannot say which
action of a controller it calls, but every candidate is in the same file — and
"which file" is the question `impact_of` asks. A prefix reaching two *files*
stays unresolved rather than picking one.

Matching ignores case, query strings and the scheme/host of an absolute URL.
Route parameters (`{id}`, `{id:int}`) match any one segment and `{*rest}`
absorbs the remainder. A prefix must be strictly shorter than the route it
matches, because the interpolation it came from has to go somewhere.

**Unresolved is a real answer**, as it is for imports: the endpoint may live
outside this workspace, or the path may be declared in more than one file.
Neither means nothing calls it.

The one setting that decides whether any of this works is
`graph.http_clients` — see §2.4.

### Import graph

Each code file's imports are extracted during indexing and, where possible,
resolved to another indexed file. `status` reports both coverage (which
languages have an extractor) and resolution (how many edges point at a file).

Resolved within a repository only. Python relative and absolute imports,
and JS/TS relative specifiers — including TypeScript's ESM convention where
`import './x.js'` names a file that is actually `x.ts`.

**Not** resolved, deliberately: packages (`react`, `pydantic`), tsconfig path
aliases (`@/lib/utils`), and C# namespaces, which name no path at all. Those
need a build system or a workspace-wide symbol table; until then they resolve
to nothing rather than to something plausible. An unresolved edge is not a
missing dependency.

The reverse edge — *which files import this one* — spans every repository in
the workspace, which is what a per-project language server cannot answer.

### Recorded calls

Every tool call is recorded twice: as an `mcp.tool_call` event in the rolling
log, and as a row in the manifest's `mcp_calls` table. Both carry the tool,
the parameters, the returned paths, `dropped_for_budget`, any note and the
duration.

`status` summarises them and names recent calls worth turning into eval cases.
The harvesting query is the point:

```sql
SELECT tool, query, returned FROM mcp_calls
WHERE returned = 0 OR dropped_for_budget > 0
ORDER BY called_at DESC;
```

Those are queries a real agent asked that the index could not answer, which is
a better source of eval cases than any dataset written from imagination.

**What it cannot tell you.** A call that returns three confidently irrelevant
results looks identical here to one that returns three good ones — only the
count is recorded, not whether the answer was right. Empty and clipped
responses are detectable; *wrong* ones still need a human or an eval.

Recording never fails a call: if the manifest write raises, the log entry has
already succeeded and the tool returns normally.

### Result shape

Every result is `path:start-end` anchored, so the next action is a read with
nothing to work out. Responses are token-budgeted; anything dropped is reported
in `dropped_for_budget` rather than silently cut, and an empty response carries
a note saying what was filtered and what to try next.

### Document types

`normative` · `design` · `guide` · `reference` · `record` · `implementation` ·
`test` · `generated` · `unknown`

Call `list_document_types` for the definitions and live counts — they come from
the index, not from this page.

---

## 5. Choosing a vector store

Two backends behind one `VectorStore` protocol. Nothing above `storage/` knows
which is configured, the payload is built and read by the same two functions in
both, and collections carry the same name — so a workspace indexed into both is
recognisably the same index in each.

### Qdrant

The default. Embedded for a single process, server for anything concurrent.
`docs/deployment.md` covers running it.

### MongoDB Atlas

`VECTOR_STORE=mongodb`, plus `poetry install --extras mongo`. Two Atlas indexes
are created automatically on first run: a `vectorSearch` index over the dense
vector and a `search` index over the text.

They build **asynchronously**. A collection can hold every document and answer
every query with nothing for a minute afterwards, which looks exactly like a
broken query. `workspace-indexer status` reports whether each is queryable yet.

**Hybrid search works differently here, and it is worth knowing how.** On
Qdrant the keyword branch scores a BM25 sparse vector we compute locally with
fastembed. Atlas has its own inverted index and no way to accept ours, so the
keyword branch is a `$search` stage and the scoring is Lucene's. Same idea, same
collection-wide IDF, different implementation — so a recall difference between
the two backends is worth attributing to that before blaming the vectors. The
sparse vector is still computed and simply discarded on this backend.

Fusion is `$rankFusion`, Atlas's native RRF, where the cluster has it. That is a
rolling deployment across the 8.0 fleet, so the first hybrid search tries it and
falls back to fusing ranks client-side if the server rejects the stage — same
RRF constant, same ordering, one extra round trip. The fallback is decided once
per process, by asking the server rather than by reading a version number.

### Where reranking runs

`RERANK_MODEL` names the *provider*, and the provider already says where the
model runs -- `local:` in this process, `voyageai:` over the network. A third,
`database:`, says the store reranks inside its own query:

```
RERANK_MODEL=voyageai:rerank-2.5-lite   # a call from here, after retrieval
RERANK_MODEL=local:BAAI/bge-reranker-base
RERANK_MODEL=database:rerank-2.5-lite   # a $rerank stage in the aggregation
```

One knob rather than two. A separate `DATABASE_RERANKING` boolean would create
four states of which two contradict each other, and leave "is it off?" needing
both to be read; `RERANK_ENABLED=false` still turns everything off whichever
provider is named.

Nothing branches on this. `database:` resolves the client-side reranker to the
same no-op object that `enabled: false` produces, and the store is handed a
rerank stage-builder instead -- so the search path is identical in all three
cases and never asks the question. It is the same trick `NoopReranker` already
plays.

Two things it will not do quietly:

- With `VECTOR_STORE=qdrant` it **raises at startup**. Qdrant has no
  server-side reranker, and both factories would otherwise decline to rerank,
  leaving every search returning fusion order while the config said otherwise.
- If the cluster rejects `$rankFusion`, hybrid search **raises** rather than
  falling back. Every other fallback here trades a round trip for the same
  answer; that one would return a different answer while still claiming to
  rerank.

**Requirements, and they are stricter than the toggle suggests.** `$rerank`
needs a cluster on **MongoDB 8.3 or later** -- "Latest version with
auto-upgrades" in the cluster builder -- *and* Native Reranking enabled in
Project Settings. **8.0 with the toggle on is not enough**; measured against a
live 8.0.29 cluster, every `$rerank` is refused. Atlas sends one generic
message for all causes (`$rerank is not allowed or the syntax is incorrect`),
so the store translates it into one that names both requirements.

It is also a Preview feature, and billed separately from Automated Embedding
(200M free tokens at the organization level, then $0.02/M for
`rerank-2.5-lite`).

### Sizing a shared cluster

The two limits that bind are storage and the search-index count, and both
depend on tier. **Measured on the Flex cluster this was developed against, the
search-index limit is 10** — created scratch collections until Atlas refused.

| tier | storage | search indexes |
|---|---|---|
| Free (`M0`) | 512 MB | 3 |
| Flex | 5 GB | 10 (measured) |
| `M10`+ | 10 GB and up | far more |

This store uses **two** indexes per collection, so Free allows exactly one
indexed collection and Flex allows five.

Flex counts storage as `dataSize + indexSize` from `db.stats()` —
*uncompressed*, unlike the `storageSize` figure other tiers use. Measured with
this workspace mirrored across: 66.8 MB of documents plus 2.1 MB of b-trees, so
**69 MB for 11,049 chunks** — about 6.2 KB each, and **1.3% of a Flex
cluster**.

Vectors are stored as BSON `binData`, not as arrays of doubles, and that is the
whole reason the free tier is viable. Measured on this workspace's own index of
11,049 chunks at 1024 dimensions:

| encoding | per document | 11,049 chunks |
|---|---|---|
| array of doubles | 15.07 KB | 163 MB |
| `binData` float32 (default) | 6.15 KB | 66 MB |
| `binData` int8 | 3.15 KB | 34 MB |
| payload alone, no vector | 2.14 KB | 23 MB |

At 6.2 KB per chunk, **Free runs out at roughly 80,000 chunks and Flex at
roughly 800,000** — seven times and seventy times this workspace respectively.
On Flex, storage is not the constraint; the index count is, and five indexed
collections is more than one workspace needs.

Reach for a paid tier before reaching for `int8`: quantisation costs recall,
and recall is the thing being bought.

`MONGODB_VECTOR_DTYPE=int8` exists for when storage genuinely is the binding
constraint. Not `int1`: Atlas supports it only with euclidean similarity, and
every measurement this project has taken is on cosine.

### Measured: Qdrant against Atlas

Same 11,049 vectors in both, copied with `mirror` rather than re-embedded, so
any difference is the store and not the embeddings. Same 16-case eval dataset,
same query embeddings. Numbers reproduce exactly across runs — retrieval is
deterministic given a fixed index.

| | recall@10 | MRR@10 | median end-to-end | store only |
|---|---|---|---|---|
| Qdrant, no rerank | 0.750 | 0.356 | 247 ms | **9.8 ms** |
| Atlas, no rerank | **0.844** | **0.500** | 383 ms | 95.9 ms |
| Qdrant, rerank | **0.875** | **0.679** | **560 ms** | — |
| Atlas, rerank | 0.812 | 0.661 | 740 ms | — |

**Relevance: it depends entirely on whether the reranker is on**, which is not
a hedge — it is the finding.

Without reranking Atlas is clearly better: +0.094 recall and +0.144 MRR. Its
keyword branch is Lucene, with real analysis behind it, against our locally
computed fastembed BM25.

With reranking the order reverses. The explanation is in the candidate set,
measured directly:

| depth | Qdrant recall | Atlas recall |
|---|---|---|
| @10 | 0.750 | 0.844 |
| @20 | 0.875 | 0.844 |
| @50 | **0.906** | 0.844 |

**Atlas is flat.** Whatever it misses at 10 it still misses at 50. Qdrant keeps
finding more with depth. Both genuinely return 50 hits — Atlas's are simply
concentrated in fewer files (20 distinct against Qdrant's 26 on the same
query), so extra depth buys more chunks of documents it already found.

A reranker consumes the candidate set and does the ordering itself. So Atlas's
advantage — ordering the head well — is exactly the part the reranker replaces,
while Qdrant's advantage — a more file-diverse tail — is what the reranker has
to work with. **Ordering quality stops mattering once something else does the
ordering; candidate diversity starts mattering more.**

**Latency: Qdrant, decisively, and about half of it is geography.** Store-only
medians, query embedding paid once and shared:

| branch | Qdrant | Atlas | difference |
|---|---|---|---|
| hybrid (RRF) | 9.8 ms | 95.9 ms | +86 ms |
| dense only | 5.9 ms | 71.6 ms | +66 ms |
| keyword only | 3.9 ms | 65.6 ms | +62 ms |

A bare round trip is 2.5 ms to local Qdrant and 50 ms to this Atlas cluster, so
roughly 50 ms of the gap is network distance rather than engine speed. Net of
that, Atlas does about 46 ms of work per hybrid query against Qdrant's ~7 ms.
The cluster is also `M0`, which is shared and throttled; a dedicated tier would
narrow this and cannot close the network half.

End to end the gap shrinks to about 180 ms, because the query embedding and the
rerank call are identical for both and dominate.

**The honest summary, with everything else held equal:** Qdrant is faster by a
margin no tuning will erase at this distance, and marginally better on the
retrieval quality that actually ships (reranked). Atlas is better at raw
ranking and is the stronger choice if you would otherwise run no reranker at
all — which is also the configuration where its latency hurts least, since
there is no second API call to hide behind.

Neither difference is large enough to override an operational reason. If a
project already runs MongoDB, this measurement is not an argument against using
it.

### Two Atlas features deliberately not used

**Automated Embedding** — Atlas generating embeddings server-side on insert —
is not wired up, for three separate reasons, any one of which would be enough.
It requires `M10` or higher, so it is unavailable on the free tier at all. It
supports Voyage models only, which puts the provider abstraction that
`pydantic-ai` buys us back behind a vendor lock. And its **query** rate limit is
3 requests per minute — for an MCP tool an agent calls a dozen times in one
task, that is a hard stop rather than a tuning problem.

**`$rerank`** — Atlas running a Voyage reranker inside the aggregation — is a
genuine option and would fit the existing `Reranker` protocol as a third
implementation beside Voyage and the no-op. It is not built yet, and the reason
to be careful is the same one: it is Voyage-only server-side, so adopting it
would move a swappable client-side component into the database. Worth measuring
against the current client-side reranker before adopting, not instead of it.

---

### PDFs

**Needs the extra**: `poetry install --extras pdf`. Without it a PDF is recorded
and counted but not embedded, exactly as before the feature existed — the log
says so once per run rather than per file.

One chunk per page, split further when a page is too long for the budget.
`symbol_path` carries `page 12`, and that is the anchor to act on: there are no
line numbers in a PDF worth returning, so `start_line`/`end_line` describe the
extracted text and exist only to keep `location` well-formed.

Headings are not inferred. A PDF's text layer has no heading structure, only
visual size, and guessing one from font metrics is a separate project — a wrong
heading trail is worse than none, because it reads as authoritative.

**Three kinds of PDF are recorded but not indexed**, each logged distinctly
because they call for different responses:

- **no text layer** — a scan without OCR. The document is known and counted, so
  `status` distinguishes "we have this and cannot read it" from "we never saw
  it". Those need OCR and a fix to the walk respectively.
- **encrypted** — ordinary to find in a workspace, and nothing can be read.
- **damaged** — costs that one file, never the run.

Extraction happens in the **reader**, not the chunker, and that is a security
decision rather than a layering one: `read_source` is the single point where
bytes become text, and the only place the secret scanner sees. A chunker that
opened the file itself would be a second path to the embedding API that nothing
scans. A PDF is exactly the kind of document people paste credentials into.

Staleness re-extracts the PDF to compare, so `search.check_staleness` costs a
pymupdf parse per PDF per search — once per file, not per hit. Turn it off if
that bites.

## 6. Things that surprise people

**Adding a corpus under an existing root joins the main index** and shifts
every eval number for reasons unrelated to any code change. Give it its own
`workspace.name`.

**Changing `embed_doc_type` needs `--force`.** It changes what is embedded but
not chunk identity, so a normal run finds identical ids and does nothing.

**A manifest survives a store rebuild.** Point at a fresh Qdrant and the
decision ladder skips every file because the manifest says they are done.
`index` warns; `status` reports it as `inconsistent`; `--force` fixes it.

**`data/` is derived state.** The backup story is "re-index", not "back it up".

**Opaque files produce zero chunks.** A file whose bytes do not decode as text
is recorded but not embedded, so it cannot be found by search. An unknown
*language* is fine — that falls to the text chunker and stays searchable.
