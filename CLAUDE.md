# workspace-indexer

A hybrid (dense + BM25 sparse) index over a multi-repo workspace, built so an
LLM can find code and docs by meaning instead of by grep. See
`docs/iteration-1-plan.md` for the design and the reasoning behind it, and
`docs/iteration-2-plan.md` for document classification and the MCP server.

Layers, and the seam each one owns. Kept complete on purpose: a list that
covers half the packages is worse than none, because it reads as if the others
do not exist.

- `config/` — `workspace.yaml` (what to index, committable) and `.env` (how to
  index, secret). Only the former is hot-reloaded.
- `obs/` — logging. Set up before anything else runs.
- `models/` — the shared value objects. No behaviour beyond validation.
- `discovery/` — what files exist and what we know about them. Never opens a
  file. Consults git for tracked status, because a `.gitignore` pattern has no
  effect on a file already in the index and matching patterns alone dropped
  committed source.
- `secrets/` — keeps credentials out of the index, and therefore out of API
  requests. Withholding is destructive, so a false positive is silent data loss.
- `classification/` — what *role* a document plays (spec, design, changelog),
  which is a different axis from how to chunk it.
- `chunking/` — one strategy per `FileKind`, resolved through a registry.
- `embedding/` — dense via pydantic-ai (provider-swappable), sparse via fastembed.
- `rerank/` — `Reranker` protocol; `NoopReranker` is how reranking turns off.
- `storage/` — `VectorStore` protocol over **Qdrant and MongoDB Atlas**, one
  collection per embedding space. A cross-backend contract suite keeps them
  honest.
- `state/` — SQLite manifest driving incremental reindex.
- `graph/` — what files reference, what references them, and where each
  reference points: imports, HTTP route edges that cross repositories, and the
  origin of an import — first-party, a declared dependency, the language's
  framework, or unclassified. Origin is what makes coverage mean something:
  only a first-party edge can resolve to a file here, so only there is a
  failure a defect.
- `grounding/` — whether a codebase records *why* it is the way it is, so an
  empty answer can be told from an absent one.
- `worktrees/` — reporting results as one git checkout sees them.
- `pipeline/` — the indexing run itself, including the brakes on deletion.
- `search/` — embed the query, fuse the branches, rerank, flag stale results.
- `mcp/` — the MCP surface: how an agent reaches the index mid-session.
- `watching/` — keeping the index fresh without a manual run.
- `evaluation/` — retrieval-quality measurement. Every tuning knob is settled
  here rather than by feel.

# Code organization mandates

These are not preferences to weigh against other factors. Follow them.

- **One class per file.** Every class gets its own module, named after it in
  snake_case (`class SearchHit` lives in `search_hit.py`). This holds even for
  small value objects — do not group "related" classes into one module because
  they feel cohesive.
- To keep call sites readable under that rule, group the modules into a package
  and re-export from its `__init__.py`, so callers still write
  `from workspace_indexer.models import Chunk` rather than reaching into
  `workspace_indexer.models.chunk`.
- `tests/test_one_class_per_file.py` enforces both halves of this. If it fails,
  fix the layout, not the test.

# Testing mandate

- **Tests ship with the code, in the same unit of work.** Any module with
  branching or real logic gets its tests written as part of building it, never
  deferred to a later "testing phase."
- If behaviour gets verified with a throwaway shell command or a scratch
  script, that check belongs in the test suite instead. Do not delete it.
- Modules with no logic to get wrong — plain enums, plain value objects,
  `__init__.py` re-exports — do not need tests. Say which ones were skipped and
  why rather than testing them for the sake of coverage.
- Prefer real filesystems and real git repos in fixtures over mocks. Discovery
  is about how the filesystem actually behaves; a mock confirms our assumptions
  instead of the truth.

# Typing mandate

- **Everything is typed, and pyright runs in strict mode** — `src/` and
  `tests/` alike. `typeCheckingMode = "strict"` is set in `pyproject.toml`;
  `poetry run pyright` must report zero errors before work is called done.
- Test code is held to the same bar as source. Strict mode on tests is what
  catches an untyped fixture quietly widening a parameter to `Any`, which then
  hides real type errors in the code under test.
- Prefer narrowing over silencing. Use an `assert x is not None` to narrow an
  Optional rather than a `# type: ignore`. Where a suppression is genuinely
  needed, make it a targeted `# pyright: ignore[ruleName]`, never a bare one.
- Do not reach into private attributes to make a test pass. If a behaviour is
  worth asserting, assert the observable effect instead.

# Commands

```bash
poetry run pytest -q                     # full suite
poetry run pytest -m "not integration"   # skip anything needing network/Qdrant
poetry run ruff check src/ tests/
poetry run pyright                       # strict; must be 0 errors

# All three run in CI on every push and PR (.github/workflows/ci.yml),
# with no API key and no network. If it passes locally it passes there.
poetry run workspace-indexer index --dry-run      # chunk plan + token estimate, no API calls
```
