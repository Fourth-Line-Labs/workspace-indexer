# Testing

What the suite guards, and the things that cost us time before it did.

`poetry run pytest -q -m "not integration"` is what CI runs: no network, no
credentials, no API key. `poetry run pytest -q` adds the integration tests and
needs both a running Qdrant and whatever credentials are configured; every one
of them skips cleanly when its credential is absent.

---

## 1. The three ways this project has broken

Every guard below exists because of a real failure, and all three shared a
shape: **the break had nothing to do with the code being changed, and nothing
errored.** That combination is what makes them worth automating — a loud
failure in the thing you just edited finds itself.

### A list drifted

`impact_of` shipped as a fifth MCP tool. `test_mcp_stdio_integration` went on
asserting the four that existed before it, and stayed green, because CI runs
`-m "not integration"` and nobody ran it by hand. The reference doc's tool list
was hand-typed and had the same problem in the other direction: a tool could go
undocumented and nothing said so.

**Guard:** `tests/test_no_drift.py`. One rule — *if a list appears twice, one of
them is derived, or a test proves they match.* It found two more hand-written
enumerations on its first run.

The AST parser that reads the tool list out of `server_factory` lives in
`tests/mcp_tool_names.py`, in one copy. It was previously pasted into three test
modules, which was the same bug one level up.

### State leaked between tests

`log_once` dedupes for the life of the process — right in production, wrong
across a test session, where whichever test runs first consumes the event.
Worse, `configure_logging` handed structlog a **new processor list** on every
call, which orphaned any logger cached against the previous one (see §3).

**Guards:** `tests/test_no_leaks.py` finds every module-level name the codebase
actually *mutates* and requires it to be listed with the mechanism that clears
it. Adding process-global state is then a deliberate act with a reset attached.
`tests/test_log_once_isolation.py` proves the autouse ledger reset in
`conftest.py` is load-bearing — it needs its own module, because
`test_logging_setup.py`'s local reset fixture clears the ledger as a side
effect and hid the leak entirely. *A local fixture was masking a global bug.*

### Two backends drifted apart

With one storage implementation, a bug is a bug. With two, the failure mode is
subtler: each stays internally consistent while the pair stops agreeing, and
nothing above `storage/` can see it. A filter field declared for one index and
not the other produces a search that silently ignores the filter.

**Guard:** `tests/test_vector_store_contract.py` runs the same behaviours
against Qdrant and Atlas. Every assertion is about observable behaviour through
the protocol — nothing reaches for a Qdrant filter object or a Mongo pipeline,
because those are already tested per backend.

Qdrant runs on every invocation (embedded, no network), so CI covers the
contract. Atlas is `integration`. **That asymmetry is why the Qdrant parameter
exists at all:** a contract only one backend ever runs is a contract that
notices nothing in CI.

A test asserts the contract grows when the protocol does, so adding a method to
`VectorStore` cannot leave a hole neither backend is checked against.

---

## 2. Integration tests, and why they rot

They need credentials, a live index and real services, so they cannot run on a
pull request. That is exactly how the stdio test went stale for two PRs.

Three things reduce the exposure:

- **CI collects them** (`pytest -m integration --collect-only`). That imports
  every module and evaluates every decorator, so an import error, a renamed
  fixture or a bad signature fails in CI. It cannot catch a wrong assertion.
- **The drift guard runs in CI** and does catch the wrong-assertion case, at
  least for enumerations.
- **`.github/workflows/integration.yml`** runs the real thing on manual
  dispatch and weekly. Never on a pull request: those tests cost money, and a
  fork's PR must not see the secrets. It reports skips with `-rs`, because a
  green run that verified nothing otherwise looks identical to one that
  verified everything.

**Before merging anything that changes the MCP surface, the storage seam or the
watcher, run the full suite locally** — `poetry run pytest -q`, no marker
filter. It takes about two minutes.

---

## 3. structlog: two traps worth knowing

`capture_logs` mutates the configured processor list **in place** rather than
replacing it. Its source says why:

> always keep the list instance intact to not break references held by bound
> loggers

With `cache_logger_on_first_use=True` — which we use — a logger is frozen
against whatever list was live the first time it was used, and every module
here binds its logger at import. In-place mutation is the only way a later
`capture_logs` can reach such a logger.

So handing structlog a **fresh list** on a second `configure` orphans every
cached logger: `capture_logs` mutates the new list, the logger reads the old
one, and the event goes to the real sinks instead of the capture.

```python
configure_logging(...)                        # list A
log.info("warm")                              # logger cached against list A
reset_for_tests()
configure_logging(...)                        # list B installed
with capture_logs() as logs: log.info("x")    # mutates B; logger reads A
assert logs == []                             # reproduced
```

`obs/logging.py` therefore keeps **one list for the life of the process** and
refills it by slice assignment. `reset_for_tests` reinstalls it after
`structlog.reset_defaults()` swaps in one of structlog's own.

`cache_logger_on_first_use=False` also fixes it and was measured — 38 → 47 µs
per call, 22%, about 0.1 s over a run the size of the current index. Rejected
because keeping both the cache and the list identity costs nothing, and because
the new list was the actual defect: turning off the cache only makes the defect
stop mattering.

**The reason this deserved a real fix:** `capture_logs` returning `[]` reads
exactly like *"the code never logged"*. It surfaced as two `StopIteration`s in
`test_watcher.py`; the same leak under an `assert not [...]` would have been
invisible.

### The other way `capture_logs` returns nothing

The trap above is about a *fresh list*. There is a second one with the same
symptom and a different cause, and it costs the same hour.

`capture_logs` installs its capturing processor when it is entered. Anything
that calls `configure_logging` **after** that replaces the processor list
outright and the capture is silently undone:

```python
with structlog.testing.capture_logs() as logs:
    CliRunner().invoke(app, ["watch", ...])   # calls configure_logging inside
assert logs == []                              # empty -- and the line *was* logged
```

It surfaced writing the tests for #61. `capture_logs` came back empty while
pytest's own captured output plainly showed the event being emitted a few lines
below — which is the tell: **if the event appears in captured stdout but not in
`logs`, the capture was replaced, not missed.**

The fix is not to fight it. Any test that drives a CLI command should assert on
the **log file** instead:

```python
entries = [json.loads(line) for line in (tmp_path / "l-watch.jsonl").read_text().splitlines()]
```

That is closer to the thing under test anyway — the complaint in #61 was that a
crash left no trace *in the JSONL*, so the JSONL is what has to be checked. Two
details it needs: the file name carries the command (`l-watch.jsonl`, not
`l.jsonl`), and `configure_logging` is idempotent, so a test module driving more
than one command needs `reset_for_tests()` between them or the first one pins
the path for all of them.

---

## 4. MongoDB Atlas: five things that will confuse you

All five were learned by running the tests, not by reading the docs.

**Search indexes build asynchronously.** A collection can hold every document
and answer every query with nothing for a minute afterwards. That looks exactly
like a broken pipeline. `describe_vectors` reports whether each index is
queryable yet, and the fixtures wait on it.

**The search-index budget is per cluster and small.** This store needs two per
collection (one `vectorSearch`, one `search`). Free allows 3, so exactly one
indexed collection; Flex allows 10 — measured, by creating scratch collections
until Atlas refused, rather than taken from the docs. The store contract's
"drop removes it" assertion lives in fixture teardown so it works on either.

**Dropping a collection frees the quota lazily.** Drop-then-recreate fails with
*"The maximum number of FTS indexes has been reached for this instance size"* —
which reads like a quota problem and is a timing one. Worse, once the
collection is gone there is no way to ask about its indexes, so polling for
their absence returns "none" immediately while Atlas is still letting go. The
fixtures **retry the create** instead, because trying is the only honest
question.

**The lazy release is what actually bites, not the budget.** Mirroring the real
workspace across and running `pytest -q` failed 36 setups with *"maximum number
of FTS indexes"* — on a Flex cluster with a budget of 10 and at most 6 in use.
The cause is not capacity: a full run creates and drops several index pairs,
and the dropped ones are still counted for minutes afterwards, so they
accumulate past the limit.

I first wrote this up as a Free-tier capacity limit. That was wrong, and wrong
in the way worth recording: the error message says "maximum number of FTS
indexes", which reads as a quota, and the fix that worked (drop the mirror)
was consistent with the wrong explanation. The fixtures now **retry the
create** rather than polling for the absence of indexes, which cannot work --
once the collection is gone, asking about its indexes returns "none"
immediately while Atlas is still letting go. A full run self-heals.

**`AsyncMongoClient` binds to the event loop it was created on** and raises
rather than reconnecting. A module-scoped fixture therefore needs its tests on
the module's loop:

```python
pytestmark = pytest.mark.asyncio(loop_scope="module")
```

---

## 5. Conventions

**Real things over mocks.** Discovery is about how the filesystem actually
behaves; a mock confirms our assumptions instead of the truth. Fixtures use
real temp directories, real `git init`, real SQLite files, real embedded
Qdrant.

The one fake in the storage layer is the Mongo driver
(`tests/fake_mongo_*.py`), and the reasoning is written into it: Atlas Search
is a managed service, `mongomock` implements neither `$vectorSearch` nor
`$search`, and a plain `mongod` would accept the pipelines and answer them
wrongly — which is worse than not running them. So the seam is the driver and
the assertions are about **the query documents we send**, which is where the
bugs actually were. Two in the Mongo store's first draft:

- asking for `searchScore` on a `$vectorSearch` stage. Not an error — a missing
  field, so every hit scores 0.0 and the ranking collapses silently.
- a per-branch `$addFields` inside `$rankFusion` input pipelines, which the
  stage rejects.

Neither is visible in anything a mocked *result* could show.

**Check the test fails without the fix.** Every guard in this document was
verified by reverting the fix and watching the test fail. Two did not fail on
the first attempt and had to be rewritten — the `log_once` pair (a local
fixture was hiding the leak) and the reference-doc guard (a passing mention of
a tool name satisfied it, so it now requires a real entry).

**PDF fixtures must assert they fit.** `page.insert_textbox` returns the space
left over and writes *nothing* when the text overflows, so an oversized fixture
produces an empty PDF and a later assertion that fails for a reason unrelated to
the code. `write_pdf` in `test_pdf_chunker.py` raises instead. It caught two
tests on their first run.

**A wrong fixture looks like a failing feature.** The mirror tests asserted
`[2,0,0,0]` was nearer than `[1,0,0,0]`; the collection is cosine, which
ignores magnitude, so those are the same vector. The fixture was wrong, not the
code. When a new test fails, suspect it before the thing it tests.
