"""What the MCP tools do, against a real embedded Qdrant.

Deliberately not driven through an MCP client session. Everything worth
asserting -- which types each tool selects, what an empty result says, whether
a truncated list admits it was truncated -- is a property of QueryService, and
testing it here keeps a subprocess and a protocol handshake out of the loop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from tests.conftest import make_source
from tests.fake_embedding_backend import FakeEmbeddingBackend
from tests.fake_sparse_backend import FakeSparseBackend
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.config import RerankConfig, SearchSection
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.mcp import (
    GUIDANCE_TYPES,
    QueryService,
    SearchResponse,
    TaxonomyService,
)
from workspace_indexer.mcp.tool_call_recorder import ToolCallRecorder
from workspace_indexer.mcp.tool_call_sink import ToolCallSink
from workspace_indexer.models import Chunk, DocumentType, EmbeddingSpace, FileKind, ToolCall
from workspace_indexer.rerank.noop_reranker import NoopReranker
from workspace_indexer.search.search_service import SearchService
from workspace_indexer.storage.qdrant_store import QdrantStore

SPACE = EmbeddingSpace(model="fake:model", dimensions=4)

# One document per type that matters, so a tool selecting the wrong family is
# visible as the wrong path rather than as a subtly different ranking.
MD = FileKind.MARKDOWN
CODE = FileKind.CODE
DOCS: list[tuple[str, str, DocumentType, FileKind]] = [
    ("repo/CONVENTIONS.md", "modules must be one class per file", DocumentType.NORMATIVE, MD),
    ("repo/docs/design.md", "the store is a protocol, swappable", DocumentType.DESIGN, MD),
    ("repo/README.md", "install it and run the indexer", DocumentType.GUIDE, MD),
    ("repo/CHANGELOG.md", "0.2.0 modules were split per class", DocumentType.RECORD, MD),
    ("repo/src/store.py", "class Store: def put(self): 1", DocumentType.IMPLEMENTATION, CODE),
    ("repo/tests/test_store.py", "def test_put(): assert Store().put()", DocumentType.TEST, CODE),
    ("repo/poetry.lock", "generated file do not edit", DocumentType.GENERATED, FileKind.TEXT),
]


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[QdrantStore]:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="labbox", payload_indexes=False)
    sparse = FakeSparseBackend()
    chunks: list[Chunk] = []
    for index, (rel_path, body, doc_type, kind) in enumerate(DOCS):
        source = make_source(
            body,
            kind=kind,
            language="python" if kind is FileKind.CODE else None,
            rel_path=rel_path,
            unit="repo",
        )
        chunk = build_chunk(
            source,
            "labbox",
            source_text=body,
            start_line=1,
            end_line=4,
            chunker="text",
            version=1,
            chunk_index=index,
            symbol_path=f"sym{index}",
        )
        chunk.meta.doc_type = doc_type
        chunks.append(chunk)
    dense = [[1.0, 0, 0, 0]] * len(DOCS)
    await store.upsert(SPACE, chunks, dense, sparse.encode_documents([d[1] for d in DOCS]))
    yield store
    await client.close()


def _queries(
    store: QdrantStore,
    *,
    max_tokens: int = 6000,
    check_staleness: bool = False,
    sink: ToolCallSink | None = None,
) -> QueryService:
    search = SearchService(
        store=store,
        embeddings=EmbeddingService(FakeEmbeddingBackend(dimensions=4)),
        sparse=FakeSparseBackend(),
        reranker=NoopReranker(),
        config=SearchSection(rerank=RerankConfig(enabled=False, model="fake:m")),
        space=SPACE,
    )
    return QueryService(
        search=search,
        taxonomy=TaxonomyService(store, SPACE),
        max_response_tokens=max_tokens,
        # Off by default here: the fixture's files never existed on disk, so
        # leaving it on would flag every hit and say nothing about the tools.
        check_staleness=check_staleness,
        recorder=ToolCallRecorder(sink),
    )


def _paths(response: SearchResponse) -> list[str]:
    return [r.rel_path for r in response.results]


# --- search_code ---------------------------------------------------------


async def test_search_code_excludes_tests_and_generated(store: QdrantStore) -> None:
    """A test naming a symbol twenty times outranks the file defining it,
    which is the most common way code search disappoints."""
    response = await _queries(store).search_code("store", limit=20)
    assert "repo/tests/test_store.py" not in _paths(response)
    assert "repo/poetry.lock" not in _paths(response)
    assert "repo/src/store.py" in _paths(response)


async def test_include_tests_puts_them_back(store: QdrantStore) -> None:
    response = await _queries(store).search_code("store", limit=20, include_tests=True)
    assert "repo/tests/test_store.py" in _paths(response)


async def test_results_are_line_anchored(store: QdrantStore) -> None:
    """`path:start-end` is what makes the next action a Read with no guessing."""
    response = await _queries(store).search_code("store", limit=5)
    assert response.results
    for result in response.results:
        assert result.location == f"{result.rel_path}:{result.start_line}-{result.end_line}"
        assert result.start_line >= 1


# --- find_guidance -------------------------------------------------------


async def test_find_guidance_returns_only_guidance_documents(store: QdrantStore) -> None:
    response = await _queries(store).find_guidance("how must modules be structured", limit=20)
    found = set(_paths(response))
    assert found <= {"repo/CONVENTIONS.md", "repo/docs/design.md", "repo/README.md"}
    assert "repo/CONVENTIONS.md" in found


async def test_find_guidance_beats_the_changelog(store: QdrantStore) -> None:
    """The motivating case. The changelog and the convention say almost the
    same words; only one of them is a rule."""
    response = await _queries(store).find_guidance("one class per file", limit=20)
    assert "repo/CHANGELOG.md" not in _paths(response)


async def test_find_guidance_can_narrow_to_one_type(store: QdrantStore) -> None:
    response = await _queries(store).find_guidance(
        "how is the store shaped", limit=20, doc_type=DocumentType.DESIGN
    )
    assert _paths(response) == ["repo/docs/design.md"]


async def test_empty_guidance_says_what_to_do_next(store: QdrantStore) -> None:
    """An empty result set must never read as "no such thing exists"."""
    response = await _queries(store).find_guidance(
        "kubernetes ingress", limit=20, doc_type=DocumentType.REFERENCE
    )
    assert response.results == []
    assert response.note is not None
    assert "list_document_types" in response.note
    # The filters we defaulted it into are invisible to the caller otherwise.
    assert response.applied_filters


# --- get_file_context ----------------------------------------------------


async def test_file_context_returns_the_file(store: QdrantStore) -> None:
    response = await _queries(store).get_file_context("repo/src/store.py")
    assert _paths(response) == ["repo/src/store.py"]


async def test_file_context_accepts_a_path_suffix(store: QdrantStore) -> None:
    """A model that typed the path itself gives the tail of it, and telling it
    its own file does not exist is a bad answer."""
    response = await _queries(store).get_file_context("src/store.py")
    assert _paths(response) == ["repo/src/store.py"]


async def test_file_context_on_an_unknown_path_explains_itself(store: QdrantStore) -> None:
    response = await _queries(store).get_file_context("nope/missing.py")
    assert response.results == []
    assert response.note is not None
    assert "missing.py" in response.note


# --- budgeting -----------------------------------------------------------


async def test_over_budget_results_are_dropped_and_declared(store: QdrantStore) -> None:
    """A clipped list that looks complete is how an agent concludes it has seen
    everything when it has seen the first two."""
    response = await _queries(store, max_tokens=80).search_code("store", limit=20)
    assert response.dropped_for_budget > 0
    assert response.returned < response.total_matches
    assert response.note is not None
    assert "budget" in response.note


async def test_nothing_is_dropped_silently(store: QdrantStore) -> None:
    response = await _queries(store).search_code("store", limit=20)
    assert response.dropped_for_budget == 0
    assert response.returned == response.total_matches


async def test_guidance_includes_guides_not_only_specs(store: QdrantStore) -> None:
    """A measured decision, pinned so it cannot quietly regress.

    Restricting find_guidance to normative + design scored no better than plain
    search over the eight guidance eval cases (recall 0.812, MRR 0.792): the
    filter gained CLAUDE.md and lost CONTRIBUTING.md, which is a `guide`.
    Including guides gives 0.938 / 0.900. Someone tightening this list back up
    for tidiness should have to delete this test to do it.
    """
    assert DocumentType.GUIDE in GUIDANCE_TYPES

    response = await _queries(store).find_guidance("how do I install and run this", limit=20)
    assert "repo/README.md" in _paths(response)


async def test_guidance_still_excludes_the_record_and_the_code(store: QdrantStore) -> None:
    """The other half of the trade. Widening the filter must not widen it into
    the changelog, which is what the tool exists to outrank."""
    response = await _queries(store).find_guidance("how must modules be structured", limit=20)
    found = set(_paths(response))
    assert "repo/CHANGELOG.md" not in found
    assert "repo/src/store.py" not in found
    assert "repo/tests/test_store.py" not in found


async def test_staleness_can_be_turned_off_for_a_source_less_deployment(
    store: QdrantStore,
) -> None:
    """A deployment that puts the MCP server next to Qdrant rather than next to
    the code cannot read the indexed files.

    With the check on, every hit comes back flagged stale -- both wrong and
    useless, because a flag on everything carries no signal. The fixture's
    abs_paths point into a tmp dir that no longer holds these files, which is
    exactly that situation.
    """
    checked = await _queries(store, check_staleness=True).search_code("store", limit=5)
    assert checked.results
    assert all(r.stale for r in checked.results)

    unchecked = await _queries(store, check_staleness=False).search_code("store", limit=5)
    assert unchecked.results
    assert not any(r.stale for r in unchecked.results)


async def test_file_context_honours_the_same_switch(store: QdrantStore) -> None:
    response = await _queries(store, check_staleness=False).get_file_context("repo/src/store.py")
    assert response.results
    assert not any(r.stale for r in response.results)


class _Recorded(list[ToolCall]):
    """Captures what the tools recorded, without a database."""

    def record_tool_call(self, call: ToolCall) -> None:
        self.append(call)


async def test_every_tool_records_its_own_name(store: QdrantStore) -> None:
    """`search.query` cannot say which tool ran; that is the whole gap."""
    sink = _Recorded()
    queries = _queries(store, sink=sink)

    await queries.search_code("store", limit=5)
    await queries.find_guidance("how must modules be structured", limit=5)
    await queries.get_file_context("repo/src/store.py")

    assert [c.tool for c in sink] == ["search_code", "find_guidance", "get_file_context"]


async def test_the_record_carries_what_came_back(store: QdrantStore) -> None:
    sink = _Recorded()
    response = await _queries(store, sink=sink).search_code("store", limit=5)

    call = sink[0]
    assert call.returned_paths == [r.rel_path for r in response.results]
    assert call.total_matches == response.total_matches
    assert call.duration_ms > 0


async def test_an_empty_call_is_recorded_as_disappointing(store: QdrantStore) -> None:
    """The harvesting filter: an agent asked and got nothing, which is an eval
    case waiting to be written."""
    sink = _Recorded()
    await _queries(store, sink=sink).find_guidance(
        "kubernetes ingress", limit=5, doc_type=DocumentType.REFERENCE
    )

    assert sink[0].returned_paths == []
    assert sink[0].disappointed is True
    assert sink[0].note is not None


async def test_a_budget_clipped_call_is_also_disappointing(store: QdrantStore) -> None:
    """Returned nothing and returned less than it found are different
    problems, and both are worth harvesting."""
    sink = _Recorded()
    await _queries(store, max_tokens=80, sink=sink).search_code("store", limit=20)

    assert sink[0].returned_paths
    assert sink[0].dropped_for_budget > 0
    assert sink[0].disappointed is True


async def test_the_filters_the_tool_applied_are_recorded(store: QdrantStore) -> None:
    """An empty result means different things with and without a filter, so a
    replayable record has to include the ones the tool defaulted into."""
    sink = _Recorded()
    await _queries(store, sink=sink).search_code("store", limit=5)

    assert "exclude_doc_types" in sink[0].parameters


# --- locations only (#71) ------------------------------------------------


async def test_locations_only_returns_anchors_without_code(store: QdrantStore) -> None:
    """The survey case: find where something lives, then read only what
    matters. Measured on real tool calls, the responses that overflowed the
    budget were all bodied ones asking for many hits."""
    response = await _queries(store).search_code("auth", limit=8, locations_only=True)

    assert response.results, "no hits, so this asserts nothing about omission"
    assert all(r.text == "" and r.text_omitted for r in response.results)
    # The anchor survives, because it is the entire point of the mode.
    assert all(r.location and r.rel_path and r.end_line >= r.start_line for r in response.results)


async def test_locations_only_says_so_even_when_nothing_was_dropped(store: QdrantStore) -> None:
    """An agent holding results with empty bodies and no note cannot tell
    "withheld" from "these chunks are blank"."""
    response = await _queries(store).search_code("auth", limit=8, locations_only=True)

    assert response.dropped_for_budget == 0
    assert response.note is not None
    assert "locations_only" in response.note
    assert "get_file_context" in response.note


async def test_a_normal_search_says_nothing_about_omission(store: QdrantStore) -> None:
    """The note must not appear on the default path, where bodies are present
    and there is nothing to warn about."""
    response = await _queries(store).search_code("auth", limit=8)

    assert all(r.text and not r.text_omitted for r in response.results)
    assert response.note is None or "locations_only" not in response.note


async def test_a_clipped_response_reports_both_the_drop_and_the_omission(
    store: QdrantStore,
) -> None:
    """The two notes are independent facts and a caller needs both: how many
    matches it is not seeing, and that what it did get has no code in it."""
    response = await _queries(store, max_tokens=80).search_code(
        "auth", limit=8, locations_only=True
    )

    assert response.dropped_for_budget > 0
    assert response.note is not None
    assert "dropped to stay inside" in response.note
    assert "locations_only" in response.note


async def test_find_guidance_omits_bodies_too(store: QdrantStore) -> None:
    """Surveying which documents govern a topic is the same job as surveying
    where a symbol lives."""
    response = await _queries(store).find_guidance("conventions", limit=8, locations_only=True)

    assert response.results, "no hits, so this asserts nothing about omission"
    assert all(r.text == "" and r.text_omitted for r in response.results)


async def test_get_file_context_always_carries_the_code(store: QdrantStore) -> None:
    """It exists to return the body of a file already located. There is no
    locations_only for it, and adding one would make it a worse `search_code`."""
    from inspect import signature

    assert "locations_only" not in signature(QueryService.get_file_context).parameters
