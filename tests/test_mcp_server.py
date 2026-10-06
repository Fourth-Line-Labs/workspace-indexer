"""The protocol layer: what a client actually sees.

Thin by design, but three things can only go wrong here -- a tool that is not
registered, a description that does not tell the agent the vocabulary, and an
unknown document type that comes back as an empty list instead of an error.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.types import CallToolResult, InputRequiredResult, TextContent
from qdrant_client import AsyncQdrantClient

from tests.conftest import make_source
from tests.fake_embedding_backend import FakeEmbeddingBackend
from tests.fake_sparse_backend import FakeSparseBackend
from tests.mcp_tool_names import registered_tool_names
from workspace_indexer.app_context import AppContext
from workspace_indexer.chunking.chunk_factory import build_chunk
from workspace_indexer.config import RerankConfig, SearchSection, Settings
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.grounding import CoverageService
from workspace_indexer.mcp import (
    TAXONOMY_URI,
    EmptyIndexError,
    GroundingService,
    ImpactService,
    QueryService,
    TaxonomyService,
    WorkspaceServices,
    build_mcp_server,
)
from workspace_indexer.mcp.server_factory import preflight
from workspace_indexer.models import Chunk, DocumentType, EmbeddingSpace, FileKind
from workspace_indexer.rerank.noop_reranker import NoopReranker
from workspace_indexer.search.search_service import SearchService
from workspace_indexer.state import Manifest
from workspace_indexer.storage.qdrant_store import QdrantStore

SPACE = EmbeddingSpace(model="fake:model", dimensions=4)

DOCS: list[tuple[str, DocumentType]] = [
    ("repo/CONVENTIONS.md", DocumentType.NORMATIVE),
    ("repo/src/store.py", DocumentType.IMPLEMENTATION),
    ("repo/tests/test_store.py", DocumentType.TEST),
]


@pytest.fixture
async def populated(tmp_path: Path) -> AsyncIterator[QdrantStore]:
    client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    store = QdrantStore(client, workspace="labbox", payload_indexes=False)
    sparse = FakeSparseBackend()
    chunks: list[Chunk] = []
    for index, (rel_path, doc_type) in enumerate(DOCS):
        source = make_source(
            "store the thing",
            kind=FileKind.CODE,
            language="python",
            rel_path=rel_path,
            unit="repo",
        )
        chunk = build_chunk(
            source,
            "labbox",
            source_text="store the thing",
            start_line=1,
            end_line=1,
            chunker="text",
            version=1,
            chunk_index=index,
        )
        chunk.meta.doc_type = doc_type
        chunks.append(chunk)
    await store.upsert(
        SPACE,
        chunks,
        [[1.0, 0, 0, 0]] * len(chunks),
        sparse.encode_documents(["store the thing"] * len(chunks)),
    )
    yield store
    await client.close()


@pytest.fixture
def queries(populated: QdrantStore) -> QueryService:
    search = SearchService(
        store=populated,
        embeddings=EmbeddingService(FakeEmbeddingBackend(dimensions=4)),
        sparse=FakeSparseBackend(),
        reranker=NoopReranker(),
        config=SearchSection(rerank=RerankConfig(enabled=False, model="fake:m")),
        space=SPACE,
    )
    return QueryService(search=search, taxonomy=TaxonomyService(populated, SPACE))


@pytest.fixture
def server(queries: QueryService, tmp_path: Path) -> MCPServer:
    """The real server over a real (empty) manifest.

    A real SQLite file rather than a stub: the graph tools are SQL, and a mock
    would confirm our assumptions about the query instead of the query.
    """
    manifest = Manifest(tmp_path / "manifest.sqlite3")
    return build_mcp_server(
        {
            "labbox": WorkspaceServices(
                queries=queries,
                impact=ImpactService(manifest),
                grounding=GroundingService(CoverageService(manifest)),
            )
        }
    )


async def test_every_tool_is_registered(server: MCPServer) -> None:
    tools = await server.list_tools()
    assert {t.name for t in tools} == set(registered_tool_names())


async def test_tool_descriptions_carry_the_vocabulary(server: MCPServer) -> None:
    """The type list must be in context *before* the agent picks one. A round
    trip to discover it is a round trip it will skip, and then it guesses."""
    tools = {t.name: t for t in await server.list_tools()}
    guidance = tools["find_guidance"]
    schema = json.dumps(guidance.input_schema)
    for doc_type in DocumentType:
        assert doc_type.value in schema


async def test_search_code_dispatches_and_excludes_tests(server: MCPServer) -> None:
    result = await server.call_tool("search_code", {"query": "store", "limit": 10})
    body = json.dumps(_payload(result))
    assert "repo/src/store.py" in body
    assert "repo/tests/test_store.py" not in body


async def test_unknown_doc_type_is_an_error_not_an_empty_list(
    server: MCPServer,
) -> None:
    """The acceptance criterion that stops the silent-empty-result failure mode
    from shipping.

    Asserting the *message*, not just the failure. The SDK strips the text of
    an unrecognised exception and sends the model a bare "Error executing tool
    find_guidance"; only a ToolError carries its own words through. An error
    the agent cannot act on is barely better than the empty list.
    """
    with pytest.raises(ToolError) as caught:
        await server.call_tool("find_guidance", {"query": "structure", "doc_type": "blueprint"})

    message = str(caught.value)
    assert "blueprint" in message
    assert "normative" in message
    assert "spec" in message


async def test_a_plain_exception_reaches_the_model_with_its_text_removed() -> None:
    """Why the conversion above is deliberate rather than incidental.

    Pins the SDK behaviour the design works around, on a throwaway server so
    the assertion is about the SDK and not about which of our tools happens to
    fail. If a future version stops stripping the message, this test fails and
    the boundary conversion can be simplified away.
    """
    probe = MCPServer(name="probe")

    @probe.tool()
    async def explode() -> str:
        raise ValueError("a very specific and useful explanation")

    _ = explode
    with pytest.raises(UnexpectedToolError) as caught:
        await probe.call_tool("explode", {})
    assert "very specific" not in str(caught.value)


async def test_an_alias_is_not_an_error(server: MCPServer) -> None:
    result = await server.call_tool("find_guidance", {"query": "structure", "doc_type": "spec"})
    assert getattr(result, "isError", False) is False


async def test_taxonomy_resource_is_served_as_json(server: MCPServer) -> None:
    resources = await server.list_resources()
    assert str(resources[0].uri) == TAXONOMY_URI

    payload = json.loads(await _resource(server, TAXONOMY_URI))
    assert payload["taxonomy_version"] >= 1
    assert {e["name"] for e in payload["types"]} == {t.value for t in DocumentType}


async def test_resource_and_tool_agree(server: MCPServer) -> None:
    """Both surfaces exist because clients differ in which one a model reliably
    sees. They must never disagree about what is in the workspace."""
    from_resource = json.loads(await _resource(server, TAXONOMY_URI))

    result = await server.call_tool("list_document_types", {})
    from_tool = _payload(result)

    assert from_resource["types"] == from_tool["types"]


async def _resource(server: MCPServer, uri: str) -> str:
    """The body of a resource read, as text.

    The SDK returns an iterable of content parts whose payload is `str | bytes`
    depending on the mime type; ours is JSON, so anything else is a bug in the
    server rather than something to accommodate here.
    """
    result = await server.read_resource(uri)
    assert not isinstance(result, InputRequiredResult)
    parts = [part.content for part in result]
    assert all(isinstance(part, str) for part in parts)
    return "".join(part for part in parts if isinstance(part, str))


def _payload(result: CallToolResult | InputRequiredResult) -> dict[str, Any]:
    """Structured output where the SDK provides it, decoded text otherwise."""
    assert isinstance(result, CallToolResult)
    if result.structured_content is not None:
        return dict(result.structured_content)
    text = "".join(b.text for b in result.content if isinstance(b, TextContent))
    loaded: object = json.loads(text)
    assert isinstance(loaded, dict)
    return dict(loaded)  # pyright: ignore[reportUnknownArgumentType]


async def test_preflight_refuses_to_serve_an_empty_index(tmp_path: Path) -> None:
    """The deployment failure this catches is silent and total.

    An MCP client launches the server from its own working directory, so a
    relative QDRANT_PATH resolves somewhere new and an unreachable `.env`
    drops the process into embedded mode against a directory that does not
    exist. It starts up perfectly and answers every question with "nothing
    found", which the agent believes.
    """
    client = AsyncQdrantClient(path=str(tmp_path / "empty"))
    store = QdrantStore(client, workspace="labbox", payload_indexes=False)
    ctx = _stub_context(store, tmp_path)

    with pytest.raises(EmptyIndexError) as caught:
        await preflight(ctx)

    message = str(caught.value)
    # Actionable on its own: which store it looked at, and what to check.
    assert "labbox__fake_model_4" in message
    assert "status" in message
    assert "--config" in message
    await client.close()


async def test_preflight_passes_over_a_populated_index(
    populated: QdrantStore, tmp_path: Path
) -> None:
    await preflight(_stub_context(populated, tmp_path))


def _stub_context(store: QdrantStore, tmp_path: Path) -> AppContext:
    """The three fields preflight reads, with the rest left unbuilt.

    Constructing a real AppContext would load config, configure logging and
    build an embedding backend -- none of which preflight touches.
    """
    return cast(
        "AppContext",
        SimpleNamespace(
            store=store,
            space=SPACE,
            settings=Settings(state_db=tmp_path / "manifest.sqlite3"),
        ),
    )


async def test_grounding_tool_answers_over_the_real_server(server: MCPServer) -> None:
    """Registered, dispatchable, and shaped as the agent will receive it."""
    payload = _payload(await server.call_tool("grounding", {}))

    assert "repositories" in payload
    assert "note" in payload


async def test_grounding_rejects_an_unknown_repository_with_a_usable_message(
    server: MCPServer,
) -> None:
    """An empty result here would read as "this repository records no reasons"
    -- the strongest claim the tool can make, manufactured out of a typo."""
    with pytest.raises(ToolError) as caught:
        await server.call_tool("grounding", {"repo": "no-such-repo"})

    assert "no-such-repo" in str(caught.value)


async def test_the_content_tools_take_a_worktree(server: MCPServer) -> None:
    """Present on all three, absent from the checkout-agnostic ones.

    `grounding` and `list_document_types` answer about a repository rather than
    a working copy, and `impact_of` reads edges the index already resolved --
    making those ask for a checkout would tax every graph query for nothing.
    """
    tools = {t.name: t for t in await server.list_tools()}
    guarded = {"search_code", "find_guidance", "get_file_context"}

    for name in guarded:
        assert "worktree" in tools[name].input_schema["properties"], name
    for name in set(tools) - guarded:
        assert "worktree" not in tools[name].input_schema.get("properties", {}), name


async def test_a_workspace_without_worktrees_needs_no_choice(server: MCPServer) -> None:
    """The guard must stay invisible where it does not apply."""
    result = await server.call_tool("search_code", {"query": "store", "limit": 3})

    assert not getattr(result, "isError", False)


# ---- locations only (#71) ---------------------------------------------------


@pytest.mark.parametrize("tool", ["search_code", "find_guidance"])
async def test_locations_only_is_offered_on_both_search_tools(server: MCPServer, tool: str) -> None:
    """An option the agent cannot see is an option it will not use, and the
    schema is the only place it can see one."""
    tools = {t.name: t for t in await server.list_tools()}
    schema = tools[tool].input_schema

    assert "locations_only" in schema["properties"]
    assert schema["properties"]["locations_only"]["default"] is False


async def test_the_locations_only_description_warns_it_is_not_grep(
    server: MCPServer,
) -> None:
    """The mode fits more hits, which makes a short list look more complete
    than it is. Results stay ranked and capped, so absence still is not proof
    -- and the schema is where an agent reads that."""
    tools = {t.name: t for t in await server.list_tools()}
    description = json.dumps(tools["search_code"].input_schema["properties"]["locations_only"])

    assert "grep" in description
    assert "ranked" in description


async def test_locations_only_reaches_the_service_through_the_tool(
    server: MCPServer,
) -> None:
    """The parameter existing in the schema proves nothing about it being
    wired: an unpassed argument silently keeps the default."""
    bodied = _payload(await server.call_tool("search_code", {"query": "store", "limit": 5}))
    anchors = _payload(
        await server.call_tool(
            "search_code", {"query": "store", "limit": 5, "locations_only": True}
        )
    )

    assert bodied["results"], "no hits, so this compares nothing"
    assert all(r["text"] for r in bodied["results"])
    assert all(r["text"] == "" and r["text_omitted"] for r in anchors["results"])
    assert [r["location"] for r in anchors["results"]] == [r["location"] for r in bodied["results"]]


@pytest.mark.parametrize("tool", ["search_code", "find_guidance"])
async def test_the_server_instructions_advertise_locations_only_on_both(
    server: MCPServer, tool: str
) -> None:
    """The instructions are how an agent picks a tool, before it ever reads a
    schema. Advertising the mode on one of the two search tools leaves the
    other's survey case invisible."""
    instructions = server.instructions or ""
    bullet = next(
        line for line in instructions.splitlines() if line.strip().startswith(f"- {tool} --")
    )
    following = instructions.split(bullet, 1)[1].split("\n- ", 1)[0]

    assert "locations_only" in bullet + following


# ---- several workspaces on one server (#99, part two) -----------------------


@pytest.fixture
async def two_workspaces(tmp_path: Path) -> AsyncIterator[MCPServer]:
    """One server over two workspaces holding *different* documents.

    Different on purpose. Seeding both from one store would let the dispatch
    tests pass while `_for` returned the first workspace for every name --
    there would be nothing to tell the two answers apart. Each workspace holds
    exactly one file, named after it, so a wrong route is visible in the
    result rather than inferred.
    """
    client = AsyncQdrantClient(path=str(tmp_path / "two-qdrant"))
    sparse = FakeSparseBackend()
    services: dict[str, WorkspaceServices] = {}
    for name in ("alpha", "beta"):
        store = QdrantStore(client, workspace=name, payload_indexes=False, owns_client=False)
        text = f"store the {name} thing"
        source = make_source(
            text,
            kind=FileKind.CODE,
            language="python",
            rel_path=f"{name}/only.py",
            unit="repo",
        )
        chunk = build_chunk(
            source,
            name,
            source_text=text,
            start_line=1,
            end_line=1,
            chunker="text",
            version=1,
            chunk_index=0,
        )
        await store.upsert(SPACE, [chunk], [[1.0, 0, 0, 0]], sparse.encode_documents([text]))
        manifest = Manifest(tmp_path / f"{name}.sqlite3")
        services[name] = WorkspaceServices(
            queries=QueryService(
                search=SearchService(
                    store=store,
                    embeddings=EmbeddingService(FakeEmbeddingBackend(dimensions=4)),
                    sparse=FakeSparseBackend(),
                    reranker=NoopReranker(),
                    config=SearchSection(rerank=RerankConfig(enabled=False, model="fake:m")),
                    space=SPACE,
                ),
                taxonomy=TaxonomyService(store, SPACE),
                check_staleness=False,
            ),
            impact=ImpactService(manifest),
            grounding=GroundingService(CoverageService(manifest)),
        )
    yield build_mcp_server(services, ["alpha (repo-one)", "beta (repo-two)"])
    await client.close()


async def test_one_workspace_renders_the_instructions_it_always_did(
    server: MCPServer, two_workspaces: MCPServer
) -> None:
    """A deployment that gained nothing from multi-workspace support must not
    find its agent reading different instructions.

    Asserted as "the extra block is absent, and everything else is the same
    text the two-workspace server carries" -- so it holds without reaching for
    the private constant, and it fails if the per-deployment paragraph ever
    leaks into the single case.
    """
    one = server.instructions or ""
    two = two_workspaces.instructions or ""

    assert "separate indexes" not in one
    assert "workspace=<name>" not in one
    assert one.split("\n\n", 1)[0] == two.split("\n\n", 1)[0]
    assert one.split("\n\n", 1)[1] == two.split("\n\n", 2)[2]


async def test_several_workspaces_are_named_in_the_instructions(
    two_workspaces: MCPServer,
) -> None:
    """The names come from config, so the instructions cannot be a constant.
    The agent reads this before its first call -- otherwise its only route to
    the names is getting one wrong."""
    instructions = two_workspaces.instructions or ""

    assert "2 separate indexes" in instructions
    assert "alpha (repo-one), beta (repo-two)" in instructions
    assert "workspace=<name>" in instructions
    for name in registered_tool_names():
        assert name in instructions


@pytest.mark.parametrize("tool", registered_tool_names())
async def test_every_tool_accepts_a_workspace(two_workspaces: MCPServer, tool: str) -> None:
    """Every tool, not a hand-picked three. Dropping the parameter from one of
    the others used to break no test, which is the drift this PR is stamping
    out elsewhere."""
    tools = {t.name: t for t in await two_workspaces.list_tools()}
    assert "workspace" in tools[tool].input_schema["properties"]


async def test_naming_no_workspace_is_an_error_listing_them(
    two_workspaces: MCPServer,
) -> None:
    """There is no safe default: answering from whichever was listed first
    would serve one index's code to a question about another's, silently."""
    with pytest.raises(ToolError) as caught:
        await two_workspaces.call_tool("search_code", {"query": "store"})

    assert "alpha" in str(caught.value)
    assert "beta" in str(caught.value)


async def test_the_error_names_values_that_actually_dispatch(
    two_workspaces: MCPServer,
) -> None:
    """The message is read as an instruction -- it exists to buy one round
    trip. Listing the described form, "alpha (repo-one)", spends that round
    trip on a value the lookup rejects."""
    with pytest.raises(ToolError) as caught:
        await two_workspaces.call_tool("search_code", {"query": "store"})

    message = str(caught.value)
    assert "(repo-one)" not in message
    # Every name the message offers must be one the server will accept.
    for name in ("alpha", "beta"):
        assert name in message
        _payload(await two_workspaces.call_tool("search_code", {"query": "x", "workspace": name}))


async def test_naming_an_unknown_workspace_lists_the_configured_ones(
    two_workspaces: MCPServer,
) -> None:
    with pytest.raises(ToolError, match="alpha"):
        await two_workspaces.call_tool("search_code", {"query": "store", "workspace": "gamma"})


async def test_each_name_dispatches_to_its_own_index(two_workspaces: MCPServer) -> None:
    """The assertion that would catch a `_for` returning the first workspace
    for every valid name: each workspace holds one file, named after it."""
    for name, other in (("alpha", "beta"), ("beta", "alpha")):
        result = _payload(
            await two_workspaces.call_tool("search_code", {"query": "store", "workspace": name})
        )
        paths = [hit["rel_path"] for hit in result["results"]]

        assert paths == [f"{name}/only.py"], f"{name} answered with {paths}"
        assert not any(other in path for path in paths)


async def test_a_lone_workspace_still_needs_no_name(server: MCPServer) -> None:
    """Every config written before this existed has one workspace, and must
    keep working with no parameter anywhere."""
    result = _payload(await server.call_tool("search_code", {"query": "store"}))
    assert result["results"]


async def test_the_taxonomy_resource_is_withheld_when_there_is_a_choice(
    two_workspaces: MCPServer, server: MCPServer
) -> None:
    """A resource has a fixed URI and no arguments, so with several indexes it
    cannot say which it describes -- and its description claims to describe
    "this workspace". Withheld rather than answering for whichever came first,
    and rather than changing shape, which would make the payload's schema
    depend on configuration."""
    assert [str(r.uri) for r in await server.list_resources()] == [TAXONOMY_URI]
    assert await two_workspaces.list_resources() == []


async def test_describing_a_different_number_of_workspaces_is_refused() -> None:
    """Two sources for one fact. Instructions built from the labels while
    dispatch reads the services means a server that says a name is required
    and then answers anyway, or one that hides a workspace entirely."""
    with pytest.raises(ValueError, match="descriptions for"):
        build_mcp_server({"alpha": cast(Any, None)}, ["alpha", "beta"])


async def test_a_bare_string_of_labels_is_refused() -> None:
    """`Sequence[str]` accepts a plain string and `list("ab")` is `["a", "b"]`,
    so a two-character label would clear the count check and render
    per-character labels while dispatch still keys on the real names."""
    with pytest.raises(TypeError, match="not a single string"):
        build_mcp_server({"ab": cast(Any, None)}, "ab")
