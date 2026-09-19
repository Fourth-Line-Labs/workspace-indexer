"""Binding the query service to the MCP protocol.

Deliberately thin. Everything here is a decorator, a docstring and a type hint;
the behaviour lives in QueryService, which is testable without a client
session. Named `server_factory` rather than `server` so that a traceback never
leaves you wondering whether `mcp.server` is ours or the SDK's.
"""

from __future__ import annotations

import json
from typing import Annotated

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from workspace_indexer.app_context import AppContext
from workspace_indexer.grounding import CoverageService
from workspace_indexer.mcp.document_type_resolver import DocumentTypeResolver
from workspace_indexer.mcp.empty_index_error import EmptyIndexError
from workspace_indexer.mcp.grounding_report import GroundingReport
from workspace_indexer.mcp.grounding_service import GroundingService
from workspace_indexer.mcp.impact_report import ImpactReport
from workspace_indexer.mcp.impact_service import ImpactService
from workspace_indexer.mcp.query_service import QueryService
from workspace_indexer.mcp.search_response import SearchResponse
from workspace_indexer.mcp.taxonomy import Taxonomy
from workspace_indexer.mcp.taxonomy_service import TaxonomyService
from workspace_indexer.mcp.tool_call_recorder import ToolCallRecorder
from workspace_indexer.mcp.unknown_document_type_error import UnknownDocumentTypeError
from workspace_indexer.mcp.unknown_repository_error import UnknownRepositoryError
from workspace_indexer.models import DocumentType
from workspace_indexer.worktrees import (
    WorktreeChoiceError,
    WorktreeGate,
    WorktreeRegistry,
)

TAXONOMY_URI = "workspace-indexer://taxonomy"

# Spliced into the tool descriptions. The types have to be in context *before*
# the agent picks one: a round trip to discover the vocabulary is a round trip
# it will usually skip, and then it guesses.
_TYPE_LIST = ", ".join(t.value for t in DocumentType)

_INSTRUCTIONS = f"""\
A semantic + keyword index over this workspace. Use it instead of grepping when
you know *what* you want but not *where* it is.

- search_code -- implementation, with tests and generated files excluded.
  Pass locations_only=true to survey where something lives: it returns anchors
  without bodies, so many more matches fit in one response. Still ranked and
  still capped by limit, so a short list is not proof that nothing else
  matches.
- find_guidance -- specifications and design documents only. Reach for this
  before writing new code, especially when there is nothing yet to imitate.
- get_file_context -- every indexed chunk of one file, in order.
- list_document_types -- what kinds of document this workspace actually holds,
  with counts. A count of zero is a real answer: it means look at the code.
- impact_of -- what one file imports and, more usefully, what imports it.
  Call this before changing a signature, and before deleting anything.
- grounding -- whether a repository records *why* it is the way it is. Call
  this when find_guidance comes back empty, before concluding anything from
  that emptiness: it says whether the answer is missing from the index or was
  never written down. Where it reports `absent`, say the rationale is
  unrecorded rather than inferring one.

Document types: {_TYPE_LIST}.
"""


def build_impact_service(ctx: AppContext) -> ImpactService:
    return ImpactService(ctx.manifest, recorder=ToolCallRecorder(ctx.manifest))


def build_grounding_service(ctx: AppContext) -> GroundingService:
    return GroundingService(CoverageService(ctx.manifest), recorder=ToolCallRecorder(ctx.manifest))


def build_query_service(ctx: AppContext) -> QueryService:
    return QueryService(
        search=ctx.search_service(),
        taxonomy=TaxonomyService(ctx.store, ctx.space),
        check_staleness=ctx.config.search.check_staleness,
        worktrees=WorktreeGate(WorktreeRegistry(ctx.manifest)),
        # The manifest is the harvesting sink: finding the calls that returned
        # nothing is a query rather than a log scrape, and those calls are eval
        # cases waiting to be written.
        recorder=ToolCallRecorder(ctx.manifest),
    )


def build_mcp_server(
    queries: QueryService, impact: ImpactService, grounding_service: GroundingService
) -> MCPServer:
    """Wrap the services in the protocol.

    Takes the services rather than the AppContext so a test can build the real
    server -- real tool schemas, real dispatch -- over a store it seeded
    itself, without constructing every layer of the application.

    `impact` and `grounding_service` are required rather than defaulted to
    None. A tool that exists only when someone remembered to wire it is worse
    than no tool: the agent cannot tell a missing capability from a negative
    answer, and neither can the person reading the code.
    """
    resolver = DocumentTypeResolver()
    server = MCPServer(name="workspace-indexer", instructions=_INSTRUCTIONS)

    @server.tool()
    async def search_code(
        query: Annotated[str, Field(description="What you are looking for, in plain language.")],
        limit: Annotated[int, Field(description="Maximum hits.", ge=1, le=50)] = 8,
        repo: Annotated[str | None, Field(description="Restrict to one repository.")] = None,
        language: Annotated[
            str | None, Field(description="Restrict to one language, e.g. python.")
        ] = None,
        path_prefix: Annotated[
            str | None, Field(description="Restrict to a directory, e.g. src/auth.")
        ] = None,
        include_tests: Annotated[
            bool, Field(description="Include tests and generated files.")
        ] = False,
        locations_only: Annotated[
            bool,
            Field(
                description="Return file and line ranges without the code itself. "
                "Chunk bodies are most of what a hit costs, so this fits far more "
                "matches in one response -- use it to survey where something lives "
                "before reading any of it. Not a substitute for grep: results are "
                "still ranked and capped by limit, so a short list is not proof "
                "that nothing else matches."
            ),
        ] = False,
        worktree: Annotated[
            str | None,
            Field(
                description="Which checkout you are working in: a worktree name or "
                'path, or "none" for the main checkout. Required only when the '
                "repository has worktrees."
            ),
        ] = None,
    ) -> SearchResponse:
        """Search implementation code and documentation by meaning.

        Tests and generated files are excluded unless include_tests is set,
        because a test naming a symbol twenty times otherwise outranks the one
        file that defines it. Every result is anchored as path:start-end.
        """
        try:
            return await queries.search_code(
                query,
                limit=limit,
                repo=repo,
                language=language,
                path_prefix=path_prefix,
                include_tests=include_tests,
                locations_only=locations_only,
                worktree=worktree,
            )
        except WorktreeChoiceError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    async def find_guidance(
        query: Annotated[str, Field(description="The decision or topic you need rules for.")],
        limit: Annotated[int, Field(description="Maximum hits.", ge=1, le=50)] = 8,
        repo: Annotated[str | None, Field(description="Restrict to one repository.")] = None,
        doc_type: Annotated[
            str | None,
            Field(description=f"Narrow to one type. One of: {_TYPE_LIST}."),
        ] = None,
        locations_only: Annotated[
            bool,
            Field(
                description="Return file and line ranges without the code itself. "
                "Chunk bodies are most of what a hit costs, so this fits far more "
                "matches in one response -- use it to survey where something lives "
                "before reading any of it. Not a substitute for grep: results are "
                "still ranked and capped by limit, so a short list is not proof "
                "that nothing else matches."
            ),
        ] = False,
        worktree: Annotated[
            str | None,
            Field(
                description="Which checkout you are working in: a worktree name or "
                'path, or "none" for the main checkout. Required only when the '
                "repository has worktrees."
            ),
        ] = None,
    ) -> SearchResponse:
        """Find the specs, conventions and design documents governing a topic.

        Searches normative and design documents only, so a changelog describing
        how something used to work cannot outrank the standard saying how it
        must be built. Use this before writing new code -- particularly when
        there is no existing implementation to copy.

        An unrecognised doc_type is an error naming the valid ones, never an
        empty result.
        """
        try:
            selected = resolver.resolve(doc_type) if doc_type else None
        except UnknownDocumentTypeError as exc:
            # Converted at the boundary, deliberately. The SDK strips the text
            # of an arbitrary exception and sends the model a bare "Error
            # executing tool find_guidance" -- which is exactly the useless,
            # unactionable failure this whole design is built to avoid. Only a
            # ToolError carries its message through to the agent.
            raise ToolError(str(exc)) from exc
        try:
            return await queries.find_guidance(
                query,
                limit=limit,
                repo=repo,
                doc_type=selected,
                locations_only=locations_only,
                worktree=worktree,
            )
        except WorktreeChoiceError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    async def get_file_context(
        rel_path: Annotated[
            str, Field(description="Path from a search result, or a trailing portion of one.")
        ],
        limit: Annotated[int, Field(description="Maximum chunks.", ge=1, le=100)] = 20,
        worktree: Annotated[
            str | None,
            Field(
                description="Which checkout you are working in: a worktree name or "
                'path, or "none" for the main checkout. Required only when the '
                "repository has worktrees."
            ),
        ] = None,
    ) -> SearchResponse:
        """Return every indexed chunk of one file, in file order.

        Use this to expand around a hit: the search returns the matching
        function, this returns the rest of the file as it was indexed.
        """
        try:
            return await queries.get_file_context(rel_path, limit=limit, worktree=worktree)
        except WorktreeChoiceError as exc:
            raise ToolError(str(exc)) from exc

    @server.tool()
    async def list_document_types() -> Taxonomy:
        """List the document types in this workspace, with counts and examples.

        Counts describe this index, not the code's vocabulary. A type reported
        with count 0 genuinely has no documents of that kind here -- if
        `normative` is 0, this workspace has no written standards and you
        should read the implementation instead of hunting for specs.
        """
        return await queries.taxonomy()

    @server.tool()
    async def impact_of(
        rel_path: Annotated[
            str,
            Field(description="Path from a search result, or a trailing portion of one."),
        ],
        limit: Annotated[int, Field(description="Maximum edges per direction.", ge=1, le=200)] = 25,
    ) -> ImpactReport:
        """What one file imports, and what imports it.

        Answer this before changing a signature, renaming an export, or
        deleting a file. `used_by` is the expensive half: it spans every
        repository in the workspace, which a per-project language server cannot
        do, and each entry is anchored as path:line at the import statement.

        Read `note` before concluding anything from an empty result. Empty can
        mean the language has no import scanner, or that the imports naming
        this file are spelled in a way we cannot resolve to a path -- neither
        of which means nothing depends on it.
        """
        return impact.impact_of(rel_path, limit=limit)

    # The suppression below is not decoration, and it is not understood. Every
    # tool above is registered by decorator and never called by name, exactly
    # like this one, and pyright reports none of them. Verified by experiment:
    # a byte-identical clone of `list_document_types` added to this file *is*
    # reported, and renaming `list_document_types` itself makes it reported
    # too -- so the exemption tracks the name, not the shape or the position.
    # What earns a name the exemption, I could not establish; the obvious
    # candidates (the name appearing in _INSTRUCTIONS, elsewhere in src/, or in
    # a test's call_tool string) are all true of `grounding` as well.
    # Narrowed to the one rule rather than silenced, and left with this note so
    # the next person starts from what has already been ruled out.
    @server.tool()
    async def grounding(  # pyright: ignore[reportUnusedFunction]
        repo: Annotated[
            str | None,
            Field(description="Restrict to one repository, as named in a search result."),
        ] = None,
    ) -> GroundingReport:
        """Whether a repository records *why* it is the way it is.

        Call this when find_guidance returns nothing, before concluding
        anything from that. An empty search has two causes -- the index missed
        it, or nobody wrote it down -- and they call for opposite next moves.
        Nothing else here can tell them apart.

        Reports four sources per repository (design docs, normative docs,
        commit rationale, and WHY:/DECISION: markers), each `absent`, `thin` or
        `present`. Where the verdict is `absent`, the reason genuinely was not
        recorded: read the implementation, and say the rationale is unrecorded
        rather than inferring a plausible one.

        Read `notes`. They carry the findings that change what to do -- most
        importantly that a repository's reasons live in an issue tracker this
        index cannot read, which turns "no rationale" into "wrong system".

        An unrecognised repo is an error naming the indexed ones, never an
        empty result, because empty here reads as "records no reasons".
        """
        try:
            return grounding_service.grounding(repo)
        except UnknownRepositoryError as exc:
            # Same boundary conversion as find_guidance: only a ToolError
            # carries its message to the model.
            raise ToolError(str(exc)) from exc

    @server.resource(
        TAXONOMY_URI,
        name="Document taxonomy",
        description="Document types in this workspace, with counts and example paths.",
        mime_type="application/json",
    )
    async def taxonomy_resource() -> str:
        taxonomy = await queries.taxonomy()
        return json.dumps(taxonomy.model_dump(), indent=2)

    # Registered by decoration; naming them here keeps the linter honest about
    # the fact that these are the server's public surface.
    _ = (
        search_code,
        find_guidance,
        get_file_context,
        list_document_types,
        impact_of,
        taxonomy_resource,
    )
    return server


async def preflight(ctx: AppContext) -> None:
    """Fail loudly at startup rather than serving nothing.

    Deliberately a hard failure, not a warning. A warning goes to stderr, which
    an MCP client files under logs nobody reads, and the session then proceeds
    with a server that answers every question with "nothing found" -- worse
    than no server at all, because the agent believes the answer.
    """
    if await ctx.store.count(ctx.space):
        return
    raise EmptyIndexError(
        space=ctx.space.slug(),
        # Asked of the store rather than reconstructed from settings, which
        # reported every backend as Qdrant the moment a second one existed.
        mode=ctx.store.describe(),
        detail=f"collection {ctx.store.collection_name(ctx.space)} holds no points.",
    )
