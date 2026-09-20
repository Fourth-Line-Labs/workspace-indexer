"""The MCP surface: how Claude Code reaches the index mid-session."""

from __future__ import annotations

from workspace_indexer.mcp.document_type_resolver import ALIASES, DocumentTypeResolver
from workspace_indexer.mcp.empty_index_error import EmptyIndexError
from workspace_indexer.mcp.grounding_report import GroundingReport
from workspace_indexer.mcp.grounding_service import GroundingService
from workspace_indexer.mcp.impact_report import ImpactReport
from workspace_indexer.mcp.impact_service import ImpactService
from workspace_indexer.mcp.query_service import CODE_EXCLUDES, GUIDANCE_TYPES, QueryService
from workspace_indexer.mcp.result_budget import ResultBudget, anchor_tokens
from workspace_indexer.mcp.search_response import SearchResponse
from workspace_indexer.mcp.search_result import SearchResult
from workspace_indexer.mcp.server_factory import (
    TAXONOMY_URI,
    build_grounding_service,
    build_impact_service,
    build_mcp_server,
    build_query_service,
)
from workspace_indexer.mcp.taxonomy import TAXONOMY_VERSION, Taxonomy
from workspace_indexer.mcp.taxonomy_entry import TaxonomyEntry
from workspace_indexer.mcp.taxonomy_service import TaxonomyService
from workspace_indexer.mcp.tool_call_recorder import ToolCallRecorder
from workspace_indexer.mcp.tool_call_sink import ToolCallSink
from workspace_indexer.mcp.unknown_document_type_error import UnknownDocumentTypeError
from workspace_indexer.mcp.unknown_repository_error import UnknownRepositoryError

__all__ = [
    "ALIASES",
    "anchor_tokens",
    "CODE_EXCLUDES",
    "GUIDANCE_TYPES",
    "TAXONOMY_URI",
    "TAXONOMY_VERSION",
    "DocumentTypeResolver",
    "EmptyIndexError",
    "GroundingReport",
    "GroundingService",
    "ImpactReport",
    "ImpactService",
    "QueryService",
    "ResultBudget",
    "SearchResponse",
    "SearchResult",
    "Taxonomy",
    "TaxonomyEntry",
    "TaxonomyService",
    "ToolCallRecorder",
    "ToolCallSink",
    "UnknownDocumentTypeError",
    "UnknownRepositoryError",
    "build_grounding_service",
    "build_impact_service",
    "build_mcp_server",
    "build_query_service",
]
