"""The three services one workspace is served by."""

from __future__ import annotations

from dataclasses import dataclass

from workspace_indexer.mcp.grounding_service import GroundingService
from workspace_indexer.mcp.impact_service import ImpactService
from workspace_indexer.mcp.query_service import QueryService


@dataclass(frozen=True, slots=True)
class WorkspaceServices:
    """What a tool call needs once it knows which workspace it is answering for.

    Grouped rather than passed as three arguments because they always travel
    together and are always chosen together: picking the query service of one
    workspace and the impact service of another would be a silent cross-index
    answer, which is the failure separate workspaces exist to prevent.
    """

    queries: QueryService
    impact: ImpactService
    grounding: GroundingService
