"""Every graph edge one file contributes, read but not yet recorded."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.graph.import_edge import ImportEdge
from workspace_indexer.graph.namespace_declaration import NamespaceDeclaration
from workspace_indexer.graph.route_call import RouteCall
from workspace_indexer.graph.route_declaration import RouteDeclaration


class ScannedFile(BaseModel):
    """The backfill's unit of work, between reading and writing.

    Exists so the reading and parsing happen outside the transaction that
    writes them: holding the write lock across file I/O would block every other
    writer -- the MCP server records each tool call into this database -- for
    the length of an upgrade run.
    """

    root_label: str
    rel_path: str
    imports: list[ImportEdge] = []
    namespaces: list[NamespaceDeclaration] = []
    routes: list[RouteDeclaration] = []
    calls: list[RouteCall] = []
