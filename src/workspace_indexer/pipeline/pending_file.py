"""A file whose chunks are embedded but not yet written."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.classification import Classification
from workspace_indexer.graph.import_edge import ImportEdge
from workspace_indexer.graph.namespace_declaration import NamespaceDeclaration
from workspace_indexer.graph.route_call import RouteCall
from workspace_indexer.graph.route_declaration import RouteDeclaration
from workspace_indexer.models import Chunk, SourceFile
from workspace_indexer.state import ChunkDelta


class PendingFile(BaseModel):
    """Work held back so embedding batches span files.

    Embedding one file at a time would mean one API request per file: a
    five-chunk module and a forty-thousand-file workspace is forty thousand
    round trips for work that batches naturally.
    """

    source: SourceFile
    chunker: str
    chunker_version: int
    chunks: list[Chunk]
    delta: ChunkDelta
    routes: list[RouteDeclaration] = []
    calls: list[RouteCall] = []
    classification: Classification | None = None
    # Extracted with the chunks so the file is parsed once per run, and
    # written in the same transaction so the graph cannot survive a file
    # whose chunks were rolled back.
    imports: list[ImportEdge] = []
    # The other end of a C# `using`, extracted in the same pass and written in
    # the same transaction, so a file's declarations cannot outlive its chunks.
    namespaces: list[NamespaceDeclaration] = []

    @property
    def to_embed(self) -> list[Chunk]:
        wanted = set(self.delta.to_upsert)
        return [chunk for chunk in self.chunks if chunk.chunk_id in wanted]
