"""Everything the pipeline needs, with handles the tests assert against.

Shared rather than rebuilt per test module: two constructions of `Indexer`
drift, and the second one is always the one missing the argument that matters.
"""

from __future__ import annotations

from pathlib import Path

from tests.fake_embedding_backend import FakeEmbeddingBackend
from tests.fake_sparse_backend import FakeSparseBackend
from workspace_indexer.chunking import ChunkerRegistry
from workspace_indexer.classification import RuleClassifier
from workspace_indexer.config import Settings, WorkspaceConfig
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.models import EmbeddingSpace
from workspace_indexer.pipeline import Indexer
from workspace_indexer.state import Manifest
from workspace_indexer.storage.qdrant_store import QdrantStore

# Four dimensions, because nothing here measures similarity -- the embedding
# backend is fake and the vectors only have to be storable.
SPACE = EmbeddingSpace(model="fake:model", dimensions=4)
NARROW = EmbeddingSpace(model="fake:other", dimensions=4)


class Harness:
    """Everything the pipeline needs, with handles the tests assert against."""

    def __init__(
        self, config: WorkspaceConfig, store: QdrantStore, manifest: Manifest, tmp: Path
    ) -> None:
        self.config = config
        self.store = store
        self.manifest = manifest
        self.backend = FakeEmbeddingBackend(dimensions=4)
        self.embeddings = EmbeddingService(self.backend)
        self.sparse = FakeSparseBackend()
        self.tmp = tmp

    def indexer(self, space: EmbeddingSpace = SPACE) -> Indexer:
        return Indexer(
            config=self.config,
            settings=Settings(state_db=self.tmp / "manifest.sqlite3"),
            manifest=self.manifest,
            registry=ChunkerRegistry(self.config.workspace.name),
            embeddings=self.embeddings,
            sparse=self.sparse,
            store=self.store,
            space=space,
            classifier=RuleClassifier(),
            flush_chunks=8,
        )

    @property
    def documents_embedded(self) -> int:
        return self.backend.stats_documents

    def reset_counters(self) -> None:
        self.backend.batches.clear()
        self.backend.calls = 0
