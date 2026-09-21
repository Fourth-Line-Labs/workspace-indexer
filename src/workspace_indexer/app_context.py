"""Wiring: build every layer from config, once, in one place.

The CLI commands then read as what they do rather than as assembly, and the
future MCP server gets the same construction for free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from workspace_indexer.chunking import ChunkerRegistry
from workspace_indexer.classification import DocumentClassifier, RuleClassifier
from workspace_indexer.config import (
    LoggingConfig,
    RerankConfig,
    Settings,
    WorkspaceConfig,
    load_workspace_config,
)
from workspace_indexer.embedding import (
    build_embedding_service,
    build_space,
    build_sparse_backend,
)
from workspace_indexer.embedding.embedding_service import EmbeddingService
from workspace_indexer.embedding.sparse_backend import SparseBackend
from workspace_indexer.models import EmbeddingSpace
from workspace_indexer.obs.logging import configure_logging
from workspace_indexer.pipeline import Indexer
from workspace_indexer.rerank import build_reranker
from workspace_indexer.rerank.reranker import Reranker
from workspace_indexer.search import SearchService
from workspace_indexer.state import Manifest
from workspace_indexer.storage import build_vector_store
from workspace_indexer.storage.vector_store import VectorStore


@dataclass(slots=True)
class AppContext:
    config: WorkspaceConfig
    settings: Settings
    space: EmbeddingSpace
    manifest: Manifest
    registry: ChunkerRegistry
    embeddings: EmbeddingService
    sparse: SparseBackend
    store: VectorStore
    reranker: Reranker
    classifier: DocumentClassifier

    @classmethod
    def build(
        cls,
        config_path: Path | None = None,
        role: str | None = None,
        workspace: str | None = None,
    ) -> AppContext:
        """One workspace's worth of everything.

        `workspace` may be omitted when the config describes only one, which is
        every config written before workspaces became a list. Where there is a
        choice and none was made, `select` raises rather than picking: the
        workspaces are separate indexes precisely so their results do not mix,
        and answering from whichever was listed first would defeat that
        silently.
        """
        return cls._from_config(load_workspace_config(config_path).select(workspace), role=role)

    @classmethod
    def build_all(
        cls, config_path: Path | None = None, role: str | None = None
    ) -> list[AppContext]:
        """One context per configured workspace, in config order.

        For the commands that act on everything -- indexing, serving -- rather
        than on a named one. Each context is fully independent: its own
        manifest file, its own collection, and its own embedding settings if it
        overrides them.
        """
        config = load_workspace_config(config_path)
        return [cls._from_config(config.select(name), role=role) for name in config.workspace_names]

    @classmethod
    def _from_config(cls, config: WorkspaceConfig, role: str | None = None) -> AppContext:
        settings = Settings()
        # Applied to the config itself, before anything reads it, so every
        # layer downstream sees one answer. Doing it per consumer is how
        # RERANK_MODEL came to be a documented setting that nothing read.
        config = with_rerank_overrides(config, settings)
        # The workspace's embedding overrides land on `Settings` rather than
        # being carried separately, so everything derived from it moves
        # together -- the embedding space, the backend, the price, and
        # `config_hash`, which decides whether two eval runs are comparable.
        embedding = config.workspace.embedding
        if embedding is not None:
            settings = embedding.applied_to(settings)

        # Before anything else runs, so a failure during setup is still logged.
        # `role` names the command, and separates its log file from every
        # other command's. Two processes sharing a rotating file cannot both
        # roll it over on Windows -- see configure_logging.
        configure_logging(_with_env_overrides(config, settings), role)

        space = build_space(settings)
        return cls(
            config=config,
            settings=settings,
            space=space,
            # From the config where `state_dir` is set, so several workspaces
            # keep several manifests. Falls back to STATE_DB untouched, which
            # is what a single-workspace setup has always used.
            manifest=Manifest(config.manifest_path(settings.state_db)),
            registry=ChunkerRegistry(config.workspace.name),
            embeddings=build_embedding_service(settings),
            sparse=build_sparse_backend(settings),
            store=build_vector_store(settings, config.workspace.name),
            reranker=build_reranker(config.search.rerank, settings),
            classifier=RuleClassifier(),
        )

    def indexer(self) -> Indexer:
        return Indexer(
            config=self.config,
            settings=self.settings,
            manifest=self.manifest,
            registry=self.registry,
            embeddings=self.embeddings,
            sparse=self.sparse,
            store=self.store,
            space=self.space,
            classifier=self.classifier,
        )

    def search_service(self, space: EmbeddingSpace | None = None) -> SearchService:
        return SearchService(
            store=self.store,
            embeddings=self.embeddings,
            sparse=self.sparse,
            reranker=self.reranker,
            config=self.config.search,
            space=space or self.space,
        )

    async def close(self) -> None:
        await self.store.close()
        self.manifest.close()


def with_rerank_overrides(config: WorkspaceConfig, settings: Settings) -> WorkspaceConfig:
    """.env wins over workspace.yaml for reranking, as it does for logging.

    Public because it is the wiring a test has to be able to assert directly --
    the bug it fixes was invisible from the outside, since the wrong reranker
    still returns plausible results.

    Only for values actually set: `rerank_enabled` and `rerank_model` have
    non-None defaults, so applying them unconditionally would override a
    workspace.yaml that configured reranking deliberately. `model_fields_set`
    is the only thing that distinguishes "the default" from "someone typed the
    default".

    These two were declared, documented and read by nothing at all. Setting
    RERANK_MODEL had no effect, which is worse than not offering it -- and it
    stayed that way because the reranker is built from `config.search.rerank`
    while the setting sat in `Settings`. `test_settings_are_wired.py` now fails
    the build for any setting nothing reads.
    """
    provided = settings.model_fields_set
    updates: dict[str, object] = {}
    if "rerank_enabled" in provided:
        updates["enabled"] = settings.rerank_enabled
    if "rerank_model" in provided:
        updates["model"] = settings.rerank_model
    if not updates:
        return config
    # Validated rather than copied in. `model_copy(update=...)` does not run
    # field validators, so an override went straight past the check that exists
    # to catch a bad `RERANK_MODEL` at config load rather than an hour into a
    # run -- including the retired `database:` provider (#73), which then
    # surfaced from the reranker factory as "unknown provider" and named
    # neither the requirement nor the issue.
    rerank = RerankConfig.model_validate({**config.search.rerank.model_dump(), **updates})
    return config.model_copy(update={"search": config.search.model_copy(update={"rerank": rerank})})


def _with_env_overrides(config: WorkspaceConfig, settings: Settings) -> LoggingConfig:
    """.env wins over workspace.yaml for logging, so LOG_LEVEL=DEBUG works
    without editing a committed file."""
    logging_config = config.logging
    updates: dict[str, object] = {}
    if settings.log_level:
        updates["level"] = settings.log_level
    if settings.logfire_enabled is not None or settings.logfire_send_to_cloud is not None:
        updates["logfire"] = logging_config.logfire.model_copy(
            update={
                k: v
                for k, v in {
                    "enabled": settings.logfire_enabled,
                    "send_to_cloud": settings.logfire_send_to_cloud,
                }.items()
                if v is not None
            }
        )
    return logging_config.model_copy(update=updates) if updates else logging_config
