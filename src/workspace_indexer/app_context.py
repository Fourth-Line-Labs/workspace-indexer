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
        config = load_workspace_config(config_path).select(workspace)
        # Before anything else runs, so a failure during setup is still logged.
        # `role` names the command, and separates its log file from every
        # other command's. Two processes sharing a rotating file cannot both
        # roll it over on Windows -- see configure_logging.
        #
        # Here rather than in `from_config`, because a registry serving
        # several workspaces builds many contexts and wants one log, not one
        # per workspace. `logging` is a shared section, so there is only ever
        # one answer to configure with.
        configure_logging_for(config, role)
        return cls.from_config(config)

    @classmethod
    def from_config(
        cls,
        config: WorkspaceConfig,
        *,
        store: VectorStore | None = None,
        embeddings: EmbeddingService | None = None,
        sparse: SparseBackend | None = None,
    ) -> AppContext:
        """One workspace, optionally over resources somebody else owns.

        Public because `WorkspaceRegistry` is a real caller: it builds the
        shared client and backends and then asks for a context over them.

        The three optional arguments exist for serving several workspaces at
        once. A Qdrant client cannot be built twice against embedded storage,
        and loading the same local embedding model once per workspace wastes
        memory for no benefit -- so a registry builds those once and hands them
        in. Left out, each context builds its own, which is what every
        single-workspace command does.
        """
        # Order matters, and getting it wrong is silent. `applied_to` rebuilds
        # Settings from a full `model_dump()`, and pydantic sets
        # `model_fields_set` from the input dict's keys -- so afterwards every
        # one of its fields counts as explicitly provided. `with_rerank_overrides`
        # gates on exactly that signal, because it is the only thing telling
        # "the default" from "someone typed the default". Read off the derived
        # settings, it therefore overrides a workspace.yaml `search.rerank`
        # every time, re-validated so nothing errors -- and the workspace it
        # hits hardest is one barred from a hosted API, which gets hosted
        # reranking switched back on.
        base = Settings()
        config = with_rerank_overrides(config, base)
        settings = settings_for(config, base)
        name = config.workspace.name
        space = build_space(settings)
        return cls(
            config=config,
            settings=settings,
            space=space,
            # From the config where `state_dir` is set, so several workspaces
            # keep several manifests. Falls back to STATE_DB untouched, which
            # is what a single-workspace setup has always used.
            manifest=Manifest(config.manifest_path(settings.state_db)),
            registry=ChunkerRegistry(name),
            # `is not None`, not truthiness: these name resources somebody
            # else owns, and a store that one day grew `__len__` would read as
            # absent while empty -- turning sharing off for one workspace and
            # nothing else.
            embeddings=embeddings if embeddings is not None else build_embedding_service(settings),
            sparse=sparse if sparse is not None else build_sparse_backend(settings),
            store=store if store is not None else build_vector_store(settings, name),
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


def configure_logging_for(config: WorkspaceConfig, role: str | None = None) -> None:
    """Set logging up once for a command, honouring .env over workspace.yaml.

    Public because `serve` builds a registry rather than a single context and
    still needs exactly this: one log for the process, not one per workspace.
    `logging` is a shared section, so there is only ever one answer.
    """
    configure_logging(_with_env_overrides(config, Settings()), role)


def settings_for(config: WorkspaceConfig, base: Settings | None = None) -> Settings:
    """Environment settings with this workspace's embedding overrides applied.

    Pulled out so a registry can derive the same settings -- and therefore the
    same `EmbeddingSpace` -- without building a context, which is how it knows
    whether two workspaces can share one backend.

    The overrides land on `Settings` rather than being carried separately so
    everything derived from it moves together: the space, the backend, the
    price, and `config_hash`, which decides whether two eval runs are
    comparable.

    `base` lets a caller supply the pristine settings it has already read, so
    anything that must be decided *before* the overrides land -- the rerank
    precedence, which reads `model_fields_set` -- can be decided against the
    same object rather than a second one.
    """
    settings = base if base is not None else Settings()
    embedding = config.workspace.embedding
    return embedding.applied_to(settings) if embedding else settings


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
