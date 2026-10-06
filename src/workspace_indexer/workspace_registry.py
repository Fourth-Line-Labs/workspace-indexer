"""Every configured workspace, served from one process.

Separate indexes, shared machinery. Each workspace keeps its own collection,
its own manifest and its own embedding space -- that separation is the whole
point -- but the expensive objects underneath are built once and handed round.

Two reasons, and only the first is an optimisation. An embedded Qdrant locks
its storage folder to a single client, so a second client against the same path
is refused outright: a server holding several workspaces open at once *must*
share one client or it cannot start at all. Loading the same local embedding
model once per workspace is merely wasteful, and the sparse backend is the same
fastembed model for every workspace that has not overridden it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qdrant_client import AsyncQdrantClient

from workspace_indexer.app_context import AppContext, close_blocking, settings_for
from workspace_indexer.config import (
    Settings,
    WorkspaceChoiceError,
    WorkspaceConfig,
    load_workspace_config,
)
from workspace_indexer.embedding import build_embedding_service, build_sparse_backend
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.storage import build_qdrant_client, build_vector_store

if TYPE_CHECKING:
    from workspace_indexer.embedding.embedding_service import EmbeddingService
    from workspace_indexer.embedding.sparse_backend import SparseBackend
    from workspace_indexer.storage.vector_store import VectorStore

log = get_logger("workspace_indexer.registry")


class WorkspaceRegistry:
    """Contexts for every configured workspace, keyed by name."""

    def __init__(self, config: WorkspaceConfig) -> None:
        self._names = config.workspace_names
        self._client: AsyncQdrantClient | None = None
        # Stores built but not yet handed to a context, so a failure between
        # the two can still release them.
        self._unowned: list[VectorStore] = []
        dense: dict[tuple[str, int, float | None], EmbeddingService] = {}
        sparse: dict[str, SparseBackend] = {}
        contexts: dict[str, AppContext] = {}

        try:
            self._build(config, contexts, dense, sparse)
        except Exception:
            # Nothing holds a reference to a constructor that raised, so the
            # cleanup has to happen here or not at all -- and on embedded
            # Qdrant the client holds the storage-folder lock.
            self._abandon(contexts)
            raise

        self._contexts = contexts
        log.info(
            "registry.built",
            workspaces=len(contexts),
            dense_backends=len(dense),
            sparse_backends=len(sparse),
        )

    def _build(
        self,
        config: WorkspaceConfig,
        contexts: dict[str, AppContext],
        dense: dict[tuple[str, int, float | None], EmbeddingService],
        sparse: dict[str, SparseBackend],
    ) -> None:
        for name in self._names:
            selected = config.select(name)
            settings = settings_for(selected)
            # Keyed by what the backend actually depends on, not by workspace.
            # Every workspace inheriting its embedding settings from .env --
            # the ordinary case -- then shares one dense and one sparse
            # backend however many workspaces there are.
            dense_key = (
                settings.embedding_model,
                settings.embedding_dimensions,
                # The price too: `EmbeddingService` is built with a
                # `TokenPricer` over it, and `price_per_mtok` is a
                # per-workspace override. Two workspaces on one model at
                # different rates would otherwise share the first one's
                # pricer and both report at its rate -- the cross-workspace
                # leak that field exists to prevent.
                settings.embedding_price_per_mtok,
            )
            if dense_key not in dense:
                dense[dense_key] = build_embedding_service(settings)
            if settings.sparse_model not in sparse:
                sparse[settings.sparse_model] = build_sparse_backend(settings)
            # Held while it has no owner. `from_config` can raise after this
            # store exists -- an unknown rerank provider, a model that will
            # not load -- and it releases only what it created itself, so
            # without this the store would be invisible to the cleanup.
            store = build_vector_store(settings, name, client=self._qdrant(settings))
            self._unowned.append(store)
            contexts[name] = AppContext.from_config(
                selected,
                store=store,
                embeddings=dense[dense_key],
                sparse=sparse[settings.sparse_model],
            )
            self._unowned.remove(store)

    def _abandon(self, contexts: dict[str, AppContext]) -> None:
        """Release what a failed build had already taken.

        Every close is guarded individually, for the reason `close` states and
        this used to ignore: a raising close must not strand the rest. Here it
        matters twice over, because this runs inside an `except` block -- an
        unguarded failure would replace the build error the user needs to see
        (an unknown rerank provider, say) with a sqlite error about tidying up.

        Stores as well as manifests. With `VECTOR_STORE=mongodb` nothing is
        shared and every store owns a live client, so closing only the shared
        Qdrant client would release nothing at all.

        Store and client failures are reported alongside manifest ones. They
        used to be suppressed, which made the Mongo case the quiet one: a
        wedged store close leaked a live client with no signal anywhere while
        a wedged sqlite handle got a warning.
        """
        failures: list[str] = []
        for name, ctx in contexts.items():
            try:
                ctx.manifest.close()
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                failures.append(f"{name} manifest: {type(exc).__name__}: {exc}")
            failure = close_blocking(ctx.store.close)
            if failure is not None:
                failures.append(f"{name} store: {failure}")
        for index, store in enumerate(self._unowned):
            failure = close_blocking(store.close)
            if failure is not None:
                failures.append(f"unowned store {index}: {failure}")
        self._unowned.clear()
        if self._client is not None:
            failure = close_blocking(self._client.close)
            if failure is not None:
                failures.append(f"<shared client>: {failure}")
            self._client = None
        if failures:
            log.warning("registry.abandon_failed", failures=failures)

    @classmethod
    def load(cls, config_path: Path | None = None) -> WorkspaceRegistry:
        return cls(load_workspace_config(config_path))

    def _qdrant(self, settings: Settings) -> AsyncQdrantClient | None:
        """One client for every Qdrant-backed workspace, built on first use.

        None for a non-Qdrant store, which builds its own: the lock this
        exists for is Qdrant's, and a Mongo deployment is a server that is
        happy with several connections.
        """
        if settings.vector_store != "qdrant":
            return None
        if self._client is None:
            self._client = build_qdrant_client(settings)
        return self._client

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def describe(self) -> list[str]:
        """Each workspace with the roots it covers, for the agent to read.

        Bare names say *that* several indexes exist but not which holds what,
        so the first call is a guess -- and a wrong guess returns a confident
        empty result, which reads as "this does not exist" rather than as "you
        asked the wrong index".
        """
        described: list[str] = []
        for name in self._names:
            roots = [r.resolved_label for r in self._contexts[name].config.workspace.roots]
            shown = ", ".join(roots[:3])
            if len(roots) > 3:
                shown += f", +{len(roots) - 3} more"
            described.append(f"{name} ({shown})")
        return described

    def context(self, name: str | None = None) -> AppContext:
        """The named workspace, or the only one when there is no choice.

        Raises where a choice exists and none was made, rather than picking.
        The workspaces are separate indexes precisely so their results do not
        mix, and answering from whichever was listed first would defeat that
        without saying so.
        """
        if name is None:
            if len(self._names) != 1:
                raise WorkspaceChoiceError(None, self.names)
            return self._contexts[self._names[0]]
        found = self._contexts.get(name)
        if found is None:
            raise WorkspaceChoiceError(name, self.names)
        return found

    async def close(self) -> None:
        """Close every context, then the client they were sharing.

        In that order: a store handed a shared client does not close it, so
        the client outlives the contexts and has to be closed here. Each
        context is closed even if an earlier one raised -- a failure to close
        one manifest must not leak the rest.
        """
        failures: list[str] = []
        for name, ctx in self._contexts.items():
            try:
                await ctx.close()
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
        if self._client is not None:
            try:
                await self._client.close()
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                # Outside the try this would discard every failure collected
                # above, and in `serve`'s `finally` it would replace the
                # traceback of whatever actually ended the session.
                failures.append(f"<shared client>: {type(exc).__name__}: {exc}")
            self._client = None
        if failures:
            log.warning("registry.close_failed", failures=failures)
