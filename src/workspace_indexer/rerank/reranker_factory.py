"""Choose a reranker from config.

Three independent ways reranking ends up off, and none of them may break a
search: it is disabled, it is enabled but unconfigured, or the call fails at
query time. The first two resolve to NoopReranker here; the third is handled by
ScoringReranker's degrade path.
"""

from __future__ import annotations

from workspace_indexer.config import RerankConfig, Settings
from workspace_indexer.obs.logging import get_logger, log_once
from workspace_indexer.rerank.local_cross_encoder_reranker import LocalCrossEncoderReranker
from workspace_indexer.rerank.noop_reranker import NoopReranker
from workspace_indexer.rerank.reranker import Reranker
from workspace_indexer.rerank.voyage_reranker import VoyageReranker

log = get_logger("workspace_indexer.rerank.factory")

VOYAGE_PROVIDERS = frozenset({"voyageai", "voyage"})
LOCAL_PROVIDERS = frozenset({"fastembed", "local"})
# `database:` and `server:` are deliberately absent. They declared that the
# *store* reranked inside the query; that path was retired in #73 and
# `RerankConfig` now refuses such a model at config load, so nothing
# database-shaped can reach this factory. Restoring it is tracked in #94.
KNOWN_PROVIDERS = VOYAGE_PROVIDERS | LOCAL_PROVIDERS


def build_reranker(config: RerankConfig, settings: Settings) -> Reranker:
    if not config.enabled:
        log_once(log, "rerank:disabled", "rerank.skipped", reason="disabled_in_config")
        return NoopReranker()

    provider = config.provider
    if provider in LOCAL_PROVIDERS:
        return _build(lambda: LocalCrossEncoderReranker(config, config.model_id), provider)

    if provider in VOYAGE_PROVIDERS:
        if not settings.voyage_api_key:
            # A search must never fail because an optional quality enhancement
            # is unconfigured. Logged once: the condition is permanent for the
            # life of the process, and repeating it on every search would bury
            # real events.
            log_once(log, "rerank:no_key", "rerank.skipped", reason="no_voyage_api_key")
            return NoopReranker()
        return _build(
            lambda: VoyageReranker(config, config.model_id, settings.voyage_api_key), provider
        )

    raise ValueError(
        f"unknown rerank provider {provider!r} in {config.model!r}; "
        f"known providers: {', '.join(sorted(KNOWN_PROVIDERS))}. "
        "Add one by subclassing ScoringReranker and implementing _score()."
    )


def _build(make: object, provider: str) -> Reranker:
    """Import failures degrade rather than abort.

    The provider SDKs are optional extras, and a missing one is a
    configuration gap, not a reason to lose an entire index run.
    """
    try:
        assert callable(make)
        reranker = make()
    except ImportError as exc:
        log_once(
            log,
            f"rerank:import:{provider}",
            "rerank.skipped",
            reason="sdk_not_installed",
            provider=provider,
            error=str(exc),
        )
        return NoopReranker()
    log.info("rerank.backend", provider=provider)
    assert isinstance(reranker, Reranker)
    return reranker
