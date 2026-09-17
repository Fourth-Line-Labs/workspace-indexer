"""Choosing a reranker, and the three ways reranking ends up off.

This is the requirement most likely to be quietly half-implemented, so each
disable path gets its own test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.config import RerankConfig, Settings
from workspace_indexer.rerank.local_cross_encoder_reranker import LocalCrossEncoderReranker
from workspace_indexer.rerank.noop_reranker import NoopReranker
from workspace_indexer.rerank.reranker import Reranker
from workspace_indexer.rerank.reranker_factory import build_reranker
from workspace_indexer.rerank.voyage_reranker import VoyageReranker


@pytest.fixture(autouse=True)
def _isolate_env(  # pyright: ignore[reportUnusedFunction]
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)


def _config(**overrides: object) -> RerankConfig:
    base: dict[str, object] = {}
    base.update(overrides)
    return RerankConfig(**base)  # type: ignore[arg-type]


def test_disabled_gives_the_noop() -> None:
    """Path one: turned off in config. No code path differs downstream."""
    reranker = build_reranker(_config(enabled=False), Settings())
    assert isinstance(reranker, NoopReranker)


def test_enabled_without_an_api_key_gives_the_noop() -> None:
    """Path two: a search must never fail because an optional quality
    enhancement is unconfigured."""
    settings = Settings(voyage_api_key=None)
    reranker = build_reranker(_config(model="voyageai:rerank-2.5-lite"), settings)
    assert isinstance(reranker, NoopReranker)


def test_voyage_is_built_when_configured() -> None:
    settings = Settings(voyage_api_key="test-key")
    reranker = build_reranker(_config(model="voyageai:rerank-2.5-lite"), settings)
    assert isinstance(reranker, VoyageReranker)


def test_local_needs_no_credentials() -> None:
    """The offline path: no key, no network at construction."""
    reranker = build_reranker(_config(model="fastembed:Xenova/ms-marco-MiniLM-L-6-v2"), Settings())
    assert isinstance(reranker, LocalCrossEncoderReranker)


def test_unknown_provider_fails_loudly_and_says_how_to_add_one() -> None:
    """A typo should not surface an hour into indexing."""
    with pytest.raises(ValueError, match="unknown rerank provider"):
        build_reranker(_config(model="cohere:rerank-v3"), Settings(voyage_api_key="k"))


def test_bare_model_name_is_rejected_at_config_load() -> None:
    """A bare name cannot say which provider serves it, which is what made the
    abstraction unusable before."""
    with pytest.raises(ValueError, match="must be `provider:model`"):
        RerankConfig(model="rerank-2.5-lite")


def test_provider_and_model_id_split() -> None:
    config = _config(model="fastembed:Xenova/ms-marco-MiniLM-L-6-v2")
    assert config.provider == "fastembed"
    assert config.model_id == "Xenova/ms-marco-MiniLM-L-6-v2"


def test_every_implementation_satisfies_the_protocol() -> None:
    """The seam is structural, so nothing has to inherit it — but everything
    has to match it."""
    built = [
        build_reranker(_config(enabled=False), Settings()),
        build_reranker(_config(model="fastembed:Xenova/ms-marco-MiniLM-L-6-v2"), Settings()),
        build_reranker(_config(model="voyageai:rerank-2.5-lite"), Settings(voyage_api_key="k")),
    ]
    assert all(isinstance(r, Reranker) for r in built)


def test_missing_sdk_degrades_rather_than_aborting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path two again, by a different route: the provider SDKs are optional
    extras and a missing one is a configuration gap, not a lost index run."""
    import workspace_indexer.rerank.reranker_factory as factory

    def explode(*_: object, **__: object) -> object:
        raise ImportError("no module named fastembed")

    monkeypatch.setattr(factory, "LocalCrossEncoderReranker", explode)
    reranker = build_reranker(_config(model="fastembed:whatever"), Settings())
    assert isinstance(reranker, NoopReranker)


async def test_noop_preserves_order_and_respects_top_n() -> None:
    from workspace_indexer.models import SearchHit

    hits = [
        SearchHit(chunk_id=f"id-{i}", score=1.0, rel_path=f"f{i}.py", root_label="r")
        for i in range(5)
    ]
    ranked = await NoopReranker().rerank("q", hits, top_n=3)
    assert [h.chunk_id for h in ranked] == ["id-0", "id-1", "id-2"]


def test_a_database_provider_never_reaches_this_factory() -> None:
    """It used to resolve to a no-op here because the store reranked instead.
    The store's implementation is gone (#73), so the model is refused at config
    load -- this asserts the factory is not quietly carrying a second opinion
    about a provider it can no longer be handed."""
    from workspace_indexer.rerank.reranker_factory import KNOWN_PROVIDERS

    assert "database" not in KNOWN_PROVIDERS
    assert "server" not in KNOWN_PROVIDERS


def test_an_unknown_provider_still_names_what_is_known() -> None:
    """The message is the only thing telling someone what to type instead, so
    it has to list the providers that actually survive."""
    with pytest.raises(ValueError, match="unknown rerank provider") as caught:
        build_reranker(RerankConfig(model="nonesuch:some-model"), Settings())

    message = str(caught.value)
    assert "voyageai" in message
    assert "fastembed" in message
    assert "database" not in message
