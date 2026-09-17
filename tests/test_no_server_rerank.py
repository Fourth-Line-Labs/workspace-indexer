"""The store's pipeline tail, with nothing reranking inside the query.

`NoServerRerank` is the only `ServerReranker` left: the Atlas `$rerank`
implementation was retired in #73 because no environment this project can reach
could execute it. What these assert is that the surviving tail is *exactly* the
projection the store wrote before any of this existed -- the failure worth
catching is a default that quietly changes retrieval, not one that errors.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from workspace_indexer.config import RerankConfig
from workspace_indexer.storage.no_server_rerank import NoServerRerank
from workspace_indexer.storage.server_reranker import ServerReranker


def test_the_surviving_implementation_satisfies_the_protocol() -> None:
    assert isinstance(NoServerRerank(), ServerReranker)


def test_off_by_default_adds_only_the_score_projection() -> None:
    """Turning the last reranker off must not change retrieval: this is the
    tail the store wrote before server-side reranking was ever added."""
    assert NoServerRerank().stages("q", 10, "vectorSearchScore") == [
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}}
    ]


def test_the_score_metadata_name_is_taken_from_the_caller() -> None:
    """Each branch names its score differently -- `vectorSearchScore`,
    `searchScore`, `score` after a fusion -- and asking for the wrong one is
    not an error: it yields a missing field, so every hit scores 0.0 and the
    ranking silently collapses."""
    assert NoServerRerank().stages("q", 10, "searchScore") == [
        {"$addFields": {"score": {"$meta": "searchScore"}}}
    ]


def test_nothing_widens_the_candidate_set() -> None:
    """Retrieving fifty documents to return ten is only worth paying for when
    something is going to reorder them, and now nothing in the store is."""
    assert NoServerRerank().depth(10) == 10


def test_the_name_says_nothing_reranks_in_the_store() -> None:
    """`MongoStore.describe` and `search.store` read this."""
    assert NoServerRerank().name == "none"


# ---- the retirement itself (#73) -------------------------------------------


@pytest.mark.parametrize("provider", ["database", "server"])
def test_a_database_rerank_model_is_refused_at_config_load(provider: str) -> None:
    """The failure this prevents is silent, not loud. Both factories used to
    consult a provider list; with the store's implementation gone, a `database:`
    model that merely *parsed* would leave the client-side factory declining to
    rerank and every search returning unreranked fusion order while the config
    said otherwise."""
    with pytest.raises(ValidationError) as caught:
        RerankConfig(model=f"{provider}:rerank-2.5-lite")

    message = str(caught.value)
    assert "8.3" in message
    assert "Native Reranking" in message
    assert "voyageai:rerank-2.5-lite" in message


def test_the_refusal_points_at_the_issue_that_restores_it() -> None:
    """Whoever gets a capable Atlas cluster has to be able to find the work,
    or the retirement reads as "this was never possible"."""
    with pytest.raises(ValidationError, match="issue #94"):
        RerankConfig(model="database:rerank-2.5-lite")


def test_disabling_reranking_does_not_smuggle_the_model_past_the_check() -> None:
    """`enabled=False` is applied long after the field validator, so a
    `database:` model must be refused whether or not reranking is on -- a
    config that is rejected only when enabled is a trap for whoever flips it."""
    with pytest.raises(ValidationError, match="8.3"):
        RerankConfig(enabled=False, model="database:rerank-2.5-lite")


def test_client_side_providers_are_untouched() -> None:
    """The check must key on the provider, not on the word "rerank" appearing
    in the model name."""
    assert RerankConfig(model="voyageai:rerank-2.5-lite").provider == "voyageai"
    assert RerankConfig(model="fastembed:Xenova/ms-marco-MiniLM-L-6-v2").provider == "fastembed"


@pytest.mark.parametrize("model", ["DATABASE:rerank-2.5-lite", "Server:rerank-2.5", "database :x"])
def test_the_refusal_survives_casing_and_padding(model: str) -> None:
    """`.env` is where this gets typed, alongside keys that are themselves
    uppercase. An unnormalized comparison let `DATABASE:` past config load and
    into the reranker factory, which called it an unknown provider -- naming
    neither the requirement nor the way out, which is the whole failure this
    refusal replaces."""
    with pytest.raises(ValidationError, match="8.3"):
        RerankConfig(model=model)
