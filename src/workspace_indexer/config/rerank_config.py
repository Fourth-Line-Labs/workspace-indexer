"""Reranking settings."""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from workspace_indexer.config.strict import Strict

# Asked the *store* to rerank inside the query. Retired in #73: Atlas
# `$rerank` needs a cluster at MongoDB 8.3+ with Native Reranking enabled,
# and no environment this project can reach provides one -- the stage is
# absent from the local binary at every image tag, and the Flex tier cannot
# reach 8.3. Refused at config load rather than left to fail mid-query,
# because the alternative is worse: both factories decline to rerank and
# every search quietly returns fusion order while the config says otherwise.
DATABASE_PROVIDERS = frozenset({"database", "server"})


def _provider_of(model: str) -> str:
    """The provider half, canonicalized.

    One rule, used by the validator and by `provider`, because the two
    disagreeing is invisible: the validator would refuse a spelling the factory
    then failed to recognise, or the reverse. Normalized because `.env` is where
    this is typed -- `VOYAGEAI:rerank-2.5-lite` and a stray space both used to
    clear config load and die in the reranker factory as an unknown provider,
    and `'voyageai '` printed next to a list containing `voyageai` differs by an
    invisible character.

    Only the provider is lowercased. `model_id` keeps its case
    (`BAAI/bge-reranker-base` is case-sensitive) and `model` keeps whatever was
    typed, so an error quotes the input back rather than a cleaned-up version
    of it.
    """
    return model.split(":", 1)[0].strip().lower()


class RerankConfig(Strict):
    enabled: bool = True
    # `provider:model`, the same convention EMBEDDING_MODEL uses, so both
    # layers read the same way. A bare model name cannot express which
    # provider serves it, which is what made the abstraction unusable before.
    model: str = "voyageai:rerank-2.5-lite"
    candidates: int = 50
    top_n: int = 10
    # The reranker benefits from the same context header the embedder gets: a
    # bare `def upsert(...)` body is ambiguous without its file and class.
    rerank_text: Literal["embed_text", "source_text"] = "embed_text"
    # The rerank models follow instructions but expose no instruction
    # parameter, so this is prepended to the query string client-side.
    instruction: str | None = None
    # degrade: an API failure returns the fusion ordering with a WARNING.
    # fail: raise. Only the eval harness wants that — a silent degradation
    # there would quietly corrupt a measurement.
    on_error: Literal["degrade", "fail"] = "degrade"

    @field_validator("model")
    @classmethod
    def _requires_a_provider(cls, value: str) -> str:
        """Caught at config load rather than an hour into a run."""
        if ":" not in value or not all(part.strip() for part in value.split(":", 1)):
            raise ValueError(
                f"rerank model {value!r} must be `provider:model`, "
                "e.g. voyageai:rerank-2.5-lite or "
                "fastembed:Xenova/ms-marco-MiniLM-L-6-v2"
            )
        if _provider_of(value) in DATABASE_PROVIDERS:
            raise ValueError(
                f"rerank model {value!r} asks the database to rerank, which this "
                "build does not support. Atlas $rerank needs BOTH a cluster running "
                "MongoDB 8.3 or later -- 'Latest version with auto-upgrades' in the "
                "cluster builder; 8.0 is not enough even with the toggle on -- AND "
                "Native Reranking enabled in Project Settings, which requires "
                "Project Owner access. No environment this project can reach has "
                "both, so the implementation was retired (issue #73) rather than "
                "kept as code nobody could run. Use a client-side reranker, e.g. "
                "voyageai:rerank-2.5-lite. Restoring it is tracked in issue #94."
            )
        return value

    @property
    def provider(self) -> str:
        return _provider_of(self.model)

    @property
    def model_id(self) -> str:
        """Stripped, for the same reason the provider is.

        A space *after* the colon used to survive all the way to the provider:
        `voyageai: rerank-2.5-lite` cleared config load and then failed at query
        time as an invalid Voyage model, or a not-found fastembed one. Case is
        untouched -- `BAAI/bge-reranker-base` is case-sensitive.
        """
        return self.model.split(":", 1)[1].strip()
