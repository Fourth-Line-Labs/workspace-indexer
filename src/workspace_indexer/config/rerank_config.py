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
        # Normalized before the comparison: `.env` is where this is typed, next
        # to keys that are themselves uppercase, and `DATABASE:rerank-2.5-lite`
        # slipping through would surface from the reranker factory as "unknown
        # rerank provider" -- the confusing failure this refusal exists to
        # replace.
        if value.split(":", 1)[0].strip().lower() in DATABASE_PROVIDERS:
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
        return self.model.split(":", 1)[0]

    @property
    def model_id(self) -> str:
        return self.model.split(":", 1)[1]
