"""One eval run, as it is written to disk.

Flat and versioned on purpose. JSON does not enforce a shape the way a columnar
format would, so the discipline has to come from here plus a test -- otherwise
files drift and anything reading across them gets unreliable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from workspace_indexer.evaluation.eval_result import EvalResult

# Bump when the record shape changes incompatibly, so a reader can tell rather
# than silently misinterpreting an older file.
SCHEMA_VERSION = 1


class EvalRecord(BaseModel):
    schema_version: int = SCHEMA_VERSION
    recorded_at: str
    label: str

    # Everything that affects the numbers, so two runs are only comparable when
    # these match. config_hash deliberately excludes credentials -- rotating an
    # API key is not a change in configuration.
    config_hash: str
    space_slug: str
    embedding_model: str
    dimensions: int
    fusion: str
    reranker: str
    limit: int
    # Which retrieval surface produced these numbers: the raw search path, or
    # one of the MCP tools with its own document-type policy. A tool's filter
    # is the entire hypothesis under test, so a run through it is not the same
    # measurement as a run through the service beneath it.
    retriever: str = "search"
    # Which slice of the dataset ran. A guidance-only run and a full run are
    # both honest numbers and comparing them is not.
    case_filter: str = "all"

    recall_at_k: float
    mrr_at_k: float
    case_count: int
    miss_count: int

    # Latency travels with relevance because the two trade against each other
    # and a backend choice needs both. Defaulted so records written before
    # timing existed still load rather than failing the comparison outright.
    median_ms: float = 0.0
    p95_ms: float = 0.0

    # What recall@k cannot say. Defaulted to zero so records written before the
    # grep comparison still load -- and zero is the truthful value for them,
    # since a dense retriever never returns nothing and never floods.
    empty_rate: float = 0.0
    flood_rate: float = 0.0
    median_breadth: float = 0.0

    results: list[EvalResult] = Field(default_factory=list[EvalResult])

    def comparable_to(self, other: EvalRecord) -> bool:
        """Whether a delta between these two means anything.

        Different config hashes measure different systems. Reporting a delta
        across them looks authoritative and is not.

        fusion and reranker are checked separately because neither is in
        config_hash: both are per-call overrides rather than settings. Turning
        reranking off is a deliberate experiment, not a regression, and an
        automatic "vs last run" that conflated the two would report a 0.3 drop
        in MRR as though something had broken.

        retriever and case_filter are here for the same reason. find_guidance
        scored over the eight guidance cases is the number that decides whether
        document classification earns its place; the full dataset through plain
        search is a different question with a different answer, and a delta
        between them would be meaningless in either direction.
        """
        return (
            self.config_hash == other.config_hash
            and self.fusion == other.fusion
            and self.reranker == other.reranker
            and self.retriever == other.retriever
            and self.case_filter == other.case_filter
            # Different dataset, different question. A 13-case corpus run and a
            # 16-case run over this repo matched on every field above and were
            # reported as a delta twice in one session, which is how a number
            # that means nothing acquires a plus sign in front of it.
            #
            # A weak proxy for "same dataset" -- two different datasets of the
            # same size still compare -- but it costs nothing and catches the
            # case that actually happened. config_hash omitting `roots` is the
            # real gap and is tracked separately.
            and self.case_count == other.case_count
            # `flood_rate` counts matches above `limit * FLOOD_FACTOR` and
            # `median_breadth` moves with the cutoff too, so a -n 10 run and a
            # -n 50 run are not the same measurement. recall@k and MRR@k were
            # always cutoff-relative as well; the breadth metrics are what made
            # ignoring it load-bearing rather than merely untidy.
            and self.limit == other.limit
        )
