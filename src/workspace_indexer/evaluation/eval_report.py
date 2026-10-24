"""Aggregate scores for a whole dataset."""

from __future__ import annotations

from pydantic import BaseModel

from workspace_indexer.evaluation.eval_result import EvalResult

# How many times the rank cutoff a result has to exceed before it is a flood
# rather than a generous answer. Three is a judgement, not a measurement: at
# -n 10 it means thirty files, which is more than an agent will open and few
# enough that a genuinely well-targeted pattern stays under it.
FLOOD_FACTOR = 3


class EvalReport(BaseModel):
    label: str
    limit: int
    results: list[EvalResult]

    @property
    def recall_at_k(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.recall for r in self.results) / len(self.results)

    @property
    def mrr_at_k(self) -> float:
        """Mean reciprocal rank rewards putting the right file *first*, which
        recall alone cannot see: a result at rank 10 and one at rank 1 score
        the same on recall."""
        if not self.results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.results) / len(self.results)

    @property
    def misses(self) -> list[EvalResult]:
        return [r for r in self.results if r.recall < 1.0]

    @property
    def median_ms(self) -> float:
        """The typical query. Median rather than mean: one cold start or one
        retried request drags a mean over sixteen cases far enough to change
        which backend looks faster."""
        return self._percentile(0.5)

    @property
    def p95_ms(self) -> float:
        """The tail. This is the number an agent notices, because a tool it
        calls a dozen times in a task hits the tail every task."""
        return self._percentile(0.95)

    @property
    def empty_rate(self) -> float:
        """Share of queries the retriever answered with nothing at all.

        An agent can act on a wrong answer -- it reads the file and moves on.
        It cannot act on silence, and a tool that is silent one query in three
        is not a tool an agent should reach for first, whatever its recall on
        the queries it does answer.

        Over every result, unlike `flood_rate` and `median_breadth`. Returning
        nothing is visible in `found` for any retriever, while breadth is a
        count only the retrievers that can exceed the cutoff report -- so
        filtering on breadth here scored a search or tool run that answered
        nothing at all as `empty_rate` 0.0, for the surface the metric exists
        to compare the lexical tools against.
        """
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if not r.found) / len(self.results)

    @property
    def reports_breadth(self) -> bool:
        """Whether any result came from a retriever that counts its matches.

        `median_breadth` falls back to 0.0 on an empty sample, which is
        indistinguishable from a measured zero -- so a caller that renders it
        has to ask this first rather than treating a falsy value as "nothing
        to show". `empty_rate` is counted over every result and is therefore
        no longer a usable proxy.
        """
        return any(r.breadth is not None for r in self.results)

    @property
    def flood_rate(self) -> float:
        """Share of queries that matched more than `flood_factor` x the cutoff.

        The failure recall@k is blind to. A pattern matching four hundred files
        has not found anything: the right file is in there, so recall scores it
        a win, while the agent is handed a haystack and the ranking that
        decided which ten of the four hundred it sees is match count, which is
        not relevance. Counted against the cutoff rather than a flat number so
        it still means the same thing at a different -n.
        """
        scored = [r for r in self.results if r.breadth is not None]
        if not scored:
            return 0.0
        ceiling = self.limit * FLOOD_FACTOR
        return sum(1 for r in scored if (r.breadth or 0) > ceiling) / len(scored)

    @property
    def median_breadth(self) -> float:
        """Typical number of files matched, for retrievers that report it.

        Through the same nearest-rank index as `median_ms`. Taking the
        upper-middle element here while `_percentile` takes the lower-middle
        one for some even samples would print a report whose two "typical"
        numbers follow different rules -- and the rule would flip silently the
        next time the dataset gained two cases.
        """
        widths = sorted(float(r.breadth) for r in self.results if r.breadth is not None)
        return _at(widths, 0.5)

    @property
    def slowest_ms(self) -> float:
        return max((r.duration_ms for r in self.results), default=0.0)

    def _percentile(self, share: float) -> float:
        return _at(sorted(r.duration_ms for r in self.results), share)


def _at(values: list[float], share: float) -> float:
    """Nearest-rank. Exact interpolation is false precision over sixteen
    samples, and the ranking between two backends is what this decides."""
    if not values:
        return 0.0
    index = min(len(values) - 1, int(round(share * (len(values) - 1))))
    return values[index]
