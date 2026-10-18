"""How one query scored."""

from __future__ import annotations

from pydantic import BaseModel


def path_matches(expected: str, path: str) -> bool:
    """Does one returned path satisfy one expectation?

    Substring, so a case survives a file moving within a directory.
    Case-insensitive, because Linux paths are case-sensitive but a case
    mismatch in a hand-written dataset is a typo rather than a meaningful
    distinction -- and scoring a successful retrieval as a miss because the
    dataset said CONTRIBUTING.md while the file is docs/contributing.md makes
    the measurement lie.

    Deliberately takes two strings rather than a string and a list: the two
    call sites iterate in opposite directions, and a (str, list[str]) signature
    lets them be swapped silently, since both orderings typecheck.
    """
    return expected.casefold() in path.casefold()


class EvalResult(BaseModel):
    query: str
    expected: list[str]
    found: list[str]
    # 1-based position of the first expected file, or None if it never
    # appeared. This is what MRR is computed from.
    first_hit_rank: int | None = None
    # Wall time for the whole retrieval: query embedding, both branches,
    # fusion and reranking. End to end rather than store-only, because that is
    # what an agent waits for -- and because a backend that fuses server-side
    # trades a round trip against work we would otherwise do here, which a
    # store-only number would hide.
    duration_ms: float = 0.0
    # Files matched before the rank cutoff, for retrievers that can exceed it.
    # None where the notion does not apply, which is every dense retriever.
    breadth: int | None = None

    @property
    def recall(self) -> float:
        if not self.expected:
            return 0.0
        hit = sum(1 for want in self.expected if any(path_matches(want, got) for got in self.found))
        return hit / len(self.expected)

    @property
    def reciprocal_rank(self) -> float:
        return 0.0 if self.first_hit_rank is None else 1.0 / self.first_hit_rank
