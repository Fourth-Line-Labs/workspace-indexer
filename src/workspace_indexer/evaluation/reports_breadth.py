"""Optional capability: how wide a retriever's match was before the cutoff."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ReportsBreadth(Protocol):
    """How many files the retriever matched before the rank cutoff.

    Optional, because it only means something for a retriever that can return
    an unbounded set. A dense search returns `limit` hits by construction, so
    its breadth would be the cutoff restated; a lexical tool returns nothing or
    four hundred files, and both of those are failures that recall@k scores as
    an ordinary miss and an outright success respectively.

    Read after `retrieve` rather than returned from it, so `Retriever` keeps
    its contract that `limit` is a cap and not a suggestion.
    """

    last_breadth: int | None
