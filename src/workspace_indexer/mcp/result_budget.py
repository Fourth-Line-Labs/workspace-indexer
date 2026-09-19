"""Fitting hits into a token budget without lying about what was cut."""

from __future__ import annotations

from workspace_indexer.mcp.search_result import SearchResult
from workspace_indexer.models import SearchHit

# Roughly four characters per token for code and prose alike. Deliberately an
# estimate: the exact count needs the provider's tokenizer, which is an async
# network-shaped call, and spending that to trim a response is not a good
# trade. The budget is a guard rail, not an accounting record, so erring a few
# percent either way costs nothing.
_CHARS_PER_TOKEN = 4


class ResultBudget:
    """Packs hits into a fixed token budget, newest-ranked first.

    Two rules, both learned from what an over-long tool result does to a
    session. Whole hits are dropped rather than every hit being shaved to
    uselessness -- three complete chunks beat ten fragments. And a chunk that
    would overflow on its own is truncated and *flagged*, never quietly cut,
    because an agent that thinks it read a whole function will confidently
    describe the half it got.
    """

    def __init__(self, max_tokens: int, min_chunk_tokens: int = 64) -> None:
        self._max_tokens = max_tokens
        self._min_chunk_tokens = min_chunk_tokens

    def pack(
        self, hits: list[SearchHit], *, include_text: bool = True
    ) -> tuple[list[SearchResult], int]:
        """Returns the results that fit, and how many were dropped.

        `include_text=False` drops the chunk bodies, which are around ninety
        per cent of what a hit costs -- so the same budget holds far more of
        them. Measured on real tool calls: every response that overflowed did
        so with the body included, and none at the default limit overflowed at
        all.
        """
        results: list[SearchResult] = []
        spent = 0
        # A body-less hit costs tens of tokens, not hundreds, so the
        # "too small to be worth returning" floor would stop packing while
        # there was still room for a dozen more of them. There is no partial
        # location: it either fits or it does not.
        floor = self._min_chunk_tokens if include_text else 1
        for index, hit in enumerate(hits):
            remaining = self._max_tokens - spent
            # The first hit always goes in, even under a budget too small to
            # hold it. Returning nothing would be indistinguishable from "no
            # matches", and the caller would go and look somewhere else for
            # something we actually found.
            if remaining < floor and index:
                return results, len(hits) - index
            result = _to_result(hit, include_text=include_text)
            cost = _tokens(hit) if include_text else _metadata_tokens(result)
            if include_text and cost > remaining:
                result.text = _clip(hit.source_text, remaining)
                result.text_truncated = True
                cost = remaining
            results.append(result)
            spent += cost
        return results, 0


def _tokens(hit: SearchHit) -> int:
    """The indexed count where we have one, an estimate otherwise.

    `token_count` is the embedder's count of `embed_text`, which carries a
    context header we do not return -- so it overstates slightly. Overstating
    is the safe direction for a budget.
    """
    return hit.token_count or max(1, len(hit.source_text) // _CHARS_PER_TOKEN)


def _metadata_tokens(result: SearchResult) -> int:
    """What a hit costs once its body is gone.

    Measured off the serialized result rather than assumed, so adding a field
    to `SearchResult` cannot quietly make the budget optimistic -- the failure
    that would cause is an over-long response, which is the one thing this
    class exists to prevent.
    """
    return max(1, len(result.model_dump_json()) // _CHARS_PER_TOKEN)


def _clip(text: str, tokens: int) -> str:
    limit = max(0, tokens * _CHARS_PER_TOKEN)
    if len(text) <= limit:
        return text
    # On a line boundary, so the result is still readable code.
    cut = text[:limit].rsplit("\n", 1)[0]
    return (cut or text[:limit]) + "\n... (truncated to fit the response budget)"


def _to_result(hit: SearchHit, *, include_text: bool = True) -> SearchResult:
    return SearchResult(
        location=hit.location,
        rel_path=hit.rel_path,
        abs_path=hit.abs_path,
        start_line=hit.start_line,
        end_line=hit.end_line,
        doc_type=hit.doc_type.value,
        symbol_path=hit.symbol_path,
        language=hit.language,
        repo=hit.repo_name,
        text=hit.source_text if include_text else "",
        text_omitted=not include_text,
        stale=hit.stale,
        score=round(hit.rerank_score if hit.rerank_score is not None else hit.score, 4),
    )
