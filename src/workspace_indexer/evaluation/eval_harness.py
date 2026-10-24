"""Measuring retrieval quality.

Every knob in this system — dimensions, fusion mode, prefetch limit, rerank
model, chunk size, whether the context header helps — is a plausible-sounding
choice that can only be settled by measurement. Without this, tuning becomes
folklore.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import yaml

from workspace_indexer.evaluation.eval_case import EvalCase
from workspace_indexer.evaluation.eval_report import EvalReport
from workspace_indexer.evaluation.eval_result import EvalResult, path_matches
from workspace_indexer.evaluation.reports_breadth import ReportsBreadth
from workspace_indexer.evaluation.retriever import Retriever
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.evaluation")


def load_cases(path: Path) -> list[EvalCase]:
    if not path.is_file():
        raise FileNotFoundError(
            f"no eval dataset at {path}. Write ~20-30 real queries against this "
            "workspace with the files each should return."
        )
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path} must be a list of cases")
    # isinstance narrows `object` only to list[Unknown]; each element is
    # validated by EvalCase anyway, which is where a malformed case is caught.
    items = cast("list[Any]", raw)
    return [EvalCase.model_validate(item) for item in items]


class EvalHarness:
    def __init__(self, retriever: Retriever) -> None:
        self._retriever = retriever

    async def run(
        self,
        cases: list[EvalCase],
        *,
        limit: int = 10,
        label: str = "default",
    ) -> EvalReport:
        results: list[EvalResult] = []

        for case in cases:
            started = time.perf_counter()
            found = await self._retriever.retrieve(case.query, limit)
            elapsed = (time.perf_counter() - started) * 1000
            # Asked after the call rather than returned from it: a retriever
            # that cannot exceed the cutoff has no breadth worth reporting, and
            # making every retriever return one would mean inventing a number
            # for the dense path that only restates `limit`.
            breadth = (
                self._retriever.last_breadth
                if isinstance(self._retriever, ReportsBreadth)
                else None
            )
            results.append(
                # Pinned here rather than left to be recomputed from `found`:
                # the paths are rewritten by `PublicPathGuard` on the way to
                # disk, and a score derived afterwards scores the redaction
                # instead of the retrieval.
                EvalResult(
                    query=case.query,
                    expected=case.expect,
                    found=found,
                    first_hit_rank=_first_rank(case.expect, found),
                    duration_ms=elapsed,
                    breadth=breadth,
                ).measured()
            )

        report = EvalReport(label=label, limit=limit, results=results)
        log.info(
            "eval.report",
            label=label,
            retriever=self._retriever.name,
            cases=len(results),
            recall=round(report.recall_at_k, 3),
            mrr=round(report.mrr_at_k, 3),
            misses=len(report.misses),
            median_ms=round(report.median_ms, 1),
            p95_ms=round(report.p95_ms, 1),
            empty_rate=round(report.empty_rate, 3),
            flood_rate=round(report.flood_rate, 3),
        )
        return report


def _first_rank(expected: list[str], found: list[str]) -> int | None:
    for position, path in enumerate(found, start=1):
        if any(path_matches(want, path) for want in expected):
            return position
    return None
