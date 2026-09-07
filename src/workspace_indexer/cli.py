"""Command line interface."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.table import Table

from workspace_indexer.app_context import AppContext
from workspace_indexer.config import ConfigError, load_workspace_config
from workspace_indexer.config.loader import DEFAULT_CONFIG_PATH
from workspace_indexer.evaluation import (
    EvalComparison,
    EvalHarness,
    EvalRecord,
    Retriever,
    SearchRetriever,
    ToolRetriever,
    compare,
    latest_comparable,
    load_cases,
    read_records,
    write_record,
    write_report,
)
from workspace_indexer.evaluation.search_retriever import Fusion
from workspace_indexer.graph import SUPPORTED as IMPORT_LANGUAGES
from workspace_indexer.graph.origin_coverage import OriginCoverage
from workspace_indexer.grounding import CoverageService, SourceStrength, UnitCoverage
from workspace_indexer.mcp import QueryService, TaxonomyService
from workspace_indexer.models import EmbeddingSpace, FileKind, RunStats, SearchFilters
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.search import Reprojector, SearchRequest
from workspace_indexer.storage import StoreMirror, build_vector_store

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Semantic + keyword index over a multi-repo workspace.",
)
console = Console()

# Named for the command it serves, so a JSONL reader can tell a watcher's own
# failures from the indexer failures it triggers.
watch_log = get_logger("workspace_indexer.cli.watch")

ConfigOption = Annotated[Path | None, typer.Option("--config", "-c", help="Path to workspace.yaml")]


def _context(config: Path | None, role: str | None = None) -> AppContext:
    """`role` is the command's own name, and keeps its log file separate.

    Passed explicitly by each command rather than sniffed from sys.argv: a
    lookup that works and cannot be traced from the call site is worse than a
    parameter, and this one is read in a traceback more often than written.
    """
    try:
        return AppContext.build(config, role)
    except ConfigError as exc:
        # A config problem is a user problem, not a traceback.
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc


@app.command()
def index(
    config: ConfigOption = None,
    root: Annotated[str | None, typer.Option(help="Index only this root label")] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the chunk plan and token estimate, no API calls")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Ignore mtime and hash shortcuts")] = False,
    allow_deletes: Annotated[
        bool,
        typer.Option(
            "--allow-deletes",
            help="Remove a root's index even when most of it disappeared at once",
        ),
    ] = False,
) -> None:
    """Index the configured roots, reusing everything unchanged.

    Files gone from disk are removed from the index. That decision is made from
    an absence, which is also what a partial checkout looks like -- so a run
    that would remove most of a root at once stops and says so instead.
    `--allow-deletes` is how you say the files really are gone.
    """

    async def run() -> None:
        ctx = _context(config, "index")
        try:
            stats = await ctx.indexer().run(
                only_root=root, force=force, dry_run=dry_run, allow_deletes=allow_deletes
            )
        finally:
            await ctx.close()

        table = Table(title=f"{stats.mode} · {stats.run_id}", show_header=False)
        table.add_row("files seen", str(stats.files_seen))
        table.add_row("unchanged", str(stats.files_skipped))
        table.add_row("changed", str(stats.files_changed))
        table.add_row("chunks written", str(stats.chunks_upserted))
        table.add_row("chunks removed", str(stats.chunks_deleted))
        table.add_row("tokens", f"{stats.tokens_embedded:,}")
        table.add_row("estimated cost", _cost_cell(stats))
        if stats.deletions_withheld:
            table.add_row("[yellow]deletions withheld[/yellow]", str(stats.deletions_withheld))
        if stats.errors:
            table.add_row("[red]errors[/red]", str(stats.errors))
        console.print(table)
        if stats.mode == "dry-run":
            console.print("[dim]No API calls were made and nothing was stored.[/dim]")
        if stats.deletions_withheld:
            # Non-zero, because the index is now knowingly out of step with
            # disk and a warning in CI output is not a brake. The embeddings
            # this run paid for are already stored; only the pruning stopped.
            console.print(
                f"[yellow]{stats.deletions_withheld} file(s) were left in the index "
                "although they are not on disk.[/yellow] Either the checkout is "
                "incomplete, or they really are gone and this wants --allow-deletes."
            )
            raise typer.Exit(code=3)

    asyncio.run(run())


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What to look for")],
    config: ConfigOption = None,
    limit: Annotated[int | None, typer.Option("-n", "--limit")] = None,
    unit: Annotated[str | None, typer.Option(help="Only this repo or folder")] = None,
    lang: Annotated[str | None, typer.Option(help="Only this language")] = None,
    kind: Annotated[str | None, typer.Option(help="code, markdown, text...")] = None,
    path: Annotated[str | None, typer.Option(help="Only under this directory")] = None,
    fusion: Annotated[
        str | None, typer.Option(help="rrf, dense_only or sparse_only (debugging)")
    ] = None,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    full: Annotated[bool, typer.Option("--full", help="Print whole chunks")] = False,
) -> None:
    """Search the index."""

    async def run() -> None:
        ctx = _context(config, "search")
        try:
            hits = await ctx.search_service().search(
                SearchRequest(
                    query=query,
                    limit=limit,
                    fusion=fusion,  # type: ignore[arg-type]
                    rerank=rerank,
                    filters=SearchFilters(
                        unit=unit,
                        language=lang,
                        kind=FileKind(kind) if kind else None,
                        path_prefix=path,
                    ),
                )
            )
        finally:
            await ctx.close()

        if not hits:
            console.print("[yellow]No matches.[/yellow]")
            return

        for position, hit in enumerate(hits, start=1):
            score = hit.rerank_score if hit.rerank_score is not None else hit.score
            flag = " [yellow](stale)[/yellow]" if hit.stale else ""
            symbol = f" [dim]{hit.symbol_path}[/dim]" if hit.symbol_path else ""
            console.print(f"[bold]{position:>2}.[/bold] [cyan]{hit.location}[/cyan]{symbol}{flag}")
            console.print(f"    [dim]score {score:.4f} · {hit.unit or hit.root_label}[/dim]")
            body = hit.source_text if full else _preview(hit.source_text)
            for line in body.splitlines():
                console.print(f"    {line}")
            console.print()

    asyncio.run(run())


@app.command()
def status(config: ConfigOption = None) -> None:
    """What is indexed, in which spaces, and what recent runs cost."""

    async def run() -> None:
        ctx = _context(config, "status")
        try:
            roots = Table(title="files by root")
            roots.add_column("root")
            roots.add_column("files", justify="right")
            for label, count in sorted(ctx.manifest.counts_by_root().items()):
                roots.add_row(label, f"{count:,}")
            console.print(roots)

            spaces = Table(title="embedding spaces")
            spaces.add_column("space")
            spaces.add_column("chunks", justify="right")
            spaces.add_column("stored", justify="right")

            prefix = f"{ctx.config.workspace.name}__"
            collections = {
                name.removeprefix(prefix)
                for name in await ctx.store.collection_names()
                if name.startswith(prefix)
            }
            tracked = set(ctx.manifest.spaces())
            warnings: list[str] = []

            for slug in sorted(tracked):
                active = slug == ctx.space.slug()
                recorded = ctx.manifest.chunk_count(slug)
                stored = await ctx.store.count(ctx.space) if active else None
                if stored is not None and stored != recorded:
                    # Divergence has to announce itself. Printing two numbers
                    # side by side and hoping someone notices is how stale
                    # content sat in a live collection unremarked.
                    warnings.append(
                        f"{slug}: manifest has {recorded:,} chunks, store has {stored:,}"
                    )
                spaces.add_row(
                    slug + (" [green](active)[/green]" if active else ""),
                    f"{recorded:,}",
                    "-" if stored is None else f"{stored:,}",
                )
            console.print(spaces)

            for slug in sorted(tracked - collections):
                warnings.append(f"{slug}: recorded in the manifest but no collection exists")
            for slug in sorted(collections - tracked):
                warnings.append(f"{slug}: collection exists but nothing is recorded for it")

            for warning in warnings:
                console.print(f"[yellow]inconsistent[/yellow] {warning}")

            runs = Table(title="recent runs")
            for column in ("run", "mode", "files", "written", "removed", "tokens", "cost"):
                runs.add_column(column)
            for row in ctx.manifest.recent_runs(limit=5):
                runs.add_row(
                    row.run_id,
                    row.mode + (" [red](unfinished)[/red]" if row.unfinished else ""),
                    f"{row.files_seen:,}",
                    f"{row.chunks_upserted:,}",
                    f"{row.chunks_deleted:,}",
                    f"{row.tokens_embedded:,}",
                    row.cost_display,
                )
            console.print(runs)
            _print_spend(ctx)
            _print_import_coverage(ctx)
            _print_route_coverage(ctx)
            _print_tool_calls(ctx)
        finally:
            await ctx.close()

    asyncio.run(run())


@app.command()
def grounding(config: ConfigOption = None) -> None:
    """Which repositories can explain why they are the way they are."""

    ctx = _context(config, "grounding")
    try:
        units = CoverageService(ctx.manifest).coverage()
    finally:
        ctx.manifest.close()

    if not units:
        console.print("[yellow]nothing indexed yet; run `index` first[/yellow]")
        return

    for unit in units:
        console.print(_grounding_table(unit))
        for note in unit.notes:
            console.print(f"  [yellow]![/yellow] {note}")
        console.print()

    console.print(
        "Coverage is measured against what is indexed, not what is on disk. A source "
        "marked [red]absent[/red] cannot be retrieved from -- an empty answer there is "
        "the codebase's, not the index's."
    )


_STRENGTH_COLOUR = {
    SourceStrength.ABSENT: "red",
    SourceStrength.THIN: "yellow",
    SourceStrength.PRESENT: "green",
}


def _grounding_table(unit: UnitCoverage) -> Table:
    colour = _STRENGTH_COLOUR[unit.verdict]
    table = Table(
        title=f'{unit.label} \u2014 can answer "why": [{colour}]{unit.verdict.value}[/{colour}]'
    )
    table.add_column("source")
    table.add_column("found", justify="right")
    table.add_column("rate", justify="right")
    table.add_column("strength")
    table.add_column("what it is")
    for source in unit.sources:
        shade = _STRENGTH_COLOUR[source.strength]
        rate = (
            f"{source.share:.0%} of {source.population:,}"
            if source.population_unit == "commits"
            else f"{source.per_100:.1f} / 100 files"
        )
        table.add_row(
            source.name,
            f"{source.found:,}",
            rate,
            f"[{shade}]{source.strength.value}[/{shade}]",
            source.detail,
        )
    return table


@app.command()
def explain(
    path: Annotated[Path, typer.Argument(help="File to chunk")],
    config: ConfigOption = None,
) -> None:
    """Show the chunks a single file produces. The chunk-quality debugging tool."""
    from workspace_indexer.chunking import read_source
    from workspace_indexer.discovery import Walker

    ctx = _context(config, "explain")
    try:
        target = path.expanduser().resolve()
        candidate = next(
            (c for c in Walker(ctx.config).walk() if c.abs_path.resolve() == target), None
        )
        if candidate is None:
            console.print(f"[red]{target} is not inside any configured root, or is excluded.[/red]")
            raise typer.Exit(code=1)

        source = read_source(candidate)
        if source is None:
            console.print(f"[red]{target} could not be read.[/red]")
            raise typer.Exit(code=1)

        chunker = ctx.registry.resolve(source, ctx.config.chunking)
        chunks = list(chunker.chunk(source, ctx.config.chunking))
        verdict = ctx.classifier.classify(source)
        console.print(
            f"[bold]{source.rel_path}[/bold] · {source.kind.value}"
            f" · {source.language or 'no grammar'} · chunker [cyan]{chunker.name}[/cyan]"
        )
        console.print(
            f"type [cyan]{verdict.doc_type.value}[/cyan]"
            f" · confidence {verdict.confidence:.2f}"
            f" · [dim]{verdict.reason}[/dim]"
        )
        if not chunks:
            console.print("[yellow]No chunks: recorded in the manifest but not embedded.[/yellow]")
            return
        for chunk in chunks:
            meta = chunk.meta
            console.print(
                f"\n[cyan]{meta.start_line}-{meta.end_line}[/cyan]"
                f" · {meta.chunk_index + 1}/{meta.chunk_total}"
                f" · ~{meta.token_estimate} tokens"
                f" · [dim]{symbol_label(meta.symbol_kind, meta.symbol_path)}[/dim]"
                + (" [yellow](degraded parse)[/yellow]" if meta.parse_degraded else "")
            )
            for line in chunk.source_text.splitlines()[:12]:
                console.print(f"    {line}")
    finally:
        ctx.manifest.close()


def symbol_label(kind: str | None, path: str | None) -> str:
    """`kind path`, unless the path already begins with the kind.

    Public so a test can assert it directly: it is pure string formatting, and
    exercising it through the whole CLI would need an index behind it.

    A PDF chunk is kind "page" with path "page 3", which rendered as
    "page page 3". Harmless, and the sort of thing that makes output look
    careless enough to distrust the numbers beside it.
    """
    kind = kind or "-"
    path = path or ""
    if path.startswith(kind):
        return path
    return f"{kind} {path}".rstrip()


@app.command()
def mirror(
    to: Annotated[str, typer.Option("--to", help="Target backend: qdrant | mongodb")],
    config: ConfigOption = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite/--resume", help="Drop the target collection first")
    ] = False,
) -> None:
    """Copy the current collection into the other backend. No re-embedding.

    Vectors are the expensive part of an index and they are backend neutral:
    the same floats mean the same thing in Qdrant and in Atlas. So trying a
    second store, or migrating to one, costs a scroll and a write rather than
    the whole embedding bill again.

    It is also the only honest way to compare two backends. Re-embedding into
    the second one would give it a different set of vectors, and any difference
    in the numbers afterwards could be the store or could be the embeddings,
    with no way to tell which.

    Resumes by default: points are keyed by chunk id, so re-running after an
    interruption replaces rather than duplicates.
    """

    async def run() -> None:
        ctx = _context(config, "mirror")
        target_settings = ctx.settings.model_copy(update={"vector_store": to})
        if target_settings.vector_store == ctx.settings.vector_store:
            console.print(
                f"[red]--to {to} is the backend already configured "
                f"(VECTOR_STORE={ctx.settings.vector_store}).[/red]"
            )
            raise typer.Exit(code=2)

        target = build_vector_store(target_settings, ctx.config.workspace.name)
        try:
            source_count = await ctx.store.count(ctx.space)
            if not source_count:
                console.print(
                    f"[red]{ctx.store.describe()} holds no points for "
                    f"{ctx.space.slug()}; there is nothing to mirror.[/red]"
                )
                raise typer.Exit(code=2)

            console.print(
                f"Mirroring [cyan]{source_count:,}[/cyan] points "
                f"{ctx.store.describe()} -> {target.describe()}"
            )
            moved = await StoreMirror(ctx.store, target).mirror(ctx.space, overwrite=overwrite)
            console.print(
                f"Wrote [cyan]{moved:,}[/cyan] points to "
                f"[cyan]{target.collection_name(ctx.space)}[/cyan]."
            )
            # Atlas builds its search indexes asynchronously, so a mirror that
            # has finished writing is not yet a mirror that can answer.
            ready = await target.describe_vectors(ctx.space)
            if not (ready["dense"] and ready["sparse"]):
                console.print(
                    "[yellow]The target's search indexes are still building. "
                    "Queries return nothing until they are ready; check with "
                    "`workspace-indexer status`.[/yellow]"
                )
        finally:
            await target.close()
            await ctx.close()

    asyncio.run(run())


@app.command()
def reproject(
    dimensions: Annotated[int, typer.Option("--dimensions", "-d")],
    config: ConfigOption = None,
) -> None:
    """Derive a narrower collection by Matryoshka truncation. No re-embedding."""

    async def run() -> None:
        ctx = _context(config, "reproject")
        try:
            target = await Reprojector(ctx.store, ctx.manifest).reproject(ctx.space, dimensions)
            console.print(
                f"Wrote [cyan]{ctx.store.collection_name(target)}[/cyan] "
                f"({await ctx.store.count(target):,} points) from "
                f"[cyan]{ctx.store.collection_name(ctx.space)}[/cyan]."
            )
            console.print("[dim]Compare them with `workspace-indexer eval`.[/dim]")
        finally:
            await ctx.close()

    asyncio.run(run())


@app.command("eval")
def evaluate(
    config: ConfigOption = None,
    dataset: Annotated[Path | None, typer.Option("--dataset")] = None,
    limit: Annotated[int, typer.Option("-n", "--limit")] = 10,
    fusion: Annotated[str | None, typer.Option()] = None,
    rerank: Annotated[bool | None, typer.Option("--rerank/--no-rerank")] = None,
    dimensions: Annotated[
        int | None, typer.Option("--dimensions", help="Evaluate a reprojected collection")
    ] = None,
    save: Annotated[
        bool, typer.Option("--save/--no-save", help="Record the run under evals/")
    ] = True,
    compare_previous: Annotated[
        bool, typer.Option("--compare/--no-compare", help="Diff against the last comparable run")
    ] = True,
    tool: Annotated[
        str,
        typer.Option("--tool", help="Retrieval surface: search | search_code | find_guidance"),
    ] = "search",
    group: Annotated[
        str, typer.Option("--group", help="Case group: all | retrieval | guidance")
    ] = "all",
) -> None:
    """Score retrieval quality against a dataset of query/expected-file pairs.

    `--tool` scores an MCP tool rather than the raw search path, which is the
    only way to measure what a document-type filter is actually worth: an agent
    calls find_guidance, not SearchService, and the filter is the hypothesis.
    """

    async def run() -> None:
        ctx = _context(config, "eval")
        try:
            cases = load_cases(dataset or ctx.config.eval.dataset)
            if group != "all":
                cases = [case for case in cases if case.group == group]
                if not cases:
                    console.print(f"[red]no cases in group {group!r}[/red]")
                    raise typer.Exit(code=2)
            # A reprojected collection is a distinct space, so it has to be
            # addressed as one rather than by width alone.
            space = (
                ctx.space.model_copy(
                    update={"dimensions": dimensions, "derived_from": ctx.space.dimensions}
                )
                if dimensions
                else None
            )
            retriever = _retriever(ctx, space, tool=tool, fusion=fusion, rerank=rerank)
            label = (
                f"{(space or ctx.space).slug()} fusion={fusion or ctx.config.search.fusion} "
                f"tool={retriever.name} cases={group}"
            )
            report = await EvalHarness(retriever).run(cases, limit=limit, label=label)
            active = space or ctx.space
            record = EvalRecord(
                recorded_at=datetime.now(UTC).isoformat(),
                label=label,
                config_hash=ctx.settings.config_hash(ctx.config),
                space_slug=active.slug(),
                embedding_model=active.model,
                dimensions=active.dimensions,
                fusion=fusion or ctx.config.search.fusion,
                reranker="none" if rerank is False else ctx.config.search.rerank.model,
                limit=limit,
                retriever=retriever.name,
                case_filter=group,
                recall_at_k=report.recall_at_k,
                mrr_at_k=report.mrr_at_k,
                case_count=len(report.results),
                miss_count=len(report.misses),
                median_ms=report.median_ms,
                p95_ms=report.p95_ms,
                results=report.results,
            )
            # Written before the comparison, so an interrupted comparison does
            # not lose the measurement that was just paid for.
            saved = write_record(record) if save else None
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=2) from exc
        finally:
            await ctx.close()

        console.print(f"[bold]{report.label}[/bold]")
        console.print(f"  recall@{limit}: [cyan]{report.recall_at_k:.3f}[/cyan]")
        console.print(f"  MRR@{limit}:    [cyan]{report.mrr_at_k:.3f}[/cyan]")
        console.print(f"  cases: {len(report.results)}, misses: {len(report.misses)}")
        # End-to-end per query: embedding, both branches, fusion, reranking.
        # Reported beside relevance because the two trade against each other,
        # and a backend is chosen on both.
        console.print(
            f"  latency:   median [cyan]{report.median_ms:.0f} ms[/cyan], "
            f"p95 [cyan]{report.p95_ms:.0f} ms[/cyan], "
            f"slowest {report.slowest_ms:.0f} ms"
        )
        for miss in report.misses:
            console.print(f"  [yellow]miss[/yellow] {miss.query}")
            console.print(f"    expected {miss.expected}")
            console.print(f"    got      {miss.found[:3]}")

        history = read_records()
        if saved is not None:
            # Regenerated on every run, so the document cannot drift from what
            # the runs actually produced.
            write_report(history)

        previous = latest_comparable(history, record)
        if compare_previous and previous is not None:
            _print_comparison(compare(previous, record))
        elif compare_previous:
            console.print("[dim]No earlier run with a matching configuration to compare.[/dim]")

        if saved is not None:
            console.print(f"[dim]Recorded {saved}[/dim]")

    asyncio.run(run())


def _retriever(
    ctx: AppContext,
    space: EmbeddingSpace | None,
    *,
    tool: str,
    fusion: str | None,
    rerank: bool | None,
) -> Retriever:
    """Pick the surface to score.

    The MCP tools are constructed over the same search service the baseline
    uses, so a difference between them is the tool's document-type policy and
    nothing else.
    """
    if tool == "search":
        return SearchRetriever(
            ctx.search_service(space),
            fusion=cast("Fusion | None", fusion),
            rerank=rerank,
        )

    queries = QueryService(
        search=ctx.search_service(space),
        taxonomy=TaxonomyService(ctx.store, space or ctx.space),
    )
    if tool == "search_code":
        return ToolRetriever("search_code", lambda q, n: queries.search_code(q, limit=n))
    if tool == "find_guidance":
        return ToolRetriever("find_guidance", lambda q, n: queries.find_guidance(q, limit=n))

    console.print(f"[red]unknown --tool {tool!r}: search | search_code | find_guidance[/red]")
    raise typer.Exit(code=2)


def _print_comparison(comparison: EvalComparison) -> None:
    """Aggregate first, then what actually moved.

    A run that improves the average while breaking two cases is a result worth
    seeing, and the aggregate alone hides it.
    """

    def arrow(delta: float) -> str:
        colour = "green" if delta > 0 else "red" if delta < 0 else "dim"
        return f"[{colour}]{delta:+.3f}[/{colour}]"

    console.print(f"\n[bold]vs {comparison.before.label}[/bold] ({comparison.before.recorded_at})")
    console.print(f"  recall {arrow(comparison.recall_delta)}   MRR {arrow(comparison.mrr_delta)}")

    for movement in comparison.improved:
        console.print(f"  [green]better[/green] {movement}")
    for movement in comparison.regressed:
        console.print(f"  [red]worse [/red] {movement}")
    if not comparison.improved and not comparison.regressed:
        console.print("  [dim]no case changed rank[/dim]")


@app.command()
def serve(config: ConfigOption = None) -> None:
    """Run the MCP server so an agent can query the index mid-session.

    Speaks MCP over stdio: the client starts this process and talks to it down
    a pipe, so there is no port and nothing left running afterwards. stdout is
    the protocol channel and nothing else may touch it -- our console sink
    writes to stderr, which the client collects as logs, and the rolling JSONL
    file is unaffected either way.
    """
    from workspace_indexer.mcp import (
        EmptyIndexError,
        build_grounding_service,
        build_impact_service,
        build_mcp_server,
        build_query_service,
    )
    from workspace_indexer.mcp.server_factory import preflight

    ctx = _context(config, "serve")
    try:
        asyncio.run(preflight(ctx))
    except EmptyIndexError as exc:
        console.print(f"[red]{exc}[/red]")
        asyncio.run(ctx.close())
        raise typer.Exit(code=2) from exc

    try:
        build_mcp_server(
            build_query_service(ctx),
            build_impact_service(ctx),
            build_grounding_service(ctx),
        ).run()
    finally:
        asyncio.run(ctx.close())


@app.command()
def watch(config: ConfigOption = None) -> None:
    """Watch the configured roots and reindex as files change.

    A trigger, not a second indexing path: every change ends up in the same
    `index` run the CLI performs, scoped to the root that changed.
    """
    try:
        from workspace_indexer.watching import Watcher
    except ImportError as exc:  # pragma: no cover - exercised in test_cli
        # watchfiles is an optional extra, so that an install that only wants
        # `serve` does not pull a Rust binary it will never run.
        console.print(
            "[red]the watcher needs the `watch` extra: pip install 'workspace-indexer[watch]'[/red]"
        )
        raise typer.Exit(code=2) from exc

    ctx = _context(config, "watch")
    resolved = config or DEFAULT_CONFIG_PATH

    async def run() -> None:
        indexer = ctx.indexer()

        async def reindex(root: str | None) -> None:
            try:
                stats = await indexer.run(only_root=root)
            except Exception as exc:
                # A watcher that dies on one bad reindex is worse than one that
                # reports it and keeps watching.
                #
                # Logged as well as printed. The console is gone the moment the
                # terminal closes, and a watcher is meant to run unattended --
                # so a console-only report means the only record of a failure
                # is the one nobody was looking at.
                watch_log.error(
                    "watch.reindex_failed",
                    root=root,
                    error=str(exc),
                    error_type=type(exc).__name__,
                    exc_info=True,
                )
                console.print(f"[red]reindex of {root} failed: {exc}[/red]")
                return
            console.print(
                f"[dim]{root}: {stats.files_changed} changed, "
                f"{stats.chunks_upserted} written, {stats.chunks_deleted} removed[/dim]"
            )

        watcher = Watcher(
            ctx.config,
            reindex=reindex,
            config_path=resolved,
            reload_config=lambda: load_workspace_config(resolved),
        )
        for label, mode in watcher.plan().items():
            console.print(f"  {label}: [cyan]{mode.value}[/cyan]")
        console.print("[dim]Watching. Ctrl-C to stop.[/dim]")
        try:
            await watcher.run()
        except KeyboardInterrupt:
            watch_log.info("watch.stopped", reason="interrupted")
        except Exception as exc:
            # Anything raised outside the per-root reindex -- a bad path, a
            # lock, a failure inside the watcher itself. Previously this left
            # the JSONL ending at `watch.start` with nothing after it while the
            # process visibly died, so the only evidence was on screen.
            watch_log.error(
                "watch.crashed",
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=True,
            )
            console.print(f"[red]watch failed: {exc}[/red]")
            raise
        else:
            watch_log.info("watch.stopped", reason="completed")
        finally:
            await ctx.close()

    asyncio.run(run())


def _print_tool_calls(ctx: AppContext) -> None:
    """What agents actually asked, and how often they got nothing.

    The empty column is the point. Those calls are eval cases waiting to be
    written -- queries a real agent asked that this index could not answer,
    which is a better source than any dataset written from imagination.
    """
    stats = ctx.manifest.tool_call_stats()
    if not stats:
        return

    table = Table(title="mcp tool calls")
    for column in ("tool", "calls", "returned nothing"):
        table.add_column(column)
    for tool in sorted(stats):
        calls, empty = stats[tool]
        table.add_row(tool, f"{calls:,}", f"{empty:,} ({empty / calls:.0%})")
    console.print(table)

    unanswered = ctx.manifest.tool_calls(limit=3, disappointing_only=True)
    if unanswered:
        console.print("[dim]Recent calls worth turning into eval cases:[/dim]")
        for call in unanswered:
            reason = "nothing found" if not call.returned else f"{call.dropped_for_budget} dropped"
            console.print(f"[dim]  {call.tool}: {call.query[:60]!r} — {reason}[/dim]")


def _print_route_coverage(ctx: AppContext) -> None:
    """What the route graph found, and how precisely it could find it.

    Three numbers rather than one, because they fail independently. A workspace
    with declarations and no calls has an HTTP client this does not recognise --
    `graph.http_clients` is the fix, and measured on a real React workspace
    naming its wrapper took call sites from 6 to 71. A workspace with calls and
    no declarations is missing a server-side extractor. And a low exact share
    caps how precise any matching can ever be, whatever the resolver does.
    """
    coverage = ctx.manifest.route_coverage()
    if not coverage:
        return

    declarations = coverage.get("declaration", (0, 0))[1]
    resolved, calls = coverage.get("call", (0, 0))
    exact, total = ctx.manifest.route_precision()

    table = Table(title="route graph")
    for column in ("", "count"):
        table.add_column(column)
    table.add_row("endpoints declared", f"{declarations:,}")
    table.add_row("client call sites", f"{calls:,}")
    table.add_row("…with the whole URL", f"{exact:,}" + (f" of {total:,}" if total else ""))
    table.add_row("…matched to an endpoint", f"{resolved:,}")
    console.print(table)

    if declarations and not calls:
        console.print(
            "[yellow]Endpoints were found but no call sites.[/yellow] If this workspace "
            "wraps fetch, name the wrapper in `graph.http_clients` — otherwise every "
            "call through it is invisible."
        )
    if calls and not resolved:
        console.print(
            "[dim]No call has been matched to an endpoint yet: matching is not built "
            "(see #53). These are extracted and counted, nothing more.[/dim]"
        )


def _print_import_coverage(ctx: AppContext) -> None:
    """Which languages the dependency graph actually covers.

    Reported because an empty answer from the graph is ambiguous, and the two
    readings call for opposite next moves: "nothing imports this" is a fact,
    "we have no extractor for this language" is a gap. A language absent from
    this table has no edges because nobody looked.
    """
    coverage = ctx.manifest.import_coverage()
    covered = {lang: pair for lang, pair in coverage.items() if lang in IMPORT_LANGUAGES}
    if not covered:
        return

    table = Table(title="import graph coverage")
    for column in ("language", "files", "with imports"):
        table.add_column(column)
    for language in sorted(covered):
        with_edges, files = covered[language]
        table.add_row(language, f"{files:,}", f"{with_edges:,} ({with_edges / files:.0%})")
    console.print(table)

    origins = ctx.manifest.origin_coverage()
    if origins:
        resolved_table = Table(title="import resolution by origin")
        for column in (
            "language",
            "all edges",
            "non-framework",
            "first-party",
            "first-party resolved",
            "unclassified",
        ):
            resolved_table.add_column(column)
        for language in sorted(origins):
            # Not `coverage`: that name holds the import_coverage() dict this
            # function still needs after the loop.
            by_origin = origins[language]
            if not by_origin.total:
                continue
            resolved_table.add_row(
                language,
                f"{by_origin.total:,}",
                f"{by_origin.non_framework:,}",
                f"{by_origin.first_party:,}",
                _gate_cell(by_origin),
                f"{by_origin.unclassified:,}",
            )
        console.print(resolved_table)
        console.print(
            "[dim]Three nested denominators. Only first-party edges can resolve "
            "to a file here, so that column is the one that says whether the "
            "resolver works — the target is 100%. Framework and declared "
            "dependencies are correctly unresolved and are not missing "
            "dependencies.[/dim]"
        )
        unclassified = sum(c.unclassified for c in origins.values())
        if unclassified:
            console.print(
                f"[dim]{unclassified:,} edges matched no origin rule. Not an "
                "error \u2014 it is the queue for the next rule, and counting it "
                "is what keeps it from being guessed at.[/dim]"
            )
        unrecorded = sum(c.unrecorded for c in origins.values())
        if unrecorded:
            console.print(
                f"[yellow]{unrecorded:,} edges have no origin recorded[/yellow] "
                "[dim]\u2014 this index predates origin classification. Re-run "
                "`index` to populate it.[/dim]"
            )

    unsupported = sorted(set(coverage) - IMPORT_LANGUAGES)
    if unsupported:
        console.print(
            f"[dim]No import extractor for: {', '.join(unsupported)}. "
            "The graph has no edges for these, which is not the same as "
            "their having none.[/dim]"
        )


def _gate_cell(coverage: OriginCoverage) -> str:
    """The first-party resolution rate, or why there is not one.

    Below 100% is coloured because it is the one number here that means
    something is broken rather than merely unresolvable: a first-party edge
    names a file that is indexed, so failing to reach it is a defect. An em
    dash means there are no first-party edges to rate, which is neither a pass
    nor a failure.
    """
    rate = coverage.first_party_resolution
    if rate is None:
        return "[dim]\u2014[/dim]"
    cell = f"{coverage.first_party_resolved:,} ({rate:.0%})"
    return cell if rate >= 1.0 else f"[yellow]{cell}[/yellow]"


def _cost_cell(stats: RunStats) -> str:
    """`$0.0000` used to mean both "free" and "we have no idea"."""
    if stats.unpriced_requests:
        return (
            f"[yellow]unpriced[/yellow] ({stats.unpriced_requests} requests; "
            "set EMBEDDING_PRICE_PER_MTOK)"
        )
    if stats.cost_is_estimate:
        return f"~${stats.est_cost_usd:.4f} [dim](configured rate)[/dim]"
    return f"${stats.est_cost_usd:.4f}"


def _print_spend(ctx: AppContext) -> None:
    """Cumulative tokens, and the free allowance they have drawn down.

    Labelled as what it is. This counts what *this manifest* embedded, while
    the allowance belongs to the account and is spent by everything using the
    key -- so it is a floor on usage, never an authority.
    """
    total = ctx.manifest.total_tokens_embedded()
    if not total:
        return
    allowance = ctx.settings.embedding_free_tier_tokens
    if not allowance:
        console.print(f"[dim]{total:,} tokens embedded by this manifest.[/dim]")
        return
    used = min(total, allowance)
    console.print(
        f"[dim]{total:,} tokens embedded by this manifest — "
        f"{used / allowance:.1%} of a {allowance:,} free allowance. "
        "Counts this index only; the allowance is per account.[/dim]"
    )


def _preview(text: str, lines: int = 6) -> str:
    body = text.splitlines()
    if len(body) <= lines:
        return "\n".join(body)
    return "\n".join([*body[:lines], f"... ({len(body) - lines} more lines)"])


if __name__ == "__main__":
    app()
