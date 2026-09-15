"""Flagging results whose source has changed since it was indexed.

The alternative is worse than it sounds: silently re-reading the file would
show its current text at the line numbers of the version that actually matched
the query.

The check is "is this chunk's exact text still in the file", not "does the file
hash match". A file hash and a chunk hash never compare directly, and a change
elsewhere in the file does not make *this* hit wrong.

"Still in the file" has to mean the same text the chunk was built from, which
for a PDF is the extracted layer rather than the bytes. Reading a PDF as UTF-8
yields replacement characters, so every PDF hit came back flagged stale --
found by searching a real one, not by reading this.
"""

from __future__ import annotations

from pathlib import Path

from workspace_indexer.chunking.source_decoder import decode_source
from workspace_indexer.discovery.pdf_text import extract_pages
from workspace_indexer.models import FileKind, SearchHit
from workspace_indexer.obs.logging import get_logger, log_once

log = get_logger("workspace_indexer.search.staleness")


def mark_stale(hits: list[SearchHit]) -> list[SearchHit]:
    """One read per distinct file, not per hit: several chunks of one file
    routinely appear in the same result set."""
    cache: dict[str, str | None] = {}
    marked: list[SearchHit] = []

    for hit in hits:
        if not hit.abs_path or not hit.source_text:
            marked.append(hit)
            continue
        if hit.abs_path not in cache:
            cache[hit.abs_path] = _read(hit.abs_path, hit.kind)
        text = cache[hit.abs_path]
        if text is None and Path(hit.abs_path).exists():
            # Present but unreadable in the form the chunk was built from --
            # a PDF with the extra uninstalled, say. We cannot tell whether it
            # changed, and "stale" is a claim rather than an absence of one.
            log_once(
                log,
                "staleness:unjudgeable",
                "search.staleness_unknown",
                path=hit.rel_path,
                detail="cannot re-read this file in its indexed form; not flagging it either way",
            )
            marked.append(hit)
            continue
        # A file that has vanished is stale in the sense that matters: the hit
        # no longer points anywhere real.
        stale = text is None or hit.source_text not in text
        marked.append(hit.model_copy(update={"stale": True}) if stale else hit)

    if any(h.stale for h in marked):
        log.info("search.stale_hits", count=sum(1 for h in marked if h.stale))
    return marked


def _read(path: str, kind: FileKind) -> str | None:
    """The file as the chunker saw it, which is not always its bytes.

    Re-extracting a PDF on every search is real work, and the reason
    `search.check_staleness` exists as a setting. One extraction per file per
    search, not per hit.
    """
    if kind is FileKind.PDF:
        pages = extract_pages(Path(path))
        return None if pages is None else "\n\n".join(pages)
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        log.debug("staleness.unreadable", path=path, error=str(exc))
        return None
    try:
        # The same decoder the indexer used on the way in, so this compares
        # against the text that was actually chunked.
        #
        # The two halves of the old `read_text` were not equally harmless. A
        # kept byte-order mark only ever affected the first chunk of a file,
        # and substring membership survived it. Translating CRLF to LF did not
        # survive anything: a chunk's `source_text` keeps the line endings it
        # was chunked from, so every *multi-line* chunk of a CRLF file failed
        # the substring test and was flagged stale on every search. Measured on
        # a CRLF C# file: stale before, not stale after. Visual Studio writes
        # CRLF as reliably as it writes the mark, so that was most of a C#
        # corpus reporting itself as changed since indexing.
        return decode_source(raw)
    except UnicodeDecodeError:
        # Staleness is a hint, not a gate. A file that cannot be decoded
        # strictly still gets an answer here, which is the one place that
        # differs from indexing -- indexing declines the file, and this only
        # declines to be certain about it.
        return raw.decode("utf-8", errors="replace")
