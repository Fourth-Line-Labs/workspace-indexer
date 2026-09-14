"""Turn a discovery candidate into a SourceFile.

This is the first point at which a file is opened. Discovery deliberately
stops short of it, so the manifest's mtime/size fast path can skip a file
without paying for a read.

Classification up to here was extension-only. This is where a claim like "this
.txt is text" gets checked against the actual bytes, and a file that does not
decode is downgraded to OPAQUE rather than indexed as mojibake.
"""

from __future__ import annotations

from collections.abc import Sequence
from fnmatch import fnmatch

from workspace_indexer.chunking.source_decoder import decode_source
from workspace_indexer.discovery.file_candidate import FileCandidate
from workspace_indexer.discovery.pdf_text import extract_pages
from workspace_indexer.models import FileKind, SourceFile, sha256_text
from workspace_indexer.obs.logging import get_logger
from workspace_indexer.secrets import SecretWithheldError, scan

log = get_logger("workspace_indexer.chunking.reader")


def _allowed(rel_path: str, patterns: Sequence[str] | None) -> bool:
    """Whether the content scan is waived for this path."""
    if not patterns:
        return False
    probe = "/" + rel_path.lstrip("/")
    return any(fnmatch(probe, p) or fnmatch(rel_path, p) for p in patterns)


# Kinds we never decode as UTF-8: reading them as text is meaningless and, for
# a large binary, wasteful. PDFs are binary too, but they carry a text layer
# that `extract_pages` can reach, so they are handled separately below rather
# than written off here.
_NEVER_TEXT = frozenset({FileKind.IMAGE, FileKind.OPAQUE, FileKind.PDF})

# A NUL byte in the first block is the same heuristic `git diff` uses to call a
# file binary, and it is far cheaper than attempting a full decode.
_SNIFF_BYTES = 8192


def read_source(
    candidate: FileCandidate, secret_allow: Sequence[str] | None = None
) -> SourceFile | None:
    """None when the file vanished or became unreadable between the walk and
    now — a normal race on a live workspace, not an error."""
    try:
        raw = candidate.abs_path.read_bytes()
    except OSError as exc:
        log.warning("read.failed", error=str(exc))
        return None

    kind = candidate.kind
    text: str | None = None
    pages: list[str] = []

    if kind is FileKind.PDF:
        # Extracted here, not in the chunker, so the secret scan below sees the
        # whole document. A chunker that opened the file for itself would be a
        # second path to the embedding API that the scanner never inspects.
        extracted = extract_pages(candidate.abs_path)
        if extracted is None:
            # No extra installed, unreadable, encrypted, or no text layer --
            # each logged distinctly by extract_pages. Recorded as opaque so it
            # is known and counted rather than mysteriously absent.
            kind = FileKind.OPAQUE
        else:
            pages = extracted
            text = "\n\n".join(pages)

    if kind not in _NEVER_TEXT:
        if b"\x00" in raw[:_SNIFF_BYTES]:
            log.debug("read.binary_downgrade", reason="nul_byte", declared=kind.value)
            kind = FileKind.OPAQUE
        else:
            try:
                # Shared with the graph backfill, which reads the same files by
                # a different route: see `decode_source` for why it is
                # `utf-8-sig` and why it decodes bytes rather than opening in
                # text mode.
                text = decode_source(raw)
            except UnicodeDecodeError as exc:
                log.debug("read.binary_downgrade", reason="undecodable", error=str(exc.reason))
                kind = FileKind.OPAQUE

    if text is not None and not _allowed(candidate.rel_path, secret_allow):
        findings = scan(text)
        if findings:
            # Skip the whole file rather than redacting the line. A file
            # holding a live credential is not something to ship to an
            # embedding API, and partial redaction invites being clever about
            # it. The value is never logged -- only where it was and what
            # shape it had.
            log.warning(
                "read.secret_withheld",
                findings=[str(f) for f in findings],
                detail="file not indexed; its contents would be sent to the embedding provider",
            )
            raise SecretWithheldError(candidate.rel_path, findings)

    return SourceFile(
        root_label=candidate.root_label,
        unit=candidate.unit,
        abs_path=candidate.abs_path,
        rel_path=candidate.rel_path,
        kind=kind,
        language=candidate.language if kind is candidate.kind else None,
        size=len(raw),
        mtime_ns=candidate.mtime_ns,
        # Hash the bytes, not the decoded text, so a file is identified the
        # same way whether or not it happens to be decodable.
        sha256=sha256_text(raw.decode("utf-8", "surrogateescape")),
        repo=candidate.repo,
        text=text,
        pages=pages,
    )
