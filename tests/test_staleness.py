"""Flagging results whose source moved on.

Real files, because the whole question is what is on disk right now.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace_indexer.models import FileKind, SearchHit
from workspace_indexer.search.staleness import mark_stale

BODY = "def login():\n    return check_token()"


def _hit(
    path: Path,
    source_text: str = BODY,
    chunk_id: str = "id-1",
    kind: FileKind = FileKind.CODE,
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        score=1.0,
        rel_path=path.name,
        abs_path=str(path),
        root_label="root",
        source_text=source_text,
        content_sha="sha",
        kind=kind,
    )


def test_unchanged_file_is_not_stale(tmp_path: Path) -> None:
    path = tmp_path / "a.py"
    path.write_text(f"# header\n{BODY}\n", encoding="utf-8")
    assert mark_stale([_hit(path)])[0].stale is False


def test_edited_chunk_is_stale(tmp_path: Path) -> None:
    """Showing the current text at the old line numbers would display
    something that never matched the query."""
    path = tmp_path / "a.py"
    path.write_text("def login():\n    return check_session()\n", encoding="utf-8")
    assert mark_stale([_hit(path)])[0].stale is True


def test_change_elsewhere_in_the_file_does_not_make_this_hit_stale(tmp_path: Path) -> None:
    """A file hash would flag this; the hit itself is still perfectly accurate."""
    path = tmp_path / "a.py"
    path.write_text(f"# a brand new comment\n{BODY}\n\ndef other(): pass\n", encoding="utf-8")
    assert mark_stale([_hit(path)])[0].stale is False


def test_deleted_file_is_stale(tmp_path: Path) -> None:
    hit = _hit(tmp_path / "gone.py")
    assert mark_stale([hit])[0].stale is True


def test_hits_without_a_path_are_left_alone(tmp_path: Path) -> None:
    """A point written before abs_path was stored should still render."""
    hit = SearchHit(chunk_id="id", score=1.0, rel_path="a.py", root_label="r", source_text=BODY)
    assert mark_stale([hit])[0].stale is False


def test_one_read_per_file_not_per_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Several chunks of one file routinely appear in the same result set."""
    path = tmp_path / "a.py"
    path.write_text(f"{BODY}\n\ndef other():\n    pass\n", encoding="utf-8")

    reads: list[str] = []
    original = Path.read_bytes

    def counting(self: Path) -> bytes:
        reads.append(str(self))
        return original(self)

    # `read_bytes`, because the file is decoded by the same function the
    # indexer decodes with rather than opened in text mode -- a BOM kept or a
    # CRLF translated here would compare against text that was never chunked.
    monkeypatch.setattr(Path, "read_bytes", counting)
    hits = [
        _hit(path, BODY, "id-1"),
        _hit(path, "def other():\n    pass", "id-2"),
    ]
    marked = mark_stale(hits)
    assert [h.stale for h in marked] == [False, False]
    assert reads.count(str(path)) == 1


def test_empty_input(tmp_path: Path) -> None:
    assert mark_stale([]) == []


def test_a_pdf_hit_is_not_stale_when_the_document_is_unchanged(tmp_path: Path) -> None:
    """The bug this guards: `_read` decoded raw bytes as UTF-8, so a PDF came
    back as replacement characters and *every* hit was flagged stale.

    Wrong in the direction that trains an agent to ignore the flag -- and the
    flag is what stops it editing from text that no longer exists.
    """
    pymupdf = pytest.importorskip("pymupdf", reason="needs `poetry install --extras pdf`")
    path = tmp_path / "doc.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(36, 36, 560, 780),
        "Rolling back a release requires the rollback script and the previous tag.",
        fontsize=9,
    )
    document.save(path)
    document.close()

    hit = _hit(path, "Rolling back a release requires the rollback script", kind=FileKind.PDF)
    assert mark_stale([hit])[0].stale is False


def test_a_pdf_hit_is_stale_once_its_own_text_changes(tmp_path: Path) -> None:
    pymupdf = pytest.importorskip("pymupdf", reason="needs `poetry install --extras pdf`")
    path = tmp_path / "doc.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(
        pymupdf.Rect(36, 36, 560, 780), "This section has been rewritten entirely.", fontsize=9
    )
    document.save(path)
    document.close()

    hit = _hit(path, "Rolling back a release requires the rollback script", kind=FileKind.PDF)
    assert mark_stale([hit])[0].stale is True
