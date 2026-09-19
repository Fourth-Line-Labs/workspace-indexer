"""One hit, shaped for an agent rather than a terminal."""

from __future__ import annotations

from pydantic import BaseModel


class SearchResult(BaseModel):
    # `path:start-end`, so the next action is `Read(file, offset, limit)` with
    # nothing to work out. The single most valuable field here.
    location: str
    rel_path: str
    # The file to open, absolute. `location` and `rel_path` identify the chunk
    # within the index; this is where it lives on the caller's disk, and when a
    # query is scoped to a worktree it is that worktree's copy rather than the
    # main checkout's. Carried as its own field rather than by rewriting
    # rel_path, which is the index's name for the file and has to keep meaning
    # the same thing whoever is asking.
    abs_path: str = ""
    start_line: int
    end_line: int
    doc_type: str
    # Class/method trail or heading path. Tells the agent what it is looking at
    # before it reads a line of the body.
    symbol_path: str | None = None
    language: str | None = None
    repo: str | None = None
    text: str = ""
    # Set when `text` was cut to fit the budget, so the agent knows to read the
    # file rather than assuming it has the whole chunk.
    text_truncated: bool = False
    # Set when the body was never included, because the caller asked for
    # locations only. Distinct from `text_truncated` and carried as its own
    # field because an empty `text` is otherwise ambiguous: a chunk that is
    # genuinely blank and a chunk whose body was withheld look identical, and
    # an agent that reads the second as the first concludes the file is empty.
    text_omitted: bool = False
    # The file changed after it was indexed: this text matched the query, but
    # it is not what is on disk now. Never silently hidden -- an agent editing
    # from stale text writes a patch that will not apply.
    stale: bool = False
    score: float = 0.0
