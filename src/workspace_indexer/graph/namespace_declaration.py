"""A namespace a file declares."""

from __future__ import annotations

from pydantic import BaseModel


class NamespaceDeclaration(BaseModel):
    """What `namespace X.Y` says, as the source wrote it.

    The other end of a `using` edge. Kept as its own record rather than folded
    into the file row because a file may declare more than one -- rare, and
    absent from both corpora measured, but the schema that assumes otherwise
    is the one that silently drops the second.
    """

    # Fully qualified: a namespace nested inside another is recorded as the
    # joined path, because that is the name a `using` has to spell.
    symbol: str
    # `namespace` today. The column exists so type declarations can share the
    # table later without a migration -- narrowing a using to the one file
    # declaring the type it references is the follow-on this design allows.
    kind: str = "namespace"
    line: int
