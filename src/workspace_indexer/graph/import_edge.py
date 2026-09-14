"""One import statement, as written, before anything tries to resolve it."""

from __future__ import annotations

from pydantic import BaseModel


class ImportEdge(BaseModel):
    # Exactly what the source said: "workspace_indexer.models", "@/hooks/x",
    # "System.Text", "./sibling". Unresolved on purpose -- turning that into a
    # file is per-language work, and this rung exists to find out whether it is
    # worth doing before doing it.
    module: str
    # import | from | reexport | using | using_static | global_using |
    # global_using_static. Kept because they behave differently later: a
    # reexport makes a barrel file a pass-through rather than a destination,
    # which is the thing that makes Node resolution hard, and the C# forms
    # decide whether an edge is resolvable at all -- a static using names a
    # type, which a namespace table cannot answer. `import_scanner.KINDS` is
    # the list; a test holds this comment to it.
    kind: str
    # "./x", "../x" and Python's ".rel" name a neighbour rather than a package.
    # These are the ones a within-repo resolver can settle almost for free, so
    # the ratio of relative to absolute is itself a useful measurement.
    is_relative: bool
    line: int
