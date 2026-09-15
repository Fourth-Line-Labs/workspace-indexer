"""Which files a C# `using` reaches.

Rung 2 for C#, and it answers a different shape of question than
`ImportResolver` does. A Python import names one module and therefore one file;
a `using` names a namespace, which is declared across several. So this returns
a list, and the caller has to keep that distinction -- see the precision note
below.

Scoped to the using's own unit, exactly as path resolution is. Two repositories
in one workspace routinely declare the same namespace -- `Company.Shared` in a
service and in the library it was copied from -- and a resolver that guesses
between them is worse than one that declines.

**Precision.** A namespace edge is a module-level dependency, not a file-level
one. If file F declares namespace N and file D says `using N`, D depends on
something in N, not necessarily on F. Every target here is a candidate, and the
`resolved_by` column is what keeps that readable downstream. Narrowing to the
file declaring the referenced *type* is a later refinement this table's `kind`
column already allows.
"""

from __future__ import annotations

from collections.abc import Mapping

from workspace_indexer.graph.unit import unit_of


class NamespaceResolver:
    def __init__(self, declarations: Mapping[tuple[str, str], Mapping[str, list[str]]]) -> None:
        """`declarations` maps (root_label, unit) to symbol -> declaring paths.

        Loaded once for the run rather than queried per edge, mirroring
        `ImportResolver`'s file set: resolution runs over thousands of edges
        against a table that fits in memory many times over.
        """
        self._declarations = declarations

    def targets(self, symbol: str, *, root_label: str, from_path: str) -> list[str]:
        """Every file in the same unit declaring this namespace.

        Empty means the namespace is declared nowhere here -- which for a
        `using` is the ordinary case, since most of them name the framework or
        a package. It is not a failure, and the origin classifier is what tells
        the two apart.

        The declaring file itself is not excluded: a file may use a namespace
        it also declares, and dropping the self-edge would be a claim about
        C# rather than about this workspace.
        """
        unit = self._declarations.get((root_label, unit_of(from_path)))
        if not unit:
            return []
        return list(unit.get(symbol, ()))
