"""Import resolution counted against three nested denominators.

One percentage cannot answer "is the resolver working", because most of what
it divides by was never resolvable. `System.Text` and `MyApp.Data` both arrive
as an edge with no target and only one of them is a defect, so a single figure
reports a resolver as broken when it is merely pointed at a lot of packages.

Three denominators, narrowing:

- `total` -- every edge extracted.
- `non_framework` -- drops the language's own standard library.
- `first_party` -- code in this workspace, and the only one where an
  unresolved edge is a bug. Resolution here is a lookup against symbols
  extracted from the same workspace, so the target is 100% and anything less
  is something to chase rather than a limit to accept.

`unrecorded` is deliberately not folded into `unclassified`. The first means
classification has never run over these rows -- an index predating it -- and
the second means it ran and no rule claimed them. Reporting them as one number
is how "we never looked" comes to read as "we looked and found nothing", which
is the failure this project keeps having to design against.
"""

from __future__ import annotations

from pydantic import BaseModel


class OriginCoverage(BaseModel):
    total: int
    first_party: int
    # Of `first_party`, how many reached a file. The gate.
    first_party_resolved: int
    declared_dependency: int
    framework: int
    unclassified: int
    # origin IS NULL: classification has not run over these rows.
    unrecorded: int

    @property
    def non_framework(self) -> int:
        """Everything that is not the standard library.

        Derived by subtraction rather than summed from the other buckets, so an
        origin added later is counted here without this needing to learn its
        name.
        """
        return self.total - self.framework

    @property
    def first_party_resolution_percent(self) -> int | None:
        """The gate in whole percentage points, or None when there is no gate.

        Floored, never rounded, and derived from the counts rather than the
        float: 1,234 of 1,240 edges is 99%, not 100%. This is the one column
        where "100%" has to mean *complete*, so it is returned only when every
        first-party edge resolved -- reached by comparing the counts, so no
        rounding rule can produce it by accident.

        Returned as a number rather than formatted so a caller cannot round a
        second time and end up disagreeing with itself, which is exactly how a
        `.0%` format came to print a coloured "100%".
        """
        if not self.first_party:
            return None
        if self.first_party_resolved >= self.first_party:
            return 100
        return min(99, self.first_party_resolved * 100 // self.first_party)

    @property
    def first_party_resolution(self) -> float | None:
        """The gate, or None when there is nothing to gate on.

        None rather than 0.0 or 1.0: a language with no first-party edges has
        not passed and has not failed, and picking either would make an empty
        corpus look like a result.
        """
        if not self.first_party:
            return None
        return self.first_party_resolved / self.first_party
