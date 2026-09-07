"""Where an import points, and therefore whether failing to resolve it is a bug.

Resolution coverage on its own cannot tell a resolver defect from a package
reference: `using System.Text` and `using MyApp.Data` both arrive as an edge
with no target, and only one of them is a problem. This is the axis that says
which.

That distinction is what makes the number actionable. Measured against every
edge, C# reports 0% and reads as broken; measured against first-party edges
alone, the same resolver reports the figure that says whether it works.
"""

from __future__ import annotations

from enum import StrEnum


class ImportOrigin(StrEnum):
    # Code in this workspace. The only bucket where an unresolved edge is a
    # defect: the target is indexed, so resolution is a lookup that either
    # succeeds or has a bug behind it.
    FIRST_PARTY = "first_party"
    # A dependency the project declares -- a NuGet PackageReference, a
    # package.json entry, a pyproject requirement. Never resolvable to a file
    # in this workspace, and correctly so.
    DECLARED_DEPENDENCY = "declared_dependency"
    # The language's own standard library or shipped framework.
    FRAMEWORK = "framework"
    # No rule claimed it. A category rather than a fold into FRAMEWORK,
    # because guessing here is how a resolver comes to report coverage it has
    # not earned -- and because the size of this bucket is the work queue for
    # the next rule.
    UNCLASSIFIED = "unclassified"
