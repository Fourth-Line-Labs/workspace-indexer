"""What one language's fixture tree is expected to produce."""

from __future__ import annotations

from pydantic import BaseModel


class LanguageBaseline(BaseModel):
    """Exact counts, not bands.

    The fixtures are authored, so the ground truth is known and a range would
    only hide a change. Every field here is a pure function of the file bytes --
    tree-sitter parses locally, resolution is a query against the manifest --
    so this is reproducible on a runner with no network and no API key, which
    is what makes it a gate rather than a report.
    """

    files_indexed: int
    files_with_imports: int
    import_edges: int
    namespace_declarations: int
    chunks: int

    # The origin buckets, which are the three nested denominators `status`
    # reports. `first_party_resolved` against `first_party` is the gate: only a
    # first-party edge can reach a file here, so only there is a shortfall a
    # defect rather than a package reference.
    first_party: int
    first_party_resolved: int
    declared_dependency: int
    framework: int
    unclassified: int

    # Edges no resolver at this rung can answer -- a C# `using static` names a
    # type. Counted separately because it is terminal: it is neither resolved
    # nor waiting for something to land.
    declined: int
