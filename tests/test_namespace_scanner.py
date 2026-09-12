"""Which namespaces a C# file declares.

The other end of a `using` edge. Nothing here infers a namespace from a path:
namespace-to-directory correspondence measured 96-99% across two real corpora,
which is a convention holding most of the time, and resolving 4,710 edges on a
convention is how a graph comes to be confidently wrong about the rest.
"""

from __future__ import annotations

import pytest

from workspace_indexer.graph import NamespaceScanner


@pytest.fixture
def scanner() -> NamespaceScanner:
    return NamespaceScanner()


def test_a_file_scoped_namespace(scanner: NamespaceScanner) -> None:
    """The modern spelling, and the one both measured corpora use."""
    found = scanner.scan("namespace MyApp.Data;\n\nclass Repo {}\n", "csharp")
    assert [(d.symbol, d.kind, d.line) for d in found] == [("MyApp.Data", "namespace", 1)]


def test_a_block_namespace(scanner: NamespaceScanner) -> None:
    found = scanner.scan("namespace MyApp.Data\n{\n    class Repo {}\n}\n", "csharp")
    assert [d.symbol for d in found] == ["MyApp.Data"]


def test_a_nested_namespace_is_recorded_fully_qualified(scanner: NamespaceScanner) -> None:
    """`using Inner` reaches nothing -- the name a using has to spell is
    `Outer.Inner`. Recording the inner name alone would resolve edges that do
    not exist and fail to resolve the one that does, both silently."""
    found = scanner.scan("namespace Outer { namespace Inner { class X {} } }", "csharp")
    assert [d.symbol for d in found] == ["Outer", "Outer.Inner"]


def test_two_namespaces_in_one_file(scanner: NamespaceScanner) -> None:
    """Absent from both corpora measured, which is exactly why the schema
    allows it: the design that assumes one silently drops the second."""
    source = "namespace A\n{\n    class X {}\n}\n\nnamespace B\n{\n    class Y {}\n}\n"
    found = scanner.scan(source, "csharp")
    assert [(d.symbol, d.line) for d in found] == [("A", 1), ("B", 6)]


def test_a_file_declaring_nothing(scanner: NamespaceScanner) -> None:
    """Top-level statements and global-namespace files are ordinary. No
    declarations is an answer, not a failure."""
    assert scanner.scan("class Loose {}\n", "csharp") == []


def test_other_languages_are_not_scanned(scanner: NamespaceScanner) -> None:
    """Python and the JS family resolve by path and need nothing from here.
    A `namespace` block in TypeScript is a different thing with different
    rules, and claiming it would resolve TS edges against C# reasoning."""
    assert scanner.scan("namespace Foo { export const x = 1; }", "typescript") == []
    assert scanner.scan("import os\n", "python") == []


def test_empty_text_and_unparseable_source(scanner: NamespaceScanner) -> None:
    """A half-written file costs its declarations, never the run."""
    assert scanner.scan("", "csharp") == []
    assert isinstance(scanner.scan("namespace {{{ (", "csharp"), list)


def test_usings_above_the_declaration_do_not_shift_the_line(
    scanner: NamespaceScanner,
) -> None:
    """The line is where the reader has to look, so it is the declaration's
    own line rather than the file's first."""
    source = "using System;\nusing System.Linq;\n\nnamespace MyApp.Services;\n"
    assert [(d.symbol, d.line) for d in scanner.scan(source, "csharp")] == [("MyApp.Services", 4)]
