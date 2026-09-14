"""What each file imports, taken from the source and not resolved.

Rung 1 of the dependency graph. The value being tested is narrow on purpose:
the module string exactly as written, with no attempt to turn it into a file.
"""

from __future__ import annotations

import sys

import pytest

from workspace_indexer.graph import SUPPORTED, ImportScanner
from workspace_indexer.graph.import_scanner import KINDS
from workspace_indexer.graph.parse import parse
from workspace_indexer.obs.logging import get_logger


@pytest.fixture
def scanner() -> ImportScanner:
    return ImportScanner()


def _modules(scanner: ImportScanner, code: str, language: str) -> list[str]:
    return [e.module for e in scanner.scan(code, language)]


# --- python --------------------------------------------------------------


def test_python_absolute_and_from_imports(scanner: ImportScanner) -> None:
    code = "from workspace_indexer.models import Chunk\nimport os\n"
    assert _modules(scanner, code, "python") == ["workspace_indexer.models", "os"]


def test_one_statement_can_be_two_edges(scanner: ImportScanner) -> None:
    """`import os, sys.path` imports two modules."""
    assert _modules(scanner, "import os, sys.path\n", "python") == ["os", "sys.path"]


def test_python_alias_records_the_target_not_the_alias(scanner: ImportScanner) -> None:
    """`numpy` is the edge; `np` is a local name nobody else can follow."""
    assert _modules(scanner, "import numpy as np\n", "python") == ["numpy"]


def test_python_relative_imports_are_flagged(scanner: ImportScanner) -> None:
    """Relative imports name a neighbour rather than a package, so a
    within-repo resolver can settle them almost for free -- which makes their
    share a useful measurement in itself."""
    edges = scanner.scan("from .rel import thing\nfrom . import sib\n", "python")
    assert [e.module for e in edges] == [".rel", "."]
    assert all(e.is_relative for e in edges)


def test_an_absolute_import_is_not_relative(scanner: ImportScanner) -> None:
    edges = scanner.scan("import os\n", "python")
    assert edges[0].is_relative is False


# --- javascript family ---------------------------------------------------


def test_the_module_specifier_is_extracted_not_the_statement(
    scanner: ImportScanner,
) -> None:
    """The reason this does its own parse: `process(imports=True)` reports the
    whole statement text, so the specifier still has to be dug out."""
    code = "import { A, B } from '@/hooks/useThing';\n"
    assert _modules(scanner, code, "typescript") == ["@/hooks/useThing"]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("import React from 'react';", "react"),
        ("import * as fs from 'node:fs';", "node:fs"),
        ("import './side-effect';", "./side-effect"),
        ("import type { X } from './types';", "./types"),
    ],
)
def test_every_import_form(scanner: ImportScanner, code: str, expected: str) -> None:
    assert _modules(scanner, code, "tsx") == [expected]


def test_a_reexport_is_an_edge_and_is_labelled_as_one(scanner: ImportScanner) -> None:
    """Barrel files are pass-throughs rather than destinations, which is much
    of what makes Node resolution hard. Losing the distinction now would mean
    rediscovering it later."""
    edges = scanner.scan("export * from './barrel';\n", "typescript")
    assert [(e.module, e.kind) for e in edges] == [("./barrel", "reexport")]


def test_a_plain_export_is_not_an_edge(scanner: ImportScanner) -> None:
    """`export const x = 1` is the other end of an edge, not one."""
    assert _modules(scanner, "export const x = 1;\n", "typescript") == []


# --- c# ------------------------------------------------------------------


def test_csharp_using_directives(scanner: ImportScanner) -> None:
    """The library extracts nothing at all for C#, which is the other reason
    this is our own pass."""
    code = "using System;\nusing MyApp.Data.Core;\n"
    assert _modules(scanner, code, "csharp") == ["System", "MyApp.Data.Core"]


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("using static System.Math;", "System.Math"),
        ("global using X.Y;", "X.Y"),
        # The alias is a local name; the namespace is the edge.
        ("using Alias = My.Long.Name;", "My.Long.Name"),
    ],
)
def test_every_using_form(scanner: ImportScanner, code: str, expected: str) -> None:
    assert _modules(scanner, code, "csharp") == [expected]


# --- boundaries ----------------------------------------------------------


def test_line_numbers_are_one_based(scanner: ImportScanner) -> None:
    edges = scanner.scan("\n\nimport os\n", "python")
    assert edges[0].line == 3


def test_unsupported_languages_do_no_work(scanner: ImportScanner) -> None:
    """A second parse for a language with no resolver planned is a cost on
    every file of that language for nothing."""
    assert scanner.scan("@import 'x';", "scss") == []
    assert scanner.scan("#include <stdio.h>", "c") == []


@pytest.mark.parametrize("language", sorted(SUPPORTED))
def test_every_supported_language_parses(scanner: ImportScanner, language: str) -> None:
    """A language in the list with no working grammar would report zero edges
    forever and look like a codebase that imports nothing."""
    code = "using A.B;" if language == "csharp" else "import os"
    if language in {"typescript", "tsx", "javascript"}:
        code = "import { a } from 'b';"
    assert _modules(scanner, code, language)


def test_empty_and_broken_input_do_not_raise(scanner: ImportScanner) -> None:
    assert scanner.scan("", "python") == []
    assert isinstance(scanner.scan("from . import", "python"), list)


def test_a_tree_too_deep_to_walk_costs_the_edges_not_the_run() -> None:
    """Same exposure as the namespace scanner, and the same contract: a stack
    exhausted by deeply nested source must cost this file's edges rather than
    the run that is walking thousands of files."""
    source = "using System;\n" + "class C { " * 400 + "}" * 400
    # Parsed before the limit drops, and handed in, so the only thing that can
    # exhaust the stack is the walk. `parse` swallows `RecursionError` along
    # with everything else, so parsing under the lowered limit would make this
    # pass through the wrong guard.
    tree = parse(source, "csharp", log=get_logger("tests.import_scanner"))
    assert tree is not None

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(120)
    try:
        assert ImportScanner().scan(source, "csharp", tree) == []
    finally:
        sys.setrecursionlimit(limit)


def test_the_csharp_directive_form_is_kept() -> None:
    """All four forms were recorded as `using`, which reads as one thing and
    is three: `using static` names a type rather than a namespace, and a
    `global using` applies to the whole compilation unit rather than to the
    file declaring it. Flattening them hides both facts in the data."""
    source = (
        "global using System.Linq;\n"
        "using static MyApp.Helpers;\n"
        "global using static MyApp.Both;\n"
        "using MyApp.Data;\n"
        "using Alias = MyApp.Other;\n"
    )
    found = [(e.kind, e.module) for e in ImportScanner().scan(source, "csharp")]
    assert found == [
        ("global_using", "System.Linq"),
        ("using_static", "MyApp.Helpers"),
        # Both markers, not the first one checked: `global using static` is
        # legal and carries both, and a precedence rule would record one fact
        # and silently drop the other -- the dropped one being what decides
        # whether resolution declines the edge.
        ("global_using_static", "MyApp.Both"),
        ("using", "MyApp.Data"),
        ("using", "MyApp.Other"),
    ]


def test_every_kind_the_scanner_emits_is_a_declared_one() -> None:
    """`kind` is a contract across three modules -- this scanner, the decline
    set in the pipeline, and `ImportEdge`'s own documentation. A new form added
    here without the others learning about it would be resolved as though it
    were a plain import, which is how `using static` came to be resolved as a
    namespace."""
    sources = {
        "python": "import os\nfrom pathlib import Path\n",
        "typescript": "import { a } from './a';\nexport { b } from './b';\n",
        "csharp": (
            "global using System.Linq;\n"
            "using static MyApp.Helpers;\n"
            "global using static MyApp.Both;\n"
            "using MyApp.Data;\n"
        ),
    }
    scanner = ImportScanner()
    emitted = {
        edge.kind for language, source in sources.items() for edge in scanner.scan(source, language)
    }
    assert emitted <= KINDS, f"undeclared kinds: {sorted(emitted - KINDS)}"
    # And the declaration is not a superset nobody produces: every C# form is
    # exercised above, so a stale entry shows up as an unreached one.
    assert {"using", "using_static", "global_using", "global_using_static"} <= emitted
