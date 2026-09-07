"""The standard-library membership test, per language.

The cases that matter are the negative ones: anything this wrongly claims is
an edge reported as correctly-unresolvable when nobody actually checked.
"""

from __future__ import annotations

import pytest

from workspace_indexer.graph.framework_modules import is_framework_module


@pytest.mark.parametrize("module", ["os", "sys", "json", "dataclasses"])
def test_python_stdlib_top_level(module: str) -> None:
    assert is_framework_module(module, "python")


def test_python_stdlib_submodule_counts_as_stdlib() -> None:
    # The list holds top-level names, so a dotted path has to be reduced.
    assert is_framework_module("os.path", "python")
    assert is_framework_module("concurrent.futures", "python")


def test_python_third_party_is_not_framework() -> None:
    assert not is_framework_module("pydantic", "python")
    assert not is_framework_module("workspace_indexer.models", "python")


def test_python_relative_import_is_not_framework() -> None:
    # Relative specifiers are claimed as first-party before this is consulted,
    # but it must not claim them if it ever is.
    assert not is_framework_module(".helpers", "python")
    assert not is_framework_module("..db.models", "python")


def test_csharp_bcl_root_and_namespaces() -> None:
    assert is_framework_module("System", "csharp")
    assert is_framework_module("System.Text.Json", "csharp")


def test_csharp_system_prefix_requires_a_separator() -> None:
    # `SystemTools` is somebody's own namespace, not the BCL.
    assert not is_framework_module("SystemTools", "csharp")
    assert not is_framework_module("SystemsBiology.Core", "csharp")


def test_csharp_microsoft_is_deliberately_not_framework() -> None:
    # Ambiguous by nature: some is BCL, some ships as a package, much arrives
    # through the shared framework. Left unclassified rather than guessed.
    assert not is_framework_module("Microsoft.Extensions.AI", "csharp")
    assert not is_framework_module("Microsoft.AspNetCore.Mvc", "csharp")


@pytest.mark.parametrize("language", ["javascript", "typescript", "tsx"])
def test_node_builtins_across_the_js_family(language: str) -> None:
    assert is_framework_module("fs", language)
    assert is_framework_module("path", language)


def test_node_prefixed_form_is_the_same_module() -> None:
    assert is_framework_module("node:fs", "typescript")
    assert is_framework_module("node:worker_threads", "typescript")


def test_node_builtin_subpath() -> None:
    assert is_framework_module("fs/promises", "typescript")


def test_node_packages_are_not_builtins() -> None:
    assert not is_framework_module("react", "tsx")
    assert not is_framework_module("@scope/pkg", "typescript")
    # A package whose name merely starts with a builtin's.
    assert not is_framework_module("pathfinder", "typescript")


def test_an_unknown_language_claims_nothing() -> None:
    # A language with no rules must report nothing rather than default to
    # framework, so the gap shows up as unclassified.
    assert not is_framework_module("System", "rust")
    assert not is_framework_module("fmt", "go")


def test_an_empty_module_claims_nothing() -> None:
    assert not is_framework_module("", "python")
