"""Which files a C# `using` reaches.

A list rather than a path, because a namespace is declared across several
files. The scope is the using's own repository, for the same reason path
resolution is scoped: two repositories in one workspace routinely declare the
same namespace, and guessing between them is worse than declining.
"""

from __future__ import annotations

import pytest

from workspace_indexer.graph import NamespaceResolver


@pytest.fixture
def resolver() -> NamespaceResolver:
    return NamespaceResolver(
        {
            ("workspace", "service"): {
                "MyApp.Data": ["service/Data/Repo.cs", "service/Data/Context.cs"],
                "MyApp.Web": ["service/Web/Startup.cs"],
            },
            # A second repository declaring the same namespace. This is the
            # case the unit scoping exists for.
            ("workspace", "library"): {"MyApp.Data": ["library/Data/Repo.cs"]},
            ("other_root", "service"): {"MyApp.Data": ["service/Data/Other.cs"]},
        }
    )


def test_every_file_declaring_the_namespace(resolver: NamespaceResolver) -> None:
    assert resolver.targets(
        "MyApp.Data", root_label="workspace", from_path="service/Web/Startup.cs"
    ) == ["service/Data/Repo.cs", "service/Data/Context.cs"]


def test_a_namespace_declared_in_another_repository_is_not_reached(
    resolver: NamespaceResolver,
) -> None:
    """The cross-unit isolation that makes an answer worth having. `library`
    declares `MyApp.Data` too, and a using in `service` means the one in
    `service`."""
    targets = resolver.targets(
        "MyApp.Data", root_label="workspace", from_path="service/Web/Startup.cs"
    )
    assert all(t.startswith("service/") for t in targets)


def test_a_namespace_declared_under_another_root_is_not_reached(
    resolver: NamespaceResolver,
) -> None:
    """Two roots can hold a repository of the same name. The key is the pair,
    so `other_root` cannot answer for `workspace`."""
    assert (
        resolver.targets(
            "MyApp.Nowhere", root_label="workspace", from_path="service/Web/Startup.cs"
        )
        == []
    )
    assert resolver.targets(
        "MyApp.Data", root_label="other_root", from_path="service/Web/Startup.cs"
    ) == ["service/Data/Other.cs"]


def test_a_framework_or_package_namespace_reaches_nothing(resolver: NamespaceResolver) -> None:
    """`using System.Text` is the ordinary case, not a failure: most usings
    name the framework or a package, and the origin classifier is what tells
    that apart from a resolver defect."""
    assert (
        resolver.targets("System.Text", root_label="workspace", from_path="service/Web/S.cs") == []
    )


def test_a_unit_with_no_declarations_at_all(resolver: NamespaceResolver) -> None:
    assert resolver.targets("MyApp.Data", root_label="workspace", from_path="scripts/Tool.cs") == []


def test_a_file_using_a_namespace_it_declares_itself(resolver: NamespaceResolver) -> None:
    """Not excluded. A file may use a namespace it also declares, and dropping
    the self-edge would be a claim about C# rather than about this workspace."""
    assert "service/Data/Repo.cs" in resolver.targets(
        "MyApp.Data", root_label="workspace", from_path="service/Data/Repo.cs"
    )
