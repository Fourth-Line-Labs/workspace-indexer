"""Which repository a file belongs to, and saying so when none does.

Real repositories throughout: the question is what git reports about a path,
and a fixture that asserts our own assumption would have hidden the bug these
tests exist for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog.testing

from tests.conftest import git_init, write
from workspace_indexer.config import WorkspaceConfig
from workspace_indexer.discovery import Walker


def config(root: Path, **index: Any) -> WorkspaceConfig:
    payload: dict[str, Any] = {
        "workspace": {"name": "w", "roots": [{"path": str(root), "recurse_into_children": True}]},
    }
    if index:
        payload["index"] = index
    return WorkspaceConfig.model_validate(payload)


def repos_by_file(root: Path, **index: Any) -> dict[str, str | None]:
    return {
        c.rel_path: (c.repo.name if c.repo else None) for c in Walker(config(root, **index)).walk()
    }


def test_a_repository_nested_below_the_unit_is_still_attributed(tmp_path: Path) -> None:
    """The bug this was written for.

    Repo metadata was read from the *unit* -- the first path segment of a root
    -- which stops being the repository the moment a workspace nests one. Every
    file under it was then attributed to no repository: no `repo_name`, no
    branch, no head sha, and the `repo` search filter could not find them.
    Measured on the real workspace when this was found: 309 files.
    """
    nested = tmp_path / "holder" / "product"
    nested.mkdir(parents=True)
    write(nested / "app.py", "x = 1\n")
    git_init(nested)

    assert repos_by_file(tmp_path)["holder/product/app.py"] == "product"


def test_a_file_genuinely_outside_any_repository_has_none(tmp_path: Path) -> None:
    """The other half: attributing everything would be as wrong as nothing."""
    holder = tmp_path / "holder"
    (holder / "product").mkdir(parents=True)
    write(holder / "notes.md", "loose\n")
    write(holder / "product" / "app.py", "x = 1\n")
    git_init(holder / "product")

    found = repos_by_file(tmp_path)

    assert found["holder/product/app.py"] == "product"
    assert found["holder/notes.md"] is None


def test_two_repositories_under_one_unit_are_told_apart(tmp_path: Path) -> None:
    """A unit-level answer could only ever name one of them."""
    for name in ("alpha", "beta"):
        path = tmp_path / "holder" / name
        path.mkdir(parents=True)
        write(path / "f.py", "x = 1\n")
        git_init(path)

    found = repos_by_file(tmp_path)

    assert found["holder/alpha/f.py"] == "alpha"
    assert found["holder/beta/f.py"] == "beta"


# --- issue #35: saying when nothing could filter ------------------------


def test_a_root_with_no_repository_and_no_excludes_is_reported(tmp_path: Path) -> None:
    """`respect_gitignore` only works inside a repository.

    Point a root at a vendored SDK and nothing filters it: an `obj/` directory
    contributes thousands of generated files that are real, parseable source
    and compete with the code on every query. The only symptom is worse results.
    """
    sdk = tmp_path / "vendor-sdk"
    sdk.mkdir()
    write(sdk / "Thing.cs", "class Thing {}\n")
    write(sdk / "obj" / "Thing.g.cs", "// generated\n")

    with structlog.testing.capture_logs() as logs:
        list(Walker(config(tmp_path)).walk())

    warned = [e for e in logs if e["event"] == "root.unfiltered"]
    assert len(warned) == 1
    assert warned[0]["log_level"] == "warning"
    assert "vendor-sdk" in warned[0]["units"]
    # The count is what makes it actionable rather than generic.
    assert warned[0]["files"] == 2


def test_a_repository_produces_no_warning(tmp_path: Path) -> None:
    """A repo has a .gitignore to respect, whether or not it uses one."""
    repo = tmp_path / "product"
    repo.mkdir()
    write(repo / "app.py", "x = 1\n")
    git_init(repo)

    with structlog.testing.capture_logs() as logs:
        list(Walker(config(tmp_path)).walk())

    assert not [e for e in logs if e["event"] == "root.unfiltered"]


def test_configured_excludes_silence_it(tmp_path: Path) -> None:
    """Someone who has written exclude patterns has already met this problem.

    A warning they cannot silence is a warning they learn to ignore, and it
    would then be ignored for the root that needed it.
    """
    sdk = tmp_path / "vendor-sdk"
    sdk.mkdir()
    write(sdk / "Thing.cs", "class Thing {}\n")

    with structlog.testing.capture_logs() as logs:
        list(Walker(config(tmp_path, exclude=["**/obj/**"])).walk())

    assert not [e for e in logs if e["event"] == "root.unfiltered"]


def test_nothing_is_filtered_by_the_warning(tmp_path: Path) -> None:
    """It informs; it does not decide.

    Inventing a default exclude list for a plain folder would be the tool
    guessing what a user's directory contains.
    """
    sdk = tmp_path / "vendor-sdk"
    sdk.mkdir()
    write(sdk / "Thing.cs", "class Thing {}\n")
    write(sdk / "obj" / "Thing.g.cs", "// generated\n")

    found = repos_by_file(tmp_path)

    assert "vendor-sdk/obj/Thing.g.cs" in found


def test_it_is_said_once_however_many_units_are_unprotected(tmp_path: Path) -> None:
    """One line naming all of them, not one line each."""
    for name in ("sdk-a", "sdk-b", "sdk-c"):
        path = tmp_path / name
        path.mkdir()
        write(path / "f.cs", "class F {}\n")

    with structlog.testing.capture_logs() as logs:
        list(Walker(config(tmp_path)).walk())

    warned = [e for e in logs if e["event"] == "root.unfiltered"]
    assert len(warned) == 1
    for name in ("sdk-a", "sdk-b", "sdk-c"):
        assert name in warned[0]["units"]


def test_a_mixed_root_reports_only_the_unprotected_part(tmp_path: Path) -> None:
    """The common real shape: repositories beside a dropped-in folder."""
    repo = tmp_path / "product"
    repo.mkdir()
    write(repo / "app.py", "x = 1\n")
    git_init(repo)
    sdk = tmp_path / "vendor-sdk"
    sdk.mkdir()
    write(sdk / "Thing.cs", "class Thing {}\n")

    with structlog.testing.capture_logs() as logs:
        found = {c.rel_path for c in Walker(config(tmp_path)).walk()}

    warned = [e for e in logs if e["event"] == "root.unfiltered"]
    assert warned and "vendor-sdk" in warned[0]["units"]
    assert "product" not in warned[0]["units"]
    # Both halves are still indexed: the warning changed nothing.
    assert {"product/app.py", "vendor-sdk/Thing.cs"} <= found
