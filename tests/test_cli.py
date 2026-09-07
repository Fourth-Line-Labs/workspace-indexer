"""The CLI surface.

Driven through typer's runner against a real workspace, real embedded Qdrant
and a real manifest, with only the paid embedding provider swapped out. What is
being checked is the contract a person sees: exit codes, error messages that
say what to do, and commands that do not lie about what they did.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from workspace_indexer.cli import app

runner = CliRunner()


def plain(text: str) -> str:
    """Rich hard-wraps to the terminal width, so a message can be split across
    lines mid-sentence. Assertions care about the words, not the wrapping."""
    return " ".join(text.split())


@pytest.fixture
def project(workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A working directory with a config and .env pointing at the fixture."""
    home = tmp_path / "project"
    (home / "config").mkdir(parents=True)
    (home / "config" / "workspace.yaml").write_text(
        yaml.safe_dump(
            {
                "workspace": {
                    "name": "test",
                    "roots": [{"path": str(workspace), "recurse_into_children": True}],
                },
                "index": {"exclude": ["**/node_modules/**"]},
                "logging": {"console": "off", "file": None},
                "search": {"rerank": {"enabled": False, "model": "fastembed:x"}},
            }
        ),
        encoding="utf-8",
    )
    # A local model keeps the whole CLI path free of API keys and network.
    (home / ".env").write_text(
        "EMBEDDING_MODEL=fastembed:BAAI/bge-small-en-v1.5\n"
        "EMBEDDING_DIMENSIONS=384\n"
        "QDRANT_MODE=embedded\n"
        f"QDRANT_PATH={home / 'data' / 'qdrant'}\n"
        f"STATE_DB={home / 'data' / 'manifest.sqlite3'}\n"
        "RERANK_ENABLED=false\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(home)
    yield home


def test_help_lists_every_command() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("index", "search", "status", "explain", "reproject", "eval"):
        assert command in plain(result.stdout)


def test_missing_config_exits_cleanly_with_advice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config problem is a user problem, not a traceback."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 2
    assert "workspace.example.yaml" in plain(result.stdout)
    assert "Traceback" not in plain(result.stdout)


@pytest.mark.integration
def test_dry_run_reports_a_plan_without_storing_anything(project: Path) -> None:
    from workspace_indexer.state import Manifest

    result = runner.invoke(app, ["index", "--dry-run"])
    assert result.exit_code == 0, result.stdout
    assert "dry-run" in plain(result.stdout)
    assert "No API calls were made" in plain(result.stdout)
    # The manifest file is created by wiring up the context, but a dry run must
    # leave it empty and record no run.
    with Manifest(project / "data" / "manifest.sqlite3") as manifest:
        assert manifest.file_count() == 0
        assert manifest.chunk_count() == 0
        assert manifest.recent_runs() == []


@pytest.mark.integration
def test_index_then_search_finds_something(project: Path) -> None:
    indexed = runner.invoke(app, ["index"])
    assert indexed.exit_code == 0, indexed.stdout
    assert "chunks written" in plain(indexed.stdout)

    found = runner.invoke(app, ["search", "how does the widget render itself", "-n", "3"])
    assert found.exit_code == 0, found.stdout
    assert ".py" in found.stdout or ".md" in found.stdout


@pytest.mark.integration
def test_second_index_run_reports_everything_unchanged(project: Path) -> None:
    runner.invoke(app, ["index"])
    again = runner.invoke(app, ["index"])
    assert again.exit_code == 0
    assert "unchanged" in plain(again.stdout)


@pytest.mark.integration
def test_status_reports_roots_spaces_and_runs(project: Path) -> None:
    runner.invoke(app, ["index"])
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    assert "files by root" in plain(result.stdout)
    assert "embedding spaces" in plain(result.stdout)
    assert "recent runs" in plain(result.stdout)


@pytest.mark.integration
def test_status_reports_import_origins_with_nested_denominators(project: Path) -> None:
    """The gate has to be visible, not just stored.

    Asserted through the command rather than by calling the formatter, because
    what matters is that a reader of `status` can see which denominator a
    percentage is against.
    """
    runner.invoke(app, ["index"])
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.stdout
    out = plain(result.stdout)
    assert "import resolution by origin" in out
    assert "first-party" in out
    assert "non-framework" in out
    # An index that has just run must not report edges as unclassified-by-
    # omission; classification is part of the run, not a later step.
    assert "no origin recorded" not in out


@pytest.mark.integration
def test_explain_shows_chunks_for_one_file(project: Path, workspace: Path) -> None:
    target = workspace / "repo_one" / "src" / "widget.py"
    result = runner.invoke(app, ["explain", str(target)])
    assert result.exit_code == 0, result.stdout
    assert "chunker" in plain(result.stdout)
    assert "tokens" in plain(result.stdout)


@pytest.mark.integration
def test_explain_rejects_a_path_outside_the_workspace(project: Path, tmp_path: Path) -> None:
    stray = tmp_path / "stray.py"
    stray.write_text("x = 1\n", encoding="utf-8")
    result = runner.invoke(app, ["explain", str(stray)])
    assert result.exit_code == 1
    assert "not inside any configured root" in plain(result.stdout)


@pytest.mark.integration
def test_search_with_no_matches_says_so(project: Path) -> None:
    runner.invoke(app, ["index"])
    result = runner.invoke(app, ["search", "anything", "--unit", "does-not-exist"])
    assert result.exit_code == 0
    assert "No matches" in plain(result.stdout)


@pytest.mark.integration
def test_eval_without_a_dataset_explains_what_to_write(project: Path) -> None:
    runner.invoke(app, ["index"])
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 2
    assert "real queries" in plain(result.stdout)


@pytest.mark.integration
def test_eval_scores_a_dataset(project: Path) -> None:
    runner.invoke(app, ["index"])
    (project / "config" / "eval.yaml").write_text(
        "- query: how does the widget render\n  expect: [widget.py]\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", "--dataset", "config/eval.yaml"])
    assert result.exit_code == 0, result.stdout
    assert "recall@" in plain(result.stdout)
    assert "MRR@" in plain(result.stdout)


@pytest.mark.integration
def test_reproject_creates_a_narrower_collection(project: Path) -> None:
    runner.invoke(app, ["index"])
    result = runner.invoke(app, ["reproject", "--dimensions", "128"])
    assert result.exit_code == 0, result.stdout
    assert "_128" in plain(result.stdout)


def test_watch_without_the_extra_says_which_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """watchfiles is optional so a `serve`-only install does not pull a Rust
    binary it will never run. A bare ModuleNotFoundError does not say that."""
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("watchfiles"):
            raise ImportError("No module named 'watchfiles'")
        return real_import(name, *args, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

    monkeypatch.delitem(sys.modules, "workspace_indexer.watching", raising=False)
    monkeypatch.delitem(sys.modules, "workspace_indexer.watching.watcher", raising=False)
    monkeypatch.delitem(sys.modules, "watchfiles", raising=False)
    monkeypatch.setattr(builtins, "__import__", blocked)

    result = CliRunner().invoke(app, ["watch"])

    assert result.exit_code == 2
    assert "watch" in result.output


def test_asymbol_label_does_not_repeat_the_kind() -> None:
    """A PDF chunk is kind "page" with path "page 3", which rendered as
    "page page 3" -- careless enough to make a reader distrust the numbers
    printed beside it."""
    from workspace_indexer.cli import symbol_label

    assert symbol_label("page", "page 3") == "page 3"
    assert symbol_label("heading", "Setup > WSL") == "heading Setup > WSL"
    assert symbol_label(None, None) == "-"
    assert symbol_label("function", None) == "function"
