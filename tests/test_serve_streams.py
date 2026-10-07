"""`serve`'s stdout carries the MCP protocol and nothing else.

That invariant is the whole reason the command reports failures the way it
does, and until now it had no evidence in CI: nothing drove `serve` through
the CLI, and the one stdio test that would catch a stray print is marked
integration and deselected. So all three stderr sites could regress silently.

`CliRunner` separates the streams, which makes the pin cheap. All three
failure sites are driven here -- the choice gate, the registry constructor and
preflight -- because the invariant is about the command, not about one handler.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from tests.isolated_env import isolate_context_env
from workspace_indexer.cli import app
from workspace_indexer.obs.logging import reset_for_tests


@pytest.fixture(autouse=True)
def _fresh_logging() -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    # `configure_logging` is idempotent, so without this the first invocation
    # in the module pins the log path for every one after it -- and the
    # preflight assertion below would read a file from another test's tmp_path.
    reset_for_tests()
    yield
    reset_for_tests()


def _two_workspace_config(tmp_path: Path) -> Path:
    for name in ("alpha", "beta"):
        (tmp_path / name).mkdir(exist_ok=True)
    (tmp_path / "state").mkdir(exist_ok=True)
    config = tmp_path / "workspace.yaml"
    config.write_text(
        "workspaces:\n"
        f"  - name: alpha\n    roots: [{{path: {tmp_path / 'alpha'}}}]\n"
        f"  - name: beta\n    roots: [{{path: {tmp_path / 'beta'}}}]\n"
        f"state_dir: {tmp_path / 'state'}\n"
        'logging:\n  console: "off"\n'
        f"  file:\n    path: {tmp_path / 'l.jsonl'}\n",
        encoding="utf-8",
    )
    return config


def _serve_log(tmp_path: Path) -> list[dict[str, Any]]:
    """The JSONL `serve` writes -- per-role, so `-serve` suffixed.

    Read from the file rather than with `capture_logs`: `configure_logging_for`
    runs inside the invocation and reconfigures structlog, which undoes a
    capture set up outside it. Same approach `test_watch_failures_are_logged`
    takes for the same reason.
    """
    path = tmp_path / "l-serve.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_an_unknown_workspace_is_reported_on_stderr_not_stdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The likeliest first contact with `--workspace`, and the site that was
    still on stdout after two rounds of moving the others.

    On stdout the diagnosis reaches the client's JSON-RPC parser rather than a
    reader, so the user gets a bare exit 2 with the reason nowhere they look.
    """
    isolate_context_env(monkeypatch, tmp_path)
    config = _two_workspace_config(tmp_path)

    result = CliRunner().invoke(app, ["serve", "--config", str(config), "--workspace", "typo"])

    assert result.exit_code == 2
    assert "typo" in result.stderr
    assert "alpha" in result.stderr and "beta" in result.stderr
    assert result.stdout == "", f"prose on the protocol channel: {result.stdout!r}"


def test_naming_no_workspace_is_not_an_error_for_serve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`serve` is the one command that serves every workspace when none is
    named -- the asymmetry section 1 of the reference now carves out. It gets
    past the choice gate and fails later, on the empty index.
    """
    isolate_context_env(monkeypatch, tmp_path)
    config = _two_workspace_config(tmp_path)

    result = CliRunner().invoke(app, ["serve", "--config", str(config)])

    assert result.exit_code == 2
    assert "Name the one you mean" not in result.stderr
    assert result.stdout == "", f"prose on the protocol channel: {result.stdout!r}"


def test_a_preflight_failure_names_the_workspace_that_caused_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The only observable output of the in-flight tracking.

    Without this, swapping the two lines that bind the name logs the
    *previous* workspace -- `None` on the first -- and dropping the update
    logs `None` always, both with the suite green. A startup failure naming
    the wrong index is a quiet cousin of the wrong-workspace answer this
    feature exists to prevent.

    The same run as the test above: both workspaces are empty, so preflight
    fails on the first one it reaches.
    """
    isolate_context_env(monkeypatch, tmp_path)
    config = _two_workspace_config(tmp_path)

    result = CliRunner().invoke(app, ["serve", "--config", str(config)])

    assert result.exit_code == 2
    failures = [e for e in _serve_log(tmp_path) if e["event"] == "serve.preflight_failed"]
    assert failures, "the preflight failure was not logged"
    assert failures[0]["workspace"] == "alpha"
    assert failures[0]["error_type"] == "EmptyIndexError"


def test_a_registry_that_cannot_be_built_also_reports_on_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The third of the three failure sites, and the one the other tests only
    ever pass through on its happy path.

    Driven by pointing the embedded store at a path that is a file, so the
    client cannot take its storage folder -- the registry constructor raises
    before any preflight runs.
    """
    isolate_context_env(monkeypatch, tmp_path)
    config = _two_workspace_config(tmp_path)
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("", encoding="utf-8")
    monkeypatch.setenv("QDRANT_PATH", str(blocked))

    result = CliRunner().invoke(app, ["serve", "--config", str(config)])

    assert result.exit_code == 2
    assert result.stderr.strip(), "the constructor failure was reported nowhere"
    assert result.stdout == "", f"prose on the protocol channel: {result.stdout!r}"
    # Which site failed, not merely that one did. Exit 2, a red line on stderr
    # and an empty stdout are exactly what the *preflight* handler produces
    # too, so without this the test would stay green as a duplicate of the
    # preflight one if the embedded client ever stopped refusing a file path
    # at construction -- and the constructor site would lose its only cover.
    #
    # Filtered rather than asserting the log is empty: `build_qdrant_client`
    # logs `store.embedded_mode` before the client call raises, so the file
    # exists and has entries in it.
    assert not [e for e in _serve_log(tmp_path) if e["event"] == "serve.preflight_failed"], (
        "preflight ran, so this is no longer the constructor site"
    )
