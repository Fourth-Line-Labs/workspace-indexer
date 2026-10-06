"""`serve`'s stdout carries the MCP protocol and nothing else.

That invariant is the whole reason the command reports failures the way it
does, and until now it had no evidence in CI: nothing drove `serve` through
the CLI, and the one stdio test that would catch a stray print is marked
integration and deselected. So all three stderr sites could regress silently.

`CliRunner` separates the streams, which makes the pin cheap.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.isolated_env import isolate_context_env
from workspace_indexer.cli import app


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
        'logging: {console: "off"}\n',
        encoding="utf-8",
    )
    return config


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
