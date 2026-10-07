"""The registered MCP tool names, read out of the code that registers them.

One copy, because the whole point is that there should not be two. Three
places need this list -- the in-process server test, the stdio test, and the
reference-doc guard -- and a hand-written copy in any of them is a copy that
drifts. It already did: `impact_of` shipped while the stdio test went on
asserting the four tools that existed before it, green because CI deselects
integration tests.

Parsed statically rather than by building a server, so it stays usable from a
test that talks to a subprocess and from one that only reads files.
"""

from __future__ import annotations

import ast
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / "src" / "workspace_indexer" / "cli.py"

SERVER_FACTORY = (
    Path(__file__).resolve().parents[1] / "src" / "workspace_indexer" / "mcp" / "server_factory.py"
)


def registered_tool_names() -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(SERVER_FACTORY.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Attribute) and target.attr == "tool":
                found.append(node.name)
    assert found, f"no @server.tool() functions found in {SERVER_FACTORY.name}"
    return found


def registered_tool_parameters() -> dict[str, list[str]]:
    """Every tool's parameter names, read the same way as the names.

    Derived rather than listed, for the reason this module exists: the
    reference guard used to compare against a hand-typed tuple of six
    parameters, so a tool could gain one -- `workspace` did, on six tools at
    once -- and the guard that exists to catch undocumented parameters would
    not notice. A hand-written copy is a copy that drifts.
    """
    found: dict[str, list[str]] = {}
    for node in ast.walk(ast.parse(SERVER_FACTORY.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Attribute) and target.attr == "tool":
                arguments = (
                    *node.args.posonlyargs,
                    *node.args.args,
                    *node.args.kwonlyargs,
                )
                # posonlyargs included deliberately: omitting them would drop a
                # parameter from the dict, and the reference guard would then
                # stop verifying it -- the guard-silently-checks-less shape
                # this module exists to prevent.
                found[node.name] = [argument.arg for argument in arguments]
    assert found, f"no @server.tool() functions found in {SERVER_FACTORY.name}"
    return found


def cli_command_names() -> list[str]:
    """Typer commands, read out of the CLI module.

    Same reasoning as the tool list: the reference documents every command, and
    a hand-typed list of them in the guard means the guard stops noticing when
    one is added. `mirror` was added and undocumented within a minute of this
    being written.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(CLI.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            target = decorator.func
            if not (isinstance(target, ast.Attribute) and target.attr == "command"):
                continue
            # `@app.command("eval")` renames the command; the function is
            # `evaluate`, because `eval` is a builtin.
            named = [a for a in decorator.args if isinstance(a, ast.Constant)]
            found.append(str(named[0].value) if named else node.name)
    assert found, f"no @app.command() functions found in {CLI.name}"
    return found
