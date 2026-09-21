"""The reference must list everything that exists.

Documentation drifts silently: a new option ships, nobody adds it, and the
page quietly becomes a list of the options someone remembered. This is the
same guard that keeps `config/workspace.example.yaml` complete, applied to
`docs/reference.md` -- adding a config field or a CLI flag fails the build
until it is documented.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import structlog
from pydantic import BaseModel

from tests.mcp_tool_names import cli_command_names, registered_tool_names
from workspace_indexer.config import Settings, WorkspaceConfig

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "reference.md"


@pytest.fixture(scope="module")
def text() -> str:
    return REFERENCE.read_text(encoding="utf-8")


_EVENT = "orphans.mass_deletion_withheld"


def _collapse_whitespace(text: str) -> str:
    """Line breaks are not meaningful on either side of this comparison: the
    source concatenates the string to stay inside line length, and the doc
    wraps it to fit a code block."""
    return " ".join(text.split())


def _leaf_fields(model: type[BaseModel], prefix: str = "") -> list[str]:
    """Dotted paths of every settable option, descending into nested models."""
    found: list[str] = []
    for name, field in model.model_fields.items():
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            found.extend(_leaf_fields(annotation, f"{prefix}{name}."))
        else:
            found.append(f"{prefix}{name}")
    return found


def test_every_workspace_option_is_documented(text: str) -> None:
    missing = [
        field
        for field in _leaf_fields(WorkspaceConfig)
        # `roots` is documented as a shape rather than leaf by leaf.
        if not field.startswith("workspaces.roots") and field.split(".")[-1] not in text
    ]
    assert not missing, f"undocumented workspace.yaml options: {missing}"


def test_every_env_setting_is_documented(text: str) -> None:
    missing = [name for name in Settings.model_fields if name.upper() not in text]
    assert not missing, f"undocumented .env settings: {missing}"


def test_every_command_is_documented(text: str) -> None:
    commands = cli_command_names()
    # The name must end at a word boundary. A bare prefix check passed happily
    # for `mirrorX` when looking for `mirror`, which is the same
    # nearly-matching failure the guard exists to prevent.
    missing = [c for c in commands if not any(f"### `{c}{after}" in text for after in ("`", " "))]
    assert not missing, f"undocumented commands: {missing}"


def test_every_mcp_tool_and_its_parameters_are_documented(text: str) -> None:
    """Tool descriptions are written for the agent and only visible at
    runtime; a human reading the repo needs them here."""
    tools = registered_tool_names()
    assert len(tools) >= 5, f"expected to find the registered tools, got {tools}"
    for tool in tools:
        # Its own entry, not merely a mention. A passing reference in someone
        # else's paragraph is how a tool ends up "documented" with no
        # parameters listed -- which is what this test is for.
        assert f"**`{tool}`**" in text, tool
    for parameter in ("include_tests", "path_prefix", "rel_path", "doc_type", "repo", "limit"):
        assert parameter in text, parameter


def test_the_taxonomy_is_listed(text: str) -> None:
    from workspace_indexer.models import DocumentType

    missing = [t.value for t in DocumentType if t.value not in text]
    assert not missing, f"document types missing from the reference: {missing}"


def test_the_reference_is_linked_from_the_readme() -> None:
    """A page nobody can find is not documentation."""
    readme = (REFERENCE.parents[1] / "README.md").read_text(encoding="utf-8")
    assert "docs/reference.md" in readme


def test_the_testing_guide_is_linked_from_the_readme() -> None:
    """A page nobody can find is not documentation -- the same rule the
    reference is held to."""
    readme = (REFERENCE.parents[1] / "README.md").read_text(encoding="utf-8")
    assert "docs/testing.md" in readme


def _fenced_blocks(text: str) -> list[str]:
    """The contents of every fenced code block, CommonMark's pairing rules.

    Written as a scanner rather than a regex because the pairing is what
    matters: a fence closes only with the *same* character, at least as long as
    the one that opened it. A pattern that only knows ```` ``` ```` at column 0
    treats the inner fence of a four-backtick block -- the reason anyone
    reaches for one -- as a real fence, and every later block boundary shifts.
    Tilde fences, four or more backticks, up to three leading spaces, and a
    closer with trailing whitespace or extra length are all legal and all
    handled here.

    One known deviation, and it is the indent rule: CommonMark measures a
    fence's indent from its *container's* content column, so a fence inside a
    list item is legally four or more spaces from the margin and this scanner
    reads it as ordinary text. Following that would mean tracking container
    blocks, which is a markdown parser, not a doc guard. The cost is that
    moving the example into a list makes `_mass_deletion_example` report zero
    blocks -- so if it ever does, suspect this line before suspecting the doc.
    """
    blocks: list[str] = []
    marker: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip(" ")
        if len(line) - len(stripped) > 3:
            if marker is not None:
                body.append(line)
            continue
        if marker is None:
            opener = re.match(r"(`{3,}|~{3,})(.*)$", stripped)
            if opener and not (opener.group(1)[0] == "`" and "`" in opener.group(2)):
                marker, body = opener.group(1), []
            continue
        closer = re.match(r"(`{3,}|~{3,})[ \t]*$", stripped)
        if closer and closer.group(1)[0] == marker[0] and len(closer.group(1)) >= len(marker):
            blocks.append("\n".join(body))
            marker = None
        else:
            body.append(line)
    if marker is not None:
        blocks.append("\n".join(body))
    return blocks


def _mass_deletion_example(text: str) -> str:
    """The fenced block quoting the mass-deletion log line.

    Located by finding the fence that *contains* the event name rather than by
    searching the whole document, because both earlier versions of these guards
    took the first match anywhere: one would have latched onto an inline
    mention of the event name in prose, the other onto the first `detail='` in
    the file. Either fails against text that was never the example, pointing
    the reader at the wrong block.
    """
    matching = [b for b in _fenced_blocks(text) if _EVENT in b]
    assert len(matching) == 1, (
        f"expected exactly one fenced block quoting {_EVENT}, found {len(matching)}"
    )
    return matching[0]


def test_the_fence_scanner_pairs_fences_the_way_commonmark_does() -> None:
    """The cases that shift every later block boundary when they are missed.

    Measured against the regex this replaced: it saw the inner ``` of the
    four-backtick block as a real fence, so `two` and `three` were cut from the
    wrong lines. Each case here is one of those mis-cuts.
    """
    doc = "\n".join(
        [
            "````markdown",  # a longer fence, opened to show a fence
            "```",
            "inner",
            "```",
            "````",
            "~~~",  # a tilde fence
            "two",
            "~~~~",  # closing tilde run may be longer than the opener
            "  ```bash title=x",  # indented up to three spaces, with an info string
            "three",
            "  ```   ",  # a closer may carry trailing whitespace
        ]
    )
    assert _fenced_blocks(doc) == ["```\ninner\n```", "two", "three"]


def test_the_renderer_decides_the_quoting_not_this_test() -> None:
    """Why the expected text is rendered rather than built.

    An apostrophe in the detail sentence flips the quote character, and an
    escape is rendered escaped. Both used to be hardcoded as `detail='...'`,
    which would have failed against a doc that was faithful.
    """
    assert _rendered(detail="it isn't gone") == 'detail="it isn\'t gone"'
    assert _rendered(detail="a\tb") == "detail='a\\tb'"
    assert _rendered(detail="plain") == "detail=plain"
    # Sorted, not in the order passed -- so the slice has to start at the
    # field the renderer puts first, or the rest silently leaves the
    # expected text.
    assert _rendered(root="src", detail="gone") == "detail=gone root=src"


def _log_call() -> ast.Call:
    """The `log.error` call that emits the event, from `indexer.py`'s syntax
    tree.

    Parsed rather than pattern-matched: every regex written for this has been
    wrong about where the call's arguments end, and the tree knows exactly.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "workspace_indexer"
        / "pipeline"
        / "indexer.py"
    ).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        # The *logging* call, not merely a call carrying the event name. A
        # metrics emit or a re-log through a helper takes the same first
        # argument, and deriving the doc's expected text from one of those
        # would be wrong without being visibly wrong.
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "error":
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == _EVENT:
            return node
    raise AssertionError(f"no log.error call for {_EVENT} remains in indexer.py; update this guard")


def _emitted_detail() -> str:
    """The detail string as a *value*, adjacent literals joined and escapes
    decoded -- not the source text of the literals, which is what the doc would
    disagree with the moment the sentence contains a `\\n` or a `\\"`."""
    for keyword in _log_call().keywords:
        if keyword.arg == "detail":
            try:
                detail = ast.literal_eval(keyword.value)
            except ValueError:
                # An f-string or a named constant is a plausible next shape for
                # this sentence, and `literal_eval` answers that with a
                # traceback into its own internals. Say what happened instead.
                raise AssertionError(
                    f"the detail on {_EVENT} is no longer a plain string literal; update this guard"
                ) from None
            assert isinstance(detail, str)
            return detail
    raise AssertionError("that log call no longer passes a detail string")


def _emitted_fields() -> set[str]:
    """The keyword fields that log call passes, read from the call itself.

    Derived rather than listed, because a hardcoded list is how `run_id` came
    to be missing from the example without anything noticing. `run_id` itself
    is bound through contextvars for the whole run rather than passed here, so
    it is asserted separately.
    """
    return {keyword.arg for keyword in _log_call().keywords if keyword.arg is not None}


def _rendered(**fields: str) -> str:
    """Those fields as the console actually prints them.

    The renderer is asked rather than imitated. It quotes a value only when the
    value needs it, and picks the quote character the way `repr` does -- so a
    detail sentence containing an apostrophe renders `detail="...isn't..."`,
    and any escape renders escaped. Every one of those decisions used to be
    hardcoded here as a single-quoted string, which made a faithful doc fail.
    """
    assert fields, "nothing to render"
    line = structlog.dev.ConsoleRenderer(colors=False)(None, "", {"event": _EVENT, **fields})
    assert isinstance(line, str)
    # Sliced from the alphabetically first field, not the first one passed:
    # the renderer sorts keys, which is the very fact the reference cites to
    # explain why `detail` leads that log line. Slicing from the first keyword
    # would drop every field sorting ahead of it out of the expected text --
    # an assertion that still passes while checking less than it reads as.
    return line[line.index(min(fields) + "=") :]


def test_the_quoted_log_message_matches_what_the_code_emits(text: str) -> None:
    """The reference quotes the detail sentence exactly as it is emitted.

    Someone reaches that section *because* they saw the message, and the first
    thing they do is search for the sentence. An abridged quote finds nothing,
    which is worse than no example — and the abridgement that prompted this
    guard dropped the half carrying the remedy.

    Only the sentence is verbatim. The surrounding line is wrapped and has its
    timestamp and run id elided, which the prose beside it says.

    Whitespace-normalised on both sides: the source concatenates the string
    across lines and the doc wraps it to fit, so neither's line breaks are
    meaningful. Everything else must match.
    """
    # Rendered by the renderer and looked for in the doc, rather than
    # extracted from the doc and compared. Extraction needs a pattern that
    # knows where the quote ends, and every such pattern has been wrong: a lazy
    # one stops at an apostrophe in the prose, a greedy one runs to the block's
    # last quote -- which is the detail's closing quote only while no later
    # field happens to be quoted. Constructing the expected text by hand was no
    # better: it fixed the quote character and used the literals' source text.
    # Asking `ConsoleRenderer` assumes nothing about either.
    block = _mass_deletion_example(text)
    expected = _collapse_whitespace(_rendered(detail=_emitted_detail()))
    assert expected in _collapse_whitespace(block), (
        "the example's detail string no longer matches what indexer.py emits"
    )


def test_the_quoted_log_line_shows_every_field_the_event_carries(text: str) -> None:
    """A field missing from the example is a field the reader does not know to
    look for. `recorded` says what the share was computed against, which is the
    difference between "most of a root" and "most of what we had recorded for a
    root"; `run_id` is what ties the line to the rest of its run.
    """
    block = _mass_deletion_example(text)
    for field in _emitted_fields():
        assert f"{field}=" in block, f"the quoted log line omits {field}="
    assert "run_id=" in block, "contextvars binds run_id on every line of a run"
