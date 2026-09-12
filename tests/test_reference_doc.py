"""The reference must list everything that exists.

Documentation drifts silently: a new option ships, nobody adds it, and the
page quietly becomes a list of the options someone remembered. This is the
same guard that keeps `config/workspace.example.yaml` complete, applied to
`docs/reference.md` -- adding a config field or a CLI flag fails the build
until it is documented.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
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
        if not field.startswith("workspace.roots") and field.split(".")[-1] not in text
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


def _mass_deletion_example(text: str) -> str:
    """The fenced block quoting the mass-deletion log line.

    Located by finding the fence that *contains* the event name rather than by
    searching the whole document, because both earlier versions of these guards
    took the first match anywhere: one would have latched onto an inline
    mention of the event name in prose, the other onto the first `detail='` in
    the file. Either fails against text that was never the example, pointing
    the reader at the wrong block.
    """
    # Line-anchored, and the opener accepts the *whole* info string. A bare
    # ```\n opener is not enough -- this file has ```bash and ```sql blocks --
    # but neither is an alphanumeric-only tag: CommonMark allows anything but a
    # backtick there, so `c++` or a ```bash title=x``` fence would be rejected
    # as an opener while its closing fence still matched as one, shifting every
    # later pairing. Measured: with such a fence added above the example, an
    # alphanumeric-only opener sees four blocks and *none* of them contains the
    # event, because the example block is mis-cut.
    blocks = re.findall(r"^```[^\n`]*\n(.*?)^```$", text, re.M | re.S)
    matching = [b for b in blocks if _EVENT in b]
    assert len(matching) == 1, (
        f"expected exactly one fenced block quoting {_EVENT}, found {len(matching)}"
    )
    return matching[0]


def _emitted_detail() -> str:
    """The detail string as `indexer.py` actually passes it."""
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "workspace_indexer"
        / "pipeline"
        / "indexer.py"
    ).read_text(encoding="utf-8")
    call = re.search(rf'log\.error\(\s*"{re.escape(_EVENT)}",(.*?)\n        \)', source, re.S)
    assert call is not None, f"the {_EVENT} log call moved; update this guard"
    # Capture the run of adjacent string literals Python concatenates, rather
    # than anchoring on what follows. An earlier version matched `",\n` and
    # could not work: `detail` is the last argument, so the captured body ends
    # at the comma with no newline after it.
    literals = re.search(r'detail=((?:\s*"(?:[^"\\]|\\.)*")+)', call.group(1), re.S)
    assert literals is not None, "that log call no longer passes a detail string"
    return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', literals.group(1), re.S))


def _emitted_fields() -> set[str]:
    """The keyword fields that log call passes, read from the call itself.

    Derived rather than listed, because a hardcoded list is how `run_id` came
    to be missing from the example without anything noticing. `run_id` itself
    is bound through contextvars for the whole run rather than passed here, so
    it is asserted separately.
    """
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "workspace_indexer"
        / "pipeline"
        / "indexer.py"
    ).read_text(encoding="utf-8")
    call = re.search(rf'log\.error\(\s*"{re.escape(_EVENT)}",(.*?)\n        \)', source, re.S)
    assert call is not None
    return {name for name in re.findall(r"^\s{12}(\w+)=", call.group(1), re.M)}


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
    # Built from the source and looked for in the doc, rather than extracted
    # from the doc and compared. Extraction needs a pattern that knows where
    # the quote ends, and every such pattern has been wrong: a lazy one stops
    # at an apostrophe in the prose, a greedy one runs to the block's last
    # quote -- which is the detail's closing quote only while no later field
    # happens to be quoted, and `ConsoleRenderer` quotes any value containing a
    # space. Constructing the expected text depends on none of that.
    block = _mass_deletion_example(text)
    expected = _collapse_whitespace(f"detail='{_emitted_detail()}'")
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
