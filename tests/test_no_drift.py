"""Lists that exist in more than one place must agree.

Every entry here is a real failure this project has already had, generalised.
The pattern is always the same: something is enumerated in code and enumerated
again somewhere else -- a doc, a test, a second backend's index definition --
and the copy drifts. Nothing errors; the second list simply describes a system
that no longer exists.

Three of these were found the expensive way. `impact_of` shipped and the stdio
test kept asserting the four tools that existed before it, green because CI
deselects integration tests. The reference doc's tool list was hand-typed and
passed happily while a tool went undocumented. A filter field declared in one
index and not the other produces a search that silently ignores the filter.

The rule this file encodes: if a list appears twice, one of them is derived, or
a test proves they match.
"""

from __future__ import annotations

from pathlib import Path

from tests.mcp_tool_names import registered_tool_names
from workspace_indexer.config import Settings, WorkspaceConfig
from workspace_indexer.models import DocumentType
from workspace_indexer.storage.mongo_filter import exact_terms
from workspace_indexer.storage.mongo_index_spec import FILTER_FIELDS
from workspace_indexer.storage.payload import INDEXED_FIELDS

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "workspace_indexer"


# ---- the MCP surface --------------------------------------------------------


def test_no_test_hardcodes_the_tool_list() -> None:
    """The failure that shipped: a test asserting four tools while five exist.

    Any test naming every tool must derive the list. Grepping for the literal
    set is crude and exactly right -- the thing being banned is a literal.
    """
    offenders: list[str] = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        # Asking the server what tools it has, and then checking the answer
        # against something written by hand, is the precise shape of the bug.
        # A test naming one or two tools is testing those tools; a test calling
        # list_tools is enumerating the surface, and must derive the list.
        if "list_tools()" in text and "registered_tool_names" not in text:
            offenders.append(path.name)
    assert not offenders, (
        "these tests enumerate the server's tools and check the result against a "
        "hand-written list; call registered_tool_names() instead: " + ", ".join(offenders)
    )


def test_every_tool_is_named_in_the_server_instructions() -> None:
    """The instructions block is the only description of the tool set an agent
    sees before choosing one. A tool missing from it is a tool that exists and
    is never called."""
    source = (SRC / "mcp" / "server_factory.py").read_text(encoding="utf-8")
    instructions = source.split('_INSTRUCTIONS = f"""')[1].split('"""')[0]
    missing = [name for name in registered_tool_names() if name not in instructions]
    assert not missing, f"tools absent from the server instructions: {missing}"


def test_every_tool_has_an_entry_in_the_reference() -> None:
    reference = (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
    missing = [name for name in registered_tool_names() if f"**`{name}`**" not in reference]
    assert not missing, f"tools with no entry in docs/reference.md: {missing}"


# ---- configuration ----------------------------------------------------------


def test_every_env_setting_appears_in_the_example() -> None:
    """`.env.example` is what a stranger copies. A setting missing from it is a
    setting nobody outside this repo knows exists."""
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    missing = [name for name in Settings.model_fields if name.upper() not in example]
    assert not missing, f"settings missing from .env.example: {missing}"


def test_every_document_type_is_in_the_reference() -> None:
    reference = (ROOT / "docs" / "reference.md").read_text(encoding="utf-8")
    missing = [t.value for t in DocumentType if t.value not in reference]
    assert not missing, f"document types missing from the reference: {missing}"


def test_the_workspace_schema_and_its_example_agree() -> None:
    example = (ROOT / "config" / "workspace.example.yaml").read_text(encoding="utf-8")
    missing = [
        name
        for name in WorkspaceConfig.model_fields
        # `roots` is documented as a shape rather than field by field.
        if name != "roots" and name not in example
    ]
    assert not missing, f"top-level workspace.yaml sections missing from the example: {missing}"


# ---- the two backends -------------------------------------------------------


def test_both_backends_can_filter_on_the_same_fields() -> None:
    """A field indexed for one store and not the other is a filter that works
    on Qdrant and is silently ignored on Atlas -- or errors there, which is the
    lucky case."""
    qdrant = set(INDEXED_FIELDS)
    mongo = set(FILTER_FIELDS)
    only_qdrant = sorted(qdrant - mongo)
    assert not only_qdrant, (
        f"indexed for Qdrant but not declared filterable on Atlas: {only_qdrant}"
    )


def test_every_field_the_translator_produces_is_declared() -> None:
    """The filter translator and the index definition are two places one list
    has to be right in. Atlas refuses to filter on an undeclared field."""
    from workspace_indexer.models import FileKind, SearchFilters

    every = SearchFilters(
        root_label="r",
        unit="u",
        repo_name="repo",
        language="python",
        kind=FileKind.CODE,
        path_prefix="src",
        symbol_kind="function",
        doc_type=DocumentType.NORMATIVE,
    )
    undeclared = sorted(set(exact_terms(every)) - set(FILTER_FIELDS))
    assert not undeclared, f"filter terms no Atlas index declares: {undeclared}"


def test_the_payload_writer_and_the_qdrant_indexes_agree() -> None:
    """An indexed field the payload never writes is an index over nothing, and
    a filter on it matches nothing at all."""
    written = (SRC / "storage" / "payload.py").read_text(encoding="utf-8")
    body = written.split("def to_payload(")[1].split("\ndef ")[0]
    missing = [field for field in INDEXED_FIELDS if f'"{field}"' not in body]
    assert not missing, f"indexed but never written into the payload: {missing}"


def test_both_factories_agree_on_what_database_reranking_means() -> None:
    """Two lists decide this, and they must decide the same way.

    The reranker factory returns a no-op for these providers because the store
    reranks; the store factory builds the store's reranker for them. If the
    lists diverged, one side would decline to rerank and the other would too,
    and every search would quietly return fusion order while the configuration
    said `database:`.
    """
    from workspace_indexer.rerank.reranker_factory import (
        DATABASE_PROVIDERS as rerank_side,
    )
    from workspace_indexer.storage.store_factory import DATABASE_PROVIDERS as store_side

    assert rerank_side == store_side


def test_every_package_is_named_in_the_layer_list() -> None:
    """CLAUDE.md's layer list is the map a newcomer reads first.

    The failure it already had: eleven of nineteen packages were missing,
    including two added weeks earlier. A list covering half the packages is
    worse than no list, because the omissions read as "these do not exist"
    rather than "nobody updated this".

    It drifted for a findable reason. Every other list in this repository is
    either derived or guarded, and `docs/reference.md` stayed correct through
    the same period *because* a test enforced it. This is that test, pointed at
    the other file.
    """
    documented = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    packages = sorted(
        path.name
        for path in SRC.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file() and path.name != "__pycache__"
    )
    missing = [name for name in packages if f"`{name}/`" not in documented]
    assert not missing, (
        f"packages with no entry in CLAUDE.md's layer list: {missing}. "
        "Add a line saying what seam each one owns, or the map stops matching "
        "the territory."
    )


def test_the_layer_list_names_no_package_that_is_gone() -> None:
    """The other direction, which is the quieter half.

    A deleted package leaves a line describing a seam that no longer exists,
    and a reader trusts it precisely because the rest of the list is accurate.
    """
    import re

    documented = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^- `([a-z_]+)/`", documented, re.MULTILINE))
    real = {
        path.name for path in SRC.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
    }
    stale = sorted(listed - real)
    assert not stale, f"CLAUDE.md describes packages that no longer exist: {stale}"


def test_ci_installs_every_extra_so_optional_tests_actually_run() -> None:
    """Tests behind an optional dependency skip silently without it.

    `test_pdf_chunker.py` calls `importorskip("pymupdf")`, which is the right
    behaviour for a contributor who did not install the extra and the wrong
    outcome entirely in CI: a green run that skipped the whole feature looks
    identical to one that verified it. Same shape as the integration tests that
    rotted unnoticed.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--all-extras" in workflow, (
        "CI must install every extra, or tests guarded by importorskip are "
        "skipped without anything saying so"
    )
