"""Which namespaces a file declares.

The other half of C# import resolution. `using MyApp.Data` names no path at
all -- there is no convention tying a namespace to a directory that can be
relied on, and the one that looks reliable (namespace mirrors folder) is a
style, not a rule. So the namespace is read from the declaration instead of
inferred from where the file sits.

Deliberately only C#. Python and the JS family resolve by path and need
nothing from here; adding a language means adding its declaration node types
and nothing else.
"""

from __future__ import annotations

import tree_sitter_language_pack as tslp
from tree_sitter import Node

from workspace_indexer.graph.namespace_declaration import NamespaceDeclaration
from workspace_indexer.obs.logging import get_logger

log = get_logger("workspace_indexer.graph.namespaces")

SUPPORTED = frozenset({"csharp"})

# Both spellings. The block form nests, the file-scoped form cannot.
_DECLARATIONS = frozenset({"namespace_declaration", "file_scoped_namespace_declaration"})


class NamespaceScanner:
    def scan(self, text: str, language: str) -> list[NamespaceDeclaration]:
        if language not in SUPPORTED or not text:
            return []
        try:
            tree = tslp.get_parser(language).parse(text.encode())  # pyright: ignore[reportArgumentType]
        except Exception as exc:
            # Same rule as the import scanner: a grammar cache miss with no
            # network costs the edges, never the file.
            log.debug("namespaces.parse_failed", language=language, error=str(exc))
            return []

        found: list[NamespaceDeclaration] = []
        _walk(tree.root_node, prefix="", out=found)
        return found


def _walk(node: Node, *, prefix: str, out: list[NamespaceDeclaration]) -> None:
    """Descend, carrying the enclosing namespace as a prefix.

    `namespace Outer { namespace Inner { } }` declares `Outer.Inner`, which is
    what a `using` spells -- `using Inner` reaches nothing. Recording the inner
    name alone would resolve the wrong edges and fail to resolve the right one,
    both silently.
    """
    inner = prefix
    if node.type in _DECLARATIONS:
        name = _name_of(node)
        if name:
            inner = f"{prefix}.{name}" if prefix else name
            out.append(
                NamespaceDeclaration(symbol=inner, kind="namespace", line=node.start_point[0] + 1)
            )
    for child in node.children:
        _walk(child, prefix=inner, out=out)


def _name_of(node: Node) -> str:
    """The declared name, by field first and by shape second.

    `name` is the field in both forms, but falling back to the first qualified
    name or identifier keeps this working against a grammar that renames the
    field -- the failure would otherwise be zero declarations extracted, which
    reads exactly like a repository that declares none.
    """
    named = node.child_by_field_name("name")
    if named is None:
        named = next((c for c in node.children if c.type in ("qualified_name", "identifier")), None)
    if named is None or named.text is None:
        return ""
    return named.text.decode("utf-8", errors="replace").strip()
